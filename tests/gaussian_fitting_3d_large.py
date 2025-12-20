#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state, get_expon_lr_func, safe_interact
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
from contextlib import contextmanager
from PIL import Image

from functools import partial
from solver.gaussian_model_vector import GaussianModelVector
from solver.training_loss import scalar_training_loss
from solver.batch_training_loss import batch_training_loss
from solver.training_loss_hessian import scalar_training_loss_hessian
from solver.reference_training_loss import reference_training_loss
from solver.conjugate_gradient import conjugate_gradient
from solver.diagonal_estimator import restarted_squared_hutchinson, restarted_hutchinson
from solver.solver_functions import construct_loss_func, construct_g_func, construct_JTJv_func, dot, saxpy, construct_Dhat_func
from solver.adam_optimizer import AdamOptimizer
from solver.sophia_optimizer import SophiaOptimizer
from solver.KL_clip import clip_kl

from utils.gif_renderer import GifRenderer

from copy import deepcopy

from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("Debug disabling FusedSSIM")
    FUSED_SSIM_AVAILABLE = False
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

@contextmanager
def temp_seed(seed):
    """
    Context manager to temporarily set a seed for reproducibility.
    """
    np_state = np.random.get_state()
    np.random.seed(seed)
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

def init_invisible_gaussians(num_points, sh_degree, opt):
    gaussians = GaussianModel(sh_degree, opt.optimizer_type)

    bd = 2
    xyz = bd * (torch.rand(num_points, 3, device="cuda") - 0.5)
    scales = torch.rand(num_points, 3, device="cuda")
    d = 3
    features_dc = torch.rand(num_points, 1, d, device="cuda")
    features_rest = torch.zeros(num_points, 15, 3, device="cuda")
    u = torch.rand(num_points, 1, device="cuda")
    v = torch.rand(num_points, 1, device="cuda")
    w = torch.rand(num_points, 1, device="cuda")
    quats = torch.cat(
        [torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
         torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
         torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
         torch.sqrt(u) * torch.cos(2.0 * math.pi * w), ], -1,)
    opacities = torch.ones((num_points, 1), device="cuda")

    gaussians._xyz = nn.Parameter(xyz.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(features_dc.requires_grad_(True))
    gaussians._features_rest = nn.Parameter(features_rest.requires_grad_(True))
    gaussians._scaling = nn.Parameter(scales.requires_grad_(True))
    gaussians._rotation = nn.Parameter(quats.requires_grad_(True))
    gaussians._opacity = nn.Parameter(opacities.requires_grad_(True))
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    gaussians.exposure_mapping = {"image": 0}
    gaussians.pretrained_exposures = None
    exposure = torch.eye(3, 4, device="cuda")[None].repeat(1, 1, 1)
    gaussians._exposure = nn.Parameter(exposure.requires_grad_(True))
    gaussians.training_setup(opt)

    gaussians._opacity.data -= 100000.0  # Init invisible gaussians

    return gaussians

def add_gaussian(gaussians, xyz, scaling, rotation, feature_dc, features_rest, opacity, idx=0):
    gaussians._xyz.data[idx] = xyz
    gaussians._scaling.data[idx] = scaling
    gaussians._rotation.data[idx] = rotation
    gaussians._features_dc.data[idx] = feature_dc
    gaussians._features_rest.data[idx] = features_rest
    gaussians._opacity.data[idx] = opacity
    return idx + 1

def build_camera(image_torch, viewmat):

    image_np = (image_torch.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_pil = Image.fromarray(image_np)
    W, H = image_pil.size
    fx = math.pi / 2.0
    FoVx = 0.5 * float(W) / math.tan(0.5 * fx)
    FoVy = FoVx * H / W
    camera = Camera(resolution=(W, H), 
                    colmap_id=0, 
                    R=viewmat[:3, :3], T=viewmat[:3, 3], 
                    FoVx=FoVx, FoVy=FoVy, 
                    depth_params=None, 
                    image=image_pil, 
                    invdepthmap=None, 
                    image_name="image", 
                    uid=0, 
                    data_device="cuda",)
    return camera

import numpy as np

def six_axis_view_matrices(O, r):
    """
    Args:
        O : (3,) array, look-at point in world coordinates
        r : distance from O

    Returns:
        (6, 4, 4) array of view matrices (world -> camera)
    """
    O = np.asarray(O, dtype=np.float64)
    views = []

    offsets = [
        np.array([ 1.0, 0.0, 0.0]),   # +X
        np.array([-1.0, 0.0, 0.0]),   # -X
        np.array([ 0.0, 1.0, 0.0]),   # +Y
        np.array([ 0.0,-1.0, 0.0]),   # -Y
        np.array([ 0.0, 0.0, 1.0]),   # +Z
        np.array([ 0.0, 0.0,-1.0]),   # -Z
    ]

    for off in offsets:
        forward = off

        C = O - r * off   # camera center

        if forward[1] == 0.0:
            up = np.array([0, 1, 0])
        elif forward[1] > 0.0:
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 0, -1])
        right = np.cross(forward, up)

        # view matrix: world -> camera
        V = np.eye(4)
        V[:3, :3] = np.stack([right / np.linalg.norm(right),
                             up / np.linalg.norm(up),
                             forward / np.linalg.norm(forward)], axis=0)
        V[:3, 3] = -V[:3, :3] @ C

        views.append(V)

    return np.stack(views)

import numpy as np

# def six_axis_view_matrices(r):
#     O = np.zeros(3)
#     views = []
# 
#     positions = [
#         np.array([ r, 0, 0]),   # +X
#         np.array([-r, 0, 0]),   # -X
#         np.array([ 0, r, 0]),   # +Y
#         np.array([ 0,-r, 0]),   # -Y
#         np.array([ 0, 0, r]),   # +Z
#         np.array([ 0, 0,-r]),   # -Z
#     ]
# 
#     for C in positions:
#         # camera forward (+Z looks toward origin)
#         z = (O - C)
#         z = z / np.linalg.norm(z)
# 
#         # choose a non-degenerate up
#         if abs(z[1]) > 0.99:
#             up = np.array([0, 0, 1])
#         else:
#             up = np.array([0, 1, 0])
# 
#         x = np.cross(up, z)
#         x = x / np.linalg.norm(x)
# 
#         y = np.cross(z, x)
# 
#         # world -> camera (view matrix)
#         V = np.eye(4)
#         V[:3, :3] = np.stack([x, y, z], axis=0)
#         V[:3, 3] = -V[:3, :3] @ C
# 
#         views.append(V)
# 
#     return np.stack(views)


class ExponentialLRScheduler:
    def __init__(self, init_lr, final_lr, max_iter):
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.max_iter = max_iter

        # compute decay factor γ such that:
        # final_lr = init_lr * γ^(max_iter)
        self.gamma = (final_lr / init_lr) ** (1.0 / max_iter)

    def get_lr(self, current_iter):
        # Clamp to [0, max_iter]
        current_iter = max(0, min(current_iter, self.max_iter))
        return self.init_lr * (self.gamma ** current_iter)

def run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init,
                  pipe, cameras, primary_cams, background, lr,
                  num_images_per_iter, gif_interval,
                  loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                  name=""):
    adam_optimizer = AdamOptimizer(lr=lr, betas=(0.9, 0.999), eps=1e-15, clip=False)
    sophia_optimizer = SophiaOptimizer(lr=lr, betas=(0.9, 0.99), eps=1e-15, clip=True,
                                       diagonal_update_interval=5,)

    num_cameras = len(cameras)
    num_val_cameras = len(primary_cams)

    with torch.no_grad():

        adam_optimizer.reset()

        adam_losses = [[] for _ in range(num_val_cameras)]
        adam_images = [[] for _ in range(num_val_cameras)]

        gaussians = gaussians_init.clone()
        for iteration in range(NUM_ITERATIONS):
            loss_adam_sum = 0.0
            for i in range(num_val_cameras):
                loss_adam, batch_stats = loss_func(gaussians=gaussians, 
                                                   viewpoint_cams=[primary_cams[i]], 
                                                   return_stats=True)
                loss_adam_sum += loss_adam
                if iteration % gif_interval == 0:
                    adam_losses[i].append(loss_adam.item())
                    adam_images[i].append(batch_stats[0]["images"][0])
            print(f"Adam loss: {loss_adam_sum.item():.10f}")

            rand_camera_indices = np.random.choice(range(num_cameras), size=num_images_per_iter, replace=False)
            viewpoint_batch = [cameras[i] for i in rand_camera_indices]
            g = g_func(gaussians=gaussians, viewpoint_cams=viewpoint_batch)
            s_adam = adam_optimizer.get_update(g)

            gaussians.update_step(s_adam)

        sophia_optimizer.reset()
        sophia_optimizer.set_clip(False)

        sophia_tr_losses = [[] for _ in range(num_val_cameras)]
        sophia_tr_images = [[] for _ in range(num_val_cameras)]

        gaussians = gaussians_init.clone()

        for iteration in range(NUM_ITERATIONS):
            rand_camera_indices = np.random.choice(range(num_cameras), size=num_images_per_iter, replace=False)
            viewpoint_batch = [cameras[i] for i in rand_camera_indices]
            JTJv_func1 = partial(JTJv_func, gaussians=gaussians, 
                                 viewpoint_cams=viewpoint_batch, 
                                 S=S, scale=1)
            Dhat_func1 = partial(Dhat_func, gaussians=gaussians, 
                                 viewpoint_cams=viewpoint_batch)

            loss_sophia_tr_sum = 0.0
            for i in range(num_val_cameras):
                loss_sophia_tr, batch_stats = loss_func(gaussians=gaussians, 
                                                        viewpoint_cams=[primary_cams[i]],
                                                        return_stats=True)
                loss_sophia_tr_sum += loss_sophia_tr
                if iteration % gif_interval == 0:
                    sophia_tr_losses[i].append(loss_sophia_tr.item())
                    sophia_tr_images[i].append(batch_stats[0]["images"][0])
            print(f"Sophia TR loss: {loss_sophia_tr_sum.item():.10f}")

            g = g_func(gaussians=gaussians, viewpoint_cams=viewpoint_batch)
            # safe_interact(local=locals(), banner="Before Sophia TR step")

            s_sophia_tr = sophia_optimizer.get_update(g, JTJv_func1, Dhat_func1, z_gen_func, S)
            s_sophia_tr_old = s_sophia_tr.clone()

            s_sophia_tr = clip_kl(gaussians, s_sophia_tr, kl_threshold,
                                  lr.features_dc, lr.features_rest, lr.opacity)

            gaussians.update_step(s_sophia_tr)


    gif_renderer = GifRenderer(num_rows=num_val_cameras, num_cols=3, figsize=(15, 10))
    for i in range(num_val_cameras):
        gif_renderer.add_gt(i, 0, cameras[i].original_image)
        gif_renderer.add_series(i, 1, adam_images[i], adam_losses[i], title="Adam")
        # gif_renderer.add_series(i, 2, sophia_images[i], sophia_losses[i], title="Sophia")
        gif_renderer.add_series(i, 2, sophia_tr_images[i], sophia_tr_losses[i], title="Sophia TR")

    print("Saving GIF...")
    gif_renderer.animate(f"figures/gaussian_fitting_3d_{name}.gif", interval=500)
    print(f"Saved figures/gaussian_fitting_3d_{name}.gif")

def training(opt, pipe):
    

    ####### Some fixed parameters #########
    num_images = 1
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    model_path = ""
    NUM_ITERATIONS = 100
    gif_interval = 2
    H, W = 200, 240
    max_num_points = 20
    kl_threshold = 0.1
    ####### Some fixed parameters #########

    first_iter = 0
    sh_degree = 3
    tb_writer = None

    """
    1. Generate random cameras looking at the origin
    2. Initialize some GT gaussians around the origin
    3. Render GT gaussians and build cameras with GT images
    """

    O1 = np.array([0, 0, 0])
    viewmats1 = six_axis_view_matrices(O=O1, r=20)
    viewmats = viewmats1

    image_zeros = torch.zeros((3, H, W), dtype=torch.float32, device="cuda")
    gaussians_gt = init_invisible_gaussians(max_num_points, sh_degree, opt)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    render_args = {"iteration": 0,
                   "opt": opt,
                   "pipe": pipe,
                   "bg": background,
                   "train_test_exp": train_test_exp,
                   "depth_l1_weight": depth_l1_weight,
                   "loss_type": opt.loss_type,
                   "huber_delta": opt.huber_delta,
                   "disable_ssim": opt.disable_ssim,
                   "batch_size": 1,
                   "pixel_mask": None,}
    loss_func = construct_loss_func(**render_args)
    g_func = construct_g_func(**render_args)
    JTJv_func = construct_JTJv_func(**render_args)
    Dhat_func = construct_Dhat_func(**render_args)
    z_gen_func = partial(GaussianModelVector.rademacher_like, gaussians_gt)

    xyz_lr = opt.xyz_lr_init
    xyz_lr_scheduler = ExponentialLRScheduler(opt.xyz_lr_init, opt.xyz_lr_final, opt.xyz_lr_max_steps)
    xyz_lr = xyz_lr_scheduler.get_lr(first_iter)
    lr = GaussianModelVector(xyz=xyz_lr,  
                             features_dc=opt.features_dc_lr, 
                             features_rest=opt.features_rest_lr,
                             scaling=opt.scaling_lr,
                             rotation=opt.rotation_lr,
                             opacity=opt.opacity_lr,
                             exposure=opt.exposure_lr,
                             gaussians=gaussians_gt) * 10

    S = GaussianModelVector(xyz=1.0, 
                            features_dc=1.0,
                            features_rest=1.0,
                            scaling=1.0,
                            rotation=1.0,
                            opacity=1.0,
                            exposure=1.0,
                            gaussians=gaussians_gt)
    S = (1 / S).sqrt()

    """
    For each test, do
        0) Initialize subplots
        1) Initialize some GT gaussians
        2) Render GT gaussians and generate an image
        3) Use the image to build camera with GT image
        4) Initialize some gaussians from the GT gaussians with some variations
        5) Reset optimizers
        6) For each optimizer:
            a) Run training for N iterations
            b) Save each image and loss value per iteration
        7) Save GIF
    """


    with torch.no_grad():
        if True:
            ############## Test 1: Short-side shift ##############

            gaussians_gt._opacity.data -= 100000.0  # Init invisible gaussians

            idx = 0
            for _ in range(max_num_points):
                offset = torch.randn(3, device="cuda") * 2.0
                scaling = torch.randn(3, device="cuda") * 0.5 - 0.5
                rotation = torch.randn(4, device="cuda")
                feature_dc = torch.rand(1, 3, device="cuda") * 2.0
                features_rest = 0.02 * torch.ones((1, 15, 3), device="cuda")
                opacity = torch.randn(1, device="cuda") + 1.0

                idx = add_gaussian(gaussians_gt,
                                   xyz=torch.tensor(O1, device="cuda") + offset,
                                   scaling=scaling,
                                   rotation=rotation / torch.norm(rotation),
                                   feature_dc=feature_dc,
                                   features_rest=features_rest,
                                   opacity=opacity.unsqueeze(0),
                                   idx=idx)

            gaussians_init = gaussians_gt.clone()
            for idx in range(max_num_points):
                offset_noise = torch.randn(3, device="cuda") * 0.5
                scaling_noise = torch.randn(3, device="cuda") * 0.1
                rotation_noise = torch.randn(4, device="cuda") * 0.5
                gaussians_init._xyz.data[idx] += offset_noise
                gaussians_init._scaling.data[idx] += scaling_noise
                gaussians_init._rotation.data[idx] += rotation_noise


            cameras = []
            for i, viewmat in enumerate(viewmats):
                image_gt_torch = render(viewpoint_camera=build_camera(image_zeros, viewmat), pc=gaussians_gt, pipe=pipe, bg_color=background)["render"]
                cameras.append(build_camera(image_gt_torch, viewmat))

            primary_cams = cameras[:6]

            run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init, 
                          pipe, cameras, primary_cams, background, lr,
                          num_images, gif_interval,
                          loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                          name=f"large")

        if False:
            ############## Test 2: Long-side shift ##############
            gif_renderer = GifRenderer(num_rows=2, num_cols=2, figsize=(6, 6))

            gaussians_gt._opacity.data -= 100000.0  # Init invisible gaussians
            idx = add_gaussian(gaussians_gt,
                               xyz=torch.tensor([-0.066, 0.21, -0.63], device="cuda"),
                               scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[2.5, 0.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=0)

            gaussians_init = gaussians_gt.clone()
            gaussians_init._xyz.data[0, 0] -= 0.9

            run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init,
                          pipe, cameras, background, lr,
                          loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                          gif_renderer, name="longside")

            torch.cuda.empty_cache()

        if False:
            ############## Test 3: Rotation ##############
            gif_renderer = GifRenderer(num_rows=2, num_cols=2, figsize=(6, 6))

            gaussians_gt._opacity.data -= 100000.0  # Init invisible gaussians
            idx = add_gaussian(gaussians_gt,
                               xyz=torch.tensor([-0.066, 0.21, -0.63], device="cuda"),
                               scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[2.5, 0.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=0)

            gaussians_init = gaussians_gt.clone()
            gaussians_init._rotation.data[0] += torch.tensor([0.0, 0.5, 0.0, 0.7], device="cuda")

            run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init,
                          pipe, cameras, background, lr,
                          loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                          gif_renderer, name="rotation")

        if False:
            ############## Test 4: Shrink and shift ##############
            gif_renderer = GifRenderer(num_rows=2, num_cols=2, figsize=(6, 6))

            gaussians_gt._xyz.data += 100000.0           # Init invisible gaussians
            gaussians_gt._opacity.data -= 100000.0  # Init invisible gaussians
            idx = add_gaussian(gaussians_gt,
                               xyz=torch.tensor([-0.066, 0.21, -0.63], device="cuda"),
                               scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[2.5, 0.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=0)

            gaussians_init = gaussians_gt.clone()
            gaussians_init._xyz.data[0] += torch.tensor([0.5, -0.3, 0.1], device="cuda")
            gaussians_init._scaling.data[0] += torch.tensor([0.6, 0.9, 0.01], device="cuda")

            run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init,
                          pipe, cameras, background, lr,
                          loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                          gif_renderer, name="shrink")

        if False:
            ############## Test 5: Expand and shift ##############
            gif_renderer = GifRenderer(num_rows=2, num_cols=2, figsize=(6, 6))

            gaussians_gt._xyz += 100000.0           # Init invisible gaussians
            gaussians_gt._opacity.data -= 100000.0  # Init invisible gaussians
            idx = add_gaussian(gaussians_gt,
                               xyz=torch.tensor([-0.066, 0.21, -0.63], device="cuda"),
                               scaling=torch.tensor([0.4, 0.3, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[2.5, 0.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=0)

            gaussians_init = gaussians_gt.clone()
            gaussians_init._xyz.data[0] += torch.tensor([0.5, -0.3, 0.1], device="cuda")
            gaussians_init._scaling.data[0] -= torch.tensor([0.4, 0.5, 0.01], device="cuda")

            run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init,
                          pipe, cameras, background, lr,
                          loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                          gif_renderer, name="expand")

        if False:
            ############## Test 6: Underrepresented ##############
            gif_renderer = GifRenderer(num_rows=2, num_cols=2, figsize=(6, 6))

            gaussians_gt._opacity.data -= 100000.0  # Init invisible gaussians
            idx = add_gaussian(gaussians_gt,
                               xyz=torch.tensor([-0.566, 0.21, -0.63], device="cuda"),
                               scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[2.5, 0.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=0)
            idx = add_gaussian(gaussians_gt,
                               xyz=torch.tensor([0.566, 0.21, -0.93], device="cuda"),
                               scaling=torch.tensor([-0.6, -1.6, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[0.4, 2.5, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=idx)

            gaussians_init = init_invisible_gaussians(max_num_points, sh_degree, opt)
            idx = add_gaussian(gaussians_init,
                               xyz=torch.tensor([-0.066, 0.21, -0.63], device="cuda"),
                               scaling=torch.tensor([0.4, 0.1, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[2.5, 0.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=0)

            run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init,
                          pipe, cameras, background, lr,
                          loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                          gif_renderer, name="underrepresented")

        if False:
            ############## Test 7: Two Gaussians ##############
            gif_renderer = GifRenderer(num_rows=2, num_cols=2, figsize=(6, 6))

            gaussians_gt._opacity.data -= 100000.0  # Init invisible gaussians
            idx = add_gaussian(gaussians_gt,
                               xyz=torch.tensor([-0.566, 0.21, -0.63], device="cuda"),
                               scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[2.5, 0.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=0)
            idx = add_gaussian(gaussians_gt,
                               xyz=torch.tensor([0.766, 0.21, -0.203], device="cuda"),
                               scaling=torch.tensor([-1.6, -0.3, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[0.4, 2.5, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=idx)

            gaussians_init = init_invisible_gaussians(max_num_points, sh_degree, opt)
            idx = add_gaussian(gaussians_init,
                               xyz=torch.tensor([-0.366, 0.31, -0.63], device="cuda"),
                               scaling=torch.tensor([0.0, -0.6, -1.2], device="cuda"),
                               rotation=torch.tensor([0.0, 0.2, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[2.5, 0.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=0)
            idx = add_gaussian(gaussians_init,
                               xyz=torch.tensor([0.366, 0.11, -0.203], device="cuda"),
                               scaling=torch.tensor([-1.7, -0.5, -1.2], device="cuda"),
                               rotation=torch.tensor([0.1, 0.0, 0.0, 1.0], device="cuda"),
                               feature_dc=torch.tensor([[0.3, 2.1, 0.1]], device="cuda"),
                               features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                               opacity=torch.tensor([[10.0]], device="cuda"), 
                               idx=1)

            run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init,
                          pipe, cameras, background, lr,
                          loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                          gif_renderer, name="two_gaussians")

    exit()
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, cameras, gaussians, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set

    if iteration in testing_iterations or iteration % 1 == 0:
        torch.cuda.empty_cache()
        num_val_images = 30
        val_stride = max(1, len(cameras) // num_val_images)
        val_indices = list(range(0, len(cameras), val_stride))
        validation_configs = ({'name': 'test', 'cameras' : cameras}, 
                              {'name': 'train', 'cameras' : [cameras[idx] for idx in val_indices]} )
        print(f"\n[ITER {iteration}] val_indices: {val_indices}")

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def plot_loss_vs_step_size(iteration, l1_loss, cameras, gaussians, gaussians_start, renderFunc, renderArgs, train_test_exp, s):
    torch.cuda.empty_cache()
    num_val_images = 30
    val_stride = max(1, len(cameras) // num_val_images)
    val_indices = list(range(0, len(cameras), val_stride))
    validation_configs = ({'name': 'test', 'cameras' : cameras}, 
                          {'name': 'train', 'cameras' : [cameras[idx] for idx in val_indices]} )

    test_l1_losses= []
    test_psnrs = []
    train_l1_losses= []
    train_psnrs = []

    with torch.no_grad():
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                gaussians = deepcopy(gaussians_start)
                step_size = 0.2
                for i in range(20):
                    alpha = i * step_size
                    gaussians.update_step(step_size * s)
                    
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if train_test_exp:
                            image = image[..., image.shape[-1] // 2:]
                            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])          

                    print(f"alpha {alpha:.3f} l1 {l1_test:.6f} psnr {psnr_test:.2f}")
                    if config['name'] == 'test':
                        test_l1_losses.append(l1_test.item())
                        test_psnrs.append(psnr_test.item())
                    else:
                        train_l1_losses.append(l1_test.item())
                        train_psnrs.append(psnr_test.item())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(train_l1_losses)) * step_size, train_l1_losses, label='Train L1 Loss')
    plt.plot(np.arange(0, len(test_l1_losses)) * step_size, test_l1_losses, label='Test L1 Loss')
    plt.xlabel('Step size')
    plt.ylabel('L1 Loss')
    plt.title('L1 Loss vs Step Size (Normalized to ADAM step)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, len(train_psnrs)) * step_size, train_psnrs, label='Train PSNR')
    plt.plot(np.arange(0, len(test_psnrs)) * step_size, test_psnrs, label='Test PSNR')
    plt.xlabel('Step size')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Step Size (Normalized to ADAM step)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(f"figures/loss_vs_step_size_{iteration}.png"))

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1] + list(range(0, 30001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--num_points", type=int, default=2_000)
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--all_columns", action="store_true", default=False)
    parser.add_argument("--sum_column", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(op.extract(args), 
             pp.extract(args),)

    # All done
    print("\nTraining complete.")
