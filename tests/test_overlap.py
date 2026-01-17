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
from solver.KL_clip2 import clip_kl
from solver.hellinger_clip import clip_hellinger, compute_hellinger_distance
from scene.gaussian_model import build_scaling_rotation

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

def build_camera(image_torch):

    image_np = (image_torch.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_pil = Image.fromarray(image_np)
    W, H = image_pil.size
    viewmat = np.array([1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 8,
                        0, 0, 0, 1], dtype=np.float32).reshape(4, 4)
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

def debug_hellinger(gaussians, update, debug=False):
    for param_group in ["xyz", "scaling", "rotation", "opacity", "features_dc", "features_rest"]:
        update_group = GaussianModelVector.zeros_like(gaussians)
        setattr(update_group, param_group, getattr(update, param_group).clone())

        hellinger_dist = compute_hellinger_distance(gaussians, update_group, debug=debug)
        print(f"H dist ({param_group}): max = {hellinger_dist.max().item():.6f}, mean = {hellinger_dist.mean().item():.6f}")

    total_hellinger_dist = compute_hellinger_distance(gaussians, update, debug=debug)
    print(f"H dist (total): max = {total_hellinger_dist.max().item():.6f}, mean = {total_hellinger_dist.mean().item():.6f}")

def compute_JTJ_col(JTJv_func, index, gaussians):
    v = GaussianModelVector.zeros_like(gaussians)
    v_vec = v.as_1d_tensor()
    v_vec[index] = 1.0
    v.load_1d_tensor(v_vec)
    return JTJv_func(v=v).as_1d_tensor()

def densify_gaussians(gaussians, optimizer, cap_max, opacity_prune_thresh=0.005, color_noise_lr=0.0, preserve_gaussian=False):

    dead_mask = (gaussians.get_opacity <= opacity_prune_thresh).squeeze(-1)
    if preserve_gaussian:
        gaussians.relocate_gs2(dead_mask=dead_mask, start_opacity=0.01, position_noise=0.2)
    else:
        gaussians.relocate_gs(dead_mask=dead_mask)

    num_gaussians = gaussians.get_xyz.shape[0]

    if preserve_gaussian:
        gaussians.add_new_gs2(cap_max=cap_max, growth_factor=(num_gaussians + 1)/num_gaussians)
    else:
        gaussians.add_new_gs(cap_max=cap_max, growth_factor=(num_gaussians + 1)/num_gaussians)

    optimizer.reset_indices(dead_mask)

    prune_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    optimizer.densify_and_prune(prune_mask)

    return 

def add_noise(gaussians, noise_lr, xyz_lr):
    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
    actual_covariance = L @ L.transpose(1, 2)

    def op_sigmoid(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))
    
    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*noise_lr*xyz_lr
    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
    gaussians._xyz.add_(noise)

def run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init,
                  pipe, cameras, background, lr,
                  loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                  gif_renderer, name="",
                  with_densification=True, densification_interval=10,
                  cap_max=5,
                  ):
    adam_optimizer = AdamOptimizer(lr=lr, betas=(0.9, 0.999), eps=1e-15, clip=False)
    sophia_optimizer = SophiaOptimizer(lr=lr, betas=(0.9, 0.99), eps=1e-15, clip=True,
                                       diagonal_update_interval=10,)

    densify_noise_lr = 0.0

    print(f"Running test: {name}")

    if isinstance(NUM_ITERATIONS, tuple):
        adam_iterations, sophia_iterations, sophia_tr_iterations = NUM_ITERATIONS
    else:
        adam_iterations = NUM_ITERATIONS
        sophia_iterations = NUM_ITERATIONS
        sophia_tr_iterations = NUM_ITERATIONS

    gif_interval = int(math.ceil(sophia_tr_iterations / 100))

    with torch.no_grad():
        image_gt_torch = render(viewpoint_camera=cameras[0], pc=gaussians_gt, pipe=pipe, bg_color=background)["render"]
        cameras = [build_camera(image_gt_torch)]

        adam_optimizer.reset()

        adam_losses = []
        adam_images = []

        opacities_adam = []
        opacities_sophia_tr = []

        gaussians = gaussians_init.clone()
        for it in range(adam_iterations):
            loss_adam, batch_stats = loss_func(gaussians=gaussians, viewpoint_cams=cameras, return_stats=True)
            image_adam = batch_stats[0]["images"][0]
            opacities_adam.append(gaussians.get_opacity)
            if it % gif_interval == 0:
                adam_losses.append(loss_adam.item())
                adam_images.append(image_adam)
            print(f"Adam loss: {loss_adam.item():.10f}", end="\r" if it < adam_iterations - 1 else "\n")

            g = g_func(gaussians=gaussians, viewpoint_cams=cameras)
            s_adam = adam_optimizer.get_update(g)

            gaussians.update_step(s_adam)

            if with_densification and (it + 1) % densification_interval == 0 and it < 120:
                densify_gaussians(gaussians, adam_optimizer, cap_max=cap_max, preserve_gaussian=False)


        # safe_interact(local=locals(), banner="\nAfter adam step in run_optimizer")

        adam_gaussians = gaussians
        adam_optimizer.reset()
        sophia_optimizer.reset()
        sophia_optimizer.set_clip(False)

        sophia_tr_losses = []
        sophia_tr_images = []

        gaussians = gaussians_init.clone()

        for it in range(sophia_tr_iterations):
            z_gen_func = partial(GaussianModelVector.rademacher_like, gaussians)
            JTJv_func1 = partial(JTJv_func, gaussians=gaussians, viewpoint_cams=cameras, S=S, scale=1)
            Dhat_func1 = partial(Dhat_func, gaussians=gaussians, viewpoint_cams=cameras)

            loss_sophia_tr, batch_stats = loss_func(gaussians=gaussians, viewpoint_cams=cameras, return_stats=True)
            image_sophia_tr = batch_stats[0]["images"][0]
            opacities_sophia_tr.append(gaussians.get_opacity)
            if it % gif_interval == 0:
                sophia_tr_losses.append(loss_sophia_tr.item())
                sophia_tr_images.append(image_sophia_tr)

            g = g_func(gaussians=gaussians, viewpoint_cams=cameras)

            s_adam = adam_optimizer.get_update(g)
            s_sophia_tr = sophia_optimizer.get_update(g, JTJv_func1, Dhat_func1, z_gen_func, S)

            s_sophia_tr_old = s_sophia_tr.clone()

            # s_sophia_tr = clip_kl(gaussians, s_sophia_tr, kl_threshold, 0.01, lr)
            s_sophia_tr = clip_hellinger(gaussians, s_sophia_tr, kl_threshold, lr)

            # debug_hellinger(gaussians, s_sophia_tr_old)
            # print("After KL clipping:")
            # debug_hellinger(gaussians, s_sophia_tr)

            print(f"Sophia TR loss: {loss_sophia_tr.item():.10f}", end="\r" if it < sophia_tr_iterations - 1 else "\n")
            # safe_interact(local=locals(), banner="\nAfter KL clipping in run_optimizer")

            if it > 0 and False:

                # D_est = sophia_optimizer.D_est.opacity

                # N1, N2, N3, N4, N5, N6, N7 = g._get_param_group_lengths()
                # offset = 0
                # D = []
                # for group, N in enumerate([N1, N2, N3, N4, N5, N6, N7]):
                #     if group != 5:
                #         pass
                #     else:
                #         for group_i in range(N):
                #             i = offset + group_i
                #             Di = compute_JTJ_col(JTJv_func1, i, gaussians)[i].item()
                #             D.append(Di)
                #     offset += N

                # D = torch.tensor(D, device="cuda").unsqueeze(-1)

                # s_sophia_tr.opacity = -sophia_optimizer.m.opacity / (D + sophia_optimizer.eps)

                # # safe_interact(local=locals(), banner="\nAfter Sophia step in run_optimizer - Adam update")
                # gaussians.update_step(s_sophia_tr)
                gaussians.update_step(s_adam)

            else:
                # safe_interact(local=locals(), banner="\nAfter Sophia step in run_optimizer")
                # s_sophia_tr.features_dc = s_adam.features_dc
                # s_sophia_tr.features_rest = s_adam.features_rest
                # s_sophia_tr.opacity = s_adam.opacity
                gaussians.update_step(s_sophia_tr)

            if with_densification and (it + 1) % densification_interval == 0 and it < 120:
                densify_gaussians(gaussians, sophia_optimizer, cap_max=cap_max, preserve_gaussian=True)
                # else:
                #     # Remove all but one gaussian
                #     gaussians._opacity.data[1:] = -100000.0


        opacities_adam = np.array([opacities.squeeze().cpu().numpy() for opacities in opacities_adam]).squeeze()
        opacities_sophia_tr = np.array([opacities.cpu().numpy() for opacities in opacities_sophia_tr]).squeeze()

        figure, axes = plt.subplots(2, 2, figsize=(10, 5))

        axes[0, 0].plot(range(len(adam_losses)), adam_losses, label="Adam Loss")
        axes[0, 0].set_title("Adam Loss")
        axes[0, 1].plot(range(len(sophia_tr_losses)), sophia_tr_losses, label="Sophia TR Loss", color='orange')
        axes[0, 1].set_title("Sophia TR Loss")
        axes[1, 0].plot(opacities_adam)
        axes[1, 0].set_title("Adam Opacities")
        axes[1, 1].plot(opacities_sophia_tr)
        axes[1, 1].set_title("Sophia TR Opacities")

        plt.suptitle("Overlapping Gaussians Optimization")

        plt.tight_layout()
        plt.savefig(f"figures/{name}-opacities.png")
        print(f"Saved figures/{name}-opacities.png")

        gif_renderer = GifRenderer(num_rows=1, num_cols=3, figsize=(12, 6))
        
        gif_renderer.add_gt(0, 0, image_gt_torch)
        gif_renderer.add_series(0, 1, adam_images, adam_losses, title="Adam")
        gif_renderer.add_series(0, 2, sophia_tr_images, sophia_tr_losses, title="Sophia TR")

        gif_renderer.animate(f"figures/gaussian_fitting_{name}.gif", interval=500)
        print(f"Saved figures/gaussian_fitting_{name}.gif")

        # safe_interact(local=locals(), banner="\nAfter Sophia step in run_optimizer")


        # print("Computing diagonal values")

        # figure, ax = plt.subplots(figsize=(8, 6))
        # D_est = sophia_optimizer.D_est.as_1d_tensor(with_features_rest=False, with_exposure=False).cpu().numpy()
        # ax.plot(D_est, label="Estimated diagonal")

        # N1, N2, N3, N4, N5, N6, N7 = g._get_param_group_lengths()
        # line_offset = 0
        # offset = 0
        # D = []
        # for group, N in enumerate([N1, N2, N3, N4, N5, N6, N7]):
        #     if group == 2 or group == 6:
        #         pass
        #     else:
        #         for group_i in range(N):
        #             i = offset + group_i
        #             Di = compute_JTJ_col(JTJv_func1, i, gaussians)[i].item()
        #             D.append(Di)
        #         line_offset += N
        #     offset += N

        #     ax.axvline(x=line_offset, color='gray', linestyle='--', linewidth=0.5)

        # D = np.array(D)

        # ax.plot(D, label="True diagonal")
        # ax.set_title("Diagonal of Hessian comparison")
        # ax.set_yscale("log")
        # ax.legend()
        # plt.savefig("figures/single_densification_no-noise_diagonal.png")

        safe_interact(local=locals(), banner="\nEnd of run_optimizer")

    exit()

def training(opt, pipe):
    

    ####### Some fixed parameters #########
    num_images = 1
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    model_path = ""
    NUM_ITERATIONS = (200, 10, 200)
    H, W = 200, 240
    max_num_points = 10
    kl_threshold = 0.001


    run_shortside = False
    run_longside = False
    run_rotation = False
    run_shrink = False
    run_expand = False
    run_underrepresented = False
    run_two_gaussians = False
    run_double_gaussian = False
    run_small_color_shift = False
    run_large_color_shift = False
    run_increase_opacity = True

    ####### Some fixed parameters #########

    first_iter = 0
    sh_degree = 3
    tb_writer = None

    image_zeros = torch.zeros((3, H, W), dtype=torch.float32, device="cuda")
    cameras = [build_camera(image_zeros)]
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
                   "opacity_reg": opt.opacity_reg,
                   "scale_reg": opt.scale_reg,
                   "color_reg": opt.color_reg,
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
    S = None

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
        ############## Test 1: Single Densfication ##############
        gif_renderer = GifRenderer(num_rows=2, num_cols=2, figsize=(6, 6))

        gaussians_gt._opacity.data -= 100000.0  # Init invisible gaussians
        idx = add_gaussian(gaussians_gt,
                           xyz=torch.tensor([-0.066, 0.21, -0.63], device="cuda"),
                           scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
                           rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                           feature_dc=torch.tensor([[1.2, -0.1, 0.1]], device="cuda"),
                           features_rest=0.02 * torch.ones((1, 15, 3), device="cuda"),
                           opacity=torch.tensor([[10.0]], device="cuda"), 
                           idx=0)

        gaussians_init = gaussians_gt.clone()
        gaussians_init._opacity.data -= 100000.0  # Init invisible gaussians
        # gaussians_init._xyz[0] += torch.tensor([0.0, 0.3, 0.0], device="cuda")
        # gaussians_init._features_dc[0] = torch.tensor([-1.5, -1.5, -1.5], device="cuda")
        # gaussians_init._opacity[0] = torch.tensor([-1.0], device="cuda")
        _ = add_gaussian(gaussians_init,
                           xyz=torch.tensor([-0.066, 0.21, -0.63], device="cuda"),
                           scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
                           rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
                           feature_dc=torch.tensor([[0.7, 0.1, 0.1]], device="cuda"),
                           features_rest=0.00 * torch.ones((1, 15, 3), device="cuda"),
                           opacity=torch.tensor([[0.0]], device="cuda"), 
                           idx=0)
        # _ = add_gaussian(gaussians_init,
        #                    xyz=torch.tensor([-0.066, 0.21, -0.63], device="cuda"),
        #                    scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
        #                    rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
        #                    feature_dc=torch.tensor([[0.7, 0.1, 0.1]], device="cuda"),
        #                    features_rest=0.00 * torch.ones((1, 15, 3), device="cuda"),
        #                    opacity=torch.tensor([[1]], device="cuda"), 
        #                    idx=1)

        # for idx in range(max_num_points):
        #     _ = add_gaussian(gaussians_init,
        #                      xyz=torch.tensor([-0.066, 0.61, -0.63], device="cuda"),
        #                      scaling=torch.tensor([-0.2, -1.2, -1.2], device="cuda"),
        #                      rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
        #                      feature_dc=torch.tensor([[0.0, -0.1, -0.1]], device="cuda"),
        #                      features_rest=0.00 * torch.ones((1, 15, 3), device="cuda"),
        #                      opacity=torch.tensor([[-2.0]], device="cuda"), 
        #                      idx=idx)

        run_optimizer(NUM_ITERATIONS, kl_threshold, gaussians_gt, gaussians_init, 
                      pipe, cameras, background, lr,
                      loss_func, g_func, JTJv_func, Dhat_func, z_gen_func, S,
                      gif_renderer, name="overlap-gaussians", with_densification=False,
                      cap_max=max_num_points,)

    

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
