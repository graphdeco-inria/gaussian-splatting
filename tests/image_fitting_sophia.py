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

from copy import deepcopy

from matplotlib import pyplot as plt
import matplotlib.colors as colors

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

def per_gaussian_clip_(s, abs_thresh):
    P = s.xyz.shape[0]

    xyz = s.xyz
    features_dc = s.features_dc
    features_rest = s.features_rest
    scaling = s.scaling
    rotation = s.rotation

    xyz_max = xyz.abs().max(dim=1, keepdim=True)[0]
    xyz_max[xyz_max < abs_thresh] = abs_thresh
    s.xyz = (xyz / xyz_max * abs_thresh).view_as(s.xyz)

    features_dc_max = features_dc.abs().max(dim=2, keepdim=True)[0]
    features_dc_max[features_dc_max < abs_thresh] = abs_thresh
    s.features_dc = (features_dc / features_dc_max * abs_thresh).view_as(s.features_dc)

    features_rest_max = features_rest.abs().max(dim=2, keepdim=True)[0]
    features_rest_max[features_rest_max < abs_thresh] = abs_thresh
    s.features_rest = (features_rest / features_rest_max * abs_thresh).view_as(s.features_rest)

    scaling_max = scaling.abs().max(dim=1, keepdim=True)[0]
    scaling_max[scaling_max < abs_thresh] = abs_thresh
    s.scaling = (scaling / scaling_max * abs_thresh).view_as(s.scaling)

    rotation_max = rotation.abs().max(dim=1, keepdim=True)[0]
    rotation_max[rotation_max < abs_thresh] = abs_thresh
    s.rotation = (rotation / rotation_max * abs_thresh).view_as(s.rotation)

    s.opacity.clip_(-abs_thresh, abs_thresh)

    # safe_interact(local=locals(), banner="Per-gaussian clipping")
    

def init_uniform_gaussians(num_points, sh_degree, opt):
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

    return gaussians

def build_camera(image_path):
    image_pil = Image.open(image_path)
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

def training(opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, image_path, num_points, all_columns, sum_column):
    
    cameras = [build_camera(image_path)]

    ####### Some fixed parameters #########
    num_images = 1
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    opt.model_path = ""
    ####### Some fixed parameters #########

    first_iter = 0
    sh_degree = 0
    tb_writer = None
    # tb_writer = prepare_output_and_logger(opt)
    gaussians = init_uniform_gaussians(num_points, sh_degree, opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    xyz_lr = opt.xyz_lr_init
    xyz_lr_scheduler = ExponentialLRScheduler(opt.xyz_lr_init, opt.xyz_lr_final, opt.xyz_lr_max_steps)
    xyz_lr = xyz_lr_scheduler.get_lr(first_iter)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = cameras
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    print("before train")
    with torch.no_grad():
        training_report(None, first_iter, 0, 0, l1_loss, 0, testing_iterations, cameras, gaussians, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, train_test_exp), train_test_exp)

    S = GaussianModelVector(xyz=1.0, 
                            features_dc=1.0,
                            features_rest=1.0,
                            scaling=1.0,
                            rotation=1.0,
                            opacity=1.0,
                            exposure=1.0,
                            gaussians=gaussians)
    S = (1 / S).sqrt()

    lr = GaussianModelVector(xyz=xyz_lr,  
                             features_dc=opt.features_dc_lr, 
                             features_rest=opt.features_rest_lr,
                             scaling=opt.scaling_lr,
                             rotation=opt.rotation_lr,
                             opacity=opt.opacity_lr,
                             exposure=opt.exposure_lr,
                             gaussians=gaussians)

    active_gidx = []
    prev_active_gidx_set = set()
    prev_active_gidx = []

    g_smoothed = 0
    g_denom_iter = 0
    D_denom_iter = 0
    D_est_smoothed = 0
    D_est_smoothed2 = 0
    adam_g_smoothed = 0
    adam_g_denom_iter = 0
    adam_D_denom_iter = 0
    adam_D_est_smoothed = 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, image_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        viewpoint_cams = cameras

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        pixel_sample_rate = 1.0
        if pixel_sample_rate >= 1.0:
            pixel_mask = None
        else:
            B = len(viewpoint_cams)
            H, W = viewpoint_cams[0].image_height, viewpoint_cams[0].image_width
            pixel_mask = torch.rand((B, H, W), device="cuda") > pixel_sample_rate

        render_args = {"iteration": iteration,
                       "opt": opt,
                       "pipe": pipe,
                       "bg": bg,
                       "train_test_exp": train_test_exp,
                       "depth_l1_weight": depth_l1_weight,
                       "loss_type": opt.loss_type,
                       "huber_delta": opt.huber_delta,
                       "disable_ssim": opt.disable_ssim,
                       "batch_size": 1,
                       "pixel_mask": None,
                       "regularize_visibility_mask": list(prev_active_gidx),
                       }

        loss_func = construct_loss_func(**render_args)
        g_func = construct_g_func(**render_args)
        JTJv_func = construct_JTJv_func(**render_args)
        Dhat_func = construct_Dhat_func(**render_args)
        z_gen_func = partial(GaussianModelVector.rademacher_like, gaussians)

        beta1, beta2 = opt.adahessian_beta1, opt.adahessian_beta2

        g = g_func(gaussians=gaussians, viewpoint_cams=viewpoint_cams, regularize=True, pixel_mask=pixel_mask)

        g_denom_iter += 1
        g_smoothed = beta1 * g_smoothed + (1 - beta1) * g
        g_est = g_smoothed / (1 - beta1 ** g_denom_iter)

        # ADAM
        if iteration == first_iter or iteration % 1 == 0:
            adam_D_denom_iter += 1
            adam_D_est_smoothed = opt.adam_beta2 * adam_D_est_smoothed + (1 - opt.adam_beta2) * (g * g)
            adam_v = (adam_D_est_smoothed / (1 - opt.adam_beta2 ** adam_D_denom_iter)).sqrt()

        s_adam = -(1 / (adam_v + 1e-15)) * g_est
        s_adam_vec = s_adam.as_1d_tensor()
        print(f"ADAM Num clipped: ", (s_adam_vec.abs() > 1.0).sum().item(), " / ", s_adam_vec.shape[0])
        # s_adam.clip_(-1, 1)
        s_adam = s_adam * lr
        # END ADAM

        if iteration != first_iter:
            g_vec = g.as_1d_tensor()
            D_est_smoothed_vec = D_est_smoothed.as_1d_tensor()
            new_params_mask = (D_est_smoothed_vec == 0.0) & (g_vec != 0.0)
            if new_params_mask.any():
                adam_v_vec = adam_v.as_1d_tensor()
                D_est_smoothed_vec[new_params_mask] = g_vec[new_params_mask].abs()
                D_est_smoothed.load_1d_tensor(D_est_smoothed_vec)
                D_est_vec = D_est.as_1d_tensor()
                D_est_vec[new_params_mask] = g_vec[new_params_mask].abs()
                D_est.load_1d_tensor(D_est_vec)
                print(f"New parameters detected at iteration {iteration}, initializing their D_est_smoothed values.")

        if iteration == first_iter or iteration % 10 == 0:
            JTJv = partial(JTJv_func, gaussians=gaussians, viewpoint_cams=viewpoint_cams, S=S, scale=1, regularize=True)
            Dhat = Dhat_func(gaussians=gaussians, viewpoint_cams=viewpoint_stack, regularize=True)

            print("Running Diagonal Estimation...")
            if iteration == first_iter:
                num_diag_iter = 20
                restart_iter = 3
                D_init = Dhat
            else:
                num_diag_iter = 2
                restart_iter = 1
                D_init = D_est.abs()

            D_est_t = restarted_hutchinson(Hz_func=JTJv,
                                         z_gen_func=z_gen_func,
                                         D_init=D_init,
                                         restart_iter=restart_iter,
                                         num_iters=num_diag_iter,
                                         )
            D_est_t = D_est_t.abs() / (S * S)

            D_denom_iter += 1
            D_est_smoothed = beta2 * D_est_smoothed + (1 - beta2) * D_est_t
            D_est = D_est_smoothed.abs() / (1 - beta2 ** D_denom_iter)
            # v = opt.sophia_gamma * D_est
            v = opt.sophia_gamma * D_est

            # Naive averaging is better

            # kernel3 = torch.ones((1, 1, 3), device=v.xyz.device) / 3.0
            # kernel33 = torch.ones((1, 1, 3, 3), device=v.features_rest.device) / 9.0
            # v.xyz = torch.nn.functional.conv1d(v.xyz.unsqueeze(1), kernel3, padding=1).squeeze(1)
            # v.features_dc = torch.nn.functional.conv1d(v.features_dc, kernel3, padding=1)
            # v.features_rest = torch.nn.functional.conv2d(v.features_rest.unsqueeze(1), kernel33, padding=(1,1)).squeeze(1)
            # v.scaling = torch.nn.functional.conv1d(v.scaling.unsqueeze(1), kernel3, padding=1).squeeze(1)
            # v.rotation = torch.nn.functional.conv1d(v.rotation.unsqueeze(1), kernel3, padding=1).squeeze(1)

            # v.xyz = v.xyz.mean(axis=1, keepdim=True).repeat(1, 3)
            # v.features_dc = v.features_dc.mean(axis=2, keepdim=True).repeat(1, 1, 3)
            # v.features_rest = v.features_rest.mean(axis=2, keepdim=True).repeat(1, 1, 3)
            # v.scaling = v.scaling.mean(axis=1, keepdim=True).repeat(1, 3)
            # v.rotation = v.rotation.mean(axis=1, keepdim=True).repeat(1, 4)

            # v.clip_(1e-16, 1e20)


        # v.xyz = adam_v.xyz
        # v.rotation = adam_v.rotation
        # v.scaling = adam_v.scaling
        # v.opacity = adam_v.opacity
        # v.xyz = u.xyz.clamp(adam_v.xyz / 1.1, adam_v.xyz * 1.1)
        # v.features_dc = u.features_dc.clamp(adam_v.features_dc / 1.01, adam_v.features_dc * 1.01)
        # v.features_rest = u.features_rest.clamp(adam_v.features_rest / 1.01, adam_v.features_rest * 1.01)
        # v.scaling = u.scaling.clamp(adam_v.scaling / 1.001, adam_v.scaling * 1.001)
        # v.opacity = u.opacity.clamp(adam_v.opacity / 1.01, adam_v.opacity * 1.01)


        # g_est_vec = g_est.as_1d_tensor()
        # D_est_vec = D_est.as_1d_tensor()
        # noise = torch.randn(g_est_vec.shape, device=g_est_vec.device) * 1e-6 * (1 / D_est_vec)
        # g_est_vec = g_est_vec * (1.0 + noise)
        # g_est.load_1d_tensor(g_est_vec)

        # with torch.enable_grad():
        #     loss_full = batch_training_loss(gaussians=gaussians, viewpoint_cams=viewpoint_cams, **render_args)
        #     loss_vec = loss_full.Ll1_per_pixel.flatten()
        #     n = g.as_1d_tensor().shape[0]
        #     m = loss_vec.shape[0]
        #     J_ref = torch.zeros((m, n), device=g.as_1d_tensor().device)
        #     for i in range(m):
        #         print(f"Computing Jacobian row {i+1}/{m}", end="\r")
        #         gaussians.zero_grad()
        #         loss_vec[i].backward(retain_graph=True)
        #         Ji = GaussianModelVector.from_gaussians_grad(gaussians).as_1d_tensor()
        #         J_ref[i, :] = Ji
        #     H_ref = J_ref.T @ J_ref
        #     D_ref = torch.diagonal(H_ref)

        #     s_sophia = -(1 / (v + 1e-9)) * g_est

        s_sophia = -(1 / (v + 1e-12)) * g_est
        # clip_thresh = 0.01
        # s_sophia_vec = s_sophia.as_1d_tensor()
        # print(f"Num clipped: ", (s_sophia_vec.abs() > clip_thresh).sum().item(), " / ", s_sophia_vec.shape[0])
        # s_sophia.clip_(-clip_thresh, clip_thresh)
        # s_sophia = s_sophia * lr
        s_sophia.xyz.clip_(-lr.xyz, lr.xyz)
        s_sophia.features_dc.clip_(-lr.features_dc, lr.features_dc)
        s_sophia.features_rest.clip_(-lr.features_rest, lr.features_rest)
        s_sophia.scaling.clip_(-lr.scaling, lr.scaling)
        s_sophia.rotation.clip_(-lr.rotation, lr.rotation)
        s_sophia.opacity.clip_(-lr.opacity, lr.opacity)

        if opt.use_adam: # or iteration > 900:
            print("Using ADAM step")
            s_sophia = s_adam

        # print("replacing scaling")
        # s_sophia.scaling = s_adam.scaling

        active_gidx = g.opacity.nonzero()[:, 0]
        prev_active_gidx_set.update(set(active_gidx.cpu().numpy().tolist()))
        prev_active_gidx = sorted(list(prev_active_gidx_set))

        with torch.no_grad():
            gaussians_copy = gaussians.clone()
            gaussians.update_step(s_sophia)

            render_pkg = render(viewpoint_cams[0], gaussians, pipe, background, scaling_modifier=1.0, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            active_gidx = g.opacity.nonzero()[:, 0]
            scaling_norm = gaussians.get_scaling.norm(dim=1)
            active_opacity = gaussians.get_opacity.flatten()

            torch.set_printoptions(sci_mode=False)

            # for idx in prev_active_gidx:
            #     active = (idx in active_gidx)
            #     print(f"{'Active' if active else 'Inactive'} {idx}: scaling = {scaling_norm[idx].item():.6f}, opacity = {active_opacity[idx].item():.6f}")

            print("num active gaussians:", active_gidx.shape[0], "out of", g.opacity.shape[0])
            print("num prev active gaussians:", len(prev_active_gidx), "out of", g.opacity.shape[0])
            # print("scaling norm: ", scaling_norm)
            # print("opacity: ", active_opacity)
            torch.set_printoptions(sci_mode=True)

            loss = 0.0
            Ll1 = 0.0

            if iteration % opt.eval_interval == 0:
                loss = loss_func(gaussians=gaussians, viewpoint_cams=viewpoint_cams, regularize=False)
                prev_loss = loss_func(gaussians=gaussians_copy, viewpoint_cams=viewpoint_cams, regularize=False)
                print("[ITER {}] Training loss: {}".format(iteration, loss.item()), " prev loss: ", prev_loss.item())
                    
                loss_full = batch_training_loss(gaussians=gaussians, viewpoint_cams=viewpoint_cams, regularize=True, debug_regularize=False, **render_args)

                print("after loss computation")

                # tb_writer.add_scalar('train_loss/total_loss', loss.item(), iteration)

        if True and iteration < 2025:

            n = g.as_1d_tensor().shape[0]
            u = GaussianModelVector.zeros_like(gaussians_copy)
            u_vec = v.as_1d_tensor()
            g_vec = g.as_1d_tensor()

            if False: # os.path.exists(f"H_picasso1_20x24_step{iteration}.pth"):
                H = torch.load(f"H_picasso1_20x24_step{iteration}.pth")
            else:
                # with torch.enable_grad():
                #     prev_loss_full = batch_training_loss(gaussians=gaussians_copy, viewpoint_cams=viewpoint_cams, **render_args)
                #     loss_vec = prev_loss_full

                #     m = loss_vec.shape[0]
                #     J_ref = torch.zeros((m, n), device=g.as_1d_tensor().device)
                #     for i in range(m):
                #         print(f"Computing Jacobian row {i+1}/{m}", end="\r")
                #         gaussians_copy.zero_grad()
                #         loss_vec[i].backward(retain_graph=True)
                #         Ji = GaussianModelVector.from_gaussians_grad(gaussians_copy).as_1d_tensor()
                #         J_ref[i, :] = Ji

                # with torch.no_grad():
                #     H_ref = J_ref.T @ J_ref
                # H = H_ref

                H = torch.zeros((n, n), device=g.as_1d_tensor().device)
                # for j in g_vec.nonzero()[:, 0]:
                #     print(f"Computing JTJ column {j+1}/{n}", end="\r")
                #     u_vec *= 0.0
                #     u_vec[j] = 1.0
                #     u.load_1d_tensor(u_vec)
                #     Hj = JTJv_func(u, gaussians=gaussians_copy, viewpoint_cams=viewpoint_cams).as_1d_tensor()
                #     H[:, j] = Hj
                # # torch.save(H, f"H_picasso1_20x24_step{iteration}.pth")

            # Plot estimator
            D = torch.diagonal(H)
            sorted_indices = torch.argsort(D, descending=True)
            plt.figure()
            plt.plot(D_est_t.abs().as_1d_tensor()[sorted_indices].cpu().numpy(), label=f"Estimated JTJ diagonal (no smoothing)")
            plt.plot(D_est.as_1d_tensor()[sorted_indices].cpu().numpy(), label=f"Estimated JTJ diagonal")
            plt.plot(v.as_1d_tensor()[sorted_indices].cpu().numpy(), label="Sophia preconditioner diagonal")
            plt.plot(adam_v.as_1d_tensor()[sorted_indices].cpu().numpy(), label="ADAM preconditioner diagonal")
            plt.plot(D[sorted_indices].cpu().numpy(), label="Computed JTJ diagonal")
            plt.yscale("log")
            plt.xlabel("Index")
            plt.ylabel("Diagonal Value (log scale)")

            # Set x_lim
            plt.xlim(0, 8000)
            plt.ylim(1e-20, 2)

            plt.title("JTJ Diagonal Comparison")
            plt.legend()
            print(f"Saving figure to figures/debug_jtj_diagonal_estimator_with_Dhat at step{iteration}.png")
            plt.savefig(f"figures/debug_jtj_diagonal_estimator_with_Dhat at step{iteration}.png")

            save_dict = {"gaussians": gaussians.capture(),
                         "gaussians_copy": gaussians_copy.capture(),
                         "s_sophia": s_sophia, "s_adam": s_adam, "iteration": iteration, 
                         "D_est": D_est, "D_est_t": D_est_t, "v": v, "adam_v": adam_v, "H": H,
                         "g": g, "g_est": g_est}
            torch.save(save_dict, f"debug_step{iteration}.pth")

            # D_est_vec = torch.diagonal(H)
            # D_est.load_1d_tensor(D_est_vec)
            # v = opt.sophia_gamma * D_est
            # v.clip_(1e-16, 1e20)
            safe_interact(local=locals(), banner="after save fig")

        safe_interact(local=locals(), banner="Computed loss vector")



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
             pp.extract(args), 
             args.test_iterations, 
             args.save_iterations, 
             args.checkpoint_iterations, 
             args.start_checkpoint, 
             args.debug_from, 
             args.image_path, 
             args.num_points,
             args.all_columns,
             args.sum_column)

    # All done
    print("\nTraining complete.")
