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
    model_path = ""
    ####### Some fixed parameters #########

    first_iter = 0
    sh_degree = 0
    tb_writer = None
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

    g_smoothed = 0
    g_denom_iter = 0
    D_denom_iter = 0
    D_est_smoothed = 0
    D_est_smoothed2 = 0
    adam_g_smoothed = 0
    adam_g_denom_iter = 0
    adam_D_denom_iter = 0
    adam_D_est_smoothed = 0

    load_dict = torch.load("debug_step2850.pth")

    model_params = load_dict["gaussians"]
    model_params_copy = load_dict["gaussians_copy"]
    s_sophia = load_dict["s_sophia"]
    s_adam = load_dict["s_adam"]
    iteration = load_dict["iteration"]
    D_est = load_dict["D_est"]
    D_est_t = load_dict["D_est_t"]
    v = load_dict["v"]
    adam_v = load_dict["adam_v"]
    H = load_dict["H"]
    g = load_dict["g"]
    g_est = load_dict["g_est"]

    gaussians = init_uniform_gaussians(num_points, sh_degree, opt)
    gaussians_copy = init_uniform_gaussians(num_points, sh_degree, opt)

    gaussians.restore(model_params, opt)
    gaussians_copy.restore(model_params_copy, opt)

    render_args = {"iteration": iteration,
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

    viewpoint_cams = cameras

    with torch.no_grad():
        loss_full = batch_training_loss(gaussians=gaussians, viewpoint_cams=viewpoint_cams, **render_args)
        loss_full_prev = batch_training_loss(gaussians=gaussians_copy, viewpoint_cams=viewpoint_cams, **render_args)

    loss_full_prev_scalar = 0.5 * (loss_full_prev.norm() ** 2)
    loss_full_scalar = 0.5 * (loss_full.norm() ** 2)

    print(f"loss_old = {loss_full_prev_scalar.item():.10f}")
    print(f"loss_current = {loss_full_scalar.item():.10f}")

    with torch.no_grad():
        num_sample_gidx = 1
        gaussian_indices = s_sophia.opacity.nonzero()[:,0]
        # gaussian_indices = [343]

        for sampled_gidx in gaussian_indices:
            s_new = GaussianModelVector.zeros_like(gaussians)
            s_new.xyz[sampled_gidx] = s_sophia.xyz[sampled_gidx]
            s_new.features_dc[sampled_gidx] = s_sophia.features_dc[sampled_gidx]
            s_new.features_rest[sampled_gidx] = s_sophia.features_rest[sampled_gidx]
            s_new.scaling[sampled_gidx] = s_sophia.scaling[sampled_gidx]
            s_new.rotation[sampled_gidx] = s_sophia.rotation[sampled_gidx]
            s_new.opacity[sampled_gidx] = s_sophia.opacity[sampled_gidx]

            gaussians_new = gaussians_copy.clone()
            gaussians_new.update_step(s_new)
            loss_full_new = batch_training_loss(gaussians=gaussians_new, viewpoint_cams=viewpoint_cams, **render_args)
            loss_full_new_scalar_sophia = 0.5 * (loss_full_new.norm() ** 2)

            s_new = GaussianModelVector.zeros_like(gaussians)
            s_new.xyz[sampled_gidx] = s_adam.xyz[sampled_gidx]
            s_new.features_dc[sampled_gidx] = s_adam.features_dc[sampled_gidx]
            s_new.features_rest[sampled_gidx] = s_adam.features_rest[sampled_gidx]
            s_new.scaling[sampled_gidx] = s_adam.scaling[sampled_gidx]
            s_new.rotation[sampled_gidx] = s_adam.rotation[sampled_gidx]
            s_new.opacity[sampled_gidx] = s_adam.opacity[sampled_gidx]

            gaussians_new = gaussians_copy.clone()
            gaussians_new.update_step(s_new)
            loss_full_new = batch_training_loss(gaussians=gaussians_new, viewpoint_cams=viewpoint_cams, **render_args)
            loss_full_new_scalar_adam = 0.5 * (loss_full_new.norm() ** 2)

            print(f"gidx {sampled_gidx}: loss_new = {loss_full_new_scalar_sophia.item():.10f} (sophia) vs {loss_full_new_scalar_adam.item():.10f} (adam)")

        gaussian_indices = s_sophia.opacity.nonzero()[:,0]
        sampled_gidx = gaussian_indices
        s_new = GaussianModelVector.zeros_like(gaussians)
        s_new.xyz[sampled_gidx] = s_sophia.xyz[sampled_gidx]
        s_new.features_dc[sampled_gidx] = s_sophia.features_dc[sampled_gidx]
        s_new.features_rest[sampled_gidx] = s_sophia.features_rest[sampled_gidx]
        s_new.scaling[sampled_gidx] = s_sophia.scaling[sampled_gidx]
        s_new.rotation[sampled_gidx] = s_sophia.rotation[sampled_gidx]
        s_new.opacity[sampled_gidx] = s_sophia.opacity[sampled_gidx]

        gaussians_new = gaussians_copy.clone()
        gaussians_new.update_step(s_new)
        loss_full_new = batch_training_loss(gaussians=gaussians_new, viewpoint_cams=viewpoint_cams, **render_args)
        loss_full_new_scalar_sophia = 0.5 * (loss_full_new.norm() ** 2)

        s_new = GaussianModelVector.zeros_like(gaussians)
        s_new.xyz[sampled_gidx] = s_adam.xyz[sampled_gidx]
        s_new.features_dc[sampled_gidx] = s_adam.features_dc[sampled_gidx]
        s_new.features_rest[sampled_gidx] = s_adam.features_rest[sampled_gidx]
        s_new.scaling[sampled_gidx] = s_adam.scaling[sampled_gidx]
        s_new.rotation[sampled_gidx] = s_adam.rotation[sampled_gidx]
        s_new.opacity[sampled_gidx] = s_adam.opacity[sampled_gidx]

        gaussians_new = gaussians_copy.clone()
        gaussians_new.update_step(s_new)
        loss_full_new = batch_training_loss(gaussians=gaussians_new, viewpoint_cams=viewpoint_cams, **render_args)
        loss_full_new_scalar_adam = 0.5 * (loss_full_new.norm() ** 2)

        print(f"gidx {sampled_gidx}: loss_new = {loss_full_new_scalar_sophia.item():.10f} (sophia) vs {loss_full_new_scalar_adam.item():.10f} (adam)")

        active_gidx = g.opacity.nonzero()[:,0]
        gaussians_new = gaussians_copy.clone()
        gaussians_new._xyz += 10000.0
        gaussians_new._scaling *= 0.0
        gaussians_new._opacity -= 100
        for gidx in active_gidx:
            gaussians_new._xyz[gidx] = gaussians_copy._xyz[gidx]
            gaussians_new._scaling[gidx] = gaussians_copy._scaling[gidx]
            gaussians_new._rotation[gidx] = gaussians_copy._rotation[gidx]
            gaussians_new._opacity[gidx] = gaussians_copy._opacity[gidx]

        active_gidx = g.opacity.nonzero()[:,0]

        print("render new")
        render_pkg = render(viewpoint_camera=viewpoint_cams[0], pc=gaussians_new, pipe=pipe, bg_color=background)

        image = render_pkg["render"]

        plt.figure()
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        plt.savefig(os.path.join(f"figures/debug_render_selected_gaussians_{iteration}.png"))


    safe_interact(local=locals(), banner="After loading checkpoint")




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
