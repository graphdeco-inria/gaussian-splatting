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
from solver.gaussian_model_state import GaussianModelState, GaussianModelScaleMatrix, GaussianModelParamGroupMask, GaussianModelSplatMask
from solver.training_loss import scalar_training_loss
from solver.batch_training_loss import batch_training_loss
from solver.training_loss_hessian import scalar_training_loss_hessian
from solver.reference_training_loss import reference_training_loss
from solver.loss_image_state import MultiBatchLossImageState
from solver.solver_functions import LinearSolverFunctions
from solver.conjugate_gradient import cg_damped, cgls_damped
from solver.preconditioner import AdaHessianPreconditioner
from solver.solver_utils import CamProvider

from copy import deepcopy

from matplotlib import pyplot as plt

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    splat_mask = None

    ####### Some fixed parameters #########
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    model_path = ""
    ####### Some fixed parameters #########

    pixel_sample_rate = opt.pixel_sample_rate_max
    xyz_lr = opt.xyz_lr_init
    damp = opt.damp_init

    rescale = GaussianModelScaleMatrix(xyz_scale=opt.xyz_scale,  
                                      features_dc_scale=opt.features_dc_scale, 
                                      features_rest_scale=opt.featuress_rest_scale, 
                                      scaling_scale=opt.scaling_scale, 
                                      rotation_scale=opt.rotation_scale, 
                                      opacity_scale=opt.opacity_scale, 
                                      exposure_scale=opt.exposure_scale)

    lr = GaussianModelScaleMatrix(xyz_scale=xyz_lr,  
                                  features_dc_scale=opt.features_dc_lr, 
                                  features_rest_scale=opt.featuress_rest_lr, 
                                  scaling_scale=opt.scaling_lr, 
                                  rotation_scale=opt.rotation_lr, 
                                  opacity_scale=opt.opacity_lr, 
                                  exposure_scale=opt.exposure_lr)

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    test_viewpoint_stack = scene.getTestCameras().copy()
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    preconditioner = None

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    iteration = first_iter

    gaussians.update_learning_rate(iteration)

    num_batch_cameras = len(viewpoint_indices) if opt.num_images == -1 else min(opt.num_images, len(viewpoint_indices))
    rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)
    scale = len(viewpoint_indices) / num_batch_cameras

    viewpoint_cams = []
    for rand_idx in rand_indices:
        viewpoint_cam = viewpoint_stack[rand_idx]
        viewpoint_cams.append(viewpoint_cam)

    num_val_images = int(len(viewpoint_stack) * opt.linesearch_val_images)
    val_stride = max(1, len(scene.getTrainCameras()) // num_val_images)
    val_indices = list(range(0, len(scene.getTrainCameras()), val_stride))
    val_viewpoint_stack = [viewpoint_stack[i] for i in val_indices]
    val_scale = len(viewpoint_stack) / len(val_viewpoint_stack)

    # Render
    if (iteration - 1) == debug_from:
        pipe.debug = True

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    if iteration > 1200:
        if iteration % opt.splat_sample_update_freq == 0:
            num_gaussians = gaussians.get_xyz.shape[0]
            splat_mask_out = torch.rand(num_gaussians, device="cuda") > opt.splat_sample_rate if opt.splat_sample_rate < 1.0 else None
            splat_mask = GaussianModelSplatMask(mask_out_filter=splat_mask_out) if splat_mask_out is not None else None
    else:
        splat_mask = None

    # Test vector loss prediction using J

    # Generate pixel mask, which is a boolean mask of shape (H*W,) with True for masking out pixels
    if pixel_sample_rate >= 1.0:
        pixel_mask = None
    else:
        B = len(viewpoint_cams)
        H, W = viewpoint_cams[0].image_height, viewpoint_cams[0].image_width
        pixel_mask = torch.rand((B, H, W), device="cuda") > pixel_sample_rate


    loss_func = partial(batch_training_loss, iteration=iteration, opt=opt, pipe=pipe, bg=background, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=opt.disable_ssim, pixel_mask=pixel_mask)
    cur_state = LinearSolverFunctions(loss_func, gaussians, batch_size=5, param_mask=None, splat_mask=splat_mask, rescale=rescale, damp=damp)

    SJTJSx = partial(cur_state.JTJv, viewpoint_cams=viewpoint_cams, scale=scale, use_rescale=True, use_damping=True)
    JTJx = partial(cur_state.JTJv, viewpoint_cams=viewpoint_cams, scale=scale, use_rescale=False, use_damping=False)

    warmup_sample_size = min(1, len(viewpoint_cams))
    warmup_scale = len(viewpoint_stack) / warmup_sample_size
    warmup_cam_provider = CamProvider(viewpoint_cams, mode="random", max_stride=1, sample_size=warmup_sample_size)

    rademacher_gen = partial(GaussianModelState.rademacher_like_gaussians, gaussians)
    preconditioner = AdaHessianPreconditioner(rademacher_gen, beta2=0.999, eps=1e-16, hessian_power=1.0)

    iters = [10, 100, 500, 1000, 10000]
    # iters = [10]
    Ds = {}

    for num_iter in iters:
        print("Estimating diagonal with num_iter =", num_iter)

        preconditioner.reset()
        preconditioner.update(SJTJSx, warmup_cam_provider, warmup_scale, num_iter=num_iter)

        # Ds[num_iter] = preconditioner.D_corrected.sqrt()
        Ds[num_iter] = preconditioner.D_corrected

        torch.save(Ds, "diagonal_estimate.pth")

    safe_interact(local=locals(), banner="After preconditioner")


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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
