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
from random import randint
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, safe_interact
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
from contextlib import contextmanager
from copy import deepcopy
import matplotlib.pyplot as plt

from functools import partial
from solver.gaussian_model_state import GaussianModelState, GaussianModelScaleMatrix, GaussianModelParamGroupMask, GaussianModelSplatMask
from solver.training_loss import scalar_training_loss
from solver.batch_training_loss import batch_training_loss
from solver.solver_functions import LinearSolverFunctions
from solver.conjugate_gradient import cg_damped, cgls_damped
from solver.preconditioner import AdaHessianPreconditioner
from solver.solver_utils import CamProvider

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, jvp_start, num_images):
    print("after training called")

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    param_mask = GaussianModelParamGroupMask(mask_xyz=False,
                                             mask_features_dc=False, 
                                             mask_features_rest=False, 
                                             mask_scaling=False, 
                                             mask_rotation=False, 
                                             mask_opacity=False, 
                                             mask_exposure=True)

    P = gaussians.get_xyz.shape[0]

    damp = GaussianModelScaleMatrix(xyz_scale=1e-1, 
                                    features_dc_scale=5e-2, 
                                    features_rest_scale=5e-2, 
                                    scaling_scale=5e-2, 
                                    rotation_scale=5e-2, 
                                    opacity_scale=5e-2, 
                                    exposure_scale=1e1) * 1e-2

    rescale = GaussianModelScaleMatrix(xyz_scale=0.0001, 
                                      features_dc_scale=0.0025, 
                                      features_rest_scale=0.0001, 
                                      scaling_scale=0.005, 
                                      rotation_scale=0.001, 
                                      opacity_scale=0.025, 
                                      exposure_scale=1.0)

    loss_func = partial(batch_training_loss, iteration=jvp_start, opt=opt, pipe=pipe, bg=background, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=False)
    solver_functions = LinearSolverFunctions(loss_func, gaussians, batch_size=10, param_mask=param_mask, damp=damp, splat_mask=None, rescale=rescale)

    rand_indices = np.random.permutation(len(viewpoint_stack))[:num_images]
    # DEBUG
    rand_indices = [rand_indices[1]]

    print(f"Using images {rand_indices} for JVP/VJP testing")
    safe_interact(local=locals())

    vcs = [viewpoint_stack[i] for i in rand_indices]

    g, start_loss = solver_functions.gradient_and_loss_est(vcs, 1)

    scalar_loss = 0
    for vc in vcs:
        scalar_loss += scalar_training_loss(iteration=jvp_start, opt=opt, viewpoint_cam=vc, gaussians=gaussians, pipe=pipe, bg=background, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight, SPARSE_ADAM_AVAILABLE=SPARSE_ADAM_AVAILABLE, FUSED_SSIM_AVAILABLE=FUSED_SSIM_AVAILABLE)[0]

    eps = 1e-5
    # assert (scalar_loss - start_loss).abs() < eps, f"Loss mismatch {scalar_loss} vs {start_loss}"

    gaussians.zero_grad()
    scalar_loss.backward()
    g_ref = GaussianModelState.from_gaussians_grad(gaussians, param_mask=param_mask, splat_mask=None) * rescale

    g_diff_abs = (g - g_ref).abs()
    diff_norm_sq = solver_functions.dot(g_diff_abs, g_diff_abs)

    print(f"Start loss = {start_loss}, scalar loss {scalar_loss}, ||g - g_ref||^2 = {diff_norm_sq}")



    # CHECK VJP
    test_u_image = GaussianModelState.zero_like_gaussians(gaussians, param_mask=param_mask, splat_mask=None)
    test_v_image = solver_functions.jvp(test_u_image, vcs)
    test_v = test_v_image.as_1d_tensor()

    TEST_ITERATIONS = 500
    for _ in range(TEST_ITERATIONS):
        test_v *= 0

        rand_row_idx = np.random.randint(0, test_v.numel())

        test_v[rand_row_idx] = 10000.0

        test_v_image.load_1d_tensor(test_v)
        vjp = solver_functions.vjp(test_v_image, vcs, 1).as_1d_tensor()

        nonzero_col_indices = torch.nonzero(vjp, as_tuple=False).squeeze()
        perm_idx = torch.randperm(nonzero_col_indices.numel())
        nonzero_col_indices = nonzero_col_indices[perm_idx]

        for col in nonzero_col_indices:

            test_u = test_u_image.as_1d_tensor() * 0
            test_u[col] = 10000.0
            test_u_image.load_1d_tensor(test_u)

            jvp = solver_functions.jvp(test_u_image, vcs).as_1d_tensor()

            print(f"vjp[{col}] = {vjp[col].item()}, jvp[{rand_row_idx}] = {jvp[rand_row_idx].item()}")


            safe_interact(local=locals())

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, jvp_start, val_indices=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set

    if iteration in testing_iterations or (iteration >= jvp_start):
        torch.cuda.empty_cache()
        if val_indices is None:
            num_val_images = 10
            val_stride = max(1, len(scene.getTrainCameras()) // num_val_images)
            val_indices = list(range(0, len(scene.getTrainCameras()), val_stride))
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in val_indices]} )
        print(f"\n[ITER {iteration}] val_indices: {val_indices}")

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
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
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def plot_loss_vs_step_size(iteration, l1_loss, scene : Scene, gaussians_start, renderFunc, renderArgs, train_test_exp, s, end_alpha, train_indices, val_indices, loss_func):

    sample_train_indices = np.random.choice(train_indices, len(val_indices), replace=False)

    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                          {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in sample_train_indices]},
                          {'name': 'val', 'cameras' : [scene.getTrainCameras()[idx] for idx in val_indices]}, )

    print(f"\n[ITER {iteration}] plotting val_indices: {val_indices}")

    test_l1_losses= []
    test_psnrs = []
    test_solver_losses = []
    val_l1_losses= []
    val_psnrs = []
    val_solver_losses = []
    train_l1_losses= []
    train_psnrs = []
    train_solver_losses = []

    step_size = (end_alpha) / 20.0
    with torch.no_grad():
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                gaussians = deepcopy(gaussians_start)
                if loss_func is not None:
                    temp_solver_functions = LinearSolverFunctions(loss_func, gaussians, batch_size=10)
                alpha = 0.0
                while alpha < end_alpha:
                    alpha = alpha + step_size
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

                    if loss_func is not None:
                        loss_scalar = temp_solver_functions.evaluate_loss(config['cameras'], 1)[0]
                    else:
                        loss_scalar = 0

                    print(f"alpha {alpha:.2f} l1 {l1_test:.6f} psnr {psnr_test:.2f}, solver loss {loss_scalar:.6f}")
                    if config['name'] == 'test':
                        test_l1_losses.append(l1_test.item())
                        test_psnrs.append(psnr_test.item())
                        test_solver_losses.append(loss_scalar.item())
                    elif config['name'] == 'val':
                        val_l1_losses.append(l1_test.item())
                        val_psnrs.append(psnr_test.item())
                        val_solver_losses.append(loss_scalar.item())
                    else:
                        train_l1_losses.append(l1_test.item())
                        train_psnrs.append(psnr_test.item())
                        train_solver_losses.append(loss_scalar.item())

    plt.figure(figsize=(17, 5))
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(0, len(train_l1_losses)) * step_size, train_l1_losses, label='Train L1 Loss')
    plt.plot(np.arange(0, len(val_l1_losses)) * step_size, val_l1_losses, label='Val L1 Loss')
    plt.plot(np.arange(0, len(test_l1_losses)) * step_size, test_l1_losses, label='Test L1 Loss')
    plt.xlabel('Step size')
    plt.ylabel('L1 Loss')
    plt.title('L1 Loss vs Step Size (Normalized to PCG Step)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(0, len(train_psnrs)) * step_size, train_psnrs, label='Train PSNR')
    plt.plot(np.arange(0, len(val_psnrs)) * step_size, val_psnrs, label='Val PSNR')
    plt.plot(np.arange(0, len(test_psnrs)) * step_size, test_psnrs, label='Test PSNR')
    plt.xlabel('Step size')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Step Size (Normalized to PCG step)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(0, len(train_solver_losses)) * step_size, train_solver_losses, label='Train Solver Loss')
    plt.plot(np.arange(0, len(val_solver_losses)) * step_size, val_solver_losses, label='Val Solver Loss')
    plt.plot(np.arange(0, len(test_solver_losses)) * step_size, test_solver_losses, label='Test Solver Loss')
    plt.xlabel('Step size')
    plt.ylabel('Solver Loss')
    plt.title('Solver Loss vs Step Size (Normalized to PCG step)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(f"figures/pcg_loss_vs_step_size_{iteration}.png"))

    torch.cuda.empty_cache()


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
    parser.add_argument("--jvp_start", type=int, default = 15001)
    parser.add_argument("--num_images", type=int, default = 5)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.jvp_start, args.num_images)

    # All done
    print("\nTraining complete.")
