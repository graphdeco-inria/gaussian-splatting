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
from solver.diagonal_estimator import restarted_squared_hutchinson
from solver.solver_functions import construct_loss_func, construct_g_func, construct_JTJv_func, dot, saxpy

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

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class CamProvider:
    def __init__(self, viewpoint_stack):
        self.viewpoint_stack = viewpoint_stack

    def sample_new(self, batch_size):
        indices = np.random.choice(len(self.viewpoint_stack), batch_size, replace=True)
        viewpoint_batch = [self.viewpoint_stack[idx] for idx in indices]
        return viewpoint_batch, len(self.viewpoint_stack) / batch_size

def JTJv_hat(v, JTJv_func, gaussians, cam_provider, batch_size, S=None, damp=0.0):
    viewpoint_batch, scale = cam_provider.sample_new(batch_size=batch_size)
    return JTJv_func(v=v, gaussians=gaussians, viewpoint_cams=viewpoint_batch, scale=scale, S=S, damp=damp)

def compute_ref_loss(ref_loss_func, gaussians, viewpoint_cams, scale):
    ref_loss = 0.0
    for vc_i, vc in enumerate(viewpoint_cams):
        ref_loss_i = ref_loss_func(gaussians=gaussians, viewpoint_cam=vc) ** 2
        ref_loss_i *= scale
        ref_loss += ref_loss_i.item()
    return ref_loss


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

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    lr = GaussianModelVector(xyz=xyz_lr,  
                             features_dc=opt.features_dc_lr, 
                             features_rest=opt.features_rest_lr,
                             scaling=opt.scaling_lr,
                             rotation=opt.rotation_lr,
                             opacity=opt.opacity_lr,
                             exposure=opt.exposure_lr,
                             gaussians=gaussians)
    S = GaussianModelVector(xyz=opt.xyz_scale,
                            features_dc=opt.features_dc_scale,
                            features_rest=opt.features_rest_scale,
                            scaling=opt.scaling_scale,
                            rotation=opt.rotation_scale,
                            opacity=opt.opacity_scale,
                            exposure=opt.exposure_scale,
                            gaussians=gaussians)

    D_est_t = GaussianModelVector.ones_like(gaussians)
    D_est_smoothed = 0
    g_smoothed = 0
    denom_iter = 0

    damp = 1e-12
                             

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

    num_batch_cameras = len(viewpoint_indices) if opt.num_images == -1 else min(opt.num_images, len(viewpoint_indices))
    # rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)
    rand_indices = [12, 15, 52, 66, 89, 101, 133, 150, 166, 176, 214, 215, 221, 223, 225, 226, 246, 261, 285, 296]
    scale = len(viewpoint_indices) / num_batch_cameras

    for rand_idx in rand_indices:
        viewpoint_cam = viewpoint_stack[rand_idx]

    ref_loss_func = partial(reference_training_loss, iteration=first_iter, opt=opt, pipe=pipe, bg=background, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=opt.disable_ssim, pixel_mask=None)

    param_groups = ["xyz", "features_dc", "features_rest", "scaling", "rotation", "opacity"]
    param_group_noise_scales = {"xyz": 1.0, "features_dc": 10, "features_rest": 1.0, "scaling": 10, "rotation": 10, "opacity": 100}
    noise_levels = [1e-6, 1e-5, 1e-4, 1e-3]
    NUM_ITERS = 1000
    losses = {}
    image_losses0 = {}

    # for param_group in param_groups:
    #     losses[param_group] = {}
    #     for noise_level in noise_levels:
    #         losses[param_group][noise_level] = {}
    #         for idx in rand_indices:
    #             losses[param_group][noise_level][idx] = []

    for idx in rand_indices:
        ref_loss_i = ref_loss_func(gaussians=gaussians, viewpoint_cam=viewpoint_stack[idx]) ** 2
        image_losses0[idx] = ref_loss_i.item()


    for param_group in param_groups:
        if param_group not in losses.keys():
            losses[param_group] = {}

        for base_noise_level in noise_levels:
            noise_level = base_noise_level * param_group_noise_scales[param_group]

            if noise_level not in losses[param_group].keys():
                losses[param_group][noise_level] = {}
                for idx in rand_indices:
                    losses[param_group][noise_level][idx] = []


            print(f"Processing param group {param_group} noise level {noise_level}")

            for _ in range(NUM_ITERS):
                gaussians_copy = deepcopy(gaussians)

                noise_shape = getattr(gaussians_copy, f"_{param_group}").shape
                noise_tensor = torch.randn(noise_shape, device="cuda") * noise_level

                noise = GaussianModelVector(xyz=0.0,
                                            features_dc=0.0,
                                            features_rest=0.0,
                                            scaling=0.0,
                                            rotation=0.0,
                                            opacity=0.0,
                                            exposure=0.0,
                                            gaussians=gaussians)

                setattr(noise, param_group, noise_tensor)

                gaussians_copy.update_step(noise)

                for idx in rand_indices:
                    ref_loss_i = ref_loss_func(gaussians=gaussians_copy, viewpoint_cam=viewpoint_stack[idx]) ** 2
                    losses[param_group][noise_level][idx].append(ref_loss_i.item())

        torch.save({"noisy_losses": losses, "ref_losses": image_losses0}, f"noise_tol_losses_iter{first_iter}.pth")


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, jvp_start, val_indices=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set

    if iteration in testing_iterations or (iteration >= jvp_start):
        torch.cuda.empty_cache()
        if val_indices is None:
            num_val_images = 100
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

def prepare_output_and_logger(args, opt):    
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
        cfg_log_f.write('\n')
        cfg_log_f.write(str(Namespace(**vars(opt))))
        cfg_log_f.write('\n')

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


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
