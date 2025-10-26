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
from solver.gaussian_model_state import GaussianModelState
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

def training(opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, image_path, num_points, all_columns, sum_column):
    rescale = GaussianModelScaleMatrix(xyz_scale=0.0001, 
                                      features_dc_scale=0.0025, 
                                      features_rest_scale=0.0001, 
                                      scaling_scale=0.005, 
                                      rotation_scale=0.001, 
                                      opacity_scale=0.025, 
                                      exposure_scale=1.0)
    
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

        gaussians.zero_grad()
        for vc in viewpoint_cams:
            ref_loss_scalar_i = reference_training_loss(iteration, opt, vc, gaussians, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight)
            ref_loss_scalar_i.backward()
        ref_g = GaussianModelState.from_gaussians_grad(gaussians)

        # Test vector loss prediction using J

        render_pkg = render(viewpoint_cams[0], gaussians, pipe, bg, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        loss_func = partial(batch_training_loss, iteration=iteration, opt=opt, pipe=pipe, bg=background, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=False)
        cur_state_gn = LinearSolverFunctions(loss_func, gaussians, batch_size=5, param_mask=None, damp=None, splat_mask=None, rescale=rescale)

        g_vec = ref_g.as_1d_tensor(with_features_rest=False, with_exposure=False)
        u = GaussianModelState.zero_like_gaussians(gaussians)
        u_vec = u.as_1d_tensor(with_features_rest=False, with_exposure=False)
        v = cur_state_gn.jvp(u, viewpoint_cams)
        v_vec = v.as_1d_tensor()
        m, n = v_vec.shape[0], u_vec.shape[0]

        # max_indices = {347: 17145, 555: 11708, 687: 6209, 1160: 5782, 1197: 9095, 1197: 9095, 1365: 18183}
        # max_indices = {0: -1, 1: -1, 403: -1, 324: -1, 617: -1}
        # max_indices = {617: -1}
        # max_indices = {1379: -1}
        max_indices = {56: -1}

        cols_list = range(u_vec.shape[0]) if all_columns else list(max_indices.keys())

        s_adam = -ref_g / (ref_g.abs() + 1e-15) * rescale
        s_adam_vec = s_adam.as_1d_tensor(with_features_rest=False, with_exposure=False)

        for i in cols_list:
            print("JVP column:", i, "/", u_vec.shape[0])
            u_vec *= 0.0
            u_vec[i] = 1.0
            u.load_1d_tensor(u_vec, with_features_rest=False, with_exposure=False)
            Ji = cur_state_gn.jvp(u, viewpoint_cams).as_1d_tensor()
            loss_vec0 = loss_func(gaussians=gaussians, viewpoint_cams=viewpoint_cams).as_1d_tensor()

            deltas = {}
            predicted_deltas = {}

            with torch.no_grad():
                if all_columns:
                    step_range = range(-10, 10, 1)
                    step_size = 1e-2
                else:
                    # step_range = range(-1000, 1000, 1)
                    # step_size = 1e-4
                    step_range = range(-10, 10, 1)
                    step_size = 1e-2

                for step in step_range:
                    alpha = step * step_size
                    gaussians_copy = deepcopy(gaussians)
                    gaussians_copy.update_step(alpha * u)
                    loss_vec = loss_func(gaussians=gaussians_copy, viewpoint_cams=viewpoint_cams).as_1d_tensor()
                    print(f"alpha = {alpha:4f}, loss = {loss_vec.norm() ** 2:4e}")

                    delta = loss_vec - loss_vec0
                    predicted_delta = alpha * Ji

                    error_vec = delta - predicted_delta

                    deltas[alpha] = delta
                    predicted_deltas[alpha] = predicted_delta

                    max_idx = error_vec.abs().argmax().item()

                if not sum_column:
                    max_idx = max_indices.get(i, max_idx)

                    alphas = list(deltas.keys())
                    max_delta = [deltas[alpha][max_idx].item() for alpha in alphas]
                    max_predicted_delta = [predicted_deltas[alpha][max_idx].item() for alpha in alphas]
                else:
                    alphas = list(deltas.keys())
                    max_delta = [deltas[alpha].sum().item() for alpha in alphas]
                    max_predicted_delta = [predicted_deltas[alpha].sum().item() for alpha in alphas]

                if g_vec[i] != 0.0:
                    plt.figure()
                    plt.plot(alphas, max_delta, label="Actual delta", alpha=0.5)
                    plt.plot(alphas, max_predicted_delta, label="Predicted delta", alpha=0.5)
                    plt.xlabel("Alpha")
                    plt.ylabel("Delta")
                    plt.title(f"JVP Column {i} sum check")
                    plt.ylim(-0.1, 0.1)

                    plt.axvline(x=s_adam_vec[i].item(), color='r', linestyle='--', label='Adam step')

                    plt.legend()

                    plt.savefig(f"figures/jvp_column_sum/jvp_column_sum_col{i:04d}.png")
                plt.close('all')

                    # safe_interact(local=locals(), banner="Saved JVP column check figure")

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

    if iteration in testing_iterations or iteration % 100 == 0:
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
