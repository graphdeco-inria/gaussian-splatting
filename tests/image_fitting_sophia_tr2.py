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

from image_fitting_utils import prepare_output_and_logger, get_image_name

from copy import deepcopy

from matplotlib import pyplot as plt

from utils.gif_renderer import GifRenderer

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, save_checkpoint_dir, checkpoint_iterations, checkpoint, debug_from, image_path, num_points, gif_interval):
    tb_writer = prepare_output_and_logger(args)
    
    cameras = [build_camera(image_path)]

    ####### Some fixed parameters #########
    num_images = 1
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    image_name = get_image_name(image_path)
    model_path = save_checkpoint_dir
    ####### Some fixed parameters #########

    if model_path != "":
        os.makedirs(model_path, exist_ok=True)

    first_iter = 0
    sh_degree = 3
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
    training_report(None, first_iter + 1, 0, 0, l1_loss, 0, testing_iterations, cameras, gaussians, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, train_test_exp), train_test_exp)

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

    sophia_optimizer = SophiaOptimizer(lr=lr, 
                                       betas=(opt.adahessian_beta1, opt.adahessian_beta2),
                                       eps=1e-15, clip=False,
                                       gamma=opt.sophia_gamma,
                                       diagonal_update_interval=opt.diagonal_update_interval,
                                       num_init_iter=opt.diagonal_init_iter,
                                       num_init_restart_iter=opt.diagonal_init_restart_iter,
                                       num_update_iter=opt.diagonal_update_iter,
                                       num_update_restart_iter=opt.diagonal_update_restart_iter
                                       )

    sophia_losses = []
    sophia_images = []

    sophia_optimizer.reset()
    sophia_optimizer.set_clip(False)

    adam_optimizer = AdamOptimizer(lr=lr, betas=(opt.adam_beta1, opt.adam_beta2), eps=1e-15, clip=False)
    adam_optimizer.reset()

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

        with torch.no_grad():

            iter_start.record()
    
            gaussians.update_learning_rate(iteration)
    
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 100 == 0:
                gaussians.oneupSHdegree()
    
            # Pick the only Camera
            viewpoint_cam = cameras[0]
    
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
    
            bg = torch.rand((3), device="cuda") if opt.random_background else background
    
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
                           }
            loss_func = construct_loss_func(**render_args)
            g_func = construct_g_func(**render_args)
            JTJv_func = construct_JTJv_func(**render_args)
            Dhat_func = construct_Dhat_func(**render_args)
            z_gen_func = partial(GaussianModelVector.rademacher_like, gaussians)
    
    
            JTJv_func1 = partial(JTJv_func, gaussians=gaussians, viewpoint_cams=cameras, S=S, scale=1)
            Dhat_func1 = partial(Dhat_func, gaussians=gaussians, viewpoint_cams=cameras)
    
            loss_sophia_tr, g, batch_stats = g_func(gaussians=gaussians, viewpoint_cams=cameras, return_stats=True, debug_loss=False)
            loss_sophia_tr = torch.tensor(loss_sophia_tr)
            # print(f"Iteration {iteration}, Sophia TR loss: {loss_sophia_tr.item():.10f}")

            if iteration % gif_interval == 0:
                sophia_losses.append(loss_sophia_tr.item())
                image_sophia_tr = batch_stats[0]["images"][0]
                sophia_images.append(image_sophia_tr.detach())

            # TODO: Fix this. Using L1 loss to get correct batch stats
            if opt.loss_type != "l1" and opt.naive_densification:
                _, _, batch_stats = g_func(gaussians=gaussians, viewpoint_cams=cameras, return_stats=True, loss_type="l1")

            image_sophia_tr = batch_stats[0]["images"][0]
            visibility_filter = batch_stats[0]["visibility_filter"]
            radii = batch_stats[0]["max_radii"]
            viewspace_point_tensor = batch_stats[0]["viewspace_point_tensor"]

            # DEBUG: Fix this more formally
            if opt.loss_type == "huber":
                viewspace_point_tensor.grad /= opt.huber_delta

            s_sophia_tr = sophia_optimizer.get_update(g, JTJv_func1, Dhat_func1, z_gen_func, S)
            s_sophia_tr_old = s_sophia_tr.clone()
    
            s_sophia_tr = clip_kl(gaussians, s_sophia_tr, opt.kl_threshold,
                                  lr.features_dc, lr.features_rest, lr.opacity)

            if opt.use_adam:
                print("Using adam step")
                s_sophia_tr = adam_optimizer.get_update(g)
    
    
            loss = loss_sophia_tr
            Ll1 = torch.tensor(0.0)
            Ll1depth = torch.tensor(0.0)
    
            iter_end.record()
            torch.cuda.synchronize()

            # safe_interact(local=locals(), banner=f"Debug prompt at iteration {iteration}")
    

            # Progress bar
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, cameras, gaussians, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, train_test_exp), train_test_exp)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # Disabling positional gradient based densification
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    prune_mask = gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold, radii)
                    sophia_optimizer.densify_and_prune(prune_mask)
                    adam_optimizer.densify_and_prune(prune_mask)
                    s_sophia_tr.densify_and_prune_(prune_mask)

                    # safe_interact(local=locals(), banner=f"After densification at iteration {iteration} prompt")
                
                if iteration % opt.opacity_reset_interval == 0 or (white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.update_step(s_sophia_tr)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                print("Model path: {}".format(model_path + "/chkpnt" + str(iteration) + ".pth"))
                torch.save((gaussians.capture(), iteration), model_path + "/chkpnt" + str(iteration) + ".pth")
                import code; code.interact(local=locals(), banner="Debug prompt after saving")
            # safe_interact(local=locals(), banner="After iteration prompt")

    gif_renderer = GifRenderer(num_rows=1, num_cols=2, figsize=(10, 6), gif_interval=1)
    gif_renderer.add_gt(0, 0, viewpoint_cam.original_image)
    gif_renderer.add_series(0, 1, sophia_images, sophia_losses, title="Sophia TR (Ours)")
    gif_renderer.animate(f"figures/image_fitting_sophia_tr.gif", interval=500)
    print(f"save figures/image_fitting_sophia_tr.gif")

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, cameras, gaussians, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set

    if iteration in testing_iterations or iteration % 100 == 0:
        print(gaussians._xyz.shape)
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
            # tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
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
    parser.add_argument("--save_checkpoint_dir", type=str, default="")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--num_points", type=int, default=2_000)
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--gif_interval", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.checkpoint_iterations != []:
        assert args.save_checkpoint_dir != "", "Please provide a directory to save checkpoints."
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args), 
             pp.extract(args), 
             args.test_iterations, 
             args.save_iterations, 
             args.save_checkpoint_dir,
             args.checkpoint_iterations, 
             args.start_checkpoint, 
             args.debug_from, 
             args.image_path, 
             args.num_points,
             args.gif_interval)

    # All done
    print("\nTraining complete.")
