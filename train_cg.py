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
    tb_writer = prepare_output_and_logger(dataset)
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
    val_indices = [i for i in range(0, len(viewpoint_stack), 20)]
    val_cameras = [viewpoint_stack[i] for i in val_indices]
    train_indices = [i for i in range(len(viewpoint_stack)) if i not in val_indices]
    train_cameras = [viewpoint_stack[i] for i in train_indices]

    del val_indices, train_indices

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    param_mask = GaussianModelParamGroupMask(mask_xyz=True,
                                             mask_features_dc=False, 
                                             mask_features_rest=False, 
                                             mask_scaling=False, 
                                             mask_rotation=False, 
                                             mask_opacity=False, 
                                             mask_exposure=True)

    P = gaussians.get_xyz.shape[0]

    damp = GaussianModelScaleMatrix(xyz_scale=5e-2, 
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
    rademacher_gen = partial(GaussianModelState.rademacher_like_gaussians, gaussians)
    preconditioner = AdaHessianPreconditioner(rademacher_gen, beta2=0.999, eps=1e-8, hessian_power=1.0)
    pcg_max_iter = 10

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        use_first_order = iteration < jvp_start

        iter_start.record()
        if use_first_order:

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack[rand_idx]
            vind = viewpoint_indices[rand_idx]

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            if iteration in testing_iterations:
                P = gaussians.get_xyz.shape[0]
                print(f"\n[ITER {iteration}] Loss: {loss.item():.6f}, P = {P}")

            loss.backward()

        else:

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 10 == 0:
                gaussians.oneupSHdegree()

            train_cam_provider = CamProvider(train_cameras, mode="random", sample_size=num_images)
            train_cam_provider.sample_new()
            batch_viewpoint_cams = train_cam_provider.get_cur_batch()

            # print(f"\n[ITER {iteration}] Using {num_batch_cameras} random cameras: {rand_indices}")

            # Same background for all cameras in the batch
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            if iteration == jvp_start:
                warmup_sample_size = 50
                warmup_cam_provider = CamProvider(train_cameras, mode="random", max_stride=1, sample_size=warmup_sample_size)
                scale = len(train_cameras) / warmup_sample_size
                preconditioner.reset()
                preconditioner.update(solver_functions.Hv, warmup_cam_provider, scale, num_iter=10)
                D_corrected = preconditioner.D_corrected
                print("preconditioner D_corrected.sqrt() norm ", solver_functions.dot(D_corrected.sqrt(), D_corrected.sqrt()))
            else:

                update_sample_size = 50
                cam_provider = CamProvider(batch_viewpoint_cams, mode="random", sample_size=update_sample_size)
                scale = len(train_cameras) / update_sample_size
                preconditioner.reset()
                preconditioner.update(solver_functions.Hv, cam_provider, scale, num_iter=5)
                D_corrected = preconditioner.D_corrected
                print("preconditioner D_corrected.sqrt() norm ", solver_functions.dot(D_corrected.sqrt(), D_corrected.sqrt()))

            train_scale = len(train_cameras) / len(batch_viewpoint_cams)
            Hx = partial(solver_functions.Hv, viewpoint_cams=batch_viewpoint_cams, scale=train_scale)
            g, start_loss = solver_functions.gradient_and_loss_est(batch_viewpoint_cams, train_scale)
            x0 = solver_functions.get_initial_solution()

            # print("DEBUG use different initial guess")
            # x0 = -g / (g.abs() + 1e-15)
            # init_scale = 0.1
            # x0.xyz_grad *= 0.0001 * init_scale
            # x0.features_dc_grad *= 0.0025 * init_scale
            # x0.features_rest_grad *= 0.0001 * init_scale
            # x0.rotation_grad *= 0.001 * init_scale
            # x0.scaling_grad *= 0.005 * init_scale
            # x0.opacity_grad *= 0.025 * init_scale
            # pcg_max_iter = 50

            s = cg_damped(Ax=Hx,
                          dot=solver_functions.dot,
                          saxpy=solver_functions.saxpy,
                          b=-g,
                          x0=x0,
                          M=preconditioner,
                          max_iter=pcg_max_iter,
                          restart_iter=50)
            
            s = rescale * s

            print("DEBUG copying old gaussians")
            gaussians_old = deepcopy(gaussians)

            # Line search
            alpha = 0.6
            cur_alpha = 0.0
            best_alpha = 0.0
            val_scale = len(val_cameras) / len(val_cameras)
            best_loss = solver_functions.evaluate_loss(val_cameras, val_scale)[0]
            print(f"[ITER {iteration}] Line search start: alpha {cur_alpha}, loss {best_loss:.6f}")
            increase_count = 0
            while True:
                gaussians.update_step(s * (alpha - cur_alpha))
                cur_alpha = alpha
                loss_scalar = solver_functions.evaluate_loss(val_cameras, val_scale)[0]

                print(f"[ITER {iteration}] alpha {cur_alpha}, loss {loss_scalar:.6f}")

                if loss_scalar < best_loss:
                    best_loss = loss_scalar
                    best_alpha = cur_alpha

                if loss_scalar > best_loss:
                    increase_count += 1

                if increase_count >= 5:
                    break

                alpha += 0.1

            gaussians.update_step(s * (best_alpha - cur_alpha))
            best_loss, best_Ll1, best_Ll1depth,  = solver_functions.evaluate_loss(val_cameras, val_scale)
            print(f"[ITER {iteration}] alpha = {best_alpha}, loss = {best_loss}")

            xyz_grad_norm = s.xyz_grad.norm().item()
            features_dc_grad_norm = s.features_dc_grad.norm().item()
            features_rest_grad_norm = s.features_rest_grad.norm().item()
            scaling_grad_norm = s.scaling_grad.norm().item()
            rotation_grad_norm = s.rotation_grad.norm().item()
            opacity_grad_norm = s.opacity_grad.norm().item()
            exposure_grad_norm = s.exposure_grad.norm().item()

            print(f"[ITER {iteration}]")
            print(f"    Gradient norms: xyz {xyz_grad_norm:.4e}, features_dc {features_dc_grad_norm:.4e}, features_rest {features_rest_grad_norm:.4e}, scaling {scaling_grad_norm:.4e}, rotation {rotation_grad_norm:.4e}, opacity {opacity_grad_norm:.4e}, exposure {exposure_grad_norm:.4e}")

            plot_loss_vs_step_size(iteration, l1_loss, scene, gaussians_old, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, s)

            loss, Ll1, Ll1depth = best_loss, best_Ll1, best_Ll1depth


        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, jvp_start)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # Disabling positional gradient based densification
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if use_first_order:
                if iteration < opt.iterations:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        safe_interact(local=locals(), banner="Debugging main optimization loop...")
            

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, jvp_start):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set

    if iteration in testing_iterations or iteration >= jvp_start:
        torch.cuda.empty_cache()
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

def plot_loss_vs_step_size(iteration, l1_loss, scene : Scene, gaussians_start, renderFunc, renderArgs, train_test_exp, s):
    torch.cuda.empty_cache()
    num_val_images = 30
    val_stride = max(1, len(scene.getTrainCameras()) // num_val_images)
    val_indices = list(range(0, len(scene.getTrainCameras()), val_stride))
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                          {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in val_indices]} )

    test_l1_losses= []
    test_psnrs = []
    train_l1_losses= []
    train_psnrs = []

    step_size = 0.1
    num_steps = 20
    with torch.no_grad():
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                gaussians = deepcopy(gaussians_start)
                for i in range(num_steps):
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
    plt.plot(np.arange(0, num_steps) * step_size, train_l1_losses, label='Train L1 Loss')
    plt.plot(np.arange(0, num_steps) * step_size, test_l1_losses, label='Test L1 Loss')
    plt.xlabel('Step size')
    plt.ylabel('L1 Loss')
    plt.title('L1 Loss vs Step Size (Normalized to PCG Step)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, num_steps) * step_size, train_psnrs, label='Train PSNR')
    plt.plot(np.arange(0, num_steps) * step_size, test_psnrs, label='Test PSNR')
    plt.xlabel('Step size')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Step Size (Normalized to PCG step)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(f"figures/pcg_loss_vs_step_size_{iteration}.png"))


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
