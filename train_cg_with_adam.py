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
from utils.ply_utils import write_gaussians_to_ply

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

    damp = GaussianModelScaleMatrix(xyz_scale=5e-2, 
                                    features_dc_scale=5e-2, 
                                    features_rest_scale=5e-2, 
                                    scaling_scale=5e-2, 
                                    rotation_scale=5e-2, 
                                    opacity_scale=5e-2, 
                                    exposure_scale=1e1) * 1e-1

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
    pcg_max_iter = 3

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

        use_first_order = iteration < jvp_start # or iteration % 5 != 1

        iter_start.record()
        if use_first_order:
            print(f"\n[ITER {iteration}] First order optimization")

            gaussians.update_learning_rate(iteration)

            gaussians._xyz.requires_grad_(True)
            gaussians._features_dc.requires_grad_(True)
            gaussians._features_rest.requires_grad_(True)
            gaussians._scaling.requires_grad_(True)
            gaussians._rotation.requires_grad_(True)
            gaussians._opacity.requires_grad_(True)
            gaussians._exposure.requires_grad_(True)

            val_indices = np.random.choice(viewpoint_indices, int(len(viewpoint_indices) * 0.1), replace=False)

            # every 1000 its we increase the levels of sh up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # render
            if (iteration - 1) == debug_from:
                pipe.debug = true

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))

            sgd_num_images = 100
            num_batch_cameras = min(sgd_num_images, len(viewpoint_indices))
            rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)

            # stride = np.random.randint(1, 4)
            # rand_index_start = np.random.randint(0, len(viewpoint_indices) - num_batch_cameras * stride)
            # rand_indices = list(range(rand_index_start, rand_index_start + num_batch_cameras * stride, stride))

            # Same background for all cameras in the batch
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            viewpoint_cams = []
            for rand_idx in rand_indices:
                viewpoint_cam = viewpoint_stack[rand_idx]
                viewpoint_cams.append(viewpoint_cam)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            sgd_loss_func = partial(scalar_training_loss, gaussians=gaussians, iteration=iteration, opt=opt, pipe=pipe, bg=bg, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight)

            gaussians.optimizer.zero_grad()
            gaussians.exposure_optimizer.zero_grad()
            for vc in viewpoint_cams:
                loss, Ll1, Ll1depth = sgd_loss_func(viewpoint_cam=vc)
                loss.backward()

            if iteration in testing_iterations:
                p = gaussians.get_xyz.shape[0]
                print(f"\n[iter {iteration}] loss: {loss.item():.6f}, p = {p}")

            gaussians._xyz.requires_grad_(True)
            gaussians._features_dc.requires_grad_(True)
            gaussians._features_rest.requires_grad_(True)
            gaussians._scaling.requires_grad_(True)
            gaussians._rotation.requires_grad_(True)
            gaussians._opacity.requires_grad_(True)
            gaussians._exposure.requires_grad_(True)

        else:

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 10 == 0:
                gaussians.oneupSHdegree()

            val_indices = np.random.choice(viewpoint_indices, int(len(viewpoint_indices) * 0.1), replace=False)
            val_cameras = [viewpoint_stack[i] for i in val_indices]
            train_indices = [i for i in range(len(viewpoint_stack)) if i not in val_indices]
            train_cameras = [viewpoint_stack[i] for i in train_indices]

            print(f"\n[ITER {iteration}] Second order optimization, training cameras = {train_indices}, val cameras = {val_indices}")

            train_cam_provider = CamProvider(train_cameras, mode="random", sample_size=-1)
            train_cam_provider.sample_new()
            batch_viewpoint_cams = train_cam_provider.get_cur_batch()

            # print(f"\n[ITER {iteration}] Using {num_batch_cameras} random cameras: {rand_indices}")

            # Same background for all cameras in the batch
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            if iteration == jvp_start:
                warmup_sample_size = 20
                warmup_cam_provider = CamProvider(train_cameras, mode="random", max_stride=1, sample_size=warmup_sample_size)
                scale = len(train_cameras) / warmup_sample_size
                preconditioner.reset()
                preconditioner.update(solver_functions.Hv, warmup_cam_provider, scale, num_iter=10)
                D_corrected = preconditioner.D_corrected
                print("preconditioner D_corrected.sqrt() norm ", solver_functions.dot(D_corrected.sqrt(), D_corrected.sqrt()))
            else:

                update_sample_size = 20
                cam_provider = CamProvider(batch_viewpoint_cams, mode="random", sample_size=update_sample_size)
                scale = len(train_cameras) / update_sample_size
                preconditioner.reset()
                preconditioner.update(solver_functions.Hv, cam_provider, scale, num_iter=10)
                D_corrected = preconditioner.D_corrected
                print("preconditioner D_corrected.sqrt() norm ", solver_functions.dot(D_corrected.sqrt(), D_corrected.sqrt()))

            train_scale = len(train_cameras) / len(batch_viewpoint_cams)
            Hx = partial(solver_functions.Hv, viewpoint_cams=batch_viewpoint_cams, scale=train_scale)
            g, start_loss = solver_functions.gradient_and_loss_est(batch_viewpoint_cams, train_scale)
            x0 = solver_functions.get_initial_solution()

            s = cg_damped(Ax=Hx,
                          dot=solver_functions.dot,
                          saxpy=solver_functions.saxpy,
                          b=-g,
                          x0=x0,
                          M=preconditioner,
                          max_iter=pcg_max_iter,
                          restart_iter=50)

            s = rescale * s

            print(f"[ITER {iteration}] Mask out foreground splats")
            xyz_min_step = 0.010
            dist_norms = s.xyz_grad.norm(dim=-1)
            dist_norms = dist_norms / dist_norms.max()
            background_indices = dist_norms < xyz_min_step
            foreground_indices = dist_norms >= xyz_min_step
            safe_interact(local=locals(), banner="Before masking out background splats")
            fg_mask = GaussianModelSplatMask(mask_out_filter=background_indices)
            bg_mask = GaussianModelSplatMask(mask_out_filter=foreground_indices)
            s_bg = g.apply_mask(splat_mask=bg_mask)
            s_bg = -s_bg / (s_bg.abs() + 1e-15) * rescale
            s = s.apply_mask(splat_mask=fg_mask) + s_bg

            print(f"[ITER {iteration}] s dot g = {solver_functions.dot(s, g):.6f}")
            # safe_interact(local=locals(), banner="Before line search")

            print("DEBUG copying old gaussians")
            gaussians_old = deepcopy(gaussians)

            # print("DEBUG writing to ply file")
            # write_gaussians_to_ply(gaussians, s, f"pcg_gaussians_iter{iteration}.ply")
            # safe_interact(local=locals(), banner="Before PCG step")

            print("Line search cameras: ", val_indices)

            # Line search
            alpha = 0.0
            cur_alpha = 0.0
            best_alpha = 0.0
            val_scale = len(val_cameras) / len(val_cameras)
            best_loss = solver_functions.evaluate_loss(val_cameras, val_scale)[0]
            print(f"[ITER {iteration}] Line search start: alpha {cur_alpha:.2f}, loss {best_loss:.6f}")
            increase_count = 0
            while True:
                gaussians.update_step(s * (alpha - cur_alpha))
                cur_alpha = alpha
                loss_scalar = solver_functions.evaluate_loss(val_cameras, val_scale)[0]

                print(f"[ITER {iteration}] alpha {cur_alpha:.2f}, loss {loss_scalar:.6f}")

                if loss_scalar < best_loss:
                    best_loss = loss_scalar
                    best_alpha = cur_alpha
                    increase_count = 0

                if loss_scalar > best_loss:
                    increase_count += 1

                if increase_count >= 5:
                    break

                if alpha == 0.0:
                    alpha += 0.1
                else:
                    alpha *= 1.2

            gaussians.update_step(s * (best_alpha - cur_alpha))
            best_loss, best_Ll1, best_Ll1depth,  = solver_functions.evaluate_loss(val_cameras, val_scale)
            print(f"[ITER {iteration}] alpha = {best_alpha}, loss = {best_loss}")

            # DEBUG
            dot = solver_functions.dot
            adam_step = -g / (g.abs() + 1e-15) * rescale
            adam_end_alpha = math.sqrt(dot(adam_step, adam_step)) / math.sqrt(dot(s, s))
            adam_step = adam_step / adam_end_alpha
            # safe_interact(local=locals(), banner="Before SGD step")

            xyz_grad_norm = s.xyz_grad.norm(dim=-1).mean()
            features_dc_grad_norm = s.features_dc_grad.norm(dim=-1).mean()
            features_rest_grad_norm = s.features_rest_grad.norm(dim=-1).mean()
            scaling_grad_norm = s.scaling_grad.norm(dim=-1).mean()
            rotation_grad_norm = s.rotation_grad.norm(dim=-1).mean()
            opacity_grad_norm = s.opacity_grad.norm(dim=-1).mean()
            exposure_grad_norm = s.exposure_grad.norm(dim=-1).mean()

            xyz_grad_norm_max = s.xyz_grad.norm(dim=-1).max()
            features_dc_grad_norm_max = s.features_dc_grad.norm(dim=-1).max()
            features_rest_grad_norm_max = s.features_rest_grad.norm(dim=-1).max()
            scaling_grad_norm_max = s.scaling_grad.norm(dim=-1).max()
            rotation_grad_norm_max = s.rotation_grad.norm(dim=-1).max()
            opacity_grad_norm_max = s.opacity_grad.norm(dim=-1).max()
            exposure_grad_norm_max = s.exposure_grad.norm(dim=-1).max()

            print(f"[ITER {iteration}]")
            print(f"    Gradient norms: xyz {xyz_grad_norm:.4e}, features_dc {features_dc_grad_norm:.4e}, features_rest {features_rest_grad_norm:.4e}, scaling {scaling_grad_norm:.4e}, rotation {rotation_grad_norm:.4e}, opacity {opacity_grad_norm:.4e}, exposure {exposure_grad_norm:.4e}")
            print(f"    Gradient max norms: xyz {xyz_grad_norm_max:.4e}, features_dc {features_dc_grad_norm_max:.4e}, features_rest {features_rest_grad_norm_max:.4e}, scaling {scaling_grad_norm_max:.4e}, rotation {rotation_grad_norm_max:.4e}, opacity {opacity_grad_norm_max:.4e}, exposure {exposure_grad_norm_max:.4e}")

            end_alpha = adam_end_alpha * 1.5 if adam_end_alpha > alpha else alpha * 1.5

            plot_loss_vs_step_size(iteration, l1_loss, scene, gaussians_old, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, s, mid_alpha=alpha, end_alpha=end_alpha, train_indices=train_indices, val_indices=val_indices, loss_func=loss_func, image_name="hybrid_all")

            plot_loss_vs_step_size(iteration, l1_loss, scene, gaussians_old, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, adam_step, mid_alpha=alpha, end_alpha=end_alpha, train_indices=train_indices, val_indices=val_indices, loss_func=loss_func, image_name="sgd_all")

            loss, Ll1, Ll1depth = best_loss, best_Ll1, best_Ll1depth

            # safe_interact(local=locals(), banner="After PCG step")


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
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, jvp_start, val_indices)
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
                    print("[ITER {}] First order optimizer step".format(iteration))
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

def plot_loss_vs_step_size(iteration, l1_loss, scene : Scene, gaussians_start, renderFunc, renderArgs, train_test_exp, s, mid_alpha, end_alpha, train_indices, val_indices, loss_func, image_name):

    sample_train_indices = np.random.choice(train_indices, len(val_indices), replace=False)

    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                          {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in sample_train_indices]},
                          {'name': 'val', 'cameras' : [scene.getTrainCameras()[idx] for idx in val_indices]}, )

    print(f"\n[ITER {iteration}] plotting val_indices: {val_indices}, mid_alpha: {mid_alpha}, end_alpha: {end_alpha}")

    alphas = []
    test_l1_losses= []
    test_psnrs = []
    test_solver_losses = []
    val_l1_losses= []
    val_psnrs = []
    val_solver_losses = []
    train_l1_losses= []
    train_psnrs = []
    train_solver_losses = []

    with torch.no_grad():
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                gaussians = deepcopy(gaussians_start)
                if loss_func is not None:
                    temp_solver_functions = LinearSolverFunctions(loss_func, gaussians, batch_size=10)
                alpha = 0.0
                while alpha < end_alpha:
                    if alpha > mid_alpha:
                        step_size = (end_alpha - mid_alpha) / 20.0
                    else:
                        step_size = (mid_alpha) / 20.0
                    
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

                    print(f"{config['name']} alpha {alpha:.2f} l1 {l1_test:.6f} psnr {psnr_test:.2f}, solver loss {loss_scalar:.6f}")
                    if config['name'] == 'test':
                        test_l1_losses.append(l1_test.item())
                        test_psnrs.append(psnr_test.item())
                        test_solver_losses.append(loss_scalar.item())
                        alphas.append(alpha)
                    elif config['name'] == 'val':
                        val_l1_losses.append(l1_test.item())
                        val_psnrs.append(psnr_test.item())
                        val_solver_losses.append(loss_scalar.item())
                    else:
                        train_l1_losses.append(l1_test.item())
                        train_psnrs.append(psnr_test.item())
                        train_solver_losses.append(loss_scalar.item())

                    gaussians.update_step(step_size * s)
                    alpha = alpha + step_size

    plt.figure(figsize=(17, 9))
    plt.subplot(1, 3, 1)
    plt.plot(alphas, train_l1_losses, label='Train L1 Loss')
    plt.plot(alphas, val_l1_losses, label='Val L1 Loss')
    plt.plot(alphas, test_l1_losses, label='Test L1 Loss')
    plt.xlabel('Step size')
    plt.ylabel('L1 Loss')
    plt.title('L1 Loss vs Step Size (Normalized to PCG Step)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(alphas, train_psnrs, label='Train PSNR')
    plt.plot(alphas, val_psnrs, label='Val PSNR')
    plt.plot(alphas, test_psnrs, label='Test PSNR')
    plt.xlabel('Step size')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Step Size (Normalized to PCG step)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(alphas, train_solver_losses, label='Train Solver Loss')
    plt.plot(alphas, val_solver_losses, label='Val Solver Loss')
    plt.plot(alphas, test_solver_losses, label='Test Solver Loss')
    plt.xlabel('Step size')
    plt.ylabel('Solver Loss')
    plt.title('Solver Loss vs Step Size (Normalized to PCG step)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(f"{image_name}_loss_vs_step_size_iter_{iteration}.png"))

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
