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

def compute_ref_loss(ref_loss_func, gaussians, viewpoint_cams, scale):
    ref_loss = 0.0
    for vc_i, vc in enumerate(viewpoint_cams):
        ref_loss_i = ref_loss_func(gaussians=gaussians, viewpoint_cam=vc) ** 2
        ref_loss_i *= scale
        ref_loss += ref_loss_i.item()
    return ref_loss

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, jvp_start, num_images):
    splat_mask = None

    ####### Some fixed parameters #########
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    model_path = ""
    ####### Some tunable parameters #########
    disable_ssim = True

    linesearch_alpha = 1e-0
    linesearch_alpha_min = 1e-2
    # linesearch_alpha = 1.0
    # linesearch_alpha_min = 1.0
    linesearch_gs_min = 1e-12
    linesearch_alpha_decrease = 0.8
    linesearch_alpha_increase = 1.2
    linesearch_alpha_c = 0.01
    linesearch_force_minstep = True

    damp_alpha_max = 0.2
    damp_alpha_min = 1e-2
    damp_increase = 1.5
    damp_increase_high = 10.0
    damp_decrease = 0.6

    pixel_sample_rate_max = 1.0
    pixel_sample_rate_min = 1.0
    pixel_sample_rate = pixel_sample_rate_max
    pixel_sample_rate_increase = 1.2
    pixel_sample_rate_decrease = 0.9

    splat_sample_update_freq = 20
    splat_sample_rate = 1.0

    pcg_num_iter = 2
    pcg_restart_iter = 5
    pcg_tol = 1e-15

    preconditioner_reset = True
    preconditioner_reset_iter = 200
    preconditioner_warmup_iter = 1

    scale_const = 1e0
    xyz_scale_init = 1.6e-4 * scale_const * 1.0
    xyz_scale_final = 1.6e-6 * scale_const * 1.0
    # xyz_scale_init = 1.6e-4 * scale_const * 1.0
    # xyz_scale_final = 1.6e-4 * scale_const * 1.0
    xyz_scale_decay = 0.999
    xyz_scale_max_steps = opt.iterations
    xyz_scale = xyz_scale_init
    # xyz_scale = xyz_scale_init * (xyz_scale_decay ** (min(opt.iterations, xyz_scale_max_steps) / xyz_scale_max_steps))

    features_dc_scale = 2.5e-3 * scale_const * 1.0
    featuress_rest_scale = features_dc_scale / 20.0
    scaling_scale = 5e-3 * scale_const * 1e+1 * 1.0
    rotation_scale = 1e-3 * scale_const * 1e-4 * 1.0
    opacity_scale = 2.5e-2 * scale_const * 1.0
    exposure_scale = 1.0 * scale_const * 1.0

    rescale = GaussianModelScaleMatrix(xyz_scale=xyz_scale,  
                                      features_dc_scale=features_dc_scale, 
                                      features_rest_scale=featuress_rest_scale, 
                                      scaling_scale=scaling_scale, 
                                      rotation_scale=rotation_scale, 
                                      opacity_scale=opacity_scale, 
                                      exposure_scale=1.0)

    # NOTE: damp needs to be relative to scale^2
    damp_init = 1e-9 * (scale_const ** 2)      
    damp_min = 1e-9 * (scale_const ** 2)       
    damp_max = 1e-2 * (scale_const ** 2)       
    # damp_init = 1e-4 * (scale_const ** 2)      
    # damp_min = 1e-5 * (scale_const ** 2)       
    # damp_max = 1e+4 * (scale_const ** 2)       
    damp_res_target = 1e-4
    damp = damp_init

    noise_opacity_thresh = 0.995
    noise_lr = 5e4
    clip_thresh = 4e0

    ####### Some tunable parameters #########

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
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

    with torch.no_grad():
        training_report(None, first_iter, None, None, l1_loss, None, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, jvp_start, val_indices=None)

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

        num_batch_cameras = min(num_images, len(viewpoint_indices))
        rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)
        scale = len(viewpoint_indices) / num_batch_cameras

        viewpoint_cams = []
        for rand_idx in rand_indices:
            viewpoint_cam = viewpoint_stack[rand_idx]
            viewpoint_cams.append(viewpoint_cam)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration > 1200:
            if iteration % splat_sample_update_freq == 0:
                num_gaussians = gaussians.get_xyz.shape[0]
                splat_mask_out = torch.rand(num_gaussians, device="cuda") > splat_sample_rate if splat_sample_rate < 1.0 else None
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

        # # DEBUG
        # ref_loss = 0.0
        # gaussians.zero_grad()
        # for vc_i, vc in enumerate(viewpoint_cams):
        #     with torch.enable_grad():
        #         ref_loss_i = reference_training_loss(iteration, opt, vc, gaussians, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=disable_ssim, pixel_mask=pixel_mask) ** 2
        #         ref_loss_i *= scale
        #         ref_loss += ref_loss_i.item()
        #         ref_loss_i.backward()
        # ref_g = GaussianModelState.from_gaussians_grad(gaussians)
        # # DEBUG END


        loss_func = partial(batch_training_loss, iteration=iteration, opt=opt, pipe=pipe, bg=background, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=disable_ssim, pixel_mask=pixel_mask)
        cur_state = LinearSolverFunctions(loss_func, gaussians, batch_size=5, param_mask=None, splat_mask=splat_mask, rescale=rescale, damp=damp)

        SHSx = partial(cur_state.JTJv, viewpoint_cams=viewpoint_cams, scale=scale, use_rescale=True, use_damping=True)

        warmup_sample_size = min(1, len(viewpoint_cams))
        warmup_scale = len(viewpoint_stack) / warmup_sample_size
        warmup_cam_provider = CamProvider(viewpoint_cams, mode="random", max_stride=1, sample_size=warmup_sample_size)

        if preconditioner is None or preconditioner_reset:
            rademacher_gen = partial(GaussianModelState.rademacher_like_gaussians, gaussians)
            preconditioner = AdaHessianPreconditioner(rademacher_gen, beta2=0.999, eps=1e-16, hessian_power=1.0)
            preconditioner.reset()
            preconditioner.update(SHSx, warmup_cam_provider, warmup_scale, num_iter=preconditioner_reset_iter)
        else:
            preconditioner.update(SHSx, warmup_cam_provider, warmup_scale, num_iter=preconditioner_warmup_iter)

        start_loss, Sg = cur_state.g(viewpoint_cams, scale, use_rescale=True, return_loss=True)
        g = cur_state.g(viewpoint_cams, scale, use_rescale=False)

        print("start loss:", start_loss)

        # Sg_vec = Sg.as_1d_tensor(with_features_rest=False, with_exposure=False)
        # safe_interact(local=locals(), banner="Before CG solve")
        # Sg_vec += torch.randn_like(Sg_vec) * 1e-10
        # Sg.load_1d_tensor(Sg_vec, with_features_rest=False, with_exposure=False)

        Sg_norm = Sg.dot(Sg)
        if Sg_norm == 0.0:
            safe_interact(local=locals(), banner="1 Zero gradient detected, stopping training.")


        x0 = cur_state.get_initial_solution()

        y, res, num_iter, res_reduc = cg_damped(Ax=SHSx,
                      dot=cur_state.dot,
                      saxpy=cur_state.saxpy,
                      b=-Sg,
                      x0=x0,
                      M=preconditioner,
                      max_iter=pcg_num_iter,
                      # max_iter=1,
                      restart_iter=pcg_restart_iter,
                      verbose=True,
                      tol=pcg_tol,)

        y_norm = math.sqrt(y.dot(y))
        print(f"y norm: {y_norm:.2e}")
        if y_norm == 0.0:
            print("Zero step detected, skipping update.")
            continue

        # DEBUG 1
        y_vec = y.as_1d_tensor()
        y_vec.clip_(min=-clip_thresh, max=clip_thresh)
        y.load_1d_tensor(y_vec)

        # # DEBUG 2
        # # safe_interact(local=locals(), banner="after CG solve")
        # low_clip_thresh = clip_thresh * 0.1
        # opacity_mask = (gaussians.get_opacity >= 0.0).squeeze(-1)
        # y.xyz_grad[opacity_mask].clip_(min=-clip_thresh, max=clip_thresh)
        # y.features_dc_grad[opacity_mask].clip_(min=-clip_thresh, max=clip_thresh)
        # y.features_rest_grad[opacity_mask].clip_(min=-clip_thresh, max=clip_thresh)
        # y.scaling_grad[opacity_mask].clip_(min=-clip_thresh, max=clip_thresh)
        # y.rotation_grad[opacity_mask].clip_(min=-clip_thresh, max=clip_thresh)
        # y.opacity_grad[opacity_mask].clip_(min=-clip_thresh, max=clip_thresh)
        # y.xyz_grad[~opacity_mask].clip_(min=-low_clip_thresh, max=low_clip_thresh)
        # y.features_dc_grad[~opacity_mask].clip_(min=-low_clip_thresh, max=low_clip_thresh)
        # y.features_rest_grad[~opacity_mask].clip_(min=-low_clip_thresh, max=low_clip_thresh)
        # y.scaling_grad[~opacity_mask].clip_(min=-low_clip_thresh, max=low_clip_thresh)
        # y.rotation_grad[~opacity_mask].clip_(min=-low_clip_thresh, max=low_clip_thresh)
        # y.opacity_grad[~opacity_mask].clip_(min=-low_clip_thresh, max=low_clip_thresh)


        # DEBUG 2
        # print("DEBUG 2: Overriding Newton step with -g")
        # s_newton = -g / (g.abs() + 1e-15) * rescale

        s_newton = rescale * y


        # DEBUG 2
        # s_newton_vec = s_newton.as_1d_tensor()
        # s_newton_xyz_max = s_newton.xyz_grad.abs().max()
        # s_newton_vec = s_newton_vec * 1e-3 / s_newton_xyz_max
        # s_newton.load_1d_tensor(s_newton_vec)


        gs_newton = g.dot(s_newton)
        negative_search_direction = res_reduc > 1e3

        print(f"gs_newton: {gs_newton:.6e}, res_reduc: {res_reduc:.6e}")

        # import code; code.interact(local=locals(), banner="after loss compute")

        ref_loss_func = partial(reference_training_loss, iteration=iteration, opt=opt, pipe=pipe, bg=bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=disable_ssim, pixel_mask=None)

        with torch.no_grad():
            loss_0 = compute_ref_loss(ref_loss_func, gaussians, viewpoint_stack, 1.0)
            loss_0_test = compute_ref_loss(ref_loss_func, gaussians, test_viewpoint_stack, 1.0)

            print("Starting linesearch from loss_0:", loss_0, " test loss_0_test:", loss_0_test)

            alpha = linesearch_alpha
            while True:

                gaussians_copy = deepcopy(gaussians)
                gaussians_copy.update_step(alpha * s_newton)

                loss_alpha = compute_ref_loss(ref_loss_func, gaussians_copy, viewpoint_stack, 1.0)
                loss_alpha_test = compute_ref_loss(ref_loss_func, gaussians_copy, test_viewpoint_stack, 1.0)
                print(f" Linesearch alpha: {alpha:.3e}, loss_alpha: {loss_alpha:.6f}, test loss_alpha_test: {loss_alpha_test:.6f}")


                if math.fabs(gs_newton) < linesearch_gs_min:
                    print("Gradient too small in linesearch.")
                    alpha = linesearch_alpha_min
                    break

                if negative_search_direction:
                    print("Non-descent direction detected in linesearch.")
                    alpha = 0.0
                    break

                # Check Armijo condition
                if loss_alpha <= loss_0 + linesearch_alpha_c * alpha * gs_newton:
                    break

                alpha *= linesearch_alpha_decrease

                if alpha < linesearch_alpha_min:
                    if linesearch_force_minstep:
                        print("Linesearch alpha below minimum. Forcing step at alpha =", alpha)
                    else:
                        alpha = 0.0
                    break

            print(f"Linesearch found alpha: {alpha:.3e}, loss_alpha: {loss_alpha:.6f}, loss_alpha_test: {loss_alpha_test:.6f}")



            print("Update with s_newton")
            gaussians.update_step(alpha * s_newton)


            iter_end.record()

            loss, Ll1, Ll1depth  = cur_state.evaluate_loss(viewpoint_cams, scale)
            print("\n\nPost-update loss (no pixel mask):", loss.item())
            print(f"damp: {damp}, pixel_sample_rate: {pixel_sample_rate}\n\n")


            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, jvp_start, val_indices=None)

            if tb_writer:
                tb_writer.add_scalar('train_loss_full/alpha', alpha, iteration)
                tb_writer.add_scalar('train_loss_full/pixel_sample_rate', pixel_sample_rate, iteration)
                tb_writer.add_scalar('train_loss_full/damping', damp, iteration)

            # Update pixel sampling rate
            if alpha < 0.01 or (y_norm == 0 and damp >= damp_max):
                pixel_sample_rate *= pixel_sample_rate_decrease
                pixel_sample_rate = max(pixel_sample_rate, pixel_sample_rate_min)
            elif alpha > 0.01:
                pixel_sample_rate *= pixel_sample_rate_increase
                pixel_sample_rate = min(pixel_sample_rate, pixel_sample_rate_max)

            # Update Damping Factor
            if num_iter == 0 or negative_search_direction:
                damp *= damp_increase_high
                damp = min(damp, damp_max)
            elif res_reduc < 1e-1:
                damp *= damp_decrease
                damp = max(damp, damp_min)
            elif res_reduc > 1e2:
                damp *= damp_increase
                damp = min(damp, damp_max)
            elif res < damp_res_target:
                damp *= damp_decrease
                damp = max(damp, damp_min)

            # Update rescale for xyz
            rescale.xyz_scale *= xyz_scale_decay
            rescale.xyz_scale = max(rescale.xyz_scale, xyz_scale_final)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            print("Inject noise")
            L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
            actual_covariance = L @ L.transpose(1, 2)

            def op_sigmoid(x, k=100, x0=noise_opacity_thresh):
                return 1 / (1 + torch.exp(-k * (x - x0)))
            
            noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*noise_lr*rescale.xyz_scale
            noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
            gaussians._xyz.add_(noise)

        # safe_interact(local=locals(), banner="after loss vs step size")

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, jvp_start, val_indices=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
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
