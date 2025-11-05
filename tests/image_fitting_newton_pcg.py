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
from image_fitting_utils import prepare_output_and_logger

from copy import deepcopy

from matplotlib import pyplot as plt

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
    tb_writer = prepare_output_and_logger(args)

    # scale_const = 1e3
    # xyz_scale = 1e-3 * scale_const
    # features_dc_scale = 1e-3 * scale_const
    # featuress_rest_scale = 1e-5 * scale_const
    # scaling_scale = 1e-3 * scale_const
    # rotation_scale = 1e-3 * scale_const
    # opacity_scale = 1e-3 * scale_const
    # exposure_scale = 1.0 * scale_const
    # damp = 1e-6 * scale_const
    scale_const = 1e0
    xyz_scale = 1e-3 * scale_const
    features_dc_scale = 2.5e-3 * scale_const
    featuress_rest_scale = 1e-4 * scale_const
    scaling_scale = 5e-3 * scale_const * 1e-14
    rotation_scale = 1e-3 * scale_const * 1e-14
    opacity_scale = 2.5e-2 * scale_const
    exposure_scale = 1.0 * scale_const

    rescale = GaussianModelScaleMatrix(xyz_scale=xyz_scale, 
                                      features_dc_scale=features_dc_scale, 
                                      features_rest_scale=featuress_rest_scale, 
                                      scaling_scale=scaling_scale, 
                                      rotation_scale=rotation_scale, 
                                      opacity_scale=opacity_scale, 
                                      exposure_scale=1.0)

    
    cameras = [build_camera(image_path)]

    ####### Some fixed parameters #########
    num_images = 1
    train_test_exp = False
    white_background = False
    cameras_extent = 7.5
    model_path = ""
    ####### Some tunable parameters #########
    linesearch_alpha = 1.0
    linesearch_alpha_max = 1.0
    linesearch_alpha_min = 1e-4
    linesearch_alpha_decrease = 0.8
    linesearch_alpha_increase = 1.2
    linesearch_alpha_c = 0.01

    damp_alpha_max = 0.2
    damp_alpha_min = 1e-2
    damp_increase = 1.2
    damp_decrease = 0.8

    pixel_sample_rate_max = 0.8
    pixel_sample_rate_min = 0.6
    pixel_sample_rate = pixel_sample_rate_max
    pixel_sample_rate_increase = 1.2
    pixel_sample_rate_decrease = 0.8

    # NOTE: damp needs to be relative to scale^2
    damp_init = 1e-6 * (scale_const ** 2)      
    damp_min = 1e-8 * (scale_const ** 2)       
    damp_max = 1e-4 * (scale_const ** 2)       
    damp = damp_init

    pcg_tol = 1e-15
    ####### Some tunable parameters #########

    first_iter = 0
    sh_degree = 0
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

        ref_g_norm2 = ref_g.dot(ref_g)
        print(f"Iteration {iteration}: ref_g norm^2: {ref_g_norm2:.6f}")

        # Test vector loss prediction using J

        # Generate pixel mask, which is a boolean mask of shape (H*W,) with True for masking out pixels
        H, W = viewpoint_cams[0].image_height, viewpoint_cams[0].image_width
        pixel_mask = torch.rand((H, W), device="cuda") > pixel_sample_rate

        render_pkg = render(viewpoint_cams[0], gaussians, pipe, bg, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        loss_func = partial(batch_training_loss, iteration=iteration, opt=opt, pipe=pipe, bg=background, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=False, pixel_mask=pixel_mask)
        loss_func_hessian = partial(scalar_training_loss_hessian, iteration=iteration, opt=opt, pipe=pipe, bg=background, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, pixel_mask=pixel_mask)
        cur_state = LinearSolverFunctions(loss_func, gaussians, batch_size=5, param_mask=None, splat_mask=None, rescale=rescale, damp=damp, loss_func_hessian=loss_func_hessian)

        ref_g_vec = ref_g.as_1d_tensor(with_features_rest=False, with_exposure=False)
        u = GaussianModelState.zero_like_gaussians(gaussians)
        n = ref_g_vec.shape[0]

        rademacher_gen = partial(GaussianModelState.rademacher_like_gaussians, gaussians)
        preconditioner = AdaHessianPreconditioner(rademacher_gen, beta2=0.999, eps=1e-16, hessian_power=1.0)

        SHSx = partial(cur_state.Hv, viewpoint_cams=viewpoint_cams, scale=1, use_rescale=True, use_damping=True)

        warmup_sample_size = min(5, len(viewpoint_cams))
        warmup_cam_provider = CamProvider(viewpoint_cams, mode="random", max_stride=1, sample_size=warmup_sample_size)
        preconditioner.reset()
        preconditioner.update(SHSx, warmup_cam_provider, len(viewpoint_cams) / warmup_sample_size, num_iter=10)

        start_loss, Sg = cur_state.g(viewpoint_cams, 1, use_rescale=True, return_loss=True)

        Sg_norm2 = Sg.dot(Sg)
        print(f"Iteration {iteration}: Sg norm^2: {Sg_norm2:.6f}")
        if Sg_norm2 == 0.0:
            safe_interact(local=locals(), banner="Zero gradient, stopping optimization")

        x0 = cur_state.get_initial_solution()

        y = cg_damped(Ax=SHSx,
                      dot=cur_state.dot,
                      saxpy=cur_state.saxpy,
                      b=-Sg,
                      x0=x0,
                      M=preconditioner,
                      # M=None,
                      max_iter=10,
                      # max_iter=1,
                      restart_iter=3,
                      verbose=True,
                      tol=pcg_tol,)

        print(f"y norm: {y.dot(y):.6f}")

        s_newton = rescale * y

        v = s_newton
        v_stepsize = math.sqrt(v.dot(v))
        v = v / (v_stepsize + 1e-15)
        v_adam = -ref_g / (ref_g.abs() + 1e-15)
        v_stepsize_adam = math.sqrt(v_adam.dot(v_adam))
        v_adam = v_adam / (v_stepsize_adam + 1e-15)

        loss_scalar, g, Hv = cur_state.Hv(v, viewpoint_cams, scale=1, use_rescale=False, return_grad_and_loss=True)

        JtJv = cur_state.JTJv(v, viewpoint_cams, scale=1, use_rescale=False)
        vJtJv = v.dot(JtJv)

        alpha = 0.0
        cur_alpha = 0.0
        best_alpha = 0.0
        best_loss = loss_scalar

        losses_first_order = []
        losses_gn = []
        losses_second_order = []
        losses_alpha = []
        losses_adam = []
        alphas = []


        ref_gv = ref_g.dot(v)
        gv = g.dot(v)
        vHv = v.dot(Hv)

        gs_newton = g.dot(s_newton)

        # import code; code.interact(local=locals(), banner="after loss compute")

        with torch.no_grad():
            loss_0 = 0.0
            loss_0_plot = 0.0
            for vc in viewpoint_cams:
                loss_0 += reference_training_loss(iteration, opt, vc, gaussians, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, pixel_mask=pixel_mask).item()
                loss_0_plot += reference_training_loss(iteration, opt, vc, gaussians, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, pixel_mask=None).item()

            print("Starting linesearch from loss_0:", loss_0)

            alpha = linesearch_alpha
            while True:
                gaussians_copy = deepcopy(gaussians)
                gaussians_copy.update_step(alpha * s_newton)

                loss_alpha = 0.0
                loss_alpha_plot = 0.0
                for vc in viewpoint_cams:
                    loss_alpha += reference_training_loss(iteration, opt, vc, gaussians_copy, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, pixel_mask=pixel_mask)
                    loss_alpha_plot += reference_training_loss(iteration, opt, vc, gaussians_copy, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, pixel_mask=None)

                # print("Linesearch alpha:", alpha, "loss_alpha:", loss_alpha.item())

                losses_alpha.append(loss_alpha_plot.item())
                alphas.append(alpha)

                # Check Armijo condition
                if loss_alpha.item() <= loss_0 + linesearch_alpha_c * alpha * gs_newton:
                    break

                alpha *= linesearch_alpha_decrease

                if alpha < linesearch_alpha_min:
                    print("Linesearch alpha below minimum. Forcing step at alpha =", alpha)
                    # alpha = 0.0
                    break

            print("Linesearch found alpha:", alpha, "loss_alpha:", loss_alpha.item())


            """
            for i in range(-20, 100, 1):
                # step_size = 0.01
                step_size = 5e-3
                alpha = i * step_size
                gaussians_copy = deepcopy(gaussians)
                gaussians_copy.update_step(alpha * v)

                loss_alpha = 0.0
                loss_alpha_plot = 0.0
                for vc in viewpoint_cams:
                    loss_alpha += reference_training_loss(iteration, opt, vc, gaussians_copy, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, pixel_mask=pixel_mask)
                    loss_alpha_plot += reference_training_loss(iteration, opt, vc, gaussians_copy, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, pixel_mask=None)

                losses_alpha.append(loss_alpha_plot.item())
                alphas.append(alpha)

                if loss_alpha.item() < best_loss:
                    best_loss = loss_alpha.item()
                    best_alpha = alpha

                losses_first_order.append(loss_0 + alpha * ref_gv)
                losses_gn.append(loss_0 + alpha * ref_gv + 0.5 * (alpha ** 2) * vJtJv)
                losses_second_order.append(loss_0 + alpha * ref_gv + 0.5 * (alpha ** 2) * vHv)

                gaussians_copy = deepcopy(gaussians)
                gaussians_copy.update_step(alpha * v_adam)

                loss_adam = 0.0
                for vc in viewpoint_cams:
                    loss_adam += scalar_training_loss(iteration, opt, vc, gaussians_copy, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight)[0]
                losses_adam.append(loss_adam.item())

                print("alpha:", alpha, "loss_alpha:", loss_alpha.item(), "loss_adam:", loss_adam.item(), "gn approx:", losses_gn[-1], "2nd order approx:", losses_second_order[-1])
            """


        """
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, losses_alpha, label="Actual loss", alpha=0.5)
        plt.plot(alphas, losses_first_order, label="First order approx", alpha=0.5)
        plt.plot(alphas, losses_gn, label="Gauss-Newton approx", alpha=0.5)
        plt.plot(alphas, losses_second_order, label="Second order approx", alpha=0.5)
        plt.plot(alphas, losses_adam, label="Adam step", alpha=0.5)

        # Plot vertical line at x = 0 and x = v_stepsize
        plt.axvline(x=0, color='k', linestyle='--', label='No update')
        # plt.axvline(x=v_stepsize, color='r', linestyle='--', label='Taken step')
        # plt.axvline(x=v_stepsize_adam, color='g', linestyle='--', label='Adam step')

        plt.xlabel("Step size")
        plt.ylabel("Loss")

        plt.legend()

        print("before savefig")
        plt.savefig("loss_vs_alpha.png")
        plt.close()
        print("after savefig")
        """

        training_report(None, iteration, None, None, l1_loss, None, [iteration], cameras, gaussians, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, train_test_exp), train_test_exp)

        print("Update with s_newton")
        gaussians.update_step(alpha * s_newton)
        # gaussians.update_step(best_alpha * v)

        # DEBUG
        loss_alpha_full = 0.0
        for vc in viewpoint_cams:
            loss_alpha_full += reference_training_loss(iteration, opt, vc, gaussians_copy, pipe, bg, train_test_exp=train_test_exp, depth_l1_weight=depth_l1_weight, pixel_mask=None)
        print("\n\nPost-update loss (no pixel mask):", loss_alpha_full.item())
        print(f"damp: {damp}, pixel_sample_rate: {pixel_sample_rate}\n\n")

        # print("best alpha:", best_alpha, "best loss:", best_loss)
        # print("max param step size = ", (best_alpha * v).as_1d_tensor(with_exposure=False, with_features_rest=False).abs().max().item())
        # print("max param step idx = ", (best_alpha * v).as_1d_tensor(with_exposure=False, with_features_rest=False).abs().argmax().item())

        training_report(tb_writer, iteration, None, loss_alpha_full, l1_loss, None, [iteration], cameras, gaussians, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, train_test_exp), train_test_exp)

        if tb_writer:
            tb_writer.add_scalar('train_loss_full/alpha', alpha, iteration)
            tb_writer.add_scalar('train_loss_full/pixel_sample_rate', pixel_sample_rate, iteration)
            tb_writer.add_scalar('train_loss_full/damping', damp, iteration)

        # Update pixel sampling rate
        if alpha < 0.001:
            pixel_sample_rate *= pixel_sample_rate_decrease
            pixel_sample_rate = max(pixel_sample_rate, pixel_sample_rate_min)
        elif alpha > 0.001:
            pixel_sample_rate *= pixel_sample_rate_increase
            pixel_sample_rate = min(pixel_sample_rate, pixel_sample_rate_max)

        # Update Damping Factor
        if alpha < damp_alpha_min or v_stepsize == 0:
            damp *= damp_increase
            damp = min(damp, damp_max)
        elif alpha > damp_alpha_max:
            damp *= damp_decrease
            damp = max(damp, damp_min)

        # safe_interact(local=locals(), banner="after loss vs step size")


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, cameras, gaussians, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

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
            # tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
            # tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
            pass

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
