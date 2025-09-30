"""
Times the Jacobian-vector product (JVP) and forward pass of the training loss function
python3 tests/test_jvp_timing.py -s <path/to/dataset> --start_checkpoint --num_images <number_of_images_in_batch>
"""

import os
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from random import randint
from utils.loss_utils import l1_loss, l1_loss_per_pixel, ssim, ssim_per_pixel
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
from functools import partial
import time
import matplotlib.pyplot as plt

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

def training(dataset, opt, pipe, checkpoint, num_images):
    rescale = GaussianModelScaleMatrix(xyz_scale=0.0001, 
                                      features_dc_scale=0.0025, 
                                      features_rest_scale=0.0001, 
                                      scaling_scale=0.005, 
                                      rotation_scale=0.001, 
                                      opacity_scale=0.025, 
                                      exposure_scale=1.0)

    np.random.seed(0)
    torch.manual_seed(0)

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        print(f"Restoring from checkpoint: {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iteration = first_iter

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))

    num_batch_cameras = min(num_images, len(viewpoint_indices))
    rand_indices = np.random.choice(viewpoint_indices, num_batch_cameras, replace=False)
    print(f"\nUsing {num_batch_cameras} random cameras: {rand_indices}")
    # Same background for all cameras in the batch
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    viewpoint_cams = []

    for i, rand_idx in enumerate(rand_indices):
        viewpoint_cam = viewpoint_stack[rand_idx]
        viewpoint_cams.append(viewpoint_cam)


    loss_func = partial(scalar_training_loss_hessian, iteration=iteration, opt=opt, pipe=pipe, bg=bg, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight)
    cur_state = LinearSolverFunctions(loss_func, gaussians, param_mask=None, damp=None, splat_mask=None, rescale=rescale)
    rademacher_gen = partial(GaussianModelState.rademacher_like_gaussians, gaussians)
    preconditioner = AdaHessianPreconditioner(rademacher_gen, beta2=0.999, eps=1e-8, hessian_power=1.0)

    loss_func_gn = partial(batch_training_loss, iteration=iteration, opt=opt, pipe=pipe, bg=background, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=False)
    cur_state_gn = LinearSolverFunctions(loss_func_gn, gaussians, param_mask=None, damp=None, splat_mask=None, rescale=rescale)

    ref_loss_scalar = 0.0
    for vc in viewpoint_cams:
        ref_loss_scalar += reference_training_loss(iteration, opt, vc, gaussians, pipe, bg, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight)

    gaussians.zero_grad()
    ref_loss_scalar.backward()
    ref_g = GaussianModelState.from_gaussians_grad(gaussians)

    loss_scalar1 = 0.0
    for vc in viewpoint_cams:
        loss_scalar1 += scalar_training_loss(iteration, opt, vc, gaussians, pipe, bg, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight)[0]
    gaussians.zero_grad()
    loss_scalar1.backward()
    g1 = GaussianModelState.from_gaussians_grad(gaussians)

    g = g1 * rescale

    warmup_cam_provider = CamProvider(viewpoint_cams, mode="random", max_stride=1, sample_size=1)
    preconditioner.reset()
    preconditioner.update(cur_state_gn.Hv, warmup_cam_provider, 1, num_iter=10)

    x0 = GaussianModelState.zero_like_gaussians(gaussians)
    Hx = partial(cur_state_gn.Hv, viewpoint_cams=viewpoint_cams, scale=1)
    pcg_max_iter = 30
    s = cg_damped(Ax=Hx,
                  dot=cur_state.dot,
                  saxpy=cur_state.saxpy,
                  b=-g,
                  x0=x0,
                  M=preconditioner,
                  max_iter=pcg_max_iter,
                  restart_iter=50)

    # v = GaussianModelState.zero_like_gaussians(gaussians)
    # v.xyz_grad[0:100, :] = 1
    # v = GaussianModelState.randn_like_gaussians(gaussians) * rescale
    # v = -ref_g / (ref_g.abs() + 1e-15) * rescale
    v = s
    v = v / math.sqrt(v.dot(v))
    loss_scalar, g, Hv = cur_state.Hv_all(v, viewpoint_cams, scale=1)

    # import code; code.interact(local=locals(), banner="before assert")
    # assert (g.xyz_grad - ref_g.xyz_grad).abs().max() < 1e-10
    # assert (g.features_dc_grad - ref_g.features_grad).abs().max() < 1e-10
    # assert (g.features_rest_grad - ref_g.features_grad).abs() < 1e-10
    # assert (g.scaling_grad - ref_g.scaling_grad).abs().max() < 1e-10
    # assert (g.rotation_grad - ref_g.rotation_grad).abs().max() < 1e-10
    # assert (g.opacity_grad - ref_g.opacity_grad).abs().max() < 1e-10

    cur_state_gn = LinearSolverFunctions(loss_func_gn, gaussians, param_mask=None, damp=0.0, splat_mask=None)
    JtJv = cur_state_gn.Hv(v, viewpoint_cams, scale=1)
    vJtJv = v.dot(JtJv)

    print(f"loss_scalar: {loss_scalar}, ref_loss_scalar: {ref_loss_scalar}")

    alpha = 0.0
    cur_alpha = 0.0

    losses_first_order = []
    losses_gn = []
    losses_second_order = []
    losses_alpha = []
    alphas = []

    loss_0 = ref_loss_scalar.item()

    gv = g.dot(v)
    vHv = v.dot(Hv)

    import code; code.interact(local=locals(), banner="after loss compute")

    for i in range(-50, 100, 1):
        step_size = 0.001
        # step_size = 1
        alpha = i * step_size
        gaussians_copy = deepcopy(gaussians)
        gaussians_copy.update_step(alpha * v)

        loss_alpha = 0.0
        for vc in viewpoint_cams:
            loss_alpha += reference_training_loss(iteration, opt, vc, gaussians_copy, pipe, bg, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight)

        losses_alpha.append(loss_alpha.item())
        alphas.append(alpha)

        losses_first_order.append(loss_0 + alpha * gv)
        losses_gn.append(loss_0 + alpha * gv + 0.5 * (alpha ** 2) * vJtJv)
        losses_second_order.append(loss_0 + alpha * gv + 0.5 * (alpha ** 2) * vHv)

        print("alpha:", alpha, "loss_alpha:", loss_alpha.item())

    plt.plot(alphas, losses_alpha, label="Actual loss")
    plt.plot(alphas, losses_first_order, label="First order approx")
    plt.plot(alphas, losses_gn, label="Gauss-Newton approx")
    plt.plot(alphas, losses_second_order, label="Second order approx")

    plt.xlabel("Step size")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig("loss_vs_alpha.png")




if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--num_images", type=int, default = 1)
    args = parser.parse_args(sys.argv[1:])
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.num_images)

