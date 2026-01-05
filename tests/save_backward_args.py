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

from solver.batch_training_loss import batch_training_loss
from solver.reference_training_loss import reference_training_loss
from solver.training_loss import training_loss
from solver.gaussian_model_state import GaussianModelState
from solver.loss_image_state import MultiBatchLossImageState
from solver.solver_functions import LinearSolverFunctions

def training(dataset, opt, pipe, checkpoint, num_images):
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

    loss_func = partial(batch_training_loss, iteration=iteration, opt=opt, pipe=pipe, bg=bg, train_test_exp=dataset.train_test_exp, depth_l1_weight=depth_l1_weight, disable_ssim=False)

    cur_state = LinearSolverFunctions(loss_func, gaussians, param_mask=None, damp=None, splat_mask=None)

    cur_state.gradient_and_loss_est(viewpoint_cams, scale=1)


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

