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
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, safe_interact, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from functools import partial
from scene.gaussian_model import build_scaling_rotation
from solver.gaussian_model_vector import GaussianModelVector
from solver.adam_optimizer import AdamOptimizer
from solver.sophia_optimizer import SophiaOptimizer
from solver.solver_functions import construct_loss_func, construct_g_func, construct_JTJv_func, dot, saxpy, construct_Dhat_func
from solver.hellinger_clip import clip_hellinger, debug_hellinger

import re
import glob

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def evaluate(dataset, opt, pipe, model_path=None):

    ####### Some fixed parameters #########
    train_test_exp = False
    ####### Some fixed parameters #########

    checkpoint_pattern = model_path + "/chkpnt*.pth"
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))

    if not checkpoint_files:
        print(f"No checkpoint files found in {model_path}.")
        exit()

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    test_cameras = scene.getTestCameras()

    with torch.no_grad():

        for checkpoint in checkpoint_files:
            print(f"Evaluating from checkpoint: {checkpoint}")
            (model_params, _) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            ssims = []
            psnrs = []
            lpipss = []

            i = 0

            for viewpoint_cam in test_cameras:
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image = render_pkg["render"].clamp(0.0, 1.0)
                gt_image = viewpoint_cam.original_image.cuda().clamp(0.0, 1.0)

                ssims.append(ssim(image, gt_image).squeeze().unsqueeze(0))
                psnrs.append(psnr(image, gt_image).squeeze())
                lpipss.append(lpips(image, gt_image, net_type='vgg').squeeze().unsqueeze(0))

            print("  SSIM : {:>12.7f}".format(torch.cat(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.cat(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.cat(lpipss).mean(), ".5"))
            print("")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    # parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.eval = True
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    evaluate(lp.extract(args), op.extract(args), pp.extract(args), args.model_path)

    # All done
    print("\nTraining complete.")
