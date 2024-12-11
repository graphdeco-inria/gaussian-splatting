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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.system_utils import mkdir_p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d
import numpy as np
import cv2
from collections import Counter
import pandas as pd
from PIL import Image


def save_ply(gaussians, path):
    mkdir_p(os.path.dirname(path))

    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def training_report(reconstructed_gaussians, scene, renderFunc, renderArgs):
    
    # 테스트 및 훈련 데이터셋 검증
    torch.cuda.empty_cache()
    validation_configs = ({
        'name': 'test',
        'cameras' : scene.getTestCameras()
    }, {
        'name': 'train',
        'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
    })


    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                # 재구성된 Gaussian 모델로 렌더링
                image = torch.clamp(renderFunc(viewpoint, reconstructed_gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                psnr_test += psnr(image, gt_image).mean().double()
            psnr_test /= len(config['cameras'])

            # PSNR과 L1 Test 값 출력
            print(f"\nEvaluating {config['name']}: PSNR {psnr_test:.4f}")

    torch.cuda.empty_cache()

def projection(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    bg_color = [1, 1, 1] if lp.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    gaussians.load_ply(checkpoint)
    print("load ply :", gaussians._xyz.shape)
    print("load ply :", gaussians._xyz)
    
    csp_data = gaussians.csp()
    print("csp :", gaussians._xyz.shape)
    print("csp :", gaussians._xyz)
    
    gaussians.unprojection(csp_data)
    print("unprojection :", gaussians._xyz.shape)
    print("unprojection :", gaussians._xyz)
    
    gaussians.save_ply('./test/lego/point_cloud/iteration_30000/test.ply')
    training_report(gaussians, scene, render, (pp.extract(args), background))
    
        
def save_features_dc_tensor(features_dc_tensor, output_dir, slice_axis='z', normalize=True):
    """
    Save features_dc_tensor as images along a specified axis.

    Args:
        features_dc_tensor (torch.Tensor): Tensor of shape [num_voxels_y, num_voxels_z, num_voxels_x, 3]
        output_dir (str): Directory to save images
        slice_axis (str): Axis along which to slice ('y', 'z', or 'x')
        normalize (bool): Whether to normalize tensor values to [0, 1]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure tensor is on CPU
    features_dc_tensor = features_dc_tensor.detach().cpu()

    if slice_axis == 'y':
        num_slices = features_dc_tensor.shape[0]
        for y in range(num_slices):
            # Slice along y-axis: [z, x, 3]
            image = features_dc_tensor[y]  # [z, x, 3]
            image = image.numpy()

            if normalize:
                min_val = image.min()
                max_val = image.max()
                if max_val > min_val:
                    image = (image - min_val) / (max_val - min_val)
                else:
                    image = image * 0.0

            # Convert to uint8
            image_uint8 = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_uint8)

            image_path = os.path.join(output_dir, f"feature_dc_y_{y:04d}.png")
            pil_image.save(image_path)

    elif slice_axis == 'z':
        num_slices = features_dc_tensor.shape[1]
        for z in range(num_slices):
            # Slice along z-axis: [y, x, 3]
            image = features_dc_tensor[:, z, :, :]  # [y, x, 3]
            image = image.numpy()

            if normalize:
                min_val = image.min()
                max_val = image.max()
                if max_val > min_val:
                    image = (image - min_val) / (max_val - min_val)
                else:
                    image = image * 0.0

            # Convert to uint8
            image_uint8 = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_uint8)

            image_path = os.path.join(output_dir, f"feature_dc_z_{z:04d}.png")
            pil_image.save(image_path)

    elif slice_axis == 'x':
        num_slices = features_dc_tensor.shape[2]
        for x in range(num_slices):
            # Slice along x-axis: [y, z, 3]
            image = features_dc_tensor[:, :, x, :]  # [y, z, 3]
            image = image.numpy()

            if normalize:
                min_val = image.min()
                max_val = image.max()
                if max_val > min_val:
                    image = (image - min_val) / (max_val - min_val)
                else:
                    image = image * 0.0

            # Convert to uint8
            image_uint8 = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_uint8)

            image_path = os.path.join(output_dir, f"feature_dc_x_{x:04d}.png")
            pil_image.save(image_path)

    else:
        raise ValueError("slice_axis must be 'y', 'z', or 'x'.")

    print(f"Saved features_dc_tensor slices along {slice_axis}-axis to {output_dir}")

# python .\projection.py -s .\data\gaussian_splatting\tandt_db\tandt\truck --start_checkpoint .\output\lego\point_cloud\iteration_30000\point_cloud.ply
# python .\projection_tensor_test.py -s ..\..\data\nerf_synthetic\nerf_synthetic\lego --start_checkpoint ..\output\lego\point_cloud\iteration_30000\point_cloud.ply

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    projection(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
