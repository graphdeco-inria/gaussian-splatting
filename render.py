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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from PIL import Image
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov, fov2focal
from utils.camera_utils import cameraList_from_camInfos
import numpy as np
from typing import List

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_set_no_gt(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                custom_cameras: List[CameraInfo]):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        if custom_cameras:
            render_set_no_gt(dataset.model_path, "custom", scene.loaded_iter, custom_cameras, gaussians, pipeline, background)

def get_cam_infos(trajectory: torch.Tensor,
                  input_width: int, input_height: int, input_focal_length: float):
    cam_infos: List[CameraInfo] = []
    for i in range(trajectory.shape[0]):
        transform = np.linalg.inv(trajectory[i, :, :].numpy())
        width = input_width
        height = input_height
        focal_length = input_focal_length
        FovX = focal2fov(focal_length, width)
        FovY = focal2fov(focal_length, height)

        # create a dummy gt image with all pixels being zero
        cam_info = CameraInfo(uid=0,
                              R=np.transpose(transform[:3, :3]), T=transform[:3, 3],
                              FovX=FovX, FovY=FovY, width=width, height=height,
                              image=Image.fromarray(np.zeros((height, width, 3), dtype=np.byte), "RGB"),
                              image_path="", image_name=f"frame_{i:03}")

        cam_infos.append(cam_info)
    return cam_infos

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--camera_trajectory", type=str, default="", help="A camera trajectory file saved as torch.save(), in shape [N, 4, 4]")
    parser.add_argument("--camera_parameters", type=str, default="976,544,581.743", help="width,height,focal: a comma separated list of numbers")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Enable custom trajectory rendering will disable train/test renders
    custom_cameras = []
    if args.camera_trajectory:
        args.skip_test = True
        args.skip_train = True
        width, height, focal = args.camera_parameters.split(",")
        width = int(width)
        height = int(height)
        focal = float(focal)
        camera_info = get_cam_infos(torch.load(args.camera_trajectory), width, height, focal)
        custom_cameras = cameraList_from_camInfos(camera_info, 1.0, args)


    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, custom_cameras)