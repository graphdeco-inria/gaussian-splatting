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
import numpy as np

from tools.camera_tool import load_views_from_lookat_torch

def render_sets_from_file(dataset : ModelParams, iteration : int, pipeline : PipelineParams, name):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, name, "ours_{}".format(scene.loaded_iter), "renders")
        gts_path = os.path.join(dataset.model_path, name, "ours_{}".format(scene.loaded_iter), "gt")
        depth_path = os.path.join(dataset.model_path, name, "ours_{}".format(scene.loaded_iter), "depth")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

        views = load_views_from_lookat_torch('./tools/cameras.lookat')
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            results = render(view, gaussians, pipeline, background)
            rendering = results["render"]
            depth = results["depth"]
            depth_img = depth / (depth.max() + 1e-5)
            depth = depth.squeeze().cpu().numpy()
            np.save(os.path.join(depth_path, f"{idx:04d}.npy"), depth)
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:04d}.jpg"))
            torchvision.utils.save_image(depth_img, os.path.join(depth_path, f"{idx:04d}.jpg"))

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets_from_file(model.extract(args), args.iteration, pipeline.extract(args), 'interactive_path')