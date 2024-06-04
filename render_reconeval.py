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
import numpy as np
import torch
from scene import Scene
import os
import shutil
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import read_write_binary as im

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        # image = view.image_name
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{}.png'.format(view.image_name)))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{}.png'.format(view.image_name)))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,override_quantization=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

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

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if os.path.exists(args.source_path):
        shutil.rmtree(args.source_path)
        os.mkdir(args.source_path)
    else:
        os.mkdir(args.source_path)

    shutil.copytree(os.path.join(args.source_path, "../sparse"), os.path.join(args.source_path, "sparse"))
    shutil.copytree(os.path.join(args.source_path, "../images"), os.path.join(args.source_path, "images"))
    shutil.copy(os.path.join(args.source_path, "../test_aligned_pose.txt"),
                os.path.join(args.source_path, "test_aligned_pose.txt"))
    data = im.read_images_binary(os.path.join(args.source_path, "sparse", "images.bin"))
    image = data[1]
    new_data = {}
    with open(os.path.join(args.source_path, "test_aligned_pose.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            n, tx, ty, tz, qx, qy, qz, qw = line.split(" ")
            name = "{}.png".format(n)
            if not os.path.exists(os.path.join(args.source_path, "images", name)):
                images = [i for i in os.listdir(os.path.join(args.source_path, "images")) if ".png" in i]
                shutil.copy(os.path.join(os.path.join(args.source_path, "images", images[0])),
                            os.path.join(os.path.join(args.source_path, "images", name)))
            i = int(n)
            qvec = [float(i) for i in [qw, qx, qy, qz]]
            tvec = [float(i) for i in [tx, ty, tz]]
            #image = data[1]
            image = image._replace(id=i, qvec=np.array(qvec), tvec=np.array(tvec), name=name)
            #data[1 + i] = image
            new_data[i] = image
    print(len(new_data))
    im.write_images_binary(new_data, os.path.join(args.source_path, "sparse/0", "images.bin"))

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)