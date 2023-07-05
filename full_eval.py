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
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
args, _ = parser.parse_known_args()

if not args.skip_training:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

    common_args = " --quiet --eval --test_iterations -1"
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python train.py -s " + source + " -m ./eval/" + scene + common_args)
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python train.py -s " + source + " -m ./eval/" + scene + common_args)
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_4 -m ./eval/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_2 -m ./eval/" + scene + common_args)

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_rendering:
    for scene in all_scenes:
        os.system("python render.py --quiet --skip_train --eval --iteration 7000 -m ./eval/" + scene)
    for scene in all_scenes:
        os.system("python render.py --quiet --skip_train --eval --iteration 30000 -m ./eval/" + scene)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + "./eval/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)