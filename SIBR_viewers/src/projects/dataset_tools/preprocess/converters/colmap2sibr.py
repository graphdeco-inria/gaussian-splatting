# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


#!/usr/bin/env python
#! -*- encoding: utf-8 -*-

""" @package dataset_tools_preprocess
This script calls meshlab to simplify a mesh

Parameters: --h help,
            --path <path to dataset>

Usage: python colmap2sibr.py --path <path to dataset> [required]

"""

import os, sys
import argparse
import os, sys, getopt
import re
from utils.commands import getProcess
from utils.paths import getBinariesPath
from utils.commands import runCommand
import subprocess
from pathlib import Path


def checkColmapConsistent(pathdir):
    colmapDir = Path(pathdir+"/colmap")

    if not os.path.isdir(colmapDir):
        return False

    # check for mesh
    colmapmesh = Path(pathdir+"/colmap/stereo/meshed-delaunay.ply")
    if not os.path.isfile(colmapmesh):
        print("SIBR_ERROR: colmap directory exists but there is no mesh ", colmapmesh)
        print("No file", colmapmesh)
        return False

    return True

def main():
    parser = argparse.ArgumentParser()

    # common arguments
    parser.add_argument("--path", type=str, required=True, help="path to dataset folder")

    args = vars(parser.parse_args())

    if not checkColmapConsistent(args['path']):
        print("SIBR_ERROR Colmap hasnt been run properly; run it first (ie dont use --noColmap)")
        sys.exit(1)

	# prepareColmap4Sibr: convert cameras and create bundle file put everything in sfm_mvs_cm, then run the normal preprocessing
    # 
    prepareColmap_app = getProcess("prepareColmap4Sibr")
    
    prepareColmap_args = [prepareColmap_app,
        "--path", args['path'],
        ]

    print("Running prepareColmap4Sibr ", prepareColmap_args)
    p_exit = subprocess.call(prepareColmap_args)
    if p_exit != 0:
        print("SIBR ERROR: prepareColmap4Sibr failed, exiting")
        sys.exit(1)

	# run rc_to_sibr process to make all images have the same size and be compatible with spixelwarp pipeline
    p_exit = subprocess.call(["python", "ibr_preprocess_rc_to_sibr.py",  "-i", args['path']+"/sfm_mvs_cm",  "-o", args['path']+"/sibr_cm"])
    if p_exit != 0:
        print("SIBR_ERROR preprocess to sibr_cm failed");
        sys.exit(1)

    prepareColmap_app = getProcess("prepareColmap4Sibr")
    
    prepareColmap_args = [prepareColmap_app,
		"--fix_metadata", 
        "--path", args['path'],
        ]

    print("Running prepareColmap4Sibr to fix scene_dataset.txt ", prepareColmap_args)
    p_exit = subprocess.call(prepareColmap_args)
    if p_exit != 0:
        print("SIBR ERROR: prepareColmap4Sibr failed, exiting")
        sys.exit(1)


    sys.exit(0)

if __name__ == "__main__":
    main()
