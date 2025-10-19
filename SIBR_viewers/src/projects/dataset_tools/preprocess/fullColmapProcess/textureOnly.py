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
This script runs a pipeline to texture a mesh after colmap has been run

Parameters: -h help,
            -path <path to your dataset folder>,
            -sibrBinariesPath <binaries directory of SIBR>,

Usage: python textureOnly.py -path <path to your dataset folder>
                                   -sibrBinariesPath <binaries directory of SIBR>

"""

import subprocess
import os, sys, getopt
import os, sys, shutil
import json
import argparse
from utils.paths import getBinariesPath 
from utils.commands import  getProcess
from utils.TaskPipeline import TaskPipeline
from simplify_mesh import simplifyMesh

def main():
    parser = argparse.ArgumentParser()

    # common arguments
    parser.add_argument("--path", type=str, required=True, help="path to your dataset folder")
    parser.add_argument("--sibrBinariesPath", type=str, default=getBinariesPath(), help="binaries directory of SIBR")
    parser.add_argument("--dry_run", action='store_true', help="run without calling commands")
    

    args = vars(parser.parse_args())


    # Fixing path values
    args["path"] = os.path.abspath(args["path"])
    args["sibrBinariesPath"] = os.path.abspath(args["sibrBinariesPath"])

    ret = simplifyMesh( args["path"] + "/colmap/stereo/unix-meshed-delaunay.ply", args["path"]  + "/colmap/stereo/unix-meshed-delaunay-simplified.ply")
    print("RET ", ret)
    if( ret.returncode != 0 ):
        print("SIBR ERROR: meshlab simplify failed, exiting")
        sys.exit(1)

    unwrap_app = getProcess("unwrapMesh")
    unwrap_args = [unwrap_app,
             "--path", args["path"] + "/colmap/stereo/unix-meshed-delaunay-simplified.ply",
             "--output", args["path"] + "/capreal/mesh.ply",
        ]

    print("Running unwrap mesh ", unwrap_args)
    p_exit = subprocess.call(unwrap_args)
    if p_exit != 0:
        print("SIBR ERROR: unwrap failed, exiting")
        sys.exit(1)

    texturemesh_app = getProcess("textureMesh")
    texturemesh_args = [texturemesh_app,
             "--path", args["path"], 
             "--output", args["path"] + "/capreal/texture.png",
             "--size", "8192",
             "--flood"
        ]

    print("Texturing mesh ", texturemesh_args)
    p_exit = subprocess.call(texturemesh_args)
    if p_exit != 0:
        print("SIBR ERROR: mesh texturing failed, exiting")
        sys.exit(1)


    print("textureonly has finished successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
