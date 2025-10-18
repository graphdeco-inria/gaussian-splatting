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

Parameters: -h help,
            -inputMesh <the mesh to simplify>,
            -outputMesh <the output mesh>,
            -meshlabPath <Meshlab binary directory>

Usage: python wedge_to_vertex_uvs.py -inputMesh <the mesh to convert>
                               -outputMesh <the output mesh>
                               -meshlabPath <Meshlab binary directory>

"""

import os, sys
import argparse
from utils.commands import runCommand, getMeshlabServer
from utils.paths import getMeshlabPath

def convertUVs(inputMesh, outputMesh, meshlabPath = getMeshlabPath()):
    mlxFile = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'meshlab/wedge_to_vertex_uvs.mlx'))

    ret = runCommand(getMeshlabServer(meshlabPath), ['-i', inputMesh,
                                                     '-o', outputMesh,
                                                     '-m', 'vt',
                                                     '-s', mlxFile])
    return ret

def main():
    parser = argparse.ArgumentParser()

    # common arguments
    parser.add_argument("--inputMesh", type=str, required=True, help="the mesh to simplify")
    parser.add_argument("--outputMesh", type=str, required=True, help="the output mesh")
    parser.add_argument("--meshlabPath", type=str, default=getMeshlabPath(), help="Meshlab binary directory")

    args = vars(parser.parse_args())

    ret = convertUVs(args['inputMesh'], args['outputMesh'], args['meshlabPath'])
    if( ret.returncode != 0 ):
        print("SIBR_ERROR meshlab error in converting UVs")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
