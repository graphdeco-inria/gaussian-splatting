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

import os
from enum import Enum, unique

@unique
class DatasetType(Enum):
    SIBR = 'sibr'
    COLMAP = 'colmap'
    CAPREAL = 'capreal'

datasetStructure = { 
    "colmap": [ "colmap", "colmap/stereo", "colmap/sparse" ],
    "capreal": [ "capreal", "capreal/undistorted" ],
    "sibr": [ "cameras", "images", "meshes" ]
}

def buildDatasetStructure(path, types):
    for folder in [folder for type in types for folder in datasetStructure[type]]:
        new_folder = os.path.abspath(os.path.join(path, folder))
        print("Creating folder %s..." % new_folder)
        os.makedirs(new_folder, exist_ok=True)