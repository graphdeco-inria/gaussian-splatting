# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------
""" @package dataset_tools_preprocess
This script creates a SIBR template dataset from the old SIBR dataset which can be fed to a SIBR application

Parameters: -h help,
            -i <path to input directory which is the output from RC> <default: ${CMAKE_INSTALL_DIR}/bin/datasets/rc_out/>,
            -o <path to output directory which can be fed into SIBR apps> <default: input directory> [optional],
            -r use release w/ debug symbols executables

Usage: python ibr_preprocess_rc_to_sibr.py -i <path_to_sibr>\sibr\install\bin\datasets\museum_sibr_old_preproc
                                           -d <path_to_sibr>\sibr\install\bin\datasets\museum_sibr_new_preproc2

"""

import subprocess
import shutil
import os, sys, getopt
import re
from utils.commands import getProcess
from utils.paths import getBinariesPath

from os import walk

#--------------------------------------------

#===============================================================================

import struct
import imghdr

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height
    
def checkOutput( output, force_continue ):
    if( output != 0):
        if( not force_continue ):
            sys.exit()
        else:
            return False
    else:
        return True
    

#===============================================================================

#--------------------------------------------
# 0. Paths, commands and options

def main(argv, path_dest):
    opts, args = getopt.getopt(argv, "hi:ro:", ["idir=", "bin="])
    executables_suffix = ""
    executables_folder = getBinariesPath()
    path_data = ""
    for opt, arg in opts:
        if opt == '-h':
            print("-i path_to_old_dataset -d path_to_new_dataset [-r (use release w/ debug symbols executables)]")
            sys.exit()
        elif opt == '-i':
            path_data = arg
            print(['Setting path_data to ', path_data])
        elif opt == '-r':
            executables_suffix = "_rwdi"
            print("Using rwdi executables.")
        elif opt == '-o':
            path_dest = arg
            print(['Setting path_dest to ', path_dest])

    return (path_data, path_dest, executables_suffix, executables_folder)

path_dest = ""
path_data, path_dest, executables_suffix, executables_folder = main(sys.argv[1:], path_dest)

if(path_data == ""):
    path_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))

if(path_dest == ""):
    path_dest = path_data

path_data = os.path.abspath(path_data + "/") + "/"
path_dest = os.path.abspath(path_dest + "/") + "/"

path_in_imgs = path_data


print(['Raw_data folder: ', path_data])
print(['Path_dest: ', path_dest])

#path_dest_pmvs    = path_dest + "pmvs/models/";
file_nameList   = path_data + "images/list_images.txt";
# path_scene_metadata = path_data + "scene_metadata.txt"


#--------------------------------------------
# Create scene metadata file from list image file
scene_metadata = "Scene Metadata File\n\n"

# read list image file
path_list_images = os.path.join(path_in_imgs, "list_images.txt")
list_images = []

print(path_list_images)
if os.path.exists(path_list_images):
    list_image_file = open(path_list_images, "r")

    for line in list_image_file:
        list_images.append(line)

    list_image_file.close()

# read clipping planes file
path_clipping_planes = os.path.join(path_data, "clipping_planes.txt")
clipping_planes = []

if os.path.exists(path_clipping_planes):
    clipping_planes_file = open(path_clipping_planes, "r")

    for line in clipping_planes_file:
        line = line.strip('\n')
        clipping_planes.append(line)

    clipping_planes_file.close()


if not os.path.exists(path_dest):
    os.mkdir(path_dest)

folder_to_create = ["images","cameras","meshes","textures"]
for f in folder_to_create:
    if not os.path.exists(os.path.join(path_dest,f)):
        os.mkdir(os.path.join(path_dest,f))

scene_metadata = scene_metadata + "[list_images]\n<filename> <image_width> <image_height> <near_clipping_plane> <far_clipping_plane>\n"

for im in list_images:
    print("copying: "+im.split(' ', 1)[0])
    shutil.copy(
        os.path.join(path_data,im.split(' ', 1)[0]),
        os.path.join(path_dest,"images",im.split(' ', 1)[0])
        )

    if len(clipping_planes) is not 0:
        scene_metadata = scene_metadata + im[:-1] + " " + clipping_planes[0] + "\n"
    else:
        scene_metadata = scene_metadata + im[:-1] + " 0.01 100\n"

shutil.copy(
        os.path.join(path_data,"list_images.txt"),
        os.path.join(path_dest,"images","list_images.txt")
        )

shutil.copy(
        os.path.join(path_data,"bundle.out"),
        os.path.join(path_dest,"cameras","bundle.out")
        )

shutil.copy(
        os.path.join(path_data,"pmvs/models/pmvs_recon.ply"),
        os.path.join(path_dest,"meshes/recon.ply")
        )

scene_metadata = scene_metadata + "\n\n// Always specify active/exclude images after list images\n\n[exclude_images]\n<image1_idx> <image2_idx> ... <image3_idx>\n"


path_scene_metadata = os.path.join(path_dest, "scene_metadata.txt")

scene_metadata_file = open(path_scene_metadata, "w")
scene_metadata_file.write(scene_metadata)
scene_metadata_file.close()