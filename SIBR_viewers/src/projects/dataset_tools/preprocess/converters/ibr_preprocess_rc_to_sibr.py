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
This script converts a Reality Capture dataset to SIBR template dataset which can be fed to an SIBR application

Parameters: -h help,
            -i <path to input directory which is the output from RC> <default: ${CMAKE_INSTALL_DIR}/bin/datasets/rc_out/>,
            -o <path to output directory which can be fed into SIBR apps> <default: input directory> [optional],
            -r use release w/ debug symbols executables

Usage: python ibr_preprocess_rc_to_sibr.py -r
                                           -i <path_to_sibr>\sibr\install\bin\datasets\museum_sibr_new_preproc_template_RCOut
                                           -o <path_to_sibr>\sibr\install\bin\datasets\museum_sibr_new_preproc2

"""

import subprocess
import shutil
import os
import re
from utils.commands import getProcess
from utils.paths import getBinariesPath
from generate_list_images import generateListImages

from os import walk

# --------------------------------------------

from tempfile import mkstemp
from shutil import move
from os import remove, close

# ===============================================================================

import sys, getopt
import struct
import imghdr

# ===============================================================================
import bundle


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
                fhandle.seek(0)  # Read 0xff next
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
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


# ===============================================================================

def replace(file_path, pattern, subst):
    # Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    close(fh)
    # Remove original file
    remove(file_path)
    # Move new file
    move(abs_path, file_path)

def checkOutput( output, force_continue ):
    """Check if an external process succeeded and if we should abort if something went wrong

    Args:
        output          (int): output of some external launched process (0=fine, 1 or greater=error)
        force_continue  (bool): should we continue anyway?

    Returns:
        bool: True if last process succeeded
    """
    if(output != 0):
        if( not force_continue ):
            sys.exit()
        else:
            return False
    else:
        return True

def get_textured_mesh_base_name (source_folder):
    """Return the base name of a textured mesh obtained with RealityCapture

    Args:
        source_folder (str): path to dataset (bundle file, textured and images)

    Returns:
        str: base name of <base name>.obj, <base name>.mtl and <base name>_u1_v1.png
    """
    default_name = "textured"
    for file_in_dir in os.listdir (source_folder):
        if (file_in_dir.lower().endswith(".mtl")):
            return os.path.splitext( file_in_dir )[0]
    return default_name


def get_scale_factor (current_res, target_res):
    """Return the scale factor needed to go from current_res to target_res.
    The value is calculated based on the idea that we don't want to crop the input dataset anymore
    and that we prefer to add black borders in order to reach the target resolution

    Args:
        current_res (vec2): current dataset resolution
        target_res  (vec2): target dataset resolution

    Returns:
        float: scale factor
    """

    # to go from current to target resolution, we have to options:
    # either scale down current width to match target width and modify height accordingly
    # or scalen down current height to match target height and modify width accordingly.
    # we take the option that doesn't crop the remaining dimension and add the least black border

    # trying scaling down width
    alpha_by_width  = (float)(target_res[0]) / current_res[0]
    adjusted_height = (int) (alpha_by_width * current_res[1])
    delta_height    = target_res[1] - adjusted_height

    # trying scaling down height
    alpha_by_height = (float)(target_res[1]) / current_res[1]
    adjusted_width  = (int) (alpha_by_height * current_res[0])
    delta_width     = target_res[0] - adjusted_width

    # since we don't want to crop the input images even more, we considerer only
    # the options were black borders need to be added

    if (delta_height < 0):
        # height would need to be cropped. take the other option
        return alpha_by_height
    elif (delta_width < 0):
        # width would need to be cropped. take the other option
        return alpha_by_width
    else:
        # none of them need to be cropped. take option that adds less black border
        return alpha_by_width if (delta_height < delta_width) else alpha_by_height



# ===============================================================================

# --------------------------------------------
# 0. Paths, commands and options

def main(argv, path_dest):
    opts, args = getopt.getopt(argv, "hi:ro:", ["idir=", "bin="])
    executables_suffix = ""
    executables_folder = getBinariesPath()
    path_data = ""
    for opt, arg in opts:
        if opt == '-h':
            print("-i path_to_rc_data_dir -o path_to_destination_dir [-r (use release w/ debug symbols executables)]")
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
        elif opt in ('-bin', '--bin'):
            executables_folder = os.path.abspath(arg)

    return (path_data, path_dest, executables_suffix, executables_folder)


path_dest = ""
path_data, path_dest, executables_suffix, executables_folder = main(sys.argv[1:], path_dest)

if(path_data == ""):
    path_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))

if(path_dest == ""):
    path_dest = path_data

path_data = os.path.abspath(path_data + "/") + "/"
path_dest = os.path.abspath(path_dest + "/") + "/"
executables_folder = os.path.abspath(executables_folder + "/") + "/"

path_in_imgs = path_data
path_out_imgs = path_dest + "images/"

print(['Raw_data folder: ', path_data])
print(['Path_dest: ', path_dest])
print(['Executables folder: ', executables_folder])

# dirs to create

raw_data = "raw/"
cameras_dir = "cameras/"
images_dir = "images/"
pmvs_model_dir	= "meshes/"
parentdir = os.path.dirname(os.path.split(path_dest)[0]) 
print("COMPARE " ,  parentdir , " AND " ,  os.path.dirname(os.path.split(path_data)[0]))
if( parentdir == os.path.dirname(os.path.split(path_data)[0])):
	capreal_dir = os.path.join(parentdir, "capreal/")
	print("CAPREAL " ,  capreal_dir)
	if not os.path.exists(capreal_dir):
		os.makedirs(capreal_dir)

dirs_to_create = [ raw_data, cameras_dir, images_dir, pmvs_model_dir]

for dir_to_create in dirs_to_create:
    path_to_dir = os.path.join(path_dest, dir_to_create)
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)


################################# GLOBALS ######################################

# half size parameters
width_limit         = 2500
create_temp_folders = False

input_bundle = bundle.Bundle(path_data + "bundle.out")


############################# RUN DISTORDCROP ##################################
# by calling distordCrop (preprocess/distordCrop), input images that have a
# resolution completely different from the average or that have too much
# black border added by RealityCapture will be listed in a exclude_images.txt file.
# A new proposed resolution will be also output in a file called cropNewSize.txt
# In order to accelerate this process and avoid loading the images multiple times,
# make sure that average resolution was already calculated and there is file called
# resolutions.txt in the dataset source folder containing the current resolution
# of each image.

# distordCrop executable
crop_app = getProcess("distordCrop" + executables_suffix, executables_folder)

# query current avera resolution in dataset
avg_resolution = input_bundle.get_avg_resolution()

# generate resolutions.txt and put it in the current dataset folder
resolutions_txt_path = os.path.join(path_data, "resolutions.txt")
input_bundle.generate_list_of_images_file(resolutions_txt_path)

# call distordCrop
p_exit = subprocess.call([crop_app, "--path", path_data, "--ratio",  "0.3", "--avg_width", str(avg_resolution[0]), "--avg_height", str(avg_resolution[1]) ])
print(crop_app, " exited with ", p_exit);
checkOutput(p_exit, False)

# read new proposed resolution and check if images were discarded
exclude = []
path_to_exclude_images_txt = os.path.join(path_data, "exclude_images.txt")
if (os.path.exists(path_to_exclude_images_txt)):
    # list of excluded cameras (one line having all the camera ids to exclude)
    exclusion_file = open(path_to_exclude_images_txt, "r")
    line = exclusion_file.readline()
    tokens = line.split()

    for cam_id in tokens:
        exclude.append(int(cam_id))
    exclusion_file.close()

# exclude cams from bundle file
input_bundle.exclude_cams (exclude)

# read proposed cropped resolution
path_to_crop_new_size_txt = os.path.join(path_data, "cropNewSize.txt")
with open(path_to_crop_new_size_txt) as crop_size_file:
    line = crop_size_file.readline()
    tokens = line.split()
    new_width   = int(tokens[0])
    new_height  = int(tokens[1])
    proposed_res = [new_width, new_height]

print("crop size:", proposed_res)

################################################################################

##################### TRANSFORM IMAGES TO TARGET RESOLUTION ####################
# we need to crop images to the previous proposed resolution (crop applied from the center).
# if a target resolution was passed as parameter, we also need to scale down the images
# and pad them with black borders in order to end up with the exact target resolution.
# Before padding the images, we need to have available the temporary result of the
# dataset scaled down in order to potentially call harmonize (which doesn't work with
# images that were padded with black borders).
# Scaled down dataset will be stored inside \<destination_folder\>/scaledDown
# and we will store the scale down factor in a scale_factor.txt file
target_res = None
if (proposed_res[0] > width_limit):
    half_width  = (int)(proposed_res[0] * 0.5)
    half_height = (int)(proposed_res[0] * 0.5)
    target_res  = [half_width, half_height]

# cropFromCenter executable
crop_from_center_app = getProcess("cropFromCenter" + executables_suffix, executables_folder)

# generate file with list of current selected images to process
path_to_transform_list_txt = os.path.join (path_data, "toTransform.txt")
input_bundle.generate_list_of_images_file(path_to_transform_list_txt)

crop_from_center_args = [crop_from_center_app,
    "--inputFile", path_to_transform_list_txt,
    "--outputPath", path_out_imgs,
    "--avgResolution", str(avg_resolution[0]), str(avg_resolution[1]),
    "--cropResolution", str(proposed_res[0]), str(proposed_res[1])
]

# calculate scale factor and how to achieve target resolution
# scaled dataset will be store in destionation_folder/scaled
if (target_res is not None):
    scale_factor = get_scale_factor(proposed_res, target_res)
    crop_from_center_args.extend([
        "--scaleDownFactor", str(scale_factor),
        "--targetResolution", str(target_res[0]), str(target_res[1])
    ])

# call cropFromCenter
p_exit = subprocess.call(crop_from_center_args)
print(crop_from_center_app, " exited with ", p_exit);
checkOutput(p_exit, False)

# write bundle file in output cameras folder
path_to_output_bundle = os.path.join (path_dest, cameras_dir, "bundle.out")
input_bundle.save(path_to_output_bundle)

# and also in scaled down output folder if needed
if (target_res is not None):
    # scale bundle file for the same factor
    input_bundle.scale(scale_factor)
    path_to_scaled_down_output_bundle = os.path.join (os.path.join (path_dest, "images/scaled"), "bundle.out")
    input_bundle.save(path_to_scaled_down_output_bundle)

################################################################################

############################ MOVE REST OF ASSETS ###############################

textured_mesh_base_name = get_textured_mesh_base_name(path_data)
print("***** TEXT * ", textured_mesh_base_name)

# copy files
files_to_move = [   #['pmvs/models/pmvs_recon.ply',''],
                    ['pmvs_recon.ply', pmvs_model_dir],
                    ['mesh.ply', pmvs_model_dir],
                    ['mesh.ply', capreal_dir],
                    ['recon.ply', pmvs_model_dir],
                    ['rc_out.csv', path_dest],
                    ["textured.obj", capreal_dir],
                    ["textured.mtl", capreal_dir],
                    ["textured_u1_v1.png", capreal_dir],
                    [textured_mesh_base_name + ".obj", capreal_dir],
                    [textured_mesh_base_name + ".mtl", capreal_dir],
                    [textured_mesh_base_name + "_u1_v1.png", capreal_dir] ]
for filename, directory_name in files_to_move:
    source_file = os.path.join (path_data, filename)
    destination_file = os.path.join (os.path.join (path_dest, directory_name), filename)
    # print("Trying ",  source_folder + file , " ", destination_folder + dir + file )
    print("Trying ",  source_file , "-->", destination_file )
    if (os.path.exists(source_file)):
        print("Moving ",  source_file , " ", destination_file )
        shutil.copy( source_file , destination_file )

################################################################################

######################## CALCULATE CLIPPING PLANES #############################
# clippingPlanes executable
clipping_planes_app = getProcess("clippingPlanes" + executables_suffix, executables_folder)

clipping_planes_args = [clipping_planes_app, path_dest]

# call clippingPlanes app
p_exit = subprocess.call(clipping_planes_args)
print(clipping_planes_app, " exited with ", p_exit);
checkOutput(p_exit, False)

################################################################################

########################## CREATE LIST IMAGES ##################################

path_images = os.path.join(path_dest, images_dir)
path_list_images = os.path.join(path_images, "list_images.txt")
generateListImages(path_images)

################################################################################

######################## CREATE SCENE METADATA #################################

# read list image file
list_images = []

if os.path.exists(path_list_images):
    list_image_file = open(path_list_images, "r")

    for line in list_image_file:
        list_images.append(line)

    list_image_file.close()

# read clipping planes file
path_clipping_planes = os.path.join(path_dest, "clipping_planes.txt")
clipping_planes = []

if os.path.exists(path_clipping_planes):
    clipping_planes_file = open(path_clipping_planes, "r")

    for line in clipping_planes_file:
        clipping_planes.append(line)

    clipping_planes_file.close()


# Create scene metadata file from list image file
scene_metadata = "Scene Metadata File\n\n"

if len(list_images) == len(clipping_planes):
    scene_metadata = scene_metadata + "[list_images]\n<filename> <image_width> <image_height> <near_clipping_plane> <far_clipping_plane>\n"
    new_list = [a[:-1] + " " + b for a, b in zip(list_images, clipping_planes)]
    for line in new_list:
        scene_metadata = scene_metadata + line

scene_metadata = scene_metadata + "\n\n// Always specify active/exclude images after list images\n\n[exclude_images]\n<image1_idx> <image2_idx> ... <image3_idx>\n"

# if len(exclude) > 0:
#     for line in exclude:
#         scene_metadata = scene_metadata + str(line) + " "

scene_metadata = scene_metadata + "\n\n\n[other parameters]"


# rename pmvs_recon.ply to recon.ply
if (os.path.exists(os.path.join(path_dest, pmvs_model_dir, "pmvs_recon.ply"))):
    shutil.copy(os.path.join(path_dest, pmvs_model_dir, "pmvs_recon.ply"), os.path.join(path_dest, pmvs_model_dir, "recon.ply"))

if (os.path.exists(os.path.join(path_dest, pmvs_model_dir, "mesh.ply"))):
    shutil.copy(os.path.join(path_dest, pmvs_model_dir, "mesh.ply"), os.path.join(path_dest, pmvs_model_dir, "recon.ply"))

path_scene_metadata = os.path.join(path_dest, "scene_metadata.txt")

scene_metadata_file = open(path_scene_metadata, "w")
scene_metadata_file.write(scene_metadata)
scene_metadata_file.close()

################################################################################

for filename in os.listdir(path_data):
    src = os.path.join(path_data, filename)
    dst = os.path.join(path_dest, raw_data)
    if not os.path.isdir(src):
        shutil.copy(src, dst)

print("Fin.")
