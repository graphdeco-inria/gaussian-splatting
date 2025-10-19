
# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
#


#!/usr/bin/env python
#! -*- encoding: utf-8 -*-

""" @package dataset_tools_preprocess
This script processes images and creates an RealityCapture (RC) reconstruction, then creates a colmap version using the RC camera registration

Parameters: -h help,
            -path <path to your dataset folder>,

Usage: python processRC.py -path <path to your dataset folder>

"""

import os, sys, shutil
os.sys.path.append('../preprocess/')
os.sys.path.append('../preprocess/realityCaptureTools')
os.sys.path.append('../preprocess/fullColmapProcess')
os.sys.path.append('../preprocess/converters')

import json
import argparse
from utils.paths import getBinariesPath, getColmapPath, getMeshlabPath
from utils.commands import  getProcess, getColmap, getRCprocess
from utils.TaskPipeline import TaskPipeline
import rc_tools
import colmap2nerf
import selective_colmap_process

def find_file(filename):
    fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
    if not os.path.exists(fname):
        fname = os.path.join("../preprocess/fullColmapProcess", filename)
    if not os.path.exists(fname):
        fname = os.path.join("../preprocess/realityCaptureTools", filename)
    if not os.path.exists(fname):
        fname = os.path.join("../preprocess/converters", filename)
    return fname
    

def main():
    parser = argparse.ArgumentParser()

    # common arguments
    parser.add_argument("--sibrBinariesPath", type=str, default=getBinariesPath(), help="binaries directory of SIBR")
    parser.add_argument("--colmapPath", type=str, default=getColmapPath(), help="path to directory colmap.bat / colmap.bin directory")
    parser.add_argument("--quality", type=str, default='default', choices=['default', 'low', 'medium', 'average', 'high', 'extreme'],
        help="quality of the reconstruction")
    parser.add_argument("--path", type=str, required=True, help="path to your dataset folder")
    parser.add_argument("--dry_run", action='store_true', help="run without calling commands")
    parser.add_argument("--rc_path", type=str, required=False, help="path to rc dataset, containing bundle.out and images")
    parser.add_argument("--out_path", type=str, required=False, help = "output path ")
    parser.add_argument("--video_name", type=str, default='default', required=False, help = "name of video file to load")
    parser.add_argument("--create_colmap", action='store_true', help="create colmap hierarchy")
    parser.add_argument("--target_width", type=str, default='default', help="colmap_target_width")
    parser.add_argument("--from_step", type=str, default='default', help="Run from this step to --to_step (or end if no to_step")
    parser.add_argument("--to_step", type=str, default='default', help="up to but *excluding* this step (from --from_step); must be unique steps")

    # RC arguments
    parser.add_argument("--do_mvs", action='store_false', help="use train folder")
    parser.add_argument("--calib_only", action='store_true', help="only do calibration")
    parser.add_argument("--hires_nerf", action='store_true', help="create hi res nerf")
    parser.add_argument("--car_data", action='store_true', help="pre(pre)process car camera data ")
    parser.add_argument("--do_train", action='store_false', help="use train folder")
    parser.add_argument("--do_validation", action='store_false', help="use validation folder")
    parser.add_argument("--no_validation_split", action='store_true', help="dont do validation split")
    parser.add_argument("--do_video", action='store_true', help="use video folder (mp4)")
    parser.add_argument("--do_test", action='store_true', help="use test folder (stills path)")
    parser.add_argument("--auto_recon_area", action='store_true', help="automatically set recon area (no user intervention)")

    parser.add_argument("--config_folder", type=str, default='default', help="folder containing configuration files; usually cwd")
    parser.add_argument("--model_name", type=str, default='default', help="Internal name of RC model")
    parser.add_argument("--path_prefix", type=str, default='default', help="Internal prefix of path images")
    parser.add_argument("--one_over_fps", type=str, default='default', help="Sampling rate for the video")
    parser.add_argument("--valid_skip", type=str, default='default', help="skip every nth image for validation")
    # "presets"
    parser.add_argument("--images_only", action='store_false', help="just process images: no validation, no test")
    parser.add_argument("--video_only", action='store_true', help="just process video: no photos, no test")

    parser.add_argument("--no_refl", action='store_true', help="dont densify mesh, dont convert_sibr, dont create nerf (def: false)")


    # needed to avoid parsing issue for passing arguments to next command (TODO)
    parser.add_argument("--video_filename", type=str, default='default', help="full path of video file (internal argument; do not set)")
    parser.add_argument("--mesh_obj_filename", type=str, default='default', help="full path of obj mesh file (internal argument; do not set)")
    parser.add_argument("--mesh_xyz_filename", type=str, default='default', help="full path of xyz point cloud file (internal argument; do not set)")
    parser.add_argument("--mesh_ply_filename", type=str, default='default', help="full path of ply mesh file (internal argument; do not set)")

    # colmap
    #colmap performance arguments
    parser.add_argument("--numGPUs", type=int, default=2, help="number of GPUs allocated to Colmap")

    # Patch match stereo
    parser.add_argument("--PatchMatchStereo.max_image_size", type=int, dest="patchMatchStereo_PatchMatchStereoDotMaxImageSize")
    parser.add_argument("--PatchMatchStereo.window_radius", type=int, dest="patchMatchStereo_PatchMatchStereoDotWindowRadius")
    parser.add_argument("--PatchMatchStereo.window_step", type=int, dest="patchMatchStereo_PatchMatchStereoDotWindowStep")
    parser.add_argument("--PatchMatchStereo.num_samples", type=int, dest="patchMatchStereo_PatchMatchStereoDotNumSamples")
    parser.add_argument("--PatchMatchStereo.num_iterations", type=int, dest="patchMatchStereo_PatchMatchStereoDotNumIterations")
    parser.add_argument("--PatchMatchStereo.geom_consistency", type=int, dest="patchMatchStereo_PatchMatchStereoDotGeomConsistency")

    # Stereo fusion
    parser.add_argument("--StereoFusion.check_num_images", type=int, dest="stereoFusion_CheckNumImages")
    parser.add_argument("--StereoFusion.max_image_size", type=int, dest="stereoFusion_MaxImageSize")


    args = vars(parser.parse_args())

    from_step = args["from_step"]
    to_step = args["to_step"]

    # Update args with quality values
    fname = find_file("ColmapQualityParameters.json")
    with open(fname, "r") as qualityParamsFile:
        qualityParams = json.load(qualityParamsFile)

        for key, value in qualityParams.items():
            if not key in args or args[key] is None:
                args[key] = qualityParams[key][args["quality"]] if args["quality"] in qualityParams[key] else qualityParams[key]["default"]

    # Get process steps
    fname = find_file("processRCSteps.json")
    with open(fname, "r") as processStepsFile:
        steps = json.load(processStepsFile)["steps"]

    # Fixing path values
    args["path"] = os.path.abspath(args["path"])
    args["sibrBinariesPath"] = os.path.abspath(args["sibrBinariesPath"])
    args["colmapPath"] = os.path.abspath(args["colmapPath"])
    args["gpusIndices"] = ','.join([str(i) for i in range(args["numGPUs"])])

    args["mesh_obj_filename"] = os.path.join(args["path"], os.path.join("rcScene", os.path.join("meshes", "mesh.obj")))
    args["mesh_xyz_filename"] = os.path.join(args["path"], os.path.join("rcScene", os.path.join("meshes", "point_cloud.xyz")))
    args["mesh_ply_filename"] = os.path.join(args["path"], os.path.join("sibr", os.path.join("capreal", "mesh.ply")))

    args["path_prefix"] = "test_"

    # fixed in preprocess
    args["video_filename"] = os.path.join(args["path"], os.path.join("raw", os.path.join("videos", "XXX.mp4")))
    if args["config_folder"] == 'default':
        if os.path.exists("registrationConfig.xml"):
            args["config_folder"] = "."
        elif os.path.exists("../preprocess/realityCaptureTools/registrationConfig.xml"):
            args["config_folder"] = "../preprocess/realityCaptureTools/"

    if args["valid_skip"] == 'default' :
        args["valid_skip"] = "10"
        
    if args["one_over_fps"] == 'default':
        args["one_over_fps"] = "0.02"

    if args["target_width"] == 'default':
        args["target_width"] = "1000"
    print("TARGET WIDTH ", args["target_width"])

    if args["no_validation_split"]:
        args["do_validation_split"] = False
    else:
        args["do_validation_split"] = True

    # presets

    exclude_steps = []

    if args["no_refl"] == True:
        exclude_steps = [ "densify_mesh", "dense_mesh", "create_nerf", "convert_sibr_mesh" ] 
        print("No densification, no sibr, no nerf, exclude:", exclude_steps)

    if args["car_data"]:
        print("Doing car data")
    else:
        print("No car data")

    if args["calib_only"]:
        to_step = "colmap_patch_match_stereo"
        args["do_mvs"] = False
        exclude_steps = [ "densify_mesh", "dense_mesh" ] 

    # either do video or do_test
    if args["do_video"]:
        args["path_prefix"] = "frame"       

    if args["video_only"]:
        args["do_train"] = False
        args["do_validation"] = False
        args["do_test"] = False
        args["do_video"] = True

    if args["do_test"]:
        args["path_prefix"] = "test_"       
        args["do_video"] = False


    if args["video_only"] and args["calib_only"]:
        exclude_steps = [ "densify_mesh", "dense_mesh",  "rc_to_colmap_path_cameras", "rc_to_colmap_validation_cameras" ]

    programs = {
        "colmap": {
            "path": getColmap(args["colmapPath"])
        },
        "RC": {
            "path": getRCprocess()
        }
    }

    # TODO: move to generic taskpipeline code; 
    if( from_step != 'default' or to_step != 'default' or exclude_steps != []):
        # check if to_step exists
        # select steps
        newsteps = []
        if from_step != 'default':
            adding_steps = False
        else:
            adding_steps = True
                
        for s in steps:
            if s['name'] == from_step :
                adding_steps = True
            if s['name'] == to_step :
                break
            if adding_steps and (not (s['name'] in exclude_steps)):
                newsteps.append(s)

        steps = newsteps

    pipeline = TaskPipeline(args, steps, programs)

    pipeline.runProcessSteps()
    
    print("selectiveColmapProcess has finished successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
