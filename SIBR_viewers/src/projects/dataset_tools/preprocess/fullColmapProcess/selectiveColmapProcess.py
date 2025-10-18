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
This script runs a pipeline to create Colmap reconstruction data; you can choose which steps to do and it also handles test path data which are only calibrated

Parameters: -h help,
            -path <path to your dataset folder>,
            -sibrBinariesPath <binaries directory of SIBR>,
            -colmapPath <colmap path directory which contains colmap.bat / colmap.bin>,
            -quality <quality of the reconstruction : 'low', 'medium', 'high', 'extreme'>,

Usage: python selectiveColmapProcess.py -path <path to your dataset folder>
                                   -sibrBinariesPath <binaries directory of SIBR>
                                   -colmapPath <colmap path directory which contains colmap.bat / colmap.bin>
                                   -quality <quality of the reconstruction : 'low', 'medium', 'high', 'extreme'>

"""

import os, sys, shutil
import json
import argparse
from utils.paths import getBinariesPath, getColmapPath, getMeshlabPath
from utils.commands import  getProcess, getColmap
from utils.TaskPipeline import TaskPipeline
import selective_colmap_process

def main():
    parser = argparse.ArgumentParser()

    # common arguments
    parser.add_argument("--path", type=str, required=True, help="path to your dataset folder")
    parser.add_argument("--sibrBinariesPath", type=str, default=getBinariesPath(), help="binaries directory of SIBR")
    parser.add_argument("--colmapPath", type=str, default=getColmapPath(), help="path to directory colmap.bat / colmap.bin directory")
    parser.add_argument("--meshlabPath", type=str, default=getMeshlabPath(), help="path to meshlabserver directory")
    parser.add_argument("--quality", type=str, default='default', choices=['default', 'low', 'medium', 'average', 'high', 'extreme'],
        help="quality of the reconstruction")
    parser.add_argument("--dry_run", action='store_true', help="run without calling commands")
    parser.add_argument("--with_texture", action='store_true', help="Add texture steps")
    parser.add_argument("--create_sibr_scene", action='store_true', help="Create SIBR scene")
    parser.add_argument("--meshsize", type=str, help="size of the output mesh in K polygons (ie 200 == 200,000 polygons). Values allowed: 200, 250, 300, 350, 400")

    # additional step choice presets
    parser.add_argument("--from_step", type=str, default='default', help="Run from this step to --to_step")
    parser.add_argument("--to_step", type=str, default='default', help="up to but *excluding* this step (from --from_step); must be unique steps")
    parser.add_argument("--exclude_steps", type=str, default='default', help="Ignore these steps (string with space separated names of steps)")
    parser.add_argument("--calibrate_only", action='store_true', help="Do calibration only (from dataset_build_structure to before colmap_path_match_stereo with no special test frame processing")
    parser.add_argument("--calibrate_only_process_test", action='store_true', help="Do calibration only (from dataset_build_structure to before colmap_path_match_stereo with test frame processing")
    parser.add_argument("--mvs_only", action='store_true', help="Do mvs only (from colmap_path_match_stereo onwards without test frame processing")
    parser.add_argument("--fix_cameras", action='store_true', help="Do fix camera step only")
    parser.add_argument("--fix_cameras_stereo", action='store_true', help="Do fix camera step only for stereo/sparse")
    
    #colmap performance arguments
    parser.add_argument("--numGPUs", type=int, default=2, help="number of GPUs allocated to Colmap")

    # Feature extractor 
    parser.add_argument("--SiftExtraction.max_image_size", type=int, dest="siftExtraction_ImageSize")
    parser.add_argument("--SiftExtraction.estimate_affine_shape", type=int, dest="siftExtraction_EstimateAffineShape") 
    parser.add_argument("--SiftExtraction.domain_size_pooling", type=int, dest="siftExtraction_DomainSizePooling")
    parser.add_argument("--SiftExtraction.max_num_features", type=int, dest="siftExtraction_MaxNumFeatures")
    parser.add_argument("--ImageReader.single_camera", type=int, dest="imageReader_SingleCamera")

    # Exhaustive matcher
    parser.add_argument("--ExhaustiveMatching.block_size", type=int, dest="exhaustiveMatcher_ExhaustiveMatchingBlockSize")

    # Mapper
    parser.add_argument("--Mapper.ba_local_max_num_iterations", type=int, dest="mapper_MapperDotbaLocalMaxNumIterations")
    parser.add_argument("--Mapper.ba_global_max_num_iterations", type=int, dest="mapper_MapperDotbaGlobalMaxNumIterations")
    parser.add_argument("--Mapper.ba_global_images_ratio", type=float, dest="mapper_MapperDotbaGlobalImagesRatio")
    parser.add_argument("--Mapper.ba_global_points_ratio", type=float, dest="mapper_MapperDotbaGlobalPointsRatio")
    parser.add_argument("--Mapper.ba_global_max_refinements", type=int, dest="mapper_MapperDotbaGlobalMaxRefinements")
    parser.add_argument("--Mapper.ba_local_max_refinements", type=int, dest="mapper_MapperDotbaLocalMaxRefinements")

    
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

    # Update args with quality values
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "ColmapQualityParameters.json"), "r") as qualityParamsFile:
        qualityParams = json.load(qualityParamsFile)

        for key, value in qualityParams.items():
            if not key in args or args[key] is None:
                args[key] = qualityParams[key][args["quality"]] if args["quality"] in qualityParams[key] else qualityParams[key]["default"]

    # Get process steps
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "SelectiveColmapProcessSteps.json"), "r") as processStepsFile:
        steps = json.load(processStepsFile)["steps"]

    # apply selective processing
    exclude_steps = []
    if( args["exclude_steps"] != 'default' ):
        exclude_steps = args["exclude_steps"].split()

    from_step = args["from_step"]
    to_step = args["to_step"]

    # treat pre-defined cases
    if( args["fix_cameras"] or args["fix_cameras_stereo"]):
        new_steps = []
        for s in steps:
            if( s['name'] == "fix_cameras" ):
                print(s)
                (s['function_args'])['photoName'] = "00"
                if args["fix_cameras_stereo"]:
                    (s['function_args'])['sparse_subdir'] = os.path.join(os.path.join("colmap", "stereo"), "sparse")
                new_steps.append(s)
                break
        steps = new_steps

    if( args["calibrate_only"] or args["calibrate_only_process_test"] ):
        from_step = "build_dataset_structure" 
        to_step = "colmap_patch_match_stereo" 
        if( args["calibrate_only"] ):
            exclude_steps = [ "extract_video_frames", "fix_cameras", "colmap_image_deleter_colmap", "remove_video_images" ]

    if( args["mvs_only"] ):
        from_step = "colmap_patch_match_stereo" 
        to_step = "use_eol_fixed_mesh"
    

    if( from_step != 'default' ):
        # check if to_step exists
        if( to_step != 'default' ):
            # select steps
            newsteps = []
            adding_steps = False
            for s in steps:
                if( s['name'] == from_step ):
                    adding_steps = True
                # special case for last step that should be added
                if( s['name'] == to_step and s['name'] != "use_eol_fixed_mesh" ):
                    break
                if adding_steps :
                    if not (s['name'] in exclude_steps):
                        newsteps.append(s)
            steps = newsteps
        else:
            print("--from_step given without --to_step; ignoring")

    if args["dry_run"]:
        print("Keeping only following steps: ")
        for s in steps:
           print (s['name'])

    # Fixing path values
    args["path"] = os.path.abspath(args["path"])
    args["sibrBinariesPath"] = os.path.abspath(args["sibrBinariesPath"])
    args["colmapPath"] = os.path.abspath(args["colmapPath"])

    args["gpusIndices"] = ','.join([str(i) for i in range(args["numGPUs"])])

    programs = {
        "colmap": {
            "path": getColmap(args["colmapPath"])
        },
        "unwrapMesh": {
            "path": getProcess("unwrapMesh", args["sibrBinariesPath"])
        },
        "textureMesh": {
            "path": getProcess("textureMesh", args["sibrBinariesPath"])
        },
    }

    pipeline = TaskPipeline(args, steps, programs)

    pipeline.runProcessSteps()
    
    print("selectiveColmapProcess has finished successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
