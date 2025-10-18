#
# RealityCapture tools
#
import os
import os.path
import sys
import argparse
import shutil
import sqlite3
import read_write_model as rwm
import pymeshlab


import cv2
print(cv2.__version__)


""" @package dataset_tools_preprocess
Library for RealityCapture treatment


"""

import bundle
import os, sys, shutil
import json
import argparse
import scipy
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.paths import getBinariesPath, getColmapPath, getMeshlabPath
from utils.commands import  getProcess, getColmap, getRCprocess, runCommand

def preprocess_for_rc(path, video_name='default', do_validation_split=True, valid_skip='10'):
    # create train/validation split (every 10 images by default now)
    print("VALID SKIP ", valid_skip)
    int_valid_skip = int(valid_skip)

    # Should exist
    rawpath = os.path.join(path, "raw")
    if not os.path.exists(rawpath):
        os.makedirs(os.path.join(path, "raw"))

    imagespath = os.path.abspath(os.path.join(rawpath, "images"))
    testpath = os.path.abspath(os.path.join(rawpath, "test"))
    videopath = os.path.abspath(os.path.join(rawpath, "videos"))
    do_test = False
    inputpath = os.path.join(path, "input")

    # If not, move around
    if not os.path.exists(imagespath):
        if os.path.exists(os.path.join(path, "images")):
            shutil.move(os.path.join(path, "images"), imagespath)
        elif not os.path.exists(os.path.join(path, "videos")) and not os.path.exists(videopath):
            print("ERROR: No images nor video, exiting. Images should be in $path/raw/images")
            exit(-1)
        # videos are optional
        if os.path.exists(os.path.join(path, "videos")):
            shutil.move(os.path.join(path, "videos"), videopath)
        # test images (stills for path)
        test_orig = os.path.join(path, "test")
#        print("TEST ", test_orig, " " , os.path.exists(test_orig) , " > ", testpath)
        if os.path.exists(test_orig):
            do_test = True
            shutil.move(test_orig, testpath)
    else:
        print("Found images {}".format(imagespath))
        if os.path.exists(videopath):
            print("Found video {}".format(videopath))
        if os.path.exists(testpath):
            print("Found test {}".format(testpath))
            do_test = True

    cnt = 0
    validation_path = os.path.abspath(os.path.join(inputpath, "validation"))
    train_path = os.path.abspath(os.path.join(inputpath, "train"))
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)
    input_test_path = os.path.abspath(os.path.join(inputpath, "test"))
    if not os.path.exists(input_test_path):
        os.makedirs(input_test_path)

    # rcScene -- will contain full bundle files from RC
    rcscenepath = os.path.join(path, "rcScene")
    if not os.path.exists(rcscenepath):
        os.makedirs(rcscenepath)

    # rcProj -- RC project save
    rcprojpath = os.path.join(path, "rcProj")
    if not os.path.exists(rcprojpath):
        os.makedirs(rcprojpath)


    # sibr -- will contain full size colmap
    sibrpath = os.path.join(path, "sibr")
    if not os.path.exists(sibrpath):
        os.makedirs(sibrpath)
        caprealpath = os.path.join(sibrpath, "capreal")
        os.makedirs(caprealpath)

    # BUG: do_validation_split is a string
    if do_validation_split != 'False':
        print("Train/Validation", train_path , " : ", validation_path)
        for filename in os.listdir(imagespath):
            ext = os.path.splitext(filename)[1]
            if ext == ".JPG" or ext == ".jpg" or ext == ".PNG" or ext == ".png" :
                image = os.path.join(imagespath, filename) 
#            print("IM ", image)
                if not(cnt % int_valid_skip ):
                    filename = "validation_"+filename
                    fname = os.path.join(validation_path, filename)
#                print("Copying ", image, " to ", fname , " in validation")
                    shutil.copyfile(image, fname)
                else:
                    filename = "train_"+filename
                    fname = os.path.join(train_path, filename)
#                print("Copying ", image, " to ", fname , " in train")
                    shutil.copyfile(image, fname)

            cnt = cnt + 1
    else:
        print("Not doing validation")
        for filename in os.listdir(imagespath):
            ext = os.path.splitext(filename)[1]
            if ext == ".JPG" or ext == ".jpg" or ext == ".PNG" or ext == ".png" :
                image = os.path.join(imagespath, filename) 
#            print("IM ", image)
          
                filename = "train_"+filename
                fname = os.path.join(train_path, filename)
#                print("Copying ", image, " to ", fname , " in train")
                shutil.copyfile(image, fname)

            cnt = cnt + 1

    if do_test:
        for filename in os.listdir(testpath):
            ext = os.path.splitext(filename)[1]
            if ext == ".JPG" or ext == ".jpg" or ext == ".PNG" or ext == ".jpg" :
                image = os.path.join(testpath, filename) 
                filename = "test_"+filename
                fname = os.path.join(input_test_path, filename)
#                print("Copying ", image, " to ", fname , " in test")
                shutil.copyfile(image, fname)
    else:
        print ("****************** NOT DOING TEST !!!")


    # extract video name -- if not given, take first
    if video_name == 'default':
        if os.path.exists(videopath):
            for filename in os.listdir(videopath):
#            print("Checking ", filename)
                if ("MP4" in filename) or ("mp4" in filename):
                    video_name = filename
    video_filename = os.path.join(path, os.path.join("raw", os.path.join("videos", video_name)))
    print("Full video path:", video_filename)

    return "video_filename", video_filename

def convert_sibr_mesh(path):
    ms = pymeshlab.MeshSet()
    mesh_path = os.path.join(os.path.join(os.path.join(path, "rcScene"), "meshes"), "mesh.obj")
    print("Loading mesh (slow...)", mesh_path)
    ms.load_new_mesh(mesh_path)
    meshply_path = out_mesh_path = os.path.join(os.path.join(os.path.join(path, "sibr"), "capreal"), "mesh.ply")
    print("Saving mesh (slow...)", out_mesh_path)
    ms.save_current_mesh(out_mesh_path, save_wedge_texcoord=False, binary=False)
    print("Done saving mesh (slow...)", out_mesh_path)
    texture_path = os.path.join(os.path.join(os.path.join(path, "sibr"), "capreal"), "mesh_u1_v1.png")
    out_texture_path = os.path.join(os.path.join(os.path.join(path, "sibr"), "capreal"), "texture.png")
    print("Copying (to allow meshlab to work) {} to {}".format(texture_path, out_texture_path))
    shutil.copyfile(texture_path, out_texture_path)
    out_mesh_path = os.path.join(os.path.join(os.path.join(os.path.join(path, "sibr"), "colmap"), "stereo"), "meshed-delaunay.ply")
    print("Copying {} to {}".format(meshply_path, out_mesh_path))
    shutil.copyfile(meshply_path, out_mesh_path)


def densify_mesh(mesh_path):
    ms = pymeshlab.MeshSet()
    subdiv_threshold = pymeshlab.Percentage(0.09)
    ms.load_new_mesh(mesh_path)
    print("Loaded mesh ", mesh_path, " Subdividing (this can take some time)...")
    ms.subdivision_surfaces_butterfly_subdivision(threshold=subdiv_threshold)
    path_split = os.path.split(mesh_path)
    dense_mesh_fname = "dense_" + path_split[1]
    fname, fname_ext = os.path.splitext(dense_mesh_fname)
    dense_mesh_fname = fname + ".obj"
    dense_mesh_path = os.path.join(path_split[0], dense_mesh_fname)
    print("Writing dense mesh ", dense_mesh_path)
    ms.save_current_mesh(dense_mesh_path)

def rc_to_colmap(rc_path, out_path, create_colmap=False, target_width=-1):

    input_bundle = bundle.Bundle(os.path.join(rc_path , "bundle.out"))
    input_bundle.generate_list_of_images_file (os.path.join(rc_path , "list_images.txt"))

    dst_image_path = os.path.join(out_path, "images")

    # create entire colmap structure
    if create_colmap:
        dir_name = os.path.join(out_path, "stereo")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        stereo_stereo_dir = os.path.join(dir_name, "stereo")
        if not os.path.exists(stereo_stereo_dir):
            os.makedirs(stereo_stereo_dir)

        dst_image_path = os.path.join(dir_name, "images")

        sparse_stereo_dir = dir_name = os.path.join(dir_name, "sparse")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


    else:
        sparse_stereo_dir = out_path

    if not os.path.exists(dst_image_path):
        os.makedirs(dst_image_path)

    # create cameras.txt
    fname = os.path.join(sparse_stereo_dir, "cameras.txt")
    print("Creating ", fname)
    numcams = len(input_bundle.list_of_input_images)

    camera_id = 1
    scale = 1.
    with open(fname, 'w') as outfile:
        outfile.write("# Camera list with one line of data per camera:\n")
        outfile.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        outfile.write("# Number of cameras: {}\n".format(numcams))
        for im in input_bundle.list_of_input_images:
            width = im.resolution[0]
            height = im.resolution[1]
            focal_length = input_bundle.list_of_cameras[camera_id-1].focal_length

            # resize images if required
            if target_width != -1:
                orig_width = width
                width = float(target_width)
                scale = float(target_width) / orig_width 
                aspect = height / orig_width
                height = width * aspect
                focal_length = scale * focal_length
               
            outfile.write("{} PINHOLE {} {} {} {} {} {}\n".format(camera_id, int(width), int(height), focal_length, focal_length, width/2.0, height/2.0))
            camera_id = camera_id + 1
        outfile.close()

    # create images.txt
    fname = os.path.join(sparse_stereo_dir, "images.txt")

    print("Creating ", fname)
    camera_id = 1
    with open(fname, 'w') as outfile:
        outfile.write( "# Image list with two lines of data per image:\n" )
        outfile.write( "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" )
        outfile.write( "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" )
        point2d_index = 0
        for cam in input_bundle.list_of_cameras:
            in_im = input_bundle.list_of_input_images[camera_id-1]
            imname = in_im.path
            name = os.path.basename(imname)
            im = cv2.imread(imname, cv2.IMREAD_UNCHANGED)
            w = im.shape[1]
            h = im.shape[0]

            # to sibr internal
            br = np.matrix(cam.rotation).transpose()
            t = -np.matmul(br , np.matrix([cam.translation[0], cam.translation[1], cam.translation[2]]).transpose())
         
            # sibr save to colmap
            br = np.matmul(br, np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            br = br.transpose()

            sci_rot = R.from_matrix(br)
            sci_quat = sci_rot.as_quat()

            t = -np.matmul(br, t)

            outfile.write("{} {} {} {} {} {} {} {} {} {}\n".format(camera_id, -sci_quat[3], -sci_quat[0], -sci_quat[1], -sci_quat[2], t[0,0], t[1,0], t[2,0], camera_id, name))
            # write out points
            first = False
            scale = 1.0
            if target_width !=1 :
                scale = float(target_width) / float(in_im.resolution[0])
            for p in cam.list_of_feature_points:
                for v in p.view_list:
                    if v[0] == camera_id-1:
                        outfile.write( str(scale*(2.*v[2]+w)) + " " + str(scale*(2.*v[3]+h))+ " -1" ) # TODO: not sure about this, seems to be -1 in all existing files
                        if not first:
                            outfile.write(" ")
                        else:
                            first = False

                        p.point2d_index[v[0]] = point2d_index
                        point2d_index = point2d_index + 1

            outfile.write("\n")
            camera_id = camera_id + 1
    outfile.close()

    # create points3D.txt
    fname = os.path.join(sparse_stereo_dir, "points3D.txt")

    print("Creating ", fname)
    camera_id = 1
    with open(fname, 'w') as outfile:
        num_points = len(input_bundle.list_of_feature_points)
#  FIX mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items()))/len(points3D)
        mean_track_length = 10 # 10 is a placeholder value
        outfile.write("# 3D point list with one line of data per point:\n" )
        outfile.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        outfile.write("# Number of points: {}, mean track length: {}\n".format(num_points, mean_track_length))
        for p in input_bundle.list_of_feature_points:
            # error set to 0.1 for all
            outfile.write(str(p.id+1)+ " " + str(p.position[0]) + " " + str(p.position[1]) + " " + str(p.position[2]) + " " + str( p.color[0])+  " " + str( p.color[1])+  " " + str( p.color[2])+ " 0.1")
            for v in p.view_list:
#                print("Cam id ", v[0], " P= ", p.id+1 , " p2dind " , p.point2d_index )
                outfile.write(" " + str(v[0]+1)+ " " + str(p.point2d_index[v[0]])  )
            outfile.write("\n")


    if create_colmap:
        fname = os.path.join(stereo_stereo_dir, "fusion.cfg")
        outfile_fusion = open(fname, 'w') 
        fname = os.path.join(stereo_stereo_dir, "patch-match.cfg")
        outfile_patchmatch = open(fname, 'w') 
        outdir = os.path.join(stereo_stereo_dir, "normal_maps")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.join(stereo_stereo_dir, "depth_maps")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.join(stereo_stereo_dir, "consistency_graphs")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # copy images
    for fname in os.listdir(rc_path):
        if fname.endswith(".jpg") or fname.endswith(".JPG") or fname.endswith(".png") or fname.endswith(".PNG") :
            src_image_fname = os.path.join(rc_path, fname)
            dst_image_fname = os.path.join(dst_image_path, os.path.basename(fname))
#            print("Copying ", src_image_fname, "to ", dst_image_fname)

            if create_colmap:
                  outfile_fusion.write(fname+"\n")
                  outfile_patchmatch.write(fname+"\n")
                  outfile_patchmatch.write("__auto__, 20\n")

            # resize if necessary
            if target_width != -1:
                im = cv2.imread(src_image_fname, cv2.IMREAD_UNCHANGED)
                orig_width = im.shape[1]
                orig_height = im.shape[0]
                width = float(target_width)
                scale = float(target_width)/ orig_width 
                aspect = orig_height / orig_width
                height = width * aspect
                dim = (int(width), int(height))
                im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(dst_image_fname, im)
            else:
                shutil.copyfile(src_image_fname, dst_image_fname)

    # copy mesh; fake it
    if create_colmap:
        outfile_patchmatch.close()
        outfile_fusion.close()

# taken from ibr_preprocess_rc_to_sibr
# TODO: pretty ugly needs rethink and cleanup
def crop_images(path_data, path_dest):
    # open calibration data
    input_bundle = bundle.Bundle(os.path.join(path_data , "bundle.out"))
    # query current average resolution of these cameras
    avg_resolution = input_bundle.get_avg_resolution()
    print("AVG resolution ", avg_resolution)

    # special case: validation_cameras take size/crop data from train cameras so they are all the same
    if "validation_" not in path_data:

        # generate resolutions.txt and put it in the current dataset folder
        resolutions_txt_path = os.path.join(path_data, "resolutions.txt")
        input_bundle.generate_list_of_images_file(resolutions_txt_path)

        # setup avg_resolution parameters for distordCrop
        print("Command: run distordCrop ARGS: ", "--path", path_data, "--ratio",  "0.3", "--avg_width", str(avg_resolution[0]), "--avg_height", str(avg_resolution[1]), ")")
        retcode = runCommand(getProcess("distordCrop"), [ "--path", path_data, "--ratio",  "0.3", "--avg_width", str(avg_resolution[0]), "--avg_height", str(avg_resolution[1]) ])
        if retcode.returncode != 0:
            print("Command: distordCrop failed, exiting (ARGS: ", "--path", path_data, "--ratio",  "0.3", "--avg_width", str(avg_resolution[0]), "--avg_height", str(avg_resolution[1]), ")")
            #exit(1)

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
        if len(exclude) > 0:
            print("Excluding ", exclude)
            input_bundle.exclude_cams (exclude)

        # read proposed cropped resolution
        path_to_crop_new_size_txt = os.path.join(path_data, "cropNewSize.txt")
    else:
        train_path_data = str.replace(path_data, "validation_", "")
        path_to_crop_new_size_txt = os.path.join(train_path_data, "cropNewSize.txt")
        print("Reading crop size from ", path_to_crop_new_size_txt )

    with open(path_to_crop_new_size_txt) as crop_size_file:
        line = crop_size_file.readline()
        tokens = line.split()
        new_width   = int(tokens[0])
        new_height  = int(tokens[1])
        proposed_res = [new_width, new_height]

    print("Crop size found:", proposed_res)
    # generate file with list of current selected images to process

    path_to_transform_list_txt = os.path.join (path_data, "toTransform.txt")
    input_bundle.generate_list_of_images_file(path_to_transform_list_txt)

    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    
    path_to_output_bundle = os.path.join (path_dest, "bundle.out")
    # write bundle file in output cameras folder
    new_width = None
    input_bundle.save(path_to_output_bundle, proposed_res)

    # setup avg_resolution and proposed_resolution parameters for distordCrop
    print("Command: run cropFromCenter ARGS:", "--inputFile", path_to_transform_list_txt, "--outputPath", path_dest, "--avgResolution", str(avg_resolution[0]), str(avg_resolution[1]), "--cropResolution", str(proposed_res[0]), str(proposed_res[1]))
    retcode = runCommand(getProcess("cropFromCenter"), [ "--inputFile", path_to_transform_list_txt, "--outputPath", path_dest, "--avgResolution", str(avg_resolution[0]), str(avg_resolution[1]), "--cropResolution", str(proposed_res[0]), str(proposed_res[1]) ])
    if retcode.returncode != 0:
        print("Command: cropFromCenter failed, exiting (ARGS:", "--inputFile", path_to_transform_list_txt, "--outputPath", path_dest, "--avgResolution", str(avg_resolution[0]), str(avg_resolution[1]), "--cropResolution", str(proposed_res[0]), str(proposed_res[1]))
        exit(1)


def fix_video_only(path):
    # TODO: currently only works for video_only + calib_only; doesnt do video only with MVS
    # verify that train is actually empty
    train_dir = os.path.join(path, os.path.join("rcScene", "train_cameras"))
    test_dir = os.path.join(path, os.path.join("rcScene", "test_path_cameras"))
    files = os.listdir(train_dir)
    if len(files) == 1: # empty bundle file
        shutil.move(train_dir, train_dir+"_save")
        print("MOVING {} to {}".format(test_dir, train_dir))
        shutil.move(test_dir, train_dir)
    else:
        print("FATAL ERROR: trying to overwrite existing train images")
        exit(1)

def car_data_process(path):
    # Contains: CAM_{BACK,FRONT}[_]{LEFT, RIGHT}
    rawpath = os.path.join(path, "raw")
    if not os.path.exists(rawpath):
        os.makedirs(rawpath)

    imagespath = os.path.abspath(os.path.join(rawpath, "images"))
    if not os.path.exists(imagespath):
        os.makedirs(imagespath)

    # read all the sets of cameras

    dirlist = [ "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT" ]
    imlists = {}
    global_im_counter = 0
    
    for dirname in dirlist:
        campath = os.path.join(path, dirname)
        first = True
# basic version
        for filename in os.listdir(campath):
            shutil.copyfile(os.path.join(campath, filename), os.path.join(imagespath, "{:06d}".format(global_im_counter)+".jpg"))
            global_im_counter += 1

"""
# code below useless
        for filename in os.listdir(campath):
            ext = os.path.splitext(filename)[1]
            if ext == ".JPG" or ext == ".jpg" or ext == ".PNG" or ext == ".png" :
                if first:
                    imlists[dirname] = [filename]
                    first = False
                else:
                    imlists[dirname].append(filename)

#                print("Adding ", filename , " to list " , dirname)
    for i in range(len(imlists["CAM_BACK"])):
        imname = imlists[ "CAM_BACK_LEFT"][i] 
        shutil.copyfile(os.path.join(path, os.path.join( "CAM_BACK_LEFT", imname)), os.path.join(imagespath, "{:06d}".format(global_im_counter)+".jpg"))
        global_im_counter += 1
        imname = imlists[ "CAM_FRONT_LEFT"][i] 
        shutil.copyfile(os.path.join(path, os.path.join( "CAM_FRONT_LEFT", imname)), os.path.join(imagespath, "{:06d}".format(global_im_counter)+".jpg"))
        global_im_counter += 1
        if i > 2:
            imname = imlists[ "CAM_FRONT"][i-2] 
            shutil.copyfile(os.path.join(path, os.path.join( "CAM_FRONT", imname)), os.path.join(imagespath, "{:06d}".format(global_im_counter)+".jpg"))
            global_im_counter += 1

    for i in range(len(imlists["CAM_BACK"])):
        imname = imlists[ "CAM_FRONT_RIGHT"][i] 
        shutil.copyfile(os.path.join(path, os.path.join( "CAM_FRONT_RIGHT", imname)), os.path.join(imagespath, "{:06d}".format(global_im_counter)+ ".jpg"))
        global_im_counter += 1
        imname = imlists[ "CAM_BACK_RIGHT"][i] 
        shutil.copyfile(os.path.join(path, os.path.join( "CAM_BACK_RIGHT", imname)), os.path.join(imagespath, "{:06d}".format(global_im_counter)+ ".jpg"))
        global_im_counter += 1
        if i < len(imlists["CAM_BACK"])-2:
            imname = imlists[ "CAM_BACK"][i+2] 
            shutil.copyfile(os.path.join(path, os.path.join( "CAM_BACK", imname)), os.path.join(imagespath, "{:06d}".format(global_im_counter)+ ".jpg"))
            global_im_counter += 1
"""
