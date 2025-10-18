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
# in python, elements declared outside __init__ are static (belong to the class)
# while ones declared inside __init__ belong to the object

import os
import get_image_size   # way faster than loading images in opencv and lightweighted (not as the case with python's pillow)
from enum import IntEnum

class InputImage:

    def __init__(self, cam_id, path_to_image):
        self.id         = cam_id
        self.path       = path_to_image         # absolute path
        self.filename   = os.path.basename(path_to_image)
        width, height   = get_image_size.get_image_size(path_to_image)
        self.resolution = [width, height]

    def __str__(self):
        return "{0}\t{1}\t{2}".format(self.path, self.resolution[0], self.resolution[1])
        # return str(self.filename) + delimiter + str(self.resolution[0]) + delimiter + str(self.resolution[1])
        #return "{0}\t{1}\t{2}".format(self.filename, self.resolution[0], self.resolution[1])


class BundleFeaturePointLine (IntEnum):
    POSITION    = 0
    COLOR       = 1
    VIEW_LIST   = 2


class BundleCamera:

    def __init__(self, cam_id, focal_length, radial_dist, rotation, translation):
        self.id                     = cam_id
        self.focal_length           = focal_length
        self.radial_dist            = radial_dist
        self.rotation               = rotation
        self.translation            = translation

        # inverse index of feature points
        self.list_of_feature_points  = []

    def add_feature_point (feature_point):
        pass

    def set_feature_points (list_of_feature_points):
        pass

    def scale_focal_length (self, factor):
        self.focal_length = self.focal_length * factor

    def __str__(self):
        first_line      = "{0:g} {1:g} {2:g}\n".format(self.focal_length, self.radial_dist[0], self.radial_dist[1])
        second_line     = "{0:g} {1:g} {2:g}\n".format(self.rotation[0][0], self.rotation[0][1], self.rotation[0][2])
        third_line      = "{0:g} {1:g} {2:g}\n".format(self.rotation[1][0], self.rotation[1][1], self.rotation[1][2])
        fourth_line     = "{0:g} {1:g} {2:g}\n".format(self.rotation[2][0], self.rotation[2][1], self.rotation[2][2])
        fifth_line      = "{0:g} {1:g} {2:g}".format(self.translation[0], self.translation[1], self.translation[2])
        return first_line + second_line + third_line + fourth_line + fifth_line


class BundleFeaturePoint:

    def __init__(self, feature_point_id, position, color, view_list):
        self.id         = feature_point_id
        self.position   = position
        self.color      = color # each channel between [0, 255]

        # list of cameras
        # each camera has the following info:
        # (<camId> <sift keypoint> <x> <y>) # (x,y) floating point value with (0,0) center of the img
        self.view_list  = view_list
        # for colmap conversion
        self.point2d_index = {}

    def remove_cam(self, cam_id):
        for index in range (len(self.view_list)):
            if (self.view_list[index][0] == cam_id):
                del self.view_list[index]
                break
        # fix all subsequent indices

        newlist = []
        nr1 = len(self.view_list)
        change = False
        for vl_item in self.view_list:
            newitem = list(vl_item)
            if (vl_item[0] > cam_id):
#                change = True
                newitem[0] = newitem[0]-1
                newlist.append(tuple(newitem))
            else:
                newlist.append(vl_item)

        if change:
            print("NEW : {}\n".format( newlist ))
            print("OLD : {}\n".format( self.view_list ))

        self.view_list = newlist

    def __str__(self):
        first_line      = "{0:g} {1:g} {2:g}\n".format(self.position[0], self.position[1], self.position[2])
        second_line     = "{0} {1} {2}\n".format(self.color[0], self.color[1], self.color[2])
        third_line      = str(len(self.view_list)) + " "
        cam_index = 0
        for view_info in self.view_list:
            third_line = third_line + "{0:g} {1:g} {2:g} {3:g}".format(view_info[0], view_info[1], view_info[2], view_info[3])
            if (cam_index != len(self.view_list) - 1):
                third_line = third_line + " "
            cam_index = cam_index + 1
        return first_line + second_line + third_line

class Bundle:

    MAX_NR_FEATURE_POINTS = 8000000   # if bundle file has more than this limit, features will not be processed at all

    def __init__(self, path_to_bundle):
        # read bundle file
        input_file = open(path_to_bundle, "r")

        # first line is the header containing bundle version
        self.header             = input_file.readline().strip()
        self.nr_cameras, self.nr_feature_points = map(int, input_file.readline().strip().split(" "))

        self.using_feature_points = (self.nr_feature_points < Bundle.MAX_NR_FEATURE_POINTS)
        if (not self.using_feature_points):
            self.nr_feature_points = 0
            print ("[bundle.py] Warning: Too many feature points. They are going to be discarded")

        self.list_of_cameras = []
        self.list_of_feature_points = []

        for i in range(self.nr_cameras):
            # read each camera
            focal_length, radial_dist_x, radial_dist_y = map(float, input_file.readline().strip().split(" "))
            r11, r12, r13 = map(float, input_file.readline().strip().split(" "))
            r21, r22, r23 = map(float, input_file.readline().strip().split(" "))
            r31, r32, r33 = map(float, input_file.readline().strip().split(" "))
            tx, ty, tz = map(float, input_file.readline().strip().split(" "))

            camera = BundleCamera(i, focal_length, (radial_dist_x, radial_dist_y), [ [r11, r12, r13], [r21, r22, r23], [r31, r32, r33] ], [tx, ty, tz])

            self.list_of_cameras.append(camera)

        if (self.using_feature_points):
            # keep reading input file
            # read feature points (sometimes there aren't as many as reported in the header)
            # display a warning when this happens
            type_of_line_to_read = BundleFeaturePointLine.POSITION

            feature_point_position  = None
            feature_point_color     = None
            feature_point_view_list = None
            feature_point_id        = 0

            for line in input_file:
                if (type_of_line_to_read == BundleFeaturePointLine.POSITION):
                    x, y, z = map(float, line.strip().split(" "))
                    feature_point_position = [x, y, z]
                elif (type_of_line_to_read == BundleFeaturePointLine.COLOR):
                    r, g, b = map(int, line.strip().split(" "))
                    feature_point_color = [r, g, b]
                elif (type_of_line_to_read == BundleFeaturePointLine.VIEW_LIST):
                    tokens = line.split()
                    nr_cams_that_see_point = int(tokens[0])
                    list_of_view_info = []
                    for i in range(nr_cams_that_see_point):
                        cam_id  = int   (tokens[1 + i*4+0])
                        sift    = int   (tokens[1 + i*4+1])
                        x_pos   = float (tokens[1 + i*4+2])
                        y_pos   = float (tokens[1 + i*4+3])
                        list_of_view_info.append( (cam_id, sift, x_pos, y_pos) )

                    # add feature point to the list of feature points contained in the bundle
                    feature_point = BundleFeaturePoint(feature_point_id, feature_point_position, feature_point_color, list_of_view_info)
                    
                    # for colmap conversion
                    for v in list_of_view_info:
                        if v[0] >= len(self.list_of_cameras):
                            print("ERROR ", v[0], "  ", len(self.list_of_cameras))
                        else:
                            self.list_of_cameras[v[0]].list_of_feature_points.append(feature_point)

                    feature_point_id = feature_point_id + 1

                    self.list_of_feature_points.append(feature_point)

                type_of_line_to_read = (type_of_line_to_read + 1) % len (BundleFeaturePointLine)

        # done processing input file
        input_file.close()

        # paths
        self.path_to_bundle_file = path_to_bundle
        self.root_directory = os.path.dirname(path_to_bundle)

        # get absolute path to input images
        image_id = 0
        self.list_of_input_images = []
        for file_in_dir in os.listdir( self.root_directory ):
            # input imgs have a [jpg|jpeg|png] extension
            if (file_in_dir.lower().endswith(".jpg" ) or file_in_dir.lower().endswith(".png" ) or file_in_dir.lower().endswith(".jpeg")):
                # input images must also have a numerical filename (avoid reading things as texture images that are stored in the same folder)
                if (os.path.splitext(file_in_dir)[0].isdigit()):
                    absolute_path = os.path.join(self.root_directory, file_in_dir)
                    image = InputImage(image_id, absolute_path)
                    self.list_of_input_images.append(image)
                    image_id = image_id + 1

        # additional data
        self.list_of_excluded_cams = []
        self.has_right_nr_feature_pts = False
        self.has_right_nr_images = (len(self.list_of_cameras) == len(self.list_of_input_images))

        if (not self.has_right_nr_images):
            print ("[bundle.py] Warning: nr cameras in bundle file (" + str(len(self.list_of_cameras)) + ") is not the same as nr of images in " + self.root_directory + " (" + str(len(self.list_of_input_images)) + ")")

        print ("[bundle.py] Message: Done reading bundle file", path_to_bundle)
        print ("[bundle.py] Message: Nr cams in bundle file", len(self.list_of_cameras))
        print ("[bundle.py] Message: Nr images in root folder", len(self.list_of_input_images))
        print ("[bundle.py] Message: Nr feature points", len(self.list_of_feature_points))

    def get_avg_resolution (self):
        result = [0, 0]
        for image in self.list_of_input_images:
            result[0] = result[0] + image.resolution[0]
            result[1] = result[1] + image.resolution[1]
        if (len(self.list_of_input_images) != 0):
            result[0] = (int)(result[0] / len(self.list_of_input_images))
            result[1] = (int)(result[1] / len(self.list_of_input_images))
        return result

    def generate_list_of_images_file (self, path_to_output):
        output_file = open (path_to_output, "w")
        for image in self.list_of_input_images:
            output_file.write(str(image) + '\n')
        output_file.close()


    def scale (self, factor):
        for cam in self.list_of_cameras:
            cam.scale_focal_length(factor)

    def exclude_cams (self, cam_list, verbose = True):
        if (verbose):
            print ("[bundle.py] Message: excluding images", cam_list)
        # calling this method twice doesn't make sense because
        # we need to make sure the index passed refer to the right cameras to remove

        # sort list of cams to exclude by decreasing order
        cam_list.sort(reverse=True)
        for index in cam_list:
            # don't forget to go through feature points and remove ref to cam
            for feature_point in self.list_of_feature_points:
                feature_point.remove_cam(index)

            # log the cam id that was removed by adding it to the internal list of excluded cams
            self.list_of_excluded_cams.append(index)

            del self.list_of_cameras[index]
            del self.list_of_input_images[index]

        # update nr_cameras attribute
        self.nr_cameras = len (self.list_of_cameras)

    def save (self, path_to_output_file, new_res=[]):
        output_file = open(path_to_output_file, "w")

        output_file.write(self.header + '\n')
        output_file.write(str(self.nr_cameras) + " " + str(self.nr_feature_points) + '\n')

        if new_res == []:
            for cam in self.list_of_cameras:
                output_file.write(str(cam) + '\n')
        else:
            # not needed TODO: verify
            #indx = 0
            for cam in self.list_of_cameras:
                #im = self.list_of_input_images[indx]
                #old_w = im.resolution[0]
                #old_h = im.resolution[1]
                #new_focal = cam.focal_length*(min(old_h/new_res[1], old_w/new_res[0]))
                #print("Old : ", cam.focal_length, " New : " , new_focal)
                #cam.focal_length = new_focal
                output_file.write(str(cam) + '\n')
                #indx = indx + 1
            
        for feature_point in self.list_of_feature_points:
#            print("Writing ", len(feature_point.view_list) , " FEATURE POINTS " )
            if len(feature_point.view_list)> 0:
                output_file.write(str(feature_point) + '\n')

