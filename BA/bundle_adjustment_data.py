import os
import sys
from tqdm import tqdm
from natsort import natsorted
import glob
import shutil
import numpy as np
from PIL import Image
from typing import NamedTuple
import pdb

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers import readColmapCameras
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from natsort import natsorted
import cv2

class CameraInfo2(NamedTuple):
    uid: int
    imageid: int
    R: np.array
    qvec: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def readColmapCameras2(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        #image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_path = os.path.join(images_folder, extr.name)
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)

        cam_info = CameraInfo2(uid=uid, imageid=extr.id, R=R, qvec=extr.qvec, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def load_colmap_data(proj_path, img_path):
    # load colmap data
    try:
        cameras_extrinsic_file = os.path.join(proj_path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(proj_path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(proj_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(proj_path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    cam_infos_unsorted = readColmapCameras2(cam_extrinsics=cam_extrinsics,cam_intrinsics=cam_intrinsics, images_folder=img_path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_path)
    
    
    
    return cam_infos


