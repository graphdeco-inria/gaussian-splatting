import os
import sys
from tqdm import tqdm
from natsort import natsorted
import glob
import shutil
import numpy as np
from PIL import Image
from typing import NamedTuple

sys.path.append('/home/xiaoyun/Code/gaussian-splatting')
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers import readColmapCameras
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

cmdname = 'colmap'
# cmdname = 'COLMAP.bat'

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

def process(proj_path, img_path):
    sparse_path = os.path.join(proj_path,'sparse')
    os.system(f'{cmdname} feature_extractor --database_path {proj_path}/ddd.db --image_path {img_path} --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --ImageReader.camera_params \"788.44, 789.29, 640, 360\" --SiftExtraction.num_threads 64')
    os.system(f'{cmdname} exhaustive_matcher --database_path {proj_path}/ddd.db')
    os.makedirs(sparse_path, exist_ok=True)
    os.system(f'{cmdname} mapper --database_path {proj_path}/ddd.db --image_path {img_path} --output_path {sparse_path}')
    os.system(f'{cmdname} model_converter --input_path {sparse_path}/0 --output_path {sparse_path}/0 --output_type TXT')

def incremental_process(proj_path, img_path):
    sparse_path = os.path.join(proj_path,'sparse')
    os.system(f'{cmdname} feature_extractor --database_path {proj_path}/ddd.db --image_path {img_path} --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --ImageReader.camera_params \"788.44, 789.29, 640, 360\" --SiftExtraction.num_threads 64')
    os.system(f'{cmdname} exhaustive_matcher --database_path {proj_path}/ddd.db')
    os.makedirs(sparse_path, exist_ok=True)
    os.system(f'{cmdname} mapper --database_path {proj_path}/ddd.db --image_path {img_path} --output_path {sparse_path}')
    os.system(f'{cmdname} model_converter --input_path {sparse_path}/0 --output_path {sparse_path}/0 --output_type TXT')

def load_colmap_cameras(proj_path, img_path):
    cameras_extrinsic_file = os.path.join(proj_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(proj_path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics,cam_intrinsics=cam_intrinsics, images_folder=img_path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_path)
    return cam_infos

def prepare_scene(img_path, proj_path, depth = 1):
    proj_img_path = os.path.join(proj_path, 'image')
    os.makedirs(proj_img_path, exist_ok=True)
    dir_depth_string = '*'
    for idx in range(depth - 1):
        dir_depth_string = f'{dir_depth_string}/*'
    pattern = os.path.join(img_path, dir_depth_string)
    print(pattern)
    dst_names = []
    for img_name in glob.glob(pattern):
        # check if the image ends with png
        if (img_name.endswith('.png') or img_name.endswith('.jpg')):
            x = img_name.split('/')
            dst_name = os.path.join(proj_img_path, f'{x[len(x) - 2]}.{x[len(x) - 1]}')
            shutil.copyfile(img_name, dst_name)
            dst_names.append(dst_name)
            print(img_name)
    return dst_names

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
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

def select_part_observes(orig_proj_path, str_prefix, cam_infos, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_proj_path = os.path.join(output_folder, 'sparse/0')
    os.makedirs(output_proj_path, exist_ok=True)
    image_dir = os.path.join(output_folder, 'image')
    os.makedirs(image_dir, exist_ok=True)

    f_cams_selected = []
    for idx in range(len(cam_infos)):
        imagename = cam_infos[idx].image_path
        if isinstance(str_prefix, list):
            for prefix_idx in range(len(str_prefix)):
                if str_prefix[prefix_idx] in imagename:
                    f_cams_selected.append(cam_infos[idx])
                    break
        else:
            if str_prefix[prefix_idx] in imagename:
                f_cams_selected.append(cam_infos[idx])
    
    with open(os.path.join(output_proj_path, 'cameras.txt'),'w') as f:
        f.writelines('# Camera list with one line of data per camera:\n')
        f.writelines('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.writelines(f'# Number of cameras: {len(f_cams_selected)}\n')
        idx_lists = []
        for idx in range(len(f_cams_selected)):
            uid = f_cams_selected[idx].uid
            if uid in idx_lists:
                continue
            else:
                idx_lists.append(uid)
            width = f_cams_selected[idx].width
            height = f_cams_selected[idx].height
            fovX = f_cams_selected[idx].FovX
            fovY = f_cams_selected[idx].FovY
            focalX = fov2focal(fovX, width)
            focalY = fov2focal(fovY, height)
            f.writelines(f'{uid} PINHOLE {width} {height} {focalX} {focalY} {width/2} {height/2}\n')

    with open(os.path.join(output_proj_path, 'images.txt'),'w') as f:
        f.writelines('# Image list with two lines of data per image:\n')
        f.writelines('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.writelines('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.writelines(f'# Number of images: {len(f_cams_selected)}\n')
        for idx in range(len(f_cams_selected)):
            imageid = f_cams_selected[idx].imageid
            qvec = f_cams_selected[idx].qvec
            T = f_cams_selected[idx].T
            cameraid = f_cams_selected[idx].uid
            imagename = f_cams_selected[idx].image_name
            f.writelines(f'{imageid} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} {cameraid} {imagename}\n')
            f.writelines('\n')
            # copy files
            shutil.copyfile(f_cams_selected[idx].image_path, f'{image_dir}/{f_cams_selected[idx].image_name}')

    open(os.path.join(output_proj_path, 'points3D.txt'),'a').close()
    output_dbpath = f'{output_folder}/ddd.db'
    shutil.copyfile(f'{orig_proj_path}/ddd.db', output_dbpath)

    # database_path = os.path.join(root_path,'ddd.db')
    # images_path = os.path.join(root_path,'images')
    os.system(f'colmap feature_extractor --database_path {output_dbpath} --image_path {image_dir}')
    os.system(f'colmap point_triangulator --database_path {output_dbpath} --image_path {image_dir} --input_path {output_proj_path} --output_path {output_proj_path}')
    os.system(f'{cmdname} model_converter --input_path {output_proj_path} --output_path {output_proj_path} --output_type TXT')
    return 0
    
# img_path = '/data/xiaoyun/dlf_data/00/img_1692936003.1167815/'
# proj_path = '/data/xiaoyun/dlf_data/00/proj_1692936003.1167815/'

# img_path = '/data/xiaoyun/dlf_data/00/img_1692935998.4679725/'
# proj_path = '/data/xiaoyun/dlf_data/00/proj_1692935998.4679725/'

img_path = '/data/xiaoyun/dlf_data/00'
proj_path = '/data/xiaoyun/dlf_result/proj_00'

img_path = '/data/xiaoyun/dlf_data_0829/colmap_00_03/images_all'
proj_path = '/data/xiaoyun/dlf_result/proj_0829_all'

dst_names = prepare_scene(img_path, proj_path, depth=2)
process(proj_path, os.path.join(proj_path, 'image'))
# cam_infos = load_colmap_cameras(proj_path, os.path.join(proj_path, 'image'))
# select_part_observes(proj_path, 'img_1692936003.1167815', cam_infos, '/data/xiaoyun/dlf_result/proj_00_img_1692936003.1167815')



