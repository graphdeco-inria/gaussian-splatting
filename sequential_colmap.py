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

cmdname = 'colmap'
# cmdname = 'COLMAP.bat'

def detect_red(image_path):
    # Load the image
    image = cv2.imread(image_path)

    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_red = np.array([0, 130, 56])
    upper_red = np.array([10, 255, 255])

    # Create a mask for red regions
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sum = 0

    contours_filter = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            contour_sum+=area 
            contours_filter.append(contour)
        #print(f"Contour Area: {area}")

    # Draw contours on the original image
    result_image = image.copy()
    cv2.drawContours(result_image, contours_filter, -1, (0, 255, 0), 2)

    return result_image, contour_sum

def get_area_all(d):
    left = [10,11,9,17,16,14,15,8,13]
    right = [1,5,4,6,12,0,3,2,7]
    left = [f'cam_{i}.png' for i in left]
    right = [f'cam_{i}.png' for i in right]
    dd_train = d + "/train/ours_10000/renders"
    dd_test = d + "/test/ours_10000/renders"
    
    
    ll_all = [os.path.join(dd_train,i)for i in os.listdir(dd_train)] + [os.path.join(dd_test,i)for i in os.listdir(dd_test)]
    
    all_area_list = []
    area_all = 0
    for ii in tqdm(ll_all):
        path = ii
        _,area = detect_red(path)
        # print(area)
        area_all += area
    return (area_all / len(ll_all))

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

def prepare_scene(img_path, proj_path, depth = 1):
    proj_img_path = os.path.join(proj_path, 'images')
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

def select_part_observes(orig_proj_path, str_prefix, cam_infos, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_proj_path = os.path.join(output_folder, 'sparse/0')
    os.makedirs(output_proj_path, exist_ok=True)
    image_dir = os.path.join(output_folder, 'images')
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
            if str_prefix in imagename:
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
    return len(f_cams_selected)
    

def process_full_pipeline(start_group, end_group, proj_path, cam_infos, times, colmap_output, model_output, train = True, render = False):
    left_ind = [10,11,9,17,16,14,15,8,13]
    right_ind = [1,5,4,6,12,0,3,2,7]
    for i in range(start_group,end_group):
        for j in range(start_group,end_group):
            left_group = times[i]
            right_group = times[j]
            selected_indices = []
            selected_indices.append(times[0])
            for ii in left_ind:
                selected_indices.append(left_group + f'.cam_{ii}.png')
            
            for ii in right_ind:
                selected_indices.append(right_group + f'.cam_{ii}.png')
            colmap_dir = os.path.join(colmap_output,f'{i}_{j}')
            select_part_observes(proj_path, selected_indices, cam_infos, colmap_dir)
            os.system(f'cp {colmap_dir}/sparse/0/points* {proj_path}/sparse/0')
            
            output = os.path.join(model_output,f'colmap_{i}_{j}')
            if train:
                os.system(f'python /home/jianing/gaussian-splatting/train.py -s {proj_path} -m {output} --iterations 10000 --eval --ls 0{i} --rs 0{j}')
            if render:
                os.system(f'python /home/jianing/gaussian-splatting/render_depth.py -m {output} --ls 0{i} --rs 0{j} --eval')
            #os.system(f'python area_test.py {output}')
            #area = get_area_all(output)
            # with open(f'/data/jianing/output_829/res_{start_group}_{end_group}.txt', 'a') as f:
            #     f.writelines(f'{i} {j} {area}\n')

def process_full_pipeline_single(proj_path, cam_infos, times, colmap_output, model_output, index = 0, train = True, render = False):
    
    selected_indices = []
    selected_indices.append(times[index])
    
    colmap_dir = os.path.join(colmap_output,f'{index}_{index}')
    cam_num = select_part_observes(proj_path, selected_indices, cam_infos, colmap_dir)
    os.system(f'cp {colmap_dir}/sparse/0/points* {proj_path}/sparse/0')
    print(cam_num)
    output = os.path.join(model_output,f'colmap_{index}_{index}')
    if train:
        os.system(f'python /home/jianing/gaussian-splatting/train.py -s {proj_path} -m {output} --iterations 10000 --eval --ls {index} --rs {index} --cam_num {cam_num}')
        #os.system(f'python /home/jianing/gaussian-splatting/train.py -s {proj_path} -m {output} --iterations 10000')
    if render:
        os.system(f'python /home/jianing/gaussian-splatting/render_depth.py -m {output} --ls 0 --rs 0 --eval --cam_num {cam_num}')

if __name__ == '__main__':
    proj_path = '/data/jianing/dlf_result/proj_0913_11_all'
    img_path = '/data/jianing/data/913/11'
    # dst_name = prepare_scene(img_path, proj_path,depth=2)
    # process(proj_path, os.path.join(proj_path,'images'))
    
    times = []
    for i in os.listdir(os.path.join(proj_path, 'images')):
        #i = i.split('.')[0] + '.' + i.split('.')[1]
        i = i.split('.')[0]
        if i not in times :
            times.append(i)
    times = natsorted(times)
    #print(times)

    ## load COLMAP results
    cam_infos = load_colmap_cameras(proj_path, os.path.join(proj_path, 'images'))
    ## select part results
    colmap_output = '/data/jianing/dlf_result/colmap_913_11'
    model_output = '/data/jianing/output_913_11'
    for i in range(len(times)):
        process_full_pipeline_single(proj_path, cam_infos, times, colmap_output, model_output, i, True, False)
    # process_full_pipeline(1, 6, proj_path, cam_infos, times, colmap_output,model_output, True, True)
    # process_full_pipeline(7, 12, proj_path, cam_infos, times, colmap_output,model_output)
    # process_full_pipeline(12, 18, proj_path, cam_infos, times, colmap_output,model_output)