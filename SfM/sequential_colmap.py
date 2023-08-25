import os
import sys
from tqdm import tqdm
from natsort import natsorted

sys.path.append('/home/xiaoyun/Code/gaussian-splatting')
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers import readColmapCameras

cmdname = 'colmap'
# cmdname = 'COLMAP.bat'

def process(proj_path, img_path):
    sparse_path = os.path.join(proj_path,'sparse')
    os.system(f'{cmdname} feature_extractor --database_path {proj_path}/ddd.db --image_path {img_path} --ImageReader.camera_model PINHOLE --SiftExtraction.num_threads 64')
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
    
img_path = '/data/xiaoyun/dlf_data/00/img_1692936003.1167815/'
proj_path = '/data/xiaoyun/dlf_data/00/proj_1692936003.1167815/'
# process(proj_path,img_path)



