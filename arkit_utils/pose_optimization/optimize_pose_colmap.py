import argparse
import logging
import numpy as np
import json
import os
import pycolmap
import shutil
from pyquaternion import Quaternion
from hloc.triangulation import create_db_from_model
from pathlib import Path
from hloc.utils.read_write_model import Camera, Image, Point3D, CAMERA_MODEL_NAMES
from hloc.utils.read_write_model import write_model, read_model
from hloc import extract_features, match_features, pairs_from_poses, triangulation

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def arkit_transform2_COLMAP(dataset_base) :
    dataset_dir = Path(dataset_base)
    
    # step1. Transorm ARKit images to COLAMP coordinate
    images = {}
    with open(dataset_base + "/sparse/0/images.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])

                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))

                c2w = np.zeros((4, 4))
                c2w[:3, :3] = qvec2rotmat(qvec)
                c2w[:3, 3] = tvec
                c2w[3, 3] = 1.0
                c2w_cv = convert_pose(c2w)

                # transform to z-up world coordinate for better visulazation
                c2w_cv = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]]) @ c2w_cv
                w2c_cv = np.linalg.inv(c2w_cv)
                R = w2c_cv[:3, :3]
                q = Quaternion(matrix=R, atol=1e-06)
                qvec = np.array([q.w, q.x, q.y, q.z])
                tvec = w2c_cv[:3, -1]
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    
    
    # step2. Write ARKit undistorted intrinsic to COLMAP cameras
    cameras = {}
    with open(dataset_base +  "/sparse/0/cameras.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)


    # Empty 3D points.
    points3D = {}

    print('Writing the COLMAP model...')
    colmap_arkit = dataset_dir / 'colmap_arkit' / 'raw'
    colmap_arkit.mkdir(exist_ok=True, parents=True)
    write_model(images=images, cameras=cameras, points3D=points3D, path=str(colmap_arkit), ext='.txt')



def optimize_pose_by_COLMAP(dataset_base) :
    feat_extracton_cmd = "colmap feature_extractor \
    --database_path " + dataset_base + "/database.db \
    --image_path "  + dataset_base + "/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.use_gpu 1"
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = "colmap exhaustive_matcher \
        --database_path " + dataset_base + "/database.db \
        --SiftMatching.use_gpu 1"
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    os.makedirs(dataset_base + "/colmap_arkit/tri", exist_ok=True)
    triangulate_cmd = "colmap point_triangulator \
    --database_path " + dataset_base + "/database.db \
    --image_path "  + dataset_base + "/images \
    --input_path " + dataset_base + "/colmap_arkit/raw \
    --output_path " + dataset_base + "/colmap_arkit/tri"
    exit_code = os.system(triangulate_cmd)
    if exit_code != 0:
        logging.error(f"Point triangulation failed with code {exit_code}. Exiting.")
        exit(exit_code)

    os.makedirs(dataset_base + "/colmap_arkit/ba", exist_ok=True)
    BA_cmd = "colmap bundle_adjuster \
        --input_path " + dataset_base + "/colmap_arkit/tri \
        --output_path " +  dataset_base + "/colmap_arkit/ba"
    exit_code = os.system(BA_cmd)
    if exit_code != 0:
        logging.error(f"BA failed with code {exit_code}. Exiting.")
        exit(exit_code)

    os.makedirs(dataset_base + "/colmap_arkit/tri2", exist_ok=True)
    triangulate_cmd = "colmap point_triangulator \
    --database_path " + dataset_base + "/database.db \
    --image_path "  + dataset_base + "/images \
    --input_path " + dataset_base + "/colmap_arkit/ba \
    --output_path " + dataset_base + "/colmap_arkit/tri2"
    exit_code = os.system(triangulate_cmd)
    if exit_code != 0:
        logging.error(f"Point triangulation failed with code {exit_code}. Exiting.")
        exit(exit_code)

    os.makedirs(dataset_base + "/colmap_arkit/ba2", exist_ok=True)
    BA_cmd = "colmap bundle_adjuster \
        --input_path " + dataset_base + "/colmap_arkit/tri2 \
        --output_path " +  dataset_base + "/colmap_arkit/ba2"
    exit_code = os.system(BA_cmd)
    if exit_code != 0:
        logging.error(f"BA failed with code {exit_code}. Exiting.")
        exit(exit_code)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ARkit pose using COLMAP")
    parser.add_argument("--input_database_path", type=str, default="data/arkit_pose/study_room/arkit_undis")
    args = parser.parse_args()

    input_database_path = args.input_database_path

    arkit_transform2_COLMAP(input_database_path)
    optimize_pose_by_COLMAP(input_database_path)
