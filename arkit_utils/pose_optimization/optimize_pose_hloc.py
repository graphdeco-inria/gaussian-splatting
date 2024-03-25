import argparse
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


def prepare_pose_and_intrinsic_prior(dataset_base) :
    dataset_dir = Path(dataset_base)
    
    # step1. Write ARKit pose (in COLMAP ccordinate) to COLMAP images
    images = {}
    with open(dataset_base + "/post/sparse/online/images.txt", "r") as fid:
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

                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    
    # step2. Write ARKit undistorted intrinsic to COLMAP cameras
    cameras = {}
    with open(dataset_base +  "/post/sparse/online/cameras.txt", "r") as fid:
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
    colmap_arkit_base = dataset_dir / 'post' / 'sparse' /'offline'
    colmap_arkit =  colmap_arkit_base / 'raw'
    colmap_arkit.mkdir(exist_ok=True, parents=True)
    write_model(images=images, cameras=cameras, points3D=points3D, path=str(colmap_arkit), ext='.bin')

    return colmap_arkit



def optimize_pose_by_hloc_and_COLMAP(dataset_base, n_ba_iterations, n_matched = 10) :
    # step1. Extract feature using hloc
    dataset_dir = Path(dataset_base)
    colmap_arkit_base = dataset_dir / 'post' / 'sparse' /'offline'
    colmap_arkit =  colmap_arkit_base / 'raw'
    outputs = colmap_arkit_base / 'hloc'
    outputs.mkdir(exist_ok=True, parents=True)

    images = dataset_dir / 'post' / 'images'
    sfm_pairs = outputs / 'pairs-sfm.txt'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    references = [str(p.relative_to(images)) for p in images.iterdir()]
    feature_conf = extract_features.confs['superpoint_inloc']
    matcher_conf = match_features.confs['superglue']

    extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    pairs_from_poses.main(colmap_arkit, sfm_pairs, n_matched)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # step2. optimize pose
    colmap_input = colmap_arkit
    for i in range(n_ba_iterations):
        colmap_sparse = outputs / 'colmap_sparse'
        colmap_sparse.mkdir(exist_ok=True, parents=True)
        reconstruction = triangulation.main(
            colmap_sparse,  # output model
            colmap_input,   # input model
            images,
            sfm_pairs,
            features,
            matches)
        
        colmap_ba = outputs / 'colmap_ba'
        colmap_ba.mkdir(exist_ok=True, parents=True)
        # BA with fix intinsics
        BA_cmd = f'colmap bundle_adjuster \
            --BundleAdjustment.refine_focal_length 0 \
            --BundleAdjustment.refine_principal_point 0 \
            --BundleAdjustment.refine_extra_params 0 \
            --input_path {colmap_sparse} \
            --output_path {colmap_ba}'
        os.system(BA_cmd)
        
        colmap_input = colmap_ba

    # step3. get ba result to outside folder
    cameras, images, point3D = read_model(colmap_ba, ext=".bin")
    write_model(cameras, images, point3D, colmap_arkit_base, ext=".txt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ARkit pose using hloc and COLMAP")
    parser.add_argument("--input_database_path", type=str, default="data/arkit_pose/study_room/arkit_undis")
    parser.add_argument("--BA_iterations", type=int, default=5)

    args = parser.parse_args()

    input_database_path = args.input_database_path
    ba_iterations = args.BA_iterations
    prepare_pose_and_intrinsic_prior(input_database_path)
    optimize_pose_by_hloc_and_COLMAP(input_database_path, ba_iterations)
