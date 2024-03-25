#!/usr/bin/env bash
set -e

# Validate the input argument
if [ -z "$1" ]; then
  echo "Usage: $0 <input_base_path> <offline_ba_iterations>"
  exit 1
fi

input_base_path=$1
ba_iterations=$2

echo "input_base_path: ${input_base_path}"

echo "=== Preprocess ARkit data === "
mkdir ${input_base_path}/post
mkdir ${input_base_path}/post/sparse/
mkdir ${input_base_path}/post/sparse/online
echo "1. undistort image uisng AVfoundation calibration data"
python arkit_utils/undistort_images/undistort_image.py --input_base ${input_base_path}
echo "2. Transform ARKit mesh to point3D"
python arkit_utils/mesh_to_points3D/arkitmeshply2point3D.py --input_base_path ${input_base_path}
echo "3. Transform ARKit pose to COLMAP coordinate"
python arkit_utils/arkit_pose_to_colmap.py --input_database_path ${input_base_path}

echo "3. Optimize pose using hloc & COLMAP"
mkdir ${input_base_path}/post/sparse/offline
python arkit_utils/pose_optimization/optimize_pose_hloc.py --input_database_path ${input_base_path}

echo "=== 3D gaussian splatting === "
echo "1. 3DGS on online data"
python train.py -s ${input_base_path}/post -t online -m ${input_base_path}/post/sparse/online/output --iterations 7000
# echo "1. 3DGS on offline data"
# CUDA_VISIBLE_DEVICS=1 python train.py -s ${input_base_path}/post -t offline -m ${input_base_path}/post/sparse/offline/output --iterations 7000
