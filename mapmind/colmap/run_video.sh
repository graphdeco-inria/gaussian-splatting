#!/usr/bin/env bash
set -e


if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

SESSION=$2
MAP_FOLDER=$1
USE_QRCODE_PRIOR=${3:-0}

# check if the model exist
SPARSE_MODEL_PATH=${MAP_FOLDER}/${SESSION}/sparse/0/
if [ -d "$SPARSE_MODEL_PATH" ]; then
    echo "SPARSE MODEL EXIST"
    exit 0
fi

# check if ordinary video exist, which require OPENCV camera model
TEST_PATH=${MAP_FOLDER}/${SESSION}/images/use_opencv_model/
TEST_PATH_GPS=${MAP_FOLDER}/${SESSION}/images/image_with_gps/
CAMERA_MODEL=PINHOLE
if [ -d "$TEST_PATH" ]; then
    echo "USE OPENCV MODEL"
    CAMERA_MODEL=OPENCV
fi

echo "====================== PROCESS FEATURE EXTRACTION " ${CAMERA_MODEL}" ======================"

TEST_PATH_VIDEO=${MAP_FOLDER}/${SESSION}/images/no_ordinary_video/
SHARE_CAMERA="--ImageReader.single_camera_per_folder 1"
if [ -d "$TEST_PATH_VIDEO" ]; then
    echo "USE SINGLE CAMERA"
    SHARE_CAMERA="--ImageReader.single_camera 1"
fi

colmap feature_extractor ${SHARE_CAMERA} \
--database_path ${MAP_FOLDER}/${SESSION}/database.db \
--image_path ${MAP_FOLDER}/${SESSION}/images \
--ImageReader.camera_model=${CAMERA_MODEL} \
--FeatureExtraction.type SIFT \
--AlikedExtraction.max_num_features 2048

echo "====================== PROCESS VIDEO MATCHER ======================"

MODELS_FOLDER=/EasyGaussianSplatting/data/models
if [ ! -d "$MODELS_FOLDER" ]; then
    mkdir -p ${MODELS_FOLDER}
fi

NUM_IMAGES=$(find ${MAP_FOLDER}/${SESSION}/images -type f | wc -l)
if [ -d "$TEST_PATH_GPS" ]; then
    echo "====================== spatial_matcher ======================"

    colmap spatial_matcher \
    --database_path ${MAP_FOLDER}/${SESSION}/database.db \
    --TwoViewGeometry.min_inlier_ratio 0.2 \
    --SpatialMatching.max_num_neighbors 50 \
    --SpatialMatching.max_distance 40 \
    --SpatialMatching.ignore_z 1 \
    --FeatureMatching.type SIFT_BRUTEFORCE
fi
if ((${NUM_IMAGES} < 500)); then
    echo "====================== exhaustive_matcher ======================"
    colmap exhaustive_matcher \
    --database_path ${MAP_FOLDER}/${SESSION}/database.db \
    --FeatureMatching.type SIFT_BRUTEFORCE
else
    COLMAP_VOC_PATH_32=vocab_tree_flickr100K_words32K_faiss.bin
    COLMAP_VOC_PATH_256=vocab_tree_flickr100K_words256K_faiss.bin
    VOC_NAME=${COLMAP_VOC_PATH_32}
    if ((${NUM_IMAGES} > 1500)); then
        VOC_NAME=${COLMAP_VOC_PATH_256}
    fi
    echo ${VOC_NAME}

    # check voc exist and download
    if [ -e "${MODELS_FOLDER}/${VOC_NAME}" ]; then
        echo "voc exists"
    else
        echo "voc not exists, process download"
        curl -L -o ${MODELS_FOLDER}/${VOC_NAME} \
          https://github.com/MapMindAI/EasyGaussianSplatting/releases/download/v0/${VOC_NAME}
    fi

    echo "====================== sequential_matcher ======================"
    colmap sequential_matcher \
    --database_path ${MAP_FOLDER}/${SESSION}/database.db \
    --TwoViewGeometry.min_inlier_ratio 0.2 \
    --SequentialMatching.loop_detection 0 \
    --SequentialMatching.loop_detection_num_images 50 \
    --SequentialMatching.loop_detection_num_nearest_neighbors 20 \
    --FeatureMatching.type SIFT_BRUTEFORCE
fi

echo "====================== PROCESS GLOMAP MAPPER ======================"

# Optional but often needed: calibrate intrinsics from the view graph.
# This modifies the database in-place, so work on a copy.
cp ${MAP_FOLDER}/${SESSION}/database.db ${MAP_FOLDER}/${SESSION}/database_global.db
colmap view_graph_calibrator \
  --database_path ${MAP_FOLDER}/${SESSION}/database_global.db

mkdir -p ${MAP_FOLDER}/${SESSION}/sparse_raw
mkdir -p ${MAP_FOLDER}/${SESSION}/sparse/0

colmap global_mapper \
  --database_path ${MAP_FOLDER}/${SESSION}/database_global.db \
  --image_path ${MAP_FOLDER}/${SESSION}/images \
  --output_path ${MAP_FOLDER}/${SESSION}/sparse_raw \

# TODO: fix its problem - rotate the model to fit UE axis
# echo "====================== ROTATE COLMAP TO UE ======================"
# python mapmind/colmap/rotate_colmap_to_UE.py \
# --colmap_model_path ${MAP_FOLDER}/${SESSION} \
# --input_sparse_path sparse_raw/0/ \
# --output_sparse_path sparse_raw/0/

if [ -d "$TEST_PATH_GPS" ]; then
  echo "====================== USE GPS TRANSFORM ======================"
  python mapmind/colmap/transform_colmap_model.py \
  --database_path ${MAP_FOLDER}/${SESSION}/database.db \
  --model_path ${MAP_FOLDER}/${SESSION}/sparse_raw/0 \
  --output_model_path ${MAP_FOLDER}/${SESSION}/sparse/0
else
  echo "====================== USE DEFAULT MODEL CONVERTER ======================"
  colmap model_converter \
  --input_path ${MAP_FOLDER}/${SESSION}/sparse_raw/0 \
  --output_path ${MAP_FOLDER}/${SESSION}/sparse/0 \
  --output_type TXT
fi

if [ -d "$TEST_PATH" ]; then
  echo "====================== PROCESS IMAGE UNDISTORTER ======================"
  colmap image_undistorter \
  --image_path ${MAP_FOLDER}/${SESSION}/images \
  --input_path ${MAP_FOLDER}/${SESSION}/sparse/0 \
  --output_path ${MAP_FOLDER}/${SESSION}/dense \
  --output_type COLMAP \
  --max_image_size 4000

  mkdir -p ${MAP_FOLDER}/${SESSION}/dense/sparse/0
  colmap model_converter \
  --input_path ${MAP_FOLDER}/${SESSION}/dense/sparse \
  --output_path ${MAP_FOLDER}/${SESSION}/dense/sparse/0 \
  --output_type TXT
fi
