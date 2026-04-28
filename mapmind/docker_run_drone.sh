#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "PLEASE SET <MAP_FOLDER> <SESSION NAME>"
    exit 1
fi

case "$(uname -s)" in
    MINGW*|CYGWIN*|MSYS*)
        MODEL_DIR=$(pwd -W)
        ;;
    *)
        MODEL_DIR=$(pwd)
        ;;
esac
SESSION=$2
MAP_FOLDER=$1


docker run --rm --name 'EasyGaussianSplatting' \
--gpus 'all,"capabilities=compute,utility,graphics,video"' \
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
-e NVIDIA_VISIBLE_DEVICES=all \
-p 8001:8001 \
-v ${MODEL_DIR}:/EasyGaussianSplatting \
ghcr.io/mapmindai/gaussiansplatting:latest \
bash -c "
cd /EasyGaussianSplatting
conda run --no-capture-output -n gaussian_splatting ./mapmind/run_drone.sh ${MAP_FOLDER} ${SESSION}
chmod -R 777 ${MAP_FOLDER}/${SESSION}
"
