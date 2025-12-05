#!/bin/bash

set -e

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "$0")")

# for dataset in bicycle bonsai counter garden kitchen stump train truck
# do
#     DIR1="saved_output/adahessian-${dataset}-huber"
#     DIR2="saved_output/adam-${dataset}"
#     OUTPUT="figures/adahessian_${dataset}_psnr.png"
#     python3 "${SCRIPT_DIR}/plot_psnr.py" "${DIR1}" "${DIR2}" "${OUTPUT}" "${dataset}"
# done

# for dataset in bicycle bonsai counter garden kitchen stump train truck
# do
#     DIR1="saved_output/adahessian-${dataset}-huber"
#     DIR2="saved_output/adam-${dataset}-huber"
#     OUTPUT="figures/adahessian_${dataset}_psnr_adam_huber.png"
#     python3 "${SCRIPT_DIR}/plot_psnr.py" "${DIR1}" "${DIR2}" "${OUTPUT}" "${dataset}"
# done

for dataset in bicycle bonsai counter garden kitchen stump train truck
do
    DIR1="saved_output/adahessian-${dataset}-huber"
    DIR2="saved_output/adam-${dataset}-huber-reset"
    OUTPUT="figures/adahessian_${dataset}_psnr_adam_huber_reset.png"
    python3 "${SCRIPT_DIR}/plot_psnr.py" "${DIR1}" "${DIR2}" "${OUTPUT}" "${dataset}"
done
