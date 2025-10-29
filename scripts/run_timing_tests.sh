#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")
DATA_DIR="$PROJECT_DIR/../3dgs_datasets/tandt/train"

cd "$PROJECT_DIR"

# echo "" > timing_results.txt
# 
# for i in 1 2 10 20 50 100; do
#     echo "num_images $i" >> timing_results.txt
#     python3 tests/test_jvp_timing.py -s "$DATA_DIR" --num_images $i >> timing_results.txt
# done

python3 scripts/parse_timing_results.py timing_results.txt

