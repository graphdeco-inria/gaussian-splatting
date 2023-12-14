#!/bin/bash

# exit if something fails
set -e

if [ "$#" -ne 3 ]
then
        echo "Usage: $0 input_dir output_dir iterations"
        echo "e.g. $0 data/colmap_train output/ 7000"
        exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
ITERATIONS=$3

echo "Input path: $INPUT_PATH, output folder: $OUTPUT_PATH, iterations: $ITERATIONS"

# activate conda virtual environment
source /opt/conda/etc/profile.d/conda.sh
conda init bash
conda activate gaussian_splatting

# Run scene optimizer
python convert.py \
    --source_path "$INPUT_PATH"

# Run training
python train.py \
    --source_path "$INPUT_PATH" \
    --model_path "$OUTPUT_PATH" \
    --iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS"
