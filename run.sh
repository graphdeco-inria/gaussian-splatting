#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda init bash
conda activate gaussian_splatting

# Run Python script
python train.py --s data/colmap_train
