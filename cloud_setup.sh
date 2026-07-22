#!/bin/bash
# Script setup va huan luyen tu dong tren Cloud GPU (Google Colab / Kaggle / RunPod)

echo "=== 1. Dang cai dat phu thuoc ==="
pip install plyfile tqdm lpips opencv-python pillow

echo "=== 2. Build CUDA Submodules cho 3D Gaussian Splatting ==="
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

echo "=== 3. Kiem tra GPU ==="
nvidia-smi

echo "=== Setup hoan tat! San sang chay batch_pipeline.py ==="
