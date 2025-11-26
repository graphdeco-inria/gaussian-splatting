# Setup with UV and CUDA 12

This project has been configured to use UV package manager with CUDA 12.x support.

## Quick Start

1. **Sync dependencies**:
   ```bash
   uv sync
   ```

2. **Install CUDA submodules**:
   ```bash
   export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
   UV_PROJECT_ENVIRONMENT=.venv uv pip install --no-build-isolation --python .venv/bin/python \
     ./submodules/diff-gaussian-rasterization \
     ./submodules/simple-knn \
     ./submodules/fused-ssim
   ```

3. **Activate the environment** (for running scripts):
   ```bash
   source .venv/bin/activate
   ```

   Or use `uv run`:
   ```bash
   uv run python train.py -s <path to COLMAP or NeRF Synthetic dataset>
   ```

## Notes

- Python 3.11 is used
- PyTorch 2.5+ with CUDA 12.1 support
- The `TORCH_CUDA_ARCH_LIST` environment variable specifies which GPU architectures to compile for
- Adjust the architecture list based on your GPU (see [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html))

## Common GPU Architectures
- RTX 40xx series: 8.9
- RTX 30xx series: 8.6
- RTX 20xx series: 7.5
- GTX 10xx series: 6.1
