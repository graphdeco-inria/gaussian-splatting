## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone https://github.com/rogerhh/gaussian-splatting-lm --recursive
```

### Setup

#### Local Setup

```shell
mamba env create -f environment.yml
conda activate gaussian_splatting-jvp
(cd submodules/diff_gaussian_rasterization && pip install -e .)
(cd submodules/fused-ssim && pip install -e .)
(cd submodules/simple-knn && pip install -e .)

```

### Running

To run the optimizer, simply use

```shell
python train_sophia_tr_hellinger.py -s <path to COLMAP or NeRF Synthetic dataset> --iterations 30000 --loss_type="l1" --gif_interval=10000000000 --num_images=1 --kl_threshold=0.000001 --eval --eval_interval=1000 --densify_until_iter=25000
```

