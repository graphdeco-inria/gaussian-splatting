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
# extra env needed
apt update && apt install libtiff5
```

### Running

To run the optimizer, simply use

```shell
python train_mcmc_sophia_hellinger.py -s <path to COLMAP or NeRF Synthetic dataset> --iterations 30000 --loss_type="l1" --noise_lr=0.0 --eval --eval_interval=1000 --cap_max <CAP_MAX>
```

cap_max is the max number of Gaussians, set to 1100000 for the train dataset

One more difference: In the 3DGS-MCMC train script, the densify_until_iter parameter is set to 25000, but it is default to 15000 in our script.
