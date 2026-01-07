## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone https://github.com/rogerhh/gaussian-splatting-lm --recursive
```

### Setup

#### Local Setup

```shell
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting
```

### Running

To run the optimizer, simply use

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

