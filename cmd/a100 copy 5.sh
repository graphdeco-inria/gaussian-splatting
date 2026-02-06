#!/bin/bash
#
#SBATCH --account=pr_373_general
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=3dgs_wd
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --output=/scratch/%u/%A:%x/%4a.out
#SBATCH --error=/scratch/%u/%A:%x/%4a.err
#SBATCH --mail-type=END

singularity exec --nv --overlay /home/$USER/overlay.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/conda/bin/activate 3dgs &&
        cd /home/$USER/gaussian-splatting &&
        python process_erank_0.3_erank_0.01_thin_train_7k.py
    "
