#!/bin/bash

mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
tanks_and_temples_scenes=("truck" "train")
deep_blending_scenes=("drjohnson" "playroom")

all_scenes=("${mipnerf360_outdoor_scenes[@]}" "${mipnerf360_indoor_scenes[@]}" "${tanks_and_temples_scenes[@]}" "${deep_blending_scenes[@]}")

for scene in "${all_scenes[@]}"; do
    # mcmc no densify
    bash srun_sub.sh train_mcmc.py $scene --densify_from_iter=50000
    # ours no densify [diagonal_accum_abs=True]
    bash srun_sub.sh train_mcmc_sophia_hellinger.py $scene --densify_from_iter=50000 --noise_lr=0.0 --diagonal_accum_abs=True
    # ours no densify [diagonal_accum_abs=False]
    bash srun_sub.sh train_mcmc_sophia_hellinger.py $scene --densify_from_iter=50000 --noise_lr=0.0 --diagonal_accum_abs=False
    # mcmc
    bash srun_sub.sh train_mcmc.py $scene
    # ours from 15k iter [diagonal_accum_abs=True]
    # bash srun_sub.sh train_mcmc_sophia_hellinger.py $scene --densify_from_iter=50000 --noise_lr=0.0 --diagonal_accum_abs=True --start_checkpoint <START_CHECKPOINT>
    # ours from 15k iter [diagonal_accum_abs=False]
    # bash srun_sub.sh train_mcmc_sophia_hellinger.py $scene --densify_from_iter=50000 --noise_lr=0.0 --diagonal_accum_abs=False --start_checkpoint <START_CHECKPOINT>
done