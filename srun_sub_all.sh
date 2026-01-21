#!/bin/bash

mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
tanks_and_temples_scenes=("truck" "train")
deep_blending_scenes=("drjohnson" "playroom")

all_scenes=("${mipnerf360_outdoor_scenes[@]}" "${mipnerf360_indoor_scenes[@]}" "${tanks_and_temples_scenes[@]}" "${deep_blending_scenes[@]}")

for scene in "${all_scenes[@]}"; do
    bash srun_sub.sh $scene mcmc
    bash srun_sub.sh $scene mcmc_no_density
    bash srun_sub.sh $scene sophia_hellinger_no_densify
    bash srun_sub.sh $scene sophia_hellinger_no_density_abs
    bash srun_sub.sh $scene sophia_hellinger_resume_from_15k # after mcmc
    bash srun_sub.sh $scene sophia_hellinger_resume_from_15k_abs # after mcmc
    bash srun_sub.sh $scene sophia_hellinger_no_densify_update10
    bash srun_sub.sh $scene sophia_hellinger_no_densify_abs_update10
    bash srun_sub.sh $scene sophia_hellinger_no_densify_update15
    bash srun_sub.sh $scene sophia_hellinger_no_densify_abs_update15
    bash srun_sub.sh $scene sophia_hellinger_no_densify_normrot
    bash srun_sub.sh $scene sophia_hellinger_no_density_abs_normrot
    bash srun_sub.sh $scene adam_tr_no_densify
done