TRAIN_SCRIPT=$1
SCENE=$2
SPARSIFY_RATIO=$3

if [ -z "$TRAIN_SCRIPT" ]; then
    TRAIN_SCRIPT="train_mcmc_sophia_hellinger.py"
fi

if [ -z "$SPARSIFY_RATIO" ]; then
    SPARSIFY_RATIO=0.01
fi

output_path="./eval/"

mipnerf360_datadir="../data/360_v2/"
tanks_and_temples_datadir="../data/tandt/"
deep_blending_datadir="../data/db/"

mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
tanks_and_temples_scenes=("truck" "train")
deep_blending_scenes=("drjohnson" "playroom")

# # baseline
# train_script="train_mcmc.py"
# common_args=" --iterations 30000 --loss_type=l1 --eval --eval_interval=1000 --quiet --cap_max 1000000 "
# output_dir=$output_path/$SCENE/mcmc

# # baseline
# train_script="train_mcmc.py"
# common_args=" --iterations 30000 --densify_from_iter 50000 --loss_type=l1 --eval --eval_interval=1000 --quiet --cap_max 1000000 "
# output_dir=$output_path/$SCENE/mcmc_nodensify

# ours
# train_script="train_mcmc_sophia_hellinger.py"
# common_args=" --iterations 30000 --loss_type=l1 --noise_lr=0.0 --kl_threshold=0.000001 --eval --eval_interval=1000 --cap_max 1000000 "
# output_dir=$output_path/$SCENE/

# # ours
# train_script="train_mcmc_sophia_hellinger.py"
# common_args=" --iterations 30000 --loss_type=l1 --noise_lr=0.0 --kl_threshold=1e-7 --eval --eval_interval=1000 --cap_max 1000000 "
# output_dir=$output_path/$SCENE/kl_1e-7

# ours
# train_script="train_mcmc_sophia_hellinger.py"
# common_args=" --iterations 30000 --loss_type=l1 --noise_lr=0.0 --eval --eval_interval=1000 --cap_max 1000000 "
# output_dir=$output_path/$SCENE/kl_1e-6-1e-8_4d33134


# train_script="train_mcmc_sophia_hellinger.py"
# common_args=" --iterations 30000 --loss_type=l1 --noise_lr=0.0 --kl_threshold_init 1e-4 --eval --eval_interval=1000 --cap_max 1000000 "
# output_dir=$output_path/$SCENE/kl_1e-4


# train_script="train_mcmc_sophia_hellinger.py"
# common_args=" --iterations 30000 --densify_from_iter 50000 --loss_type=l1 --noise_lr=0.0 --eval --eval_interval=1000 --cap_max 1000000 "
# output_dir=$output_path/$SCENE/kl_1e-6-1e-8_nodensify


# Ours new densification
train_script=$TRAIN_SCRIPT
common_args=" --iterations 30000 --loss_type=l1 --noise_lr=0.0 --eval --eval_interval=1000 --cap_max 1000000 "
common_args+=" --densify_preserve_gaussians --sparsify_gaussians --sparsify_ratio=$SPARSIFY_RATIO "
# if adam in TRAIN_SCRIPT
if [[ $TRAIN_SCRIPT == *"adam"* ]]; then
    output_dir=$output_path/$SCENE/adam_new_densify_sr$SPARSIFY_RATIO
else
    output_dir=$output_path/$SCENE/sophia_new_densify_sr$SPARSIFY_RATIO
fi

if [[ " ${mipnerf360_outdoor_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$mipnerf360_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -i images_4 -m $output_dir $common_args"
elif [[ " ${mipnerf360_indoor_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$mipnerf360_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -i images_2 -m $output_dir $common_args"
elif [[ " ${tanks_and_temples_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$tanks_and_temples_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -m $output_dir $common_args"
elif [[ " ${deep_blending_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$deep_blending_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -m $output_dir $common_args"
fi

echo "Running benchmark for scene: $SCENE"
echo "Command: $cmd"
echo "Output directory: $output_dir"
eval $cmd
