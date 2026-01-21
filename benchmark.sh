SCENE=$1
EXP_NAME=$2

output_path="./eval/"

mipnerf360_datadir="../data/360_v2/"
tanks_and_temples_datadir="../data/tandt/"
deep_blending_datadir="../data/db/"

mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
tanks_and_temples_scenes=("truck" "train")
deep_blending_scenes=("drjohnson" "playroom")


common_args=" --iterations 30000 --loss_type=l1 --eval --eval_interval=1000 --cap_max 1000000  --save_iterations 7000 15000 30000 --checkpoint_iterations 7000 15000 30000"
if [[ $EXP_NAME == "mcmc" ]]; then
    # stardard mcmc
    train_script="train_mcmc.py"
    args=$common_args
elif [[ $EXP_NAME == "mcmc_no_density" ]]; then
    # mcmc no densification
    train_script="train_mcmc.py"
    args="$common_args --densify_from_iter=50000"

elif [[ $EXP_NAME == "sophia_hellinger_no_densify" ]]; then
    # sophia hellinger no densification
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000"
elif [[ $EXP_NAME == "sophia_hellinger_no_density_abs" ]]; then
    # sophia hellinger no densification with diagonal_accum_abs
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_accum_abs"

elif [[ $EXP_NAME == "sophia_hellinger_resume_from_15k" ]]; then
    # sophia hellinger resume from 15k adam mcmc
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --start_checkpoint eval/$SCENE/mcmc/chkpnt15000.pth"
elif [[ $EXP_NAME == "sophia_hellinger_resume_from_15k_abs" ]]; then
    # sophia hellinger resume from 15k adam mcmc with diagonal_accum_abs
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --start_checkpoint eval/$SCENE/mcmc/chkpnt15000.pth --diagonal_accum_abs"

elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10" ]]; then
    # ablation: --diagonal_update_interval=10
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_abs_update10" ]]; then
    # ablation: --diagonal_update_interval=10 with diagonal_accum_abs
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --diagonal_accum_abs"

elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update15" ]]; then
    # ablation: --diagonal_update_interval=15
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=15"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_abs_update15" ]]; then
    # ablation: --diagonal_update_interval=15 with diagonal_accum_abs
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=15 --diagonal_accum_abs"

elif [[ $EXP_NAME == "sophia_hellinger_no_densify_normrot" ]]; then
    # ablation: --normalize_rotation
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --normalize_rotation"
elif [[ $EXP_NAME == "sophia_hellinger_no_density_abs_normrot" ]]; then
    # ablation: --normalize_rotation with diagonal_accum_abs
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_accum_abs --normalize_rotation"

elif [[ $EXP_NAME == "adam_tr_no_densify" ]]; then
    # ablation: Adam TR
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --use_adam --use_adam_yes --enable_adam_tr"
fi

output_dir=$output_path/$SCENE/$EXP_NAME

if [[ " ${mipnerf360_outdoor_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$mipnerf360_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -i images_4 -m $output_dir $args"
elif [[ " ${mipnerf360_indoor_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$mipnerf360_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -i images_2 -m $output_dir $args"
elif [[ " ${tanks_and_temples_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$tanks_and_temples_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -m $output_dir $args"
elif [[ " ${deep_blending_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$deep_blending_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -m $output_dir $args"
fi

echo "Running benchmark for scene: $SCENE"
echo "Command: $cmd"
echo "Output directory: $output_dir"
eval $cmd
