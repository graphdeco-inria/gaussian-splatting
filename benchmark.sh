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

# if "-eval" is in the EXP_NAME, then IS_TRAIN=false, and remove "-eval" from EXP_NAME
if [[ $EXP_NAME == *"-eval"* ]]; then
    IS_TRAIN=false
    EXP_NAME=$(echo $EXP_NAME | sed 's/-eval//g')
else
    IS_TRAIN=true
fi

common_args=" --iterations 30000 --loss_type=l1 --cap_max 1000000  --save_iterations 7000 15000 30000"
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

elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update20" ]]; then
    # ablation: --diagonal_update_interval=20
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=20"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_abs_update20" ]]; then
    # ablation: --diagonal_update_interval=20 with diagonal_accum_abs
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=20 --diagonal_accum_abs"

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

# rebuttal experiments
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_fixed_tr" ]]; then
    # ablation: --diagonal_update_interval=10 with fixed trust region
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --tr_func=uniform"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_tr_sweep1" ]]; then
    # ablation: --diagonal_update_interval=10 with different trust region
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --kl_threshold_init=1e-5 --kl_threshold_final=1e-7"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_tr_sweep2" ]]; then
    # ablation: --diagonal_update_interval=10 with different trust region
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --kl_threshold_init=1e-7 --kl_threshold_final=1e-9"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_random_init_10k" ]]; then
    # ablation: --diagonal_update_interval=10 with random init
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --init_type=random --random_init_pts=10000"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_random_init_20k" ]]; then
    # ablation: --diagonal_update_interval=10 with random init
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --init_type=random --random_init_pts=20000"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_random_init_50k" ]]; then
    # ablation: --diagonal_update_interval=10 with random init
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --init_type=random --random_init_pts=50000"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_random_init_100k" ]]; then
    # ablation: --diagonal_update_interval=10 with random init
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --init_type=random --random_init_pts=100000"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_random_init_200k" ]]; then
    # ablation: --diagonal_update_interval=10 with random init
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --init_type=random --random_init_pts=200000"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_random_init_500k" ]]; then
    # ablation: --diagonal_update_interval=10 with random init
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --init_type=random --random_init_pts=500000"
elif [[ $EXP_NAME == "sophia_hellinger_no_densify_update10_random_init_1M" ]]; then
    # ablation: --diagonal_update_interval=10 with random init
    train_script="train_mcmc_sophia_hellinger.py"
    args="$common_args --noise_lr=0.0 --densify_from_iter=50000 --diagonal_update_interval=10 --init_type=random --random_init_pts=1000000"
fi

output_dir=$output_path/$SCENE/$EXP_NAME

if [[ $IS_TRAIN == true ]]; then
    # train
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
else
    # eval
    # exclude "  --save_iterations 7000 15000 30000 --checkpoint_iterations 7000 15000 30000" from args
    args=$(echo $args | sed 's/--save_iterations 7000 15000 30000//g' | sed 's/--checkpoint_iterations 7000 15000 30000//g')
    eval_script="evaluate_from_checkpoint.py"
    if [[ " ${mipnerf360_outdoor_scenes[*]} " =~ " ${SCENE} " ]]; then
        colmap_datadir=$mipnerf360_datadir/$SCENE
        cmd="python $eval_script -s $colmap_datadir -i images_4 -m $output_dir $args --model_path $output_dir"
    elif [[ " ${mipnerf360_indoor_scenes[*]} " =~ " ${SCENE} " ]]; then
        colmap_datadir=$mipnerf360_datadir/$SCENE
        cmd="python $eval_script -s $colmap_datadir -i images_2 -m $output_dir $args --model_path $output_dir"
    elif [[ " ${tanks_and_temples_scenes[*]} " =~ " ${SCENE} " ]]; then
        colmap_datadir=$tanks_and_temples_datadir/$SCENE
        cmd="python $eval_script -s $colmap_datadir -m $output_dir $args --model_path $output_dir"
    elif [[ " ${deep_blending_scenes[*]} " =~ " ${SCENE} " ]]; then
        colmap_datadir=$deep_blending_datadir/$SCENE
        cmd="python $eval_script -s $colmap_datadir -m $output_dir $args --model_path $output_dir"
    fi

fi

echo "Running benchmark for scene: $SCENE"
echo "Command: $cmd"
echo "Output directory: $output_dir"
eval $cmd
