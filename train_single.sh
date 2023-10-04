if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_id>"
    exit 1
fi

DIR=`realpath $1`

python train.py -s $DIR \
    -m $DIR \
    -i $DIR/color \
    --data_device cpu \
    --eval \
    --position_lr_init 0.00032 \
    --position_lr_final 0.00032 \
    --position_lr_delay_mult 0.01 \
    --position_lr_max_steps 30000 \
    --feature_lr 0.005 \
    --opacity_lr 0.1 \
    --scaling_lr 0.01 \
    --rotation_lr 0.002 \
    --percent_dense 0.01 \
    --lambda_dssim 0.2 \
    --densification_interval 50 \
    --opacity_reset_interval 3000 \
    --densify_from_iter 0 \
    --densify_until_iter 30000 \
    --densify_grad_threshold 0.0002 \
    --use_ground_truth_pose  \
    --pose_path $DIR/pose \
    --no_shuffle_train
