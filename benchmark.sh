TRAIN_SCRIPT=$1
SCENE=$2
ARGS=${@:3}

# use - to join the arguments
EXP_NAME=$TRAIN_SCRIPT-$(echo ${ARGS[*]} | tr ' ' '-')

output_path="./eval/"

mipnerf360_datadir="../data/360_v2/"
tanks_and_temples_datadir="../data/tandt/"
deep_blending_datadir="../data/db/"

mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
tanks_and_temples_scenes=("truck" "train")
deep_blending_scenes=("drjohnson" "playroom")

train_script=$TRAIN_SCRIPT
common_args=" --iterations 30000 --loss_type=l1 --eval --eval_interval=1000 --cap_max 1000000  --save_iterations 7000 15000 30000"
output_dir=$output_path/$SCENE/$EXP_NAME

if [[ " ${mipnerf360_outdoor_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$mipnerf360_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -i images_4 -m $output_dir $common_args $ARGS"
elif [[ " ${mipnerf360_indoor_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$mipnerf360_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -i images_2 -m $output_dir $common_args $ARGS"
elif [[ " ${tanks_and_temples_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$tanks_and_temples_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -m $output_dir $common_args $ARGS"
elif [[ " ${deep_blending_scenes[*]} " =~ " ${SCENE} " ]]; then
    colmap_datadir=$deep_blending_datadir/$SCENE
    cmd="python $train_script -s $colmap_datadir -m $output_dir $common_args $ARGS"
fi

echo "Running benchmark for scene: $SCENE"
echo "Command: $cmd"
echo "Output directory: $output_dir"
eval $cmd
