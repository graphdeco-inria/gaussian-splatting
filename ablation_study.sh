diskpath=/mnt/disk2
exp_path=${diskpath}/auggs/experiments
colmap_path=${diskpath}/360
colmap_path_augmented=${diskpath}/360_augmented
for scene in bicycle flowers garden stump treehill
do
    # No Augmentation
    python train.py -s ${colmap_path}/${scene} -m ${exp_path}/360_augmented_noaug/${scene} \
    -i images_4 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4

    python render.py -m ${exp_path}/360_augmented_noaug/${scene} --skip_train

    python metrics.py -m ${exp_path}/360_augmented_noaug/${scene}

    # nobatch
    python train.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/360_augmented_nobatch/${scene} \
    -i images_4 \
    --eval \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4

    python render.py -m ${exp_path}/360_augmented_nobatch/${scene} --skip_train

    python metrics.py -m ${exp_path}/360_augmented_nobatch/${scene}

    # noloss
    python train.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/360_augmented_noloss/${scene} \
    -i images_4 \
    --eval \
    --bundle_training \
    --camera_order covisibility 

    python render.py -m ${exp_path}/360_augmented_noloss/${scene} --skip_train

    python metrics.py -m ${exp_path}/360_augmented_noloss/${scene}
done

for scene in bonsai counter kitchen room
do
    # No Augmentation
    python train.py -s ${colmap_path}/${scene} -m ${exp_path}/360_augmented_noaug/${scene} \
    -i images_2 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4

    python render.py -m ${exp_path}/360_augmented_noaug/${scene} --skip_train

    python metrics.py -m ${exp_path}/360_augmented_noaug/${scene}

    # nobatch
    python train.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/360_augmented_nobatch/${scene} \
    -i images_2 \
    --eval \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4

    python render.py -m ${exp_path}/360_augmented_nobatch/${scene} --skip_train

    python metrics.py -m ${exp_path}/360_augmented_nobatch/${scene}

    # noloss
    python train.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/360_augmented_noloss/${scene} \
    -i images_2 \
    --eval \
    --bundle_training \
    --camera_order covisibility 

    python render.py -m ${exp_path}/360_augmented_noloss/${scene} --skip_train

    python metrics.py -m ${exp_path}/360_augmented_noloss/${scene}
done