diskpath=/mnt/disk2
exp_path=${diskpath}/auggs/experiments
colmap_path=${diskpath}/360
colmap_path_augmented=${diskpath}/360_augmented
# for scene in bicycle flowers garden stump treehill
for scene in bicycle flowers garden stump treehill
do
    # python augment.py --colmap_path ${colmap_path}/${scene}/sparse/0 --image_path ${colmap_path}/${scene}/images_4 \
    # --augment_path ${colmap_path_augmented}/${scene}/sparse/0/points3D.bin \
    # --camera_order covisibility \
    # --visibility_aware_culling \
    # --compare_center_patch

    python train.py -s ${colmap_path}/${scene} -m ${exp_path}/360/${scene} \
    -i images_4 \
    --eval 

    python render.py -m ${exp_path}/360/${scene} --skip_train

    python metrics.py -m ${exp_path}/360/${scene}

    rm ${colmap_path_augmented}/${scene}/sparse/0/points3D.ply

    python train.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/360_augmented/${scene} \
    -i images_4 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4

    python render.py -m ${exp_path}/360_augmented/${scene} --skip_train

    python metrics.py -m ${exp_path}/360_augmented/${scene}
done

for scene in bonsai counter kitchen room
do
    # python augment.py --colmap_path ${colmap_path}/${scene}/sparse/0 --image_path ${colmap_path}/${scene}/images_2 \
    # --augment_path ${colmap_path_augmented}/${scene}_augmented/sparse/0/points3D.bin \
    # --camera_order covisibility \
    # --visibility_aware_culling \
    # --compare_center_patch

    python train.py -s ${colmap_path}/${scene} -m ${exp_path}/360/${scene} \
    -i images_2 \
    --eval 

    python render.py -m ${exp_path}/360/${scene} --skip_train

    python metrics.py -m ${exp_path}/360/${scene}

    rm ${colmap_path_augmented}/${scene}/sparse/0/points3D.ply

    python train.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/360_augmented/${scene} \
    -i images_2 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4

    python render.py -m ${exp_path}/360_augmented/${scene} --skip_train

    python metrics.py -m ${exp_path}/360_augmented/${scene}
done

