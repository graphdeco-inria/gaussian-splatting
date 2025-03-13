for scene in bicycle flowers garden room stump
do
    python augment.py --colmap_path /home/cvnar/disk4tb/360/${scene}/sparse/0 --image_path /home/cvnar/disk4tb/360/${scene}/images_4 \
    --augment_path /home/cvnar/disk4tb/360_augmented/${scene}/sparse/0/points3D.bin \
    --camera_order covisibility \
    --visibility_aware_culling \
    --compare_center_patch

    python train.py -s /home/cvnar/disk4tb/360/${scene} -m ../experiments/360/${scene} \
    -i images_4 \
    --eval 

    python render.py -m ../experiments/360/${scene} --skip_train

    python metrics.py -m ../experiments/360/${scene}

    rm /home/cvnar/disk4tb/360_augmented/${scene}/sparse/0/points3D.ply

    python train.py -s /home/cvnar/disk4tb/360_augmented/${scene} -m ../experiments/360_augmented/${scene} \
    -i images_4 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4

    python render.py -m ../experiments/360_augmented/${scene} --skip_train

    python metrics.py -m ../experiments/360_augmented/${scene}
done

for scene in bonsai counter kitchen room
do
    python augment.py --colmap_path /home/cvnar/disk4tb/360/${scene}/sparse/0 --image_path /home/cvnar/disk4tb/360/${scene}/images_2 \
    --augment_path /home/cvnar/disk4tb/360_augmented/${scene}_augmented/sparse/0/points3D.bin \
    --camera_order covisibility \
    --visibility_aware_culling \
    --compare_center_patch

    python train.py -s /home/cvnar/disk4tb/360/${scene} -m ../experiments/360/${scene} \
    -i images_2 \
    --eval 

    python render.py -m ../experiments/360/${scene} --skip_train

    python metrics.py -m ../experiments/360/${scene}

    rm /home/cvnar/disk4tb/360_augmented/${scene}/sparse/0/points3D.ply

    python train.py -s /home/cvnar/disk4tb/360_augmented/${scene} -m ../experiments/360_augmented/${scene} \
    -i images_2 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4

    python render.py -m ../experiments/360_augmented/${scene} --skip_train

    python metrics.py -m ../experiments/360_augmented/${scene}
done

