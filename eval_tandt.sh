# # Qt 관련 모든 환경 변수를 비활성화
# export QT_QPA_PLATFORM=offscreen
# unset QTDIR
# unset QT_PLUGIN_PATH
# unset LD_LIBRARY_PATH

# # KIME 입력기 관련 환경 변수도 비활성화
# unset GTK_IM_MODULE
# unset QT_IM_MODULE
# unset XMODIFIERS

for scene in Courthouse; do
    # python convert.py -s /home/cvnar/disk4tb/tandt/${scene}

    # cp -r -p /home/cvnar/disk4tb/tandt/${scene} /home/cvnar/disk4tb/tandt_augmented/${scene}

    # python augment.py --colmap_path /home/cvnar/disk4tb/tandt/${scene}/sparse/0 --image_path /home/cvnar/disk4tb/tandt/${scene}/images \
    # --augment_path /home/cvnar/disk4tb/tandt_augmented/${scene}/sparse/0/points3D.bin \
    # --camera_order covisibility \
    # --visibility_aware_culling \
    # --compare_center_patch

    python train.py -s /home/cvnar/disk4tb/tandt/${scene} -m ../experiments/tandt/${scene} \
    -i images \
    --data_device cpu \
    -r 1 \
    --eval 

    python render.py -m ../experiments/tandt/${scene} --skip_train

    python metrics.py -m ../experiments/tandt/${scene}

    # rm /home/cvnar/disk4tb/tandt_augmented/${scene}/sparse/0/points3D.ply

    # python train.py -s /home/cvnar/disk4tb/tandt_augmented/${scene} -m ../experiments/tandt_augmented/${scene} \
    # -i images \
    # -r 1 \
    # --data_device cpu \
    # --eval \
    # --bundle_training \
    # --camera_order covisibility \
    # --enable_ds_lap \
    # --lambda_ds 1.2 \
    # --lambda_lap 0.4

    # python render.py -m ../experiments/tandt_augmented/${scene} --skip_train

    # python metrics.py -m ../experiments/tandt_augmented/${scene}
done

