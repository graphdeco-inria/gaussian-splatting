for scene in bonsai counter kitchen room
do
    rm /mnt/disk2/360_augmented/${scene}/sparse/0/points3D.ply
    python augment.py --colmap_path /mnt/disk2/360/${scene}/sparse/0 --image_path /mnt/disk2/360/${scene}/images_2 \
    --augment_path /mnt/disk2/360_augmented/${scene}/sparse/0/points3D.bin \
    --camera_order covisibility \
    --visibility_aware_culling \
    --compare_center_patch
done