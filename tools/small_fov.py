import cv2
import os
def crop_fov(image):
    # 读取原始图像
    #image = cv2.imread(path)

    # 原始图像的大小
    original_height, original_width = image.shape[:2]

    # 目标视场角
    target_fov = 60  # 60度

    # 计算新的宽度和高度，保持高度不变
    new_width = int(original_width * (target_fov / 88))
    new_height = int(original_height * (target_fov / 88))

    # 计算需要裁剪的左右边界
    left_boundary = (original_width - new_width) // 2
    right_boundary = original_width - left_boundary

    top_boundary = (original_height - new_height) // 2
    down_boundary = original_height - top_boundary

    # 裁剪图像
    cropped_image = image[top_boundary:down_boundary, left_boundary:right_boundary, :]
    return cropped_image
d = '/data/jianing/dlf_result/proj_0829_all_sm/images'
#lll = [os.path.join(d, i,'images') for i in os.listdir(d) if 'colmap' in i]

for cam_name in os.listdir(d):
    print(cam_name)
    cam_name = os.path.join(d,cam_name)
    image = cv2.imread(cam_name)
    height, width = image.shape[:2]
    image = cv2.resize(image, (width // 2, height // 2))
    crop_img = crop_fov(image)
    cv2.imwrite(cam_name, crop_img)