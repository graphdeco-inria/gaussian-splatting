import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys



txt_dir = '/home/luvision/project/gaussian-splatting/test810_all4/sparse/0'


def get_rt(txt_dir,target_name):
    img_txt = os.path.join(txt_dir,'images.txt')
    cam_txt = os.path.join(txt_dir, 'cameras.txt')
    with open(img_txt,'r') as f:
        ll = f.readlines()
    ll = ll[4:]

    ll = ll[::2]

    for l in ll:
        cont = l.split()
        #cam_name = os.path.basename(cont[9]).split('.')[0]
        cam_name = cont[9]
        if target_name == cam_name:
            #print(cam_name)
            rt = cont
            cam_ind = int(cont[8])
            break
    
    cam_name = os.path.basename(rt[9]).split('.')[0]

    #QW, QX, QY, QZ = [float(rt[1]), float(rt[2]),  float(rt[3]),  float(rt[4])]
    Rq1 = np.asarray([float(rt[2]),  float(rt[3]),  float(rt[4]), float(rt[1])])
    r1 = R.from_quat(Rq1)
    Rm1 = r1.as_matrix()
    T = np.asarray([float(rt[5]), float(rt[6]), float(rt[7])])

    with open(cam_txt,'r') as f:
        ll = f.readlines()
        ll = ll[4:]
    K = ll[cam_ind - 1]
    K = K.split()[4:]
    K = [float(i) for i in K]
    return Rm1, T, K

def relative_pose(R_view1, T_view1, R_view2, T_view2):

    # 计算相对位置变换
    R_relative = np.dot(R_view2, np.linalg.inv(R_view1))
    #T_relative = T_view2 - np.dot(R_view2, T_view1)
    T_relative = T_view2 - np.dot(R_view2, T_view1)
    # 构造相对位置变换矩阵
    relative_transform = np.identity(4)
    relative_transform[:3, :3] = R_relative
    relative_transform[:3, 3] = T_relative
    return relative_transform

R1, T1, K1 = get_rt(txt_dir, '1691653061.282838/cam_0.png')
R2, T2, K2 = get_rt(txt_dir, '1691653061.282838/cam_1.png')

relative_transform = relative_pose(R1, T1, R2, T2)

#relative_transform = relative_pose(R2, T2, R1, T1)

import numpy as np
import cv2
from matplotlib import pyplot as plt

#relative_transform = np.linalg.inv(relative_transform)
#print(relative_transform)

depth_dir = '/home/luvision/project/gaussian-splatting/output_810_2/train/ours_10000/depth/00000.npy'

depth_image_A = np.load(depth_dir)
# 视角B相对于视角A的姿态变换矩阵
pose_A_to_B = np.array(relative_transform)  # 4x4姿态矩阵

fx,fy,cx,cy = K1

camera_matrix_A = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
# 视角B相对于视角A的姿态变换矩阵
pose_A_to_B = np.array(relative_transform)  # 4x4姿态矩阵
# 视角B相机内参
fx2,fy2,cx2,cy2 = K2
camera_matrix_B = np.array([[fx2, 0, cx2],
                            [0, fy2, cy2],
                            [0, 0, 1]])

# 获取视角A图像的高度和宽度
height_A, width_A = depth_image_A.shape[:2]

# 创建视角B的空白RGB和深度图像
#rgb_image_B = np.zeros((height_A, width_A, 3), dtype=np.uint8)
depth_image_B = np.zeros((height_A, width_A), dtype=np.float32)

# 遍历视角A的每个像素
for y in range(height_A):
    for x in range(width_A):
        # 获取视角A中的深度信息
        depth_A = depth_image_A[y, x]
        #print(depth_A)
        if depth_A > 0:
            # 计算视角A中像素的3D坐标
            #depth_A *= 0.1
            point_A = np.array([(x - cx) * depth_A / fx,
                                (y - cy) * depth_A / fy,
                                depth_A,1])

            point_B = np.dot(pose_A_to_B, point_A)
            # 将3D点从视角B坐标系投影回视角B图像平面
            u_B = int(point_B[0] * fx2 / point_B[2] + cx2)
            v_B = int(point_B[1] * fy2 / point_B[2] + cy2)

            if 0 <= u_B < width_A and 0 <= v_B < height_A:
                depth_image_B[v_B, u_B] = depth_A

final_depth_image_B = depth_image_B
plt.figure(1)
plt.imshow(final_depth_image_B)

# 现在final_rgb_image_B和final_depth_image_B包含了视角B的估计RGB图像和深度图像
depth_dir2 = '/home/luvision/project/gaussian-splatting/output_810_2/train/ours_10000/depth/00001.npy'
depth2 = np.load(depth_dir2)
depth2[final_depth_image_B < 0.1] = 0
plt.figure(2)
plt.imshow(depth2)

# plt.figure(3)
# plt.imshow(depth_image_A)
plt.show()