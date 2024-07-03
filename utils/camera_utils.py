#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    """
    调整当前相机对应图像的分辨率，并根据当前相机的info创建相机（包含R、T、FovY、FovX、图像数据image、image_path、image_name、width、height）
    """
    orig_w, orig_h = cam_info.image.size
    # 1. 计算下采样后的图像尺寸
    if args.resolution in [1, 2, 4, 8]:
        # 计算下采样后的图像尺寸 [1, 1/2, 1/4, 1/8]
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            # 如果用户没有指定分辨率，即默认为-1，则自动判断图片的宽度是>1.6K：如果大于，则自动进行下采样到1.6K时的采样倍率；如果小于，则采样倍率=1，即使用原图尺寸
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            # 如果用户指定了分辨率，则根据用户指定的分辨率计算采样倍率
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)    # 缩放倍率
        resolution = (int(orig_w / scale), int(orig_h / scale)) # 下采样后的图像尺寸

    # 2. 调整图片分辨率，归一化，并转换通道为torch上的 (C, H, W)
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        # 如果图片有alpha通道，则提取出来
        loaded_mask = resized_image_rgb[3:4, ...]
    # 3. 创建相机
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    '''
        cam_infos:          train或test相机info列表
        resolution_scale:   分辨率倍率
        args:               更新后的ModelParams()中的参数
    '''
    camera_list = []
    # 遍历每个camera_info（包含R、T、FovY、FovX、图像数据image、image_path、image_name、width、height）
    for id, c in enumerate(cam_infos):
        camera_list.append( loadCam(args, id, c, resolution_scale) )

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
