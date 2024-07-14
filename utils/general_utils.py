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

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0   # 归一化
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)   # 转换为 3 H W
    else:
        # 若为H W，则添加一个通道维度为 H W 1，再转换为 1 H W
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    """
    创建一个学习率调度函数，该函数根据训练进度动态调整学习率

    :param lr_init: 初始学习率
    :param lr_final: 最终学习率
    :param lr_delay_steps: 学习率延迟步数，在这些步数内学习率将被降低
    :param lr_delay_mult: 学习率延迟乘数，用于计算初始延迟学习率
    :param max_steps: 最大步数，用于规范化训练进度
    :return: 一个函数，根据当前步数返回调整后的学习率
    """

    def helper(step):
        # 如果步数小于0或学习率为0，直接返回0，表示不进行优化
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        # 如果设置了学习率延迟步数，计算延迟调整后的学习率
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        # 根据步数计算学习率的对数线性插值，实现从初始学习率到最终学习率的平滑过渡
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        # 返回调整后的学习率
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    """
    从协方差矩阵中提取6个上半对角元素，节省内存
    [ _ _ _ ]
    [   _ _ ]
    [     _ ]
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")    # N 6

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    """
    提取协方差矩阵的上半对角元素
        sym: 协方差矩阵
        return: 上半对角元素
    """
    return strip_lowerdiag(sym)

def build_rotation(r):
    '''
    旋转四元数 -> 单位化 -> 3x3的旋转矩阵
    '''
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    """
    构建3D高斯模型的 缩放-旋转矩阵
        s: 缩放因子, N 3
        r: 旋转四元素, N 4
        return: 尺度-旋转矩阵
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")   # 初始化缩放矩阵为0，N 3 3
    R = build_rotation(r)   # 旋转四元数 -> 旋转矩阵，N 3 3

    # 构建缩放矩阵，其对角线元素对应为缩放因子的s1, s2, s3
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L   # 高斯体的变化：旋转 矩阵乘 缩放
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    # 若args.quiet 为 True，不写入任何文本到标准输出管道
    sys.stdout = F(silent)

    # 设置随机种子，使得结果可复现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))   # torch 默认的 CUDA 设备为 cuda:0