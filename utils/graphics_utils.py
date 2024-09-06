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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovXleft, fovXright, fovYtop, fovYbottom):
    tanHalfFovYtop = math.tan(fovYtop)
    tanHalfFovYbottom = math.tan(fovYbottom)
    tanHalfFovXleft = math.tan(fovXleft)
    tanHalfFovXright = math.tan(fovXright)

    top = tanHalfFovYtop * znear
    bottom = tanHalfFovYbottom * znear
    left = tanHalfFovXleft * znear
    right = tanHalfFovXright * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    # note that my conventions are (fovXleft,fovYtop) negative and (fovXright,fovYbottom) positive
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (bottom - top)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = -(top + bottom) / (bottom - top)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return sidefov2focal(fov / 2, pixels / 2)

def focal2fov(focal, pixels):
    return 2 * focal2sidefov(focal, pixels / 2)

def sidefov2focal(sidefov, sidepixels):
    return sidepixels / math.tan(sidefov)

def focal2sidefov(focal, sidepixels):
    return math.atan(sidepixels / focal)
