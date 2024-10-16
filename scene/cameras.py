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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


    def get_depths_from_depth_map(self, pixel_coords):
        """
        Fetches depth values from the depth map at the given pixel coordinates using bilinear interpolation.
        """
        x_coords = pixel_coords[:, 0]
        y_coords = pixel_coords[:, 1]

        H, W = self.depth_map.shape

        # Compute integer coordinates
        x0 = torch.clamp(x_coords.floor().long(), 0, W - 1)
        x1 = torch.clamp(x0 + 1, 0, W - 1)
        y0 = torch.clamp(y_coords.floor().long(), 0, H - 1)
        y1 = torch.clamp(y0 + 1, 0, H - 1)

        # Compute fractional parts
        x_frac = x_coords - x_coords.floor()
        y_frac = y_coords - y_coords.floor()

        device = pixel_coords.device
        depth_map = self.depth_map.to(device)

        Ia = depth_map[y0, x0]
        Ib = depth_map[y1, x0]
        Ic = depth_map[y0, x1]
        Id = depth_map[y1, x1]

        depths = Ia * (1 - x_frac) * (1 - y_frac) + \
                Ib * (1 - x_frac) * y_frac + \
                Ic * x_frac * (1 - y_frac) + \
                Id * x_frac * y_frac

        return depths
    

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

