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

def composite_with_background(image, alpha_mask, background):
    if alpha_mask is None:
        return image

    alpha_mask = alpha_mask.to(device=image.device, dtype=image.dtype)
    background = background.to(device=image.device, dtype=image.dtype).view(-1, 1, 1)
    return image * alpha_mask + background * (1.0 - alpha_mask)

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
