import torch
import math
import numpy as np

#tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]] ,[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)
height = 16
width = 16
tensor = torch.randn((3, height, width), dtype=torch.float32)

residuals = torch.linalg.vector_norm(tensor, dim=(0))

def calculate_mask(residuals):
    median_residual = torch.median(residuals)
    inlier_loss = torch.where(residuals <= median_residual, 1.0, 0.0)

    kernel = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).unsqueeze(0).unsqueeze(0).cuda()
    has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)
    has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, kernel, padding = "same")
    has_inlier_neighbors = torch.where(has_inlier_neighbors >= 0.5, 1.0, 0.0)

    kernel_16 = 1/(16*16) * torch.ones((1,1,16,16)).cuda()
    if has_inlier_neighbors.shape[1] % 8 != 0:
        pad_h = 8 - (has_inlier_neighbors.shape[1] % 8) + 8
    else:
        pad_h = 8

    if has_inlier_neighbors.shape[2] % 8 != 0:
        pad_w = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
    else:
        pad_w = 8

    padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))
    padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate").cuda()

    is_inlier_patch = torch.nn.functional.conv2d(padded_weights.unsqueeze(0), kernel_16, stride = 8)

    is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch, scale_factor = 8)
    is_inlier_patch = is_inlier_patch.squeeze()

    padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

    if padding_indexing[1] == 0:
        padding_indexing[1] = has_inlier_neighbors.shape[1]
    if padding_indexing[3] == 0:
        padding_indexing[3] = has_inlier_neighbors.shape[2]

    is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]

    is_inlier_patch = torch.where(is_inlier_patch >= 0.6, 1.0, 0.0)

    mask = (is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze() >= 1e-3).cuda()

    return mask
