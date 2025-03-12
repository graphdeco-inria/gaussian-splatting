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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass
import math

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def build_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance)) * \
                     torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / \
                     (2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2D depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel

def gaussian_blur(x, kernel):
    """Apply gaussian blur to input tensor"""
    padding = (kernel.shape[-1] - 1) // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

def create_laplacian_pyramid(image, max_levels=4):
    """Create Laplacian pyramid for an image"""
    pyramids = []
    current = image
    kernel = build_gaussian_kernel().to(image.device)
    
    for _ in range(max_levels):
        # Blur and downsample
        blurred = gaussian_blur(current, kernel)
        down = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # Upsample and subtract
        up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
        laplace = current - up
        
        pyramids.append(laplace)
        current = down
    
    pyramids.append(current)  # Add the final residual
    return pyramids

def laplacian_pyramid_loss(pred, target, max_levels=4, weights=None):
    """Compute Laplacian Pyramid Loss between predicted and target images"""
    if weights is None:
        weights = [1.0] * (max_levels + 1)
    
    pred_pyramids = create_laplacian_pyramid(pred, max_levels)
    target_pyramids = create_laplacian_pyramid(target, max_levels)
    
    loss = 0
    for pred_lap, target_lap, weight in zip(pred_pyramids, target_pyramids, weights):
        loss += weight * torch.abs(pred_lap - target_lap).mean()
    
    return loss

class LaplacianPyramidLoss(torch.nn.Module):
    def __init__(self, max_levels=4, channels=3, kernel_size=5, sigma=1.0):
        super().__init__()
        self.max_levels = max_levels
        self.kernel = build_gaussian_kernel(kernel_size, sigma, channels)
        
    def forward(self, pred, target, weights=None):
        if weights is None:
            weights = [1.0] * (self.max_levels + 1)
            
        # Move kernel to the same device as input
        kernel = self.kernel.to(pred.device)
        
        pred_pyramids = self.create_laplacian_pyramid(pred, kernel)
        target_pyramids = self.create_laplacian_pyramid(target, kernel)
        
        loss = 0
        for pred_lap, target_lap, weight in zip(pred_pyramids, target_pyramids, weights):
            loss += weight * torch.abs(pred_lap - target_lap).mean()
        
        return loss
    
    @staticmethod
    def create_laplacian_pyramid(image, kernel, max_levels=4):
        pyramids = []
        current = image
        
        for _ in range(max_levels):
            # Apply Gaussian blur before downsampling to prevent aliasing
            blurred = gaussian_blur(current, kernel)
            down = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            # Upsample and subtract from the original image
            up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
            laplace = current - gaussian_blur(up, kernel)  # Apply blur to upsampled image
            
            pyramids.append(laplace)
            current = down
        
        pyramids.append(current)  # Add the final residual
        return pyramids
    
class InvDepthSmoothnessLoss(nn.Module):
    def __init__(self, alpha=10):
        super(InvDepthSmoothnessLoss, self).__init__()
        self.alpha = alpha  # 엣지 가중치 강도를 조절하는 하이퍼파라미터

    def forward(self, inv_depth, image):
        # 역깊이 맵의 그래디언트 계산
        dx_inv_depth = torch.abs(inv_depth[:, :, :-1] - inv_depth[:, :, 1:])
        dy_inv_depth = torch.abs(inv_depth[:, :-1, :] - inv_depth[:, 1:, :])

        # 이미지의 그래디언트 계산
        dx_image = torch.mean(torch.abs(image[:, :, :-1] - image[:, :, 1:]), 1, keepdim=True)
        dy_image = torch.mean(torch.abs(image[:, :-1, :] - image[:, 1:, :]), 1, keepdim=True)

        # 이미지 그래디언트에 기반한 가중치 계산
        weight_x = torch.exp(-self.alpha * dx_image)
        weight_y = torch.exp(-self.alpha * dy_image)

        # Smoothness loss 계산
        smoothness_x = dx_inv_depth * weight_x
        smoothness_y = dy_inv_depth * weight_y

        return torch.mean(smoothness_x) + torch.mean(smoothness_y)