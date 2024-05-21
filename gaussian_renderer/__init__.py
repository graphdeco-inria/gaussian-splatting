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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
import torch.nn.functional as F
import matplotlib.pyplot as plt

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           return_depth=False, return_normal=False, return_opacity=False):
    """
    渲染场景： 将高斯分布的点投影到2D屏幕上来生成渲染图像
        viewpoint_camera: 训练相机集合
        pc: 高斯模型
        pipe:   管道相关参数
        bg_color: Background tensor 必须 on GPU
        scaling_modifier:
        override_color:
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个与输入点云（高斯模型）大小相同的 零tensor，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 尝试保留张量的梯度。这是为了确保可以在反向传播过程中计算对于屏幕空间坐标的梯度
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 计算视场的 tan 值，这将用于设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵viewmatrix、投影矩阵projmatrix等
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # 创建一个高斯光栅化器对象，用于将高斯分布投影到屏幕上
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取高斯模型的三维坐标、屏幕空间坐标、透明度
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from scaling / rotation by the rasterizer.
    # 如果提供了预先计算的3D协方差矩阵，则使用它。否则，它将由光栅化器根据尺度和旋转进行计算
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预先计算的颜色，则使用它们。否则，如果希望在Python中从球谐函数中预计算颜色，请执行此操作。如果没有，则颜色将通过光栅化器进行从球谐函数到RGB的转换
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # 将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 计算相机中心到每个点的方向向量，并归一化
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # 计算相机中心到每个点的方向向量，并归一化
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 使用SH特征将方向向量转换为RGB颜色
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 将RGB颜色的范围限制在0到1之间
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 返回一个字典，包含渲染的图像、屏幕空间坐标、可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径
    return_dict={"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    if return_depth:
        #　提取相机的世界视图变换矩阵的第3列(深度方向)的前3个元素和最后一个元素。这些值将用于计算深度信息
        projvect1 = viewpoint_camera.world_view_transform[:, 2][:3].detach()
        projvect2 = viewpoint_camera.world_view_transform[:, 2][-1].detach()
        #　计算每个3D点的深度值：　means3D * projvect1.unsqueeze(0) 将 means3D (3D点坐标) 与 projvect1 (深度方向的前3个元素) 相乘,得到深度分量；
        #                   　深度分量求和,并加上 projvect2 (深度方向的最后一个元素),得到最终的深度值
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1, keepdim=True) + projvect2
        # 深度值复制成3个通道,与 colors_precomp 的尺寸匹配
        means3D_depth = means3D_depth.repeat(1, 3)
        render_depth, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=means3D_depth,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        render_depth = render_depth.mean(dim=0) # 将批量维度上的深度值取平均,得到最终的深度图
        return_dict.update({'render_depth': render_depth})

        # plt.figure(figsize=(8, 6))
        # plt.imshow(rendered_image.permute(1, 2, 0).detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Rendered Normal Map')
        # plt.show()

        # plt.figure(figsize=(8, 6))
        # plt.imshow(render_depth.detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Rendered Depth Map')
        # plt.show()


    if return_normal:
        rotations_mat = build_rotation(rotations)
        scales = pc.get_scaling
        min_scales = torch.argmin(scales, dim=1)
        indices = torch.arange(min_scales.shape[0])
        normal = rotations_mat[indices, :, min_scales]

        # convert normal direction to the camera; calculate the normal in the camera coordinate
        view_dir = means3D - viewpoint_camera.camera_center
        normal = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[..., None]

        R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)
        normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)

        render_normal, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=normal,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        render_normal = F.normalize(render_normal, dim=0)
        return_dict.update({'render_normal': render_normal})

        # plt.figure(figsize=(8, 6))
        # plt.imshow(render_normal.permute(1, 2, 0).detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Rendered Normal Map')
        # plt.show()

    if return_opacity:
        density = torch.ones_like(means3D)

        render_opacity, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=density,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        return_dict.update({'render_opacity': render_opacity.mean(dim=0)})

    return return_dict