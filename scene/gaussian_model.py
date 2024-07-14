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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        """
        定义和初始化处理高斯体模型参数的 激活函数
        """
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """
            从 缩放因子、旋转四元数 构建 各3D高斯体的 协方差矩阵
            """
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)    # 从缩放因子、旋转四元数得到高斯体的变化，N 3 3
            actual_covariance = L @ L.transpose(1, 2)   # 计算实际的 协方差矩阵
            symm = strip_symmetric(actual_covariance)   # 提取上半对角元素
            return symm

        # 初始化一些激活函数
        self.scaling_activation = torch.exp             # 缩放因子的激活函数，exp函数，确保缩放因子非负
        self.scaling_inverse_activation = torch.log     # 缩放因子的逆激活函数，用于梯度回传，log函数

        self.covariance_activation = build_covariance_from_scaling_rotation # 协方差矩阵的激活函数（实际未使用激活函数，直接构建）

        self.opacity_activation = torch.sigmoid             # 不透明的激活函数，sigmoid函数，确保不透明度在0到1之间
        self.inverse_opacity_activation = inverse_sigmoid   # 不透明度的逆激活函数

        self.rotation_activation = torch.nn.functional.normalize    # 旋转四元数的激活函数，归一化函数（取模）


    def __init__(self, sh_degree : int):
        """
        3D高斯模型的各参数 初始化为0或空
            sh_degree: 设定的 球谐函数的最大阶数，默认为3，用于控制颜色表示的复杂度
        """
        self.active_sh_degree = 0           # 当前球谐函数的阶数，初始为0
        self.max_sh_degree = sh_degree      # 允许的最大球谐阶数j

        self._xyz = torch.empty(0)          # 各3D高斯的 中心位置

        self._features_dc = torch.empty(0)  # 球谐函数的直流分量，第一个元素，用于表示基础颜色
        self._features_rest = torch.empty(0)    # 球谐函数的高阶分量，用于表示颜色的细节和变化

        self._scaling = torch.empty(0)      # 各3D高斯的 缩放因子，控制高斯的形状
        self._rotation = torch.empty(0)     # 各3D高斯的 旋转四元数
        self._opacity = torch.empty(0)      # 各3D高斯的不透明度（sigmoid前的），控制可见性
        self.max_radii2D = torch.empty(0)   # 各3D高斯投影到2D平面上的 最大半径

        self.xyz_gradient_accum = torch.empty(0)    # 3D高斯中心位置 梯度的累及值，当它太大的时候要对Gaussian进行分裂，小时代表under要复制
        self.denom = torch.empty(0)                 # 与累积梯度配合使用，表示统计了多少次累积梯度，用于计算每个高斯体的平均梯度时需除以它（denom = denominator，分母）

        self.optimizer = None   # 优化器，用于调整上述参数以改进模型（论文中采用Adam，见附录B Algorithm 1的伪代码）

        self.percent_dense = 0      # 百分比密度，控制3D高斯的密度
        self.spatial_lr_scale = 0   # 各3D高斯的位置学习率的变化因子，位置的学习率 乘以它，以抵消在不同尺度下应用同一个学习率带来的问题

        # 初始化高斯体模型各参数的 激活函数
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):  # 获取的是激活后的 缩放因子
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self): # 获取的是激活后的 旋转四元数
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):  # 获取的是激活后的 不透明度
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        # 当前球谐函数的阶数 < 规定的最大阶数，则 阶数+1
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        从稀疏点云数据pcd 初始化模型参数
            pcd: 稀疏点云，包含点的位置和颜色
            spatial_lr_scale: 位置学习率的 变化因子
        """
        # 根据scene.Scene.__init__ 以及 scene.dataset_readers.SceneInfo.nerf_normalization，即scene.dataset_readers.getNerfppNorm的代码，
        # 这个值似乎是训练相机中离它们的坐标平均值（即中心）最远距离的1.1倍，根据命名推断应该与学习率有关，防止固定的学习率适配不同尺度的场景时出现问题。
        self.spatial_lr_scale = spatial_lr_scale

        # 点云的3D位置从array转换为tensor，并放到cuda上，(N, 3)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 点云的颜色从RGB array转换为tensor，放到cuda上，再转为球谐函数直流分量系数，(N, 3)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())


        # 初始化存储 球谐系数 的张量，RGB三通道球谐的所有系数，每个通道有(max_sh_degree + 1) ** 2个球谐系数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # (P, 3, 16)
        features[:, :3, 0 ] = fused_color   # 将RGB转换后的球谐系数C0项的系数(直流分量)存入每个3D点的直流分量球谐系数中
        features[:, 3:, 1:] = 0.0   # 其余球谐系数初始化为0

        # 打印初始点的数量
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 计算点云中每个点到其最近的k个点的平均距离的 平方，用于确定高斯的尺度参数scale，且scale（的平方）不能低于1e-7
        # distCUDA2由 submodules/simple-knn/simple_knn.cu 的 SimpleKNN::knn 函数实现，KNN意思是K-Nearest Neighbor，即求每一点最近的K个点
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)  # (P,)

        # 因为scale的激活函数是exp，所以这里存的也不是真的scale，而是ln(scale)。
        # 因dist2其实是距离的平方，所以这里要开根号
        # repeat(1, 3) 标明三个方向上scale的初始值是相等的
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)    # (P, 3)

        # 初始化每个点的旋转参数为单位四元数（无旋转）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # (P, 4)
        rots[:, 0] = 1  # 四元数的实部为1，表示无旋转

        # 初始化每个点的不透明度在sigmoid前的值为0.1，inverse_sigmoid是sigmoid的反函数，等于ln(x / (1 - x))。
        # 不透明度存储的时候要取其经历sigmoid前的值，inverse_sigmoid(0.1) = -2.197
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))    # (P, 1)

        # 将以上计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))    # 高斯椭球体中心位置坐标，(N, 3)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))   # RGB三个通道球谐系数的直流分量（C0项），(N, 3, 1)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # RGB三个通道球谐系数的高阶分量，(N, 3, (最大球谐阶数 + 1)² - 1)
        self._scaling = nn.Parameter(scales.requires_grad_(True))   # 尺度(N, 3)
        self._rotation = nn.Parameter(rots.requires_grad_(True))    # 旋转四元数(N, 4)
        self._opacity = nn.Parameter(opacities.requires_grad_(True))    # 不透明度（经过sigmoid之前），(N, 1)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # 存储2D投影的最大半径，初始化为0，大小为(N,)

    def training_setup(self, training_args):
        """
        设置训练参数，包括初始化用于累积梯度的变量，配置优化器，以及创建学习率调度器
        :param training_args: 包含训练相关参数的对象
        """
        # 设置在训练过程中，用于密集化处理的3D高斯点的比例
        # 控制Gaussian的密度，在`densify_and_clone`中被使用
        self.percent_dense = training_args.percent_dense

        # 初始化用于累积3D高斯中心点位置梯度的张量，用于之后判断是否需要对3D高斯进行克隆或切分
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")    # 坐标的累积梯度

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 意义不明

        # 配置各参数的优化器，包括指定参数、学习率和参数名称
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 创建优化器，这里使用Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 创建学习率调度器，用于对中心点位置的学习率进行调整
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        # 更新Gaussian坐标的学习率
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # 模型被保存到了/point_cloud/iteration_xxx/point_cloud.ply文件中，使用PlyData.read()读取，其第一个属性，即vertex的信息为：x', 'y', 'z', 'nx', 'ny', 'nz', 3个'f_dc_x', 45个'f_rest_xx', 'opacity', 3个'scale_x', 4个'rot_x'
    def construct_list_of_attributes(self):
        # 构建ply文件的键列表
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']   # 不知道nx，ny,nz的用处
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):    # self._features_dc: (N, 3, 1)
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):    # self._features_rest: (N, 3, (最大球谐阶数 + 1)² - 1)
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):     # shape[1]: 3
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):    # shape[1]: 4
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)    # # nx, ny, nz；全是0，不知何用
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # 所有要保存的值合并成一个大数组
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        # get_opacity返回了经过exp的不透明度，是真的不透明度
        # 这句话让所有不透明度都不能超过0.01
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")    # 更新优化器中的不透明度
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        # 读取ply文件并把数据转换成torch.nn.Parameter等待优化
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        # 看样子是把优化器保存的某个名为`name`的参数的值强行替换为`tensor`
        # 这里面需要注意的是修改Adam优化器的状态变量：动量（momentum）和平方动量（second-order momentum）
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)  # 把动量清零
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)   # 把平方动量清零

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        # 根据`mask`裁剪一部分参数及其动量和二阶动量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # 删除Gaussian并移除对应的所有属性
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 重置各个参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        # 把新的张量字典添加到优化器
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # 新增Gaussian，把新属性添加到优化器中
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        '''
        被分裂的Gaussians满足两个条件：
        1. （平均）梯度过大；
        2. 在某个方向的最大缩放大于一个阈值。
        参照论文5.2节“On the other hand...”一段，大Gaussian被分裂成两个小Gaussians，
        其放缩被除以φ=1.6，且位置是以原先的大Gaussian作为概率密度函数进行采样的。
        '''

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # 算出随机采样出来的新坐标。bmm: batch matrix-matrix product
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # 提取出大于阈值`grad_threshold`且缩放参数较小（小于self.percent_dense * scene_extent）的Gaussians，在下面进行克隆
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
            max_grad    决定是否应基于 2D 位置梯度对点进行densification的限制，默认为0.0002
            min_opacity 0.005
            extent
            max_screen_size 初始为None，3000代后，即后续重置不透明度，则为20
        """
        grads = self.xyz_gradient_accum / self.denom    # 计算平均梯度
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent) # 通过克隆增加密度
        self.densify_and_split(grads, max_grad, extent) # 通过分裂增加密度

        # 接下来移除一些Gaussians，它们满足下列要求中的一个：
        # 1. 接近透明（不透明度小于min_opacity）
        # 2. 在某个相机视野里出现过的最大2D半径大于屏幕（像平面）大小
        # 3. 在某个方向的最大缩放大于0.1 * extent（也就是说很长的长条形也是会被移除的）
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size   # vs = view space?
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)    # ws = world space?
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 统计坐标的累积梯度和均值的分母（即迭代步数？）
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1