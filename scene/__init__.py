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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    """
    Scene 类用于管理场景的3D模型，包括相机参数、点云数据和高斯模型的初始化和加载
    """
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
            初始化3D场景对象

            args: 存储着与 GaussianMoedl 相关参数 的args，即包含scene/__init__.py/ModelParams()中的参数
            gaussians: 3D高斯模型对象，用于场景点的3D表示

            load_iteration: 指定加载模型的迭代次数，如果是-1，则在输出文件夹下的point_cloud/文件夹下搜索迭代次数最大的模型；如果不是None且不是-1，则加载指定迭代次数的
            shuffle: 是否在训练前打乱相机列表
            resolution_scales: 分辨率比例列表，用于处理不同分辨率的相机
        """
        self.model_path = args.model_path   # 模型文件保存路径
        self.loaded_iter = None     # 已加载的迭代次数
        self.gaussians = gaussians  # 高斯模型对象

        # 如果已有训练模型，则加载
        if load_iteration:
            if load_iteration == -1:
                # 是-1，则在输出文件夹下的point_cloud/文件夹下搜索迭代次数最大的模型，记录最大迭代次数
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                # 不是None且不是-1，则加载指定迭代次数的
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {} # 用于训练的相机
        self.test_cameras = {}  # 用于测试的相机

        # 从COLMAP或Blender的输出结果中构建 场景信息（包括点云、训练用相机、测试用相机、场景归一化参数和点云文件路径）
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # 如果没有加载模型，则将点云文件point3D.ply文件复制到input.ply文件
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                # 测试相机添加到 camlist 中
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                # 训练相机添加到 camlist 中
                camlist.extend(scene_info.train_cameras)
            # 遍历 camlist 中的所有相机,使用 camera_to_JSON 函数将每个相机转换为 JSON 格式,并添加到 json_cams 列表中，并将 json_cams 写入 cameras.json 文件中
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            # 随机打乱训练和测试相机列表
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 根据resolution_scales加载不同分辨率的训练和测试相机（包含R、T、视场角）
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            # 如果加载已训练模型，则直接读取对应（已经迭代出来的）场景
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # 不加载训练模型，则调用 GaussianModel.create_from_pcd 从稀疏点云 scene_info.point_cloud 中建立模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """
        保存当前迭代下的3D高斯模型点云。
        iteration: 当前的迭代次数
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        """
        获取指定分辨率比例的训练相机列表
        scale: 分辨率比例
        return: 指定分辨率比例的训练相机列表
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]