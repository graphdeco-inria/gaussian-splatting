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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d
import numpy as np
import cv2
from collections import Counter
import pandas as pd
import ffmpeg


def save_ply(xyz_tensor, output_file_path):
    """
    xyz_tensor: torch.Tensor of shape [N, 3]
    output_file_path: string, path to save the PLY file
    """
    # 출력 디렉토리가 존재하지 않으면 생성합니다.
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # 텐서를 CPU로 이동하고 NumPy 배열로 변환합니다.
    xyz_np = xyz_tensor.detach().cpu().numpy()

    # PLY 파일의 헤더를 작성합니다.
    num_vertices = xyz_np.shape[0]
    ply_header = f'''ply
format ascii 1.0
element vertex {num_vertices}
property float x
property float y
property float z
end_header
'''

    # PLY 파일에 데이터를 저장합니다.
    with open(output_file_path, 'w') as f:
        f.write(ply_header)
        np.savetxt(f, xyz_np, fmt='%f %f %f')

    print(f"PLY 파일이 {output_file_path} 경로에 저장되었습니다.")

def pad_image_to_even_dimensions(image):
    # image는 (height, width, channels)의 NumPy 배열입니다.
    height, width = image.shape[:2]
    new_height = height if height % 2 == 0 else height + 1
    new_width = width if width % 2 == 0 else width + 1

    # 새로운 크기의 배열을 생성하고, 패딩된 영역은 0으로 채웁니다.
    padded_image = np.zeros((new_height, new_width, *image.shape[2:]), dtype=image.dtype)
    padded_image[:height, :width, ...] = image

    return padded_image

def compression_h265(projected_data, output_directory):
    # 출력 디렉토리가 존재하지 않으면 생성합니다.
    os.makedirs(output_directory, exist_ok=True)
    
    # 압축할 텐서 목록
    tensors_to_process = {
        'feature_dc_tensor': projected_data.get('feature_dc_tensor'),
        'scaling_tensor': projected_data.get('scaling_tensor'),
        'opacity_tensor': projected_data.get('opacity_tensor'),
        'rotation_tensor': projected_data.get('rotation_tensor'),
    }
    
    # 지정된 텐서들을 처리합니다.
    for tensor_name, tensor in tensors_to_process.items():
        if tensor is None:
            print(f"{tensor_name}가 projected_data에 없습니다.")
            continue
        
        # 텐서를 CPU로 이동하고 NumPy 배열로 변환합니다.
        tensor_np = tensor.detach().cpu().numpy()
        
        # 데이터 타입이 uint8인지 확인하고 변환합니다.
        if tensor_np.dtype != np.uint8:
            tensor_np = tensor_np.astype(np.uint8)
        
        # 텐서의 차원을 확인합니다.
        if tensor_np.ndim != 4:
            print(f"{tensor_name} 텐서의 차원이 4가 아닙니다. 건너뜁니다.")
            continue
        
        num_frames, height, width, channels = tensor_np.shape
        print(f"{tensor_name} 크기: {tensor_np.shape}")
        
        # 채널 수에 따른 픽셀 포맷 설정
        if channels == 1:
            pix_fmt = 'gray'
            tensor_np = tensor_np.squeeze(-1)  # 채널 차원 제거
        elif channels == 3:
            pix_fmt = 'rgb24'
        elif channels == 4:
            pix_fmt = 'rgba'
        else:
            print(f"{tensor_name}의 채널 수 {channels}는 지원되지 않습니다. 건너뜁니다.")
            continue
        
        # 모든 프레임에 대해 패딩을 적용합니다.
        padded_frames = []
        for frame in tensor_np:
            padded_frame = pad_image_to_even_dimensions(frame)
            padded_frames.append(padded_frame)
        
        # 패딩된 프레임의 크기를 가져옵니다.
        padded_height, padded_width = padded_frames[0].shape[:2]
        print(f"{tensor_name} 패딩 후 크기: {(padded_height, padded_width)}")
        
        # 출력 파일 경로 설정
        output_video_path = os.path.join(output_directory, f"{tensor_name}.mp4")
        
        # ffmpeg 프로세스 설정
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{padded_width}x{padded_height}')
            .output(output_video_path, vcodec='libx265', pix_fmt='yuv420p', crf=23)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        
        # 패딩된 프레임 데이터를 ffmpeg 프로세스에 전달
        for frame in padded_frames:
            process.stdin.write(
                frame.tobytes()
            )
        
        # 프로세스 종료
        process.stdin.close()
        process.wait()
        
        print(f"{tensor_name}가 {output_video_path}에 저장되었습니다.")
    
    # feature_rest_tensors 처리 (15개의 텐서)
    feature_rest_tensors = projected_data.get('feature_rest_tensors')
    if feature_rest_tensors is None:
        print("feature_rest_tensors가 projected_data에 없습니다.")
    else:
        for idx, tensor in enumerate(feature_rest_tensors):
            tensor_name = f'feature_rest_tensor_{idx}'
            
            # 텐서를 CPU로 이동하고 NumPy 배열로 변환합니다.
            tensor_np = tensor.detach().cpu().numpy()
            
            # 데이터 타입이 uint8인지 확인하고 변환합니다.
            if tensor_np.dtype != np.uint8:
                tensor_np = tensor_np.astype(np.uint8)
            
            # 텐서의 차원을 확인합니다.
            if tensor_np.ndim != 4:
                print(f"{tensor_name} 텐서의 차원이 4가 아닙니다. 건너뜁니다.")
                continue
            
            num_frames, height, width, channels = tensor_np.shape
            print(f"{tensor_name} 크기: {tensor_np.shape}")
            
            # 채널 수에 따른 픽셀 포맷 설정
            if channels == 1:
                pix_fmt = 'gray'
                tensor_np = tensor_np.squeeze(-1)  # 채널 차원 제거
            elif channels == 3:
                pix_fmt = 'rgb24'
            elif channels == 4:
                pix_fmt = 'rgba'
            else:
                print(f"{tensor_name}의 채널 수 {channels}는 지원되지 않습니다. 건너뜁니다.")
                continue
            
            # 모든 프레임에 대해 패딩을 적용합니다.
            padded_frames = []
            for frame in tensor_np:
                padded_frame = pad_image_to_even_dimensions(frame)
                padded_frames.append(padded_frame)
            
            # 패딩된 프레임의 크기를 가져옵니다.
            padded_height, padded_width = padded_frames[0].shape[:2]
            print(f"{tensor_name} 패딩 후 크기: {(padded_height, padded_width)}")
            
            # 출력 파일 경로 설정
            output_video_path = os.path.join(output_directory, f"{tensor_name}.mp4")
            
            # ffmpeg 프로세스 설정
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{padded_width}x{padded_height}')
                .output(output_video_path, vcodec='libx265', pix_fmt='yuv420p', crf=23)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            
            # 패딩된 프레임 데이터를 ffmpeg 프로세스에 전달
            for frame in padded_frames:
                process.stdin.write(
                    frame.tobytes()
                )
            
            # 프로세스 종료
            process.stdin.close()
            process.wait()
            
            print(f"{tensor_name}가 {output_video_path}에 저장되었습니다.")

def compression_ffv1(projected_data, output_directory):
    # 출력 디렉토리가 존재하지 않으면 생성합니다.
    os.makedirs(output_directory, exist_ok=True)
    
    # 압축할 텐서 목록
    tensors_to_process = {
        'feature_dc_tensor': projected_data.get('feature_dc_tensor'),
        'scaling_tensor': projected_data.get('scaling_tensor'),
        'xyz_tensor': projected_data.get('xyz_tensor'),
        'opacity_tensor': projected_data.get('opacity_tensor'),
        'rotation_tensor': projected_data.get('rotation_tensor'),
    }
    
    # 지정된 텐서들을 처리합니다.
    for tensor_name, tensor in tensors_to_process.items():
        # 텐서를 CPU로 이동하고 NumPy 배열로 변환합니다.
        tensor_np = tensor.detach().cpu().numpy()
        
        # 데이터 타입이 uint16인지 확인하고 변환합니다.
        if tensor_np.dtype != np.uint16:
            tensor_np = tensor_np.astype(np.uint16)
        
        # 텐서의 차원을 확인합니다.
        if tensor_np.ndim != 4:
            print(f"{tensor_name} 텐서의 차원이 4가 아닙니다. 건너뜁니다.")
            continue
        
        num_frames, height, width, channels = tensor_np.shape
        print(f"{tensor_name} 크기: {tensor_np.shape}")
        
        # 채널 수에 따른 픽셀 포맷 설정
        if channels == 1:
            pix_fmt = 'gray16le'
            tensor_np = tensor_np.squeeze(-1)  # 채널 차원 제거
        elif channels == 3:
            pix_fmt = 'rgb48le'
        elif channels == 4:
            pix_fmt = 'rgba64le'
        else:
            print(f"{tensor_name}의 채널 수 {channels}는 지원되지 않습니다. 건너뜁니다.")
            continue
        
        # 출력 파일 경로 설정
        output_video_path = os.path.join(output_directory, f"{tensor_name}.mkv")
        
        # ffmpeg 프로세스 설정
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}')
            .output(output_video_path, format='matroska', vcodec='ffv1', pix_fmt=pix_fmt)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        
        # 프레임 데이터를 ffmpeg 프로세스에 전달
        for frame in tensor_np:
            process.stdin.write(
                frame.tobytes()
            )
        
        # 프로세스 종료
        process.stdin.close()
        process.wait()
        
        print(f"{tensor_name}가 {output_video_path}에 저장되었습니다.")
    
    # feature_rest_tensors 처리 (15개의 텐서)
    feature_rest_tensors = projected_data.get('feature_rest_tensors')
    if feature_rest_tensors is None:
        print("feature_rest_tensors가 projected_data에 없습니다.")
    else:
        for idx, tensor in enumerate(feature_rest_tensors):
            print(tensor.shape)
            tensor_name = f'feature_rest_tensor_{idx}'
            
            # 텐서를 CPU로 이동하고 NumPy 배열로 변환합니다.
            tensor_np = tensor.detach().cpu().numpy()
            
            # 데이터 타입이 uint16인지 확인하고 변환합니다.
            if tensor_np.dtype != np.uint16:
                tensor_np = tensor_np.astype(np.uint16)
            
            # 텐서의 차원을 확인합니다.
            if tensor_np.ndim != 4:
                print(f"{tensor_name} 텐서의 차원이 4가 아닙니다. 건너뜁니다.")
                continue
            
            num_frames, height, width, channels = tensor_np.shape
            print(f"{tensor_name} 크기: {tensor_np.shape}")
            
            # 채널 수에 따른 픽셀 포맷 설정
            if channels == 1:
                pix_fmt = 'gray16le'
                tensor_np = tensor_np.squeeze(-1)  # 채널 차원 제거
            elif channels == 3:
                pix_fmt = 'rgb48le'
            elif channels == 4:
                pix_fmt = 'rgba64le'
            else:
                print(f"{tensor_name}의 채널 수 {channels}는 지원되지 않습니다. 건너뜁니다.")
                continue
            
            # 출력 파일 경로 설정
            output_video_path = os.path.join(output_directory, f"{tensor_name}.mkv")
            
            # ffmpeg 프로세스 설정
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}')
                .output(output_video_path, format='matroska', vcodec='ffv1', pix_fmt=pix_fmt)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            
            # 프레임 데이터를 ffmpeg 프로세스에 전달
            for frame in tensor_np:
                process.stdin.write(
                    frame.tobytes()
                )
            
            # 프로세스 종료
            process.stdin.close()
            process.wait()
            
            print(f"{tensor_name}가 {output_video_path}에 저장되었습니다.")

    
def quantization(tensor, tensor_min=None, tensor_max=None):
    # 텐서를 CPU로 이동하고 NumPy 배열로 변환합니다.
    tensor_np = tensor.detach().cpu().numpy()

    # 최소값과 최대값을 계산합니다.
    if tensor_min is None:
        tensor_min = tensor_np.min()
    if tensor_max is None:
        tensor_max = tensor_np.max()

    # 데이터 범위를 계산합니다.
    data_range = tensor_max - tensor_min
    if data_range == 0:
        data_range = 1e-6  # 0으로 나누는 것을 방지

    # [0, 65535] 범위로 스케일링하여 uint16으로 변환합니다.
    # quantized_tensor = ((tensor_np - tensor_min) / data_range * 65535).astype(np.uint16)
    quantized_tensor = ((tensor_np - tensor_min) / data_range * 255).astype(np.uint8)

    return quantized_tensor, tensor_min, tensor_max

def dequantization(quantized_tensor, tensor_min, tensor_max, device):
    # 데이터 범위를 계산합니다.
    data_range = tensor_max - tensor_min
    if data_range == 0:
        data_range = 1e-6  # 0으로 나누는 것을 방지

    # uint16 데이터를 float32로 복원합니다.
    # tensor_np = quantized_tensor.astype(np.float32) / 65535 * data_range + tensor_min
    tensor_np = quantized_tensor.astype(np.float32) / 255 * data_range + tensor_min

    # 텐서를 생성합니다.
    tensor = torch.from_numpy(tensor_np)

    return tensor.to(device)

def slice(data_values, num_voxels_x, num_voxels_z, linear_indices, num_pixels, device):
    # 데이터 채널 수 확인
    num_channels = data_values.shape[1]

    # 이미지 합계 및 카운트 배열 생성
    image_sums = torch.zeros((num_pixels, num_channels), dtype=torch.float32, device=device)
    counts = torch.zeros((num_pixels), dtype=torch.float32, device=device)

    # 각 픽셀 위치에 데이터 값을 누적합니다.
    image_sums.index_add_(0, linear_indices, data_values)
    counts.index_add_(0, linear_indices, torch.ones_like(linear_indices, dtype=torch.float32))

    # 평균 데이터 값을 계산합니다.
    counts_mask = counts > 0
    image_means = torch.zeros_like(image_sums)
    image_means[counts_mask] = image_sums[counts_mask] / counts[counts_mask].unsqueeze(1)

    # 이미지를 (num_voxels_z, num_voxels_x, 채널 수) 형태로 변환합니다.
    image_means = image_means.view(num_voxels_z, num_voxels_x, num_channels)
    
    return image_means
    
def projection(gaussians, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, debug_from, fix_xyz, quantize):
    # xyz 좌표와 feature_dc, scaling 값을 가져옵니다.
    # print("===================")
    xyz = gaussians._xyz  # [N, 3]
    feature_dc = gaussians._features_dc.squeeze(1)  # [N, 3]
    feature_rest = gaussians._features_rest  # [N, 15, 3]
    scaling = gaussians._scaling  # [N, 3]
    rotation = gaussians._rotation  # [N, 4]
    opacity = gaussians._opacity  # [N, 1]
    
    # 각 데이터의 최소값과 최대값을 계산합니다.
    feature_dc_min = feature_dc.min(dim=0).values 
    feature_dc_max = feature_dc.max(dim=0).values 
    
    scaling_min = scaling.min(dim=0).values 
    scaling_max = scaling.max(dim=0).values  
    
    opacity_min = opacity.min()
    opacity_max = opacity.max()
    
    rotation_min = rotation.min(dim=0).values
    rotation_max = rotation.max(dim=0).values  
    
    min_vals = xyz.min(dim=0).values 
    max_vals = xyz.max(dim=0).values 
    x_min, y_min, z_min = min_vals
    x_max, y_max, z_max = max_vals

    # 복셀 크기를 설정합니다.
    voxel_size_x = 0.0035
    voxel_size_y = 0.0035
    voxel_size_z = 0.0035

    # 각 축에 대한 복셀 수를 계산합니다.
    num_voxels_x = torch.ceil((x_max - x_min) / voxel_size_x).long()
    num_voxels_y = torch.ceil((y_max - y_min) / voxel_size_y).long()
    num_voxels_z = torch.ceil((z_max - z_min) / voxel_size_z).long()
    
    print("===============================")
    print(f"Number of voxels (x, y, z): {num_voxels_x}, {num_voxels_y}, {num_voxels_z}")

    # 각 점의 복셀 인덱스를 계산합니다.
    voxel_indices_x = torch.floor((xyz[:, 0] - x_min) / voxel_size_x).long()
    voxel_indices_y = torch.floor((xyz[:, 1] - y_min) / voxel_size_y).long()
    voxel_indices_z = torch.floor((xyz[:, 2] - z_min) / voxel_size_z).long()

    # 인덱스가 범위를 벗어나지 않도록 클램핑합니다.
    voxel_indices_x = voxel_indices_x.clamp(0, num_voxels_x - 1)
    voxel_indices_y = voxel_indices_y.clamp(0, num_voxels_y - 1)
    voxel_indices_z = voxel_indices_z.clamp(0, num_voxels_z - 1)

    
    # with open("./voxel_indices.txt", "w") as f:
    #     for x, y, z in zip(voxel_indices_x, voxel_indices_y, voxel_indices_z):
    #         f.write(f"{x.item()}, {y.item()}, {z.item()}\n")
    # 복셀 인덱스를 결합하여 [N, 3] 형태로 만듭니다.
    voxel_indices = torch.stack((voxel_indices_x, voxel_indices_y, voxel_indices_z), dim=1)
    
     # 각 복셀에 포함된 포인트 수를 계산합니다.
    unique_voxels, counts = torch.unique(voxel_indices, dim=0, return_counts=True)
    total_voxels = num_voxels_x * num_voxels_y * num_voxels_z
    empty_voxels = total_voxels.item() - unique_voxels.size(0)
    
    
    save = True

    if save == True:
        device = xyz.device

        # 각 데이터를 저장할 리스트를 초기화합니다.
        feature_dc_images = []
        scaling_images = []
        xyz_images = []
        opacity_images = []
        rotation_images = []
        feature_rest_images = [[] for _ in range(15)]  # 15개의 SH 계수에 대한 리스트

        # y축 기준으로 슬라이싱
        for y in range(num_voxels_y):
            slice_mask = (voxel_indices[:, 1] == y)  # y축 기준으로 슬라이싱

            # 슬라이스된 점들의 인덱스와 값을 가져옵니다.
            x_indices = voxel_indices_x[slice_mask]  # [M]
            z_indices = voxel_indices_z[slice_mask]  # [M]
            linear_indices = z_indices * num_voxels_x + x_indices  # [M]
            num_pixels = num_voxels_z * num_voxels_x

            # feature_dc 처리
            feature_dc_values = feature_dc[slice_mask]  # [M, 3]
            image_means_dc = slice(
                data_values=feature_dc_values,
                num_voxels_x=num_voxels_x,
                num_voxels_z=num_voxels_z,
                linear_indices=linear_indices,
                num_pixels=num_pixels,
                device=device
            )
            feature_dc_images.append(image_means_dc.unsqueeze(0))  # [1, Z, X, 3]

            # scaling 처리
            scaling_values = scaling[slice_mask]  # [M, 3]
            image_means_scaling = slice(
                data_values=scaling_values,
                num_voxels_x=num_voxels_x,
                num_voxels_z=num_voxels_z,
                linear_indices=linear_indices,
                num_pixels=num_pixels,
                device=device
            )
            scaling_images.append(image_means_scaling.unsqueeze(0))

            # xyz 처리 (x와 z 좌표만 사용)
            if fix_xyz == False:
                xyz_values = xyz[slice_mask][:, [0, 2]]  # [M, 2]
                image_means_xyz = slice(
                    data_values=xyz_values,
                    num_voxels_x=num_voxels_x,
                    num_voxels_z=num_voxels_z,
                    linear_indices=linear_indices,
                    num_pixels=num_pixels,
                    device=device
                )
                xyz_images.append(image_means_xyz.unsqueeze(0))

            # opacity 처리
            opacity_values = opacity[slice_mask]  # [M, 1]
            image_means_opacity = slice(
                data_values=opacity_values,
                num_voxels_x=num_voxels_x,
                num_voxels_z=num_voxels_z,
                linear_indices=linear_indices,
                num_pixels=num_pixels,
                device=device
            )
            opacity_images.append(image_means_opacity.unsqueeze(0))

            # rotation 처리
            rotation_values = rotation[slice_mask] if slice_mask.sum() > 0 else torch.zeros((0, 4), dtype=rotation.dtype, device=device)
            image_means_rotation = slice(
                data_values=rotation_values,
                num_voxels_x=num_voxels_x,
                num_voxels_z=num_voxels_z,
                linear_indices=linear_indices,
                num_pixels=num_pixels,
                device=device
            )
            rotation_images.append(image_means_rotation.unsqueeze(0))

            # feature_rest 처리 (15개의 SH 계수)
            feature_rest_values = feature_rest[slice_mask]  # [M, 15, 3]
            for i in range(15):
                coeff_values = feature_rest_values[:, i, :]  # [M, 3]
                image_means_coeff = slice(
                    data_values=coeff_values,
                    num_voxels_x=num_voxels_x,
                    num_voxels_z=num_voxels_z,
                    linear_indices=linear_indices,
                    num_pixels=num_pixels,
                    device=device
                )
                feature_rest_images[i].append(image_means_coeff.unsqueeze(0))

        # y축 방향으로 이미지를 쌓습니다.
        feature_dc_tensor = torch.cat(feature_dc_images, dim=0)  # [Y, Z, X, 3]
        scaling_tensor = torch.cat(scaling_images, dim=0)        # [Y, Z, X, 3]
        opacity_tensor = torch.cat(opacity_images, dim=0)        # [Y, Z, X, 1]
        rotation_tensor = torch.cat(rotation_images, dim=0)      # [Y, Z, X, 4]

        
        feature_rest_tensors = []
        for i in range(15):
            coeff_tensor = torch.cat(feature_rest_images[i], dim=0)  # [Y, Z, X, 3]
            feature_rest_tensors.append(coeff_tensor)
            
        # 필요한 값들을 딕셔너리에 저장합니다.
        
        if fix_xyz == False:
            xyz_tensor = torch.cat(xyz_images, dim=0)
        else:
            xyz_tensor = gaussians._xyz
            print(xyz_tensor.shape)
        
        
        # 양자화 단계
        if quantize == True:
            xyz_tensor, xyz_min, xyz_max = quantization(xyz_tensor)
            feature_dc_tensor, feature_dc_min, feature_dc_max = quantization(feature_dc_tensor)
            scaling_tensor, scaling_min, scaling_max = quantization(scaling_tensor)
            opacity_tensor, opacity_min, opacity_max = quantization(opacity_tensor)
            rotation_tensor, rotation_min, rotation_max = quantization(rotation_tensor)

            quantized_feature_rest = []
            feature_rest_mins = []
            feature_rest_maxs = []
            for tensor in feature_rest_tensors:
                quantized_tensor, tensor_min, tensor_max = quantization(tensor)
                quantized_feature_rest.append(quantized_tensor)
                feature_rest_mins.append(tensor_min)
                feature_rest_maxs.append(tensor_max)


            # 복원 단계
            xyz_tensor = dequantization(xyz_tensor, xyz_min, xyz_max, device)
            feature_dc_tensor = dequantization(feature_dc_tensor, feature_dc_min, feature_dc_max, device)
            scaling_tensor = dequantization(scaling_tensor, scaling_min, scaling_max, device)
            opacity_tensor = dequantization(opacity_tensor, opacity_min, opacity_max, device)
            rotation_tensor = dequantization(rotation_tensor, rotation_min, rotation_max, device)
            
            feature_rest_tensors = []
            for idx, quantized_tensor in enumerate(quantized_feature_rest):
                tensor_min = feature_rest_mins[idx]
                tensor_max = feature_rest_maxs[idx]
                restored_tensor = dequantization(quantized_tensor, tensor_min, tensor_max, device)
                feature_rest_tensors.append(restored_tensor)
                
            torch.cuda.empty_cache()
        print("feature dc :",feature_dc_tensor.shape)
        print("scaling :",scaling_tensor.shape)
        print("opacity :",opacity_tensor.shape)
        print("rotation_tensor :",rotation_tensor.shape)
        
        projected_data = {
            'feature_dc_tensor': feature_dc_tensor,
            'scaling_tensor': scaling_tensor,
            'xyz_tensor': xyz_tensor,
            'opacity_tensor': opacity_tensor,
            'rotation_tensor': rotation_tensor,
            'feature_rest_tensors': feature_rest_tensors,
            'x_min': x_min,
            'y_min': y_min,
            'z_min': z_min,
            'voxel_size_x': voxel_size_x,
            'voxel_size_y': voxel_size_y,
            'voxel_size_z': voxel_size_z,
            'num_voxels_x': num_voxels_x,
            'num_voxels_y': num_voxels_y,
            'num_voxels_z': num_voxels_z,
            'voxel_indices_x': voxel_indices_x,
            'voxel_indices_y': voxel_indices_y,
            'voxel_indices_z': voxel_indices_z,
        }
            
        
        return projected_data

def unprojection(projected_data, gaussians, device):
    print("=========unprojection=========")
    # 필요한 텐서들을 가져옵니다.
    xyz_tensor = projected_data['xyz_tensor']  # [Y, Z, X, 2]
    # n,3
    feature_dc_tensor = projected_data['feature_dc_tensor']  # [Y, Z, X, 3]
    scaling_tensor = projected_data['scaling_tensor']  # [Y, Z, X, 3]
    opacity_tensor = projected_data['opacity_tensor']  # [Y, Z, X, 1]
    rotation_tensor = projected_data['rotation_tensor']  # [Y, Z, X, 4]
    feature_rest_tensors = projected_data['feature_rest_tensors']  # 리스트 형태

    y_min = projected_data['y_min']
    voxel_size_y = projected_data['voxel_size_y']

    num_voxels_x = projected_data['num_voxels_x']
    num_voxels_z = projected_data['num_voxels_z']

    # xyz_tensor에서 유효한 복셀 위치를 찾습니다.
    xyz_mask = torch.any(xyz_tensor != 0, dim=-1)  # [Y, Z, X]
    print("xyz mask, ", xyz_mask.shape)
    valid_indices = torch.nonzero(xyz_mask, as_tuple=False)  # [N, 3]
    del xyz_mask
    
    # 유효한 복셀의 인덱스를 사용하여 데이터를 추출합니다.
    linear_indices = valid_indices[:, 0] * (num_voxels_z * num_voxels_x) + valid_indices[:, 1] * num_voxels_x + valid_indices[:, 2]  # [N]

    del num_voxels_x
    del num_voxels_z
    # 각 텐서를 평탄화합니다.
    xyz_tensor = xyz_tensor.reshape(-1, 2)  # [total_voxels, 2]
    feature_dc_tensor = feature_dc_tensor.reshape(-1, 3)  # [total_voxels, 3]
    scaling_tensor = scaling_tensor.reshape(-1, 3)        # [total_voxels, 3]
    opacity_tensor = opacity_tensor.reshape(-1, 1)        # [total_voxels, 1]
    rotation_tensor = rotation_tensor.reshape(-1, 4)      # [total_voxels, 4]
    feature_rest_tensors = [tensor.reshape(-1, 3) for tensor in feature_rest_tensors]  # 리스트 형태

    # 유효한 복셀에서 데이터 추출
    linear_indices = linear_indices.to(device)
    xyz_tensor = xyz_tensor[linear_indices]  # [N, 2]
    feature_dc_tensor = feature_dc_tensor[linear_indices]  # [N, 3]
    scaling_tensor = scaling_tensor[linear_indices]        # [N, 3]
    opacity_tensor = opacity_tensor[linear_indices]        # [N, 1]
    rotation_tensor = rotation_tensor[linear_indices]      # [N, 4]
    feature_rest_tensors = [tensor[linear_indices] for tensor in feature_rest_tensors]  # 리스트 형태
    del linear_indices
    # x, y, z 좌표를 복원합니다.
    voxel_indices_y = valid_indices[:, 0].to(device)
    del valid_indices

    # x와 z 좌표는 원래 데이터에서 가져옵니다.
    x_coords = xyz_tensor[:, 0]
    z_coords = xyz_tensor[:, 1]
    # y 좌표는 복셀 인덱스로부터 복원합니다.
    y_coords = y_min + (voxel_indices_y.float() + 0.5) * voxel_size_y
    del voxel_indices_y
    
    # 최종 xyz 좌표를 생성합니다.
    xyz_tensor = torch.stack((x_coords, y_coords, z_coords), dim=1)  # [N, 3]
    del x_coords
    del z_coords
    del y_coords

    # 가우시안 모델에 데이터 할당
    gaussians._xyz = xyz_tensor  # [N, 3]
    gaussians._features_dc = feature_dc_tensor.unsqueeze(1)  # [N, 1, 3]
    gaussians._scaling = scaling_tensor  # [N, 3]
    gaussians._opacity = opacity_tensor  # [N, 1]
    gaussians._rotation = rotation_tensor  # [N, 4]

    # feature_rest를 [N, 15, 3] 형태로 결합합니다.
    feature_rest_tensors = torch.stack(feature_rest_tensors, dim=1)  # [N, 15, 3]
    gaussians._features_rest = feature_rest_tensors  # [N, 15, 3]

    return gaussians

def fix_xyz_unprojection(projected_data, gaussians, device):
    print("=========fix_xyz_unprojection=========")
    # 필요한 텐서들을 가져옵니다.
    xyz_tensor = projected_data['xyz_tensor']
    print("fix xyz unpro :",xyz_tensor.shape)
    feature_dc_tensor = projected_data['feature_dc_tensor']  # [Y, Z, X, 3]
    scaling_tensor = projected_data['scaling_tensor']  # [Y, Z, X, 3]
    opacity_tensor = projected_data['opacity_tensor']  # [Y, Z, X, 1]
    rotation_tensor = projected_data['rotation_tensor']  # [Y, Z, X, 4]
    feature_rest_tensors = projected_data['feature_rest_tensors']  # 리스트 형태

    voxel_indices_x = projected_data['voxel_indices_x']  # [N]
    voxel_indices_y = projected_data['voxel_indices_y']  # [N]
    voxel_indices_z = projected_data['voxel_indices_z']  # [N]

    # 각 포인트의 복셀 인덱스를 사용하여 속성 값을 가져옵니다.
    # 복셀 인덱스를 텐서 인덱스로 사용하기 위해 차원을 확장합니다.
    voxel_indices_y = voxel_indices_y.to(device).unsqueeze(-1)  # [N, 1]
    voxel_indices_z = voxel_indices_z.to(device).unsqueeze(-1)  # [N, 1]
    voxel_indices_x = voxel_indices_x.to(device).unsqueeze(-1)  # [N, 1]

    # 속성 텐서에서 해당 복셀의 속성 값을 가져옵니다.
    reconstructed_feature_dc = feature_dc_tensor[voxel_indices_y, voxel_indices_z, voxel_indices_x].squeeze(1)  # [N, 3]
    reconstructed_scaling = scaling_tensor[voxel_indices_y, voxel_indices_z, voxel_indices_x].squeeze(1)        # [N, 3]
    reconstructed_opacity = opacity_tensor[voxel_indices_y, voxel_indices_z, voxel_indices_x].squeeze(1)        # [N, 1]
    reconstructed_rotation = rotation_tensor[voxel_indices_y, voxel_indices_z, voxel_indices_x].squeeze(1)      # [N, 4]
    reconstructed_feature_rest = []
    
    for tensor in feature_rest_tensors:
        value = tensor[voxel_indices_y, voxel_indices_z, voxel_indices_x].squeeze(1)  # [N, 3]
        reconstructed_feature_rest.append(value)
    
    gaussians._xyz = xyz_tensor
    gaussians._features_dc = reconstructed_feature_dc.unsqueeze(1)  # [N, 1, 3]
    gaussians._scaling = reconstructed_scaling  # [N, 3]
    gaussians._opacity = reconstructed_opacity  # [N, 1]
    gaussians._rotation = reconstructed_rotation  # [N, 4]

    # feature_rest를 [N, 15, 3] 형태로 결합합니다.
    reconstructed_feature_rest = torch.stack(reconstructed_feature_rest, dim=1)  # [N, 15, 3]
    gaussians._features_rest = reconstructed_feature_rest  # [N, 15, 3]

    return gaussians


def training_report(reconstructed_gaussians, scene, renderFunc, renderArgs):
    
    # 테스트 및 훈련 데이터셋 검증
    torch.cuda.empty_cache()
    validation_configs = ({
        'name': 'test',
        'cameras' : scene.getTestCameras()
    }, {
        'name': 'train',
        'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
    })


    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                # 재구성된 Gaussian 모델로 렌더링
                image = torch.clamp(renderFunc(viewpoint, reconstructed_gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                psnr_test += psnr(image, gt_image).mean().double()
            psnr_test /= len(config['cameras'])

            # PSNR과 L1 Test 값 출력
            print(f"\nEvaluating {config['name']}: PSNR {psnr_test:.4f}")

    torch.cuda.empty_cache()
# python .\projection.py -s .\data\gaussian_splatting\tandt_db\tandt\truck --start_checkpoint .\output\lego\point_cloud\iteration_30000\point_cloud.ply
# python .\projection.py -s ..\data\nerf_synthetic\nerf_synthetic\lego --start_checkpoint .\output\lego\point_cloud\iteration_30000\point_cloud.ply

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = GaussianModel(lp.extract(args).sh_degree)
    scene = Scene(lp.extract(args), gaussians)

    fix_xyz = True
    quantize = False
    gaussians.load_ply(args.start_checkpoint)
    projected_data = projection(gaussians, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.debug_from, fix_xyz, quantize)

    output_video_path = './output/lego/point_cloud/iteration_30000/compression_uint8/'
    # compression_ffv1(projected_data, output_video_path)
    # compression_h265(projected_data, output_video_path)
    output_ply_path = './output/lego/point_cloud/iteration_30000/compression_uint8/xyz.ply'
    xyz_tensor = projected_data['xyz_tensor']
    # save_ply(xyz_tensor, output_ply_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if fix_xyz == False:
        reconstructed_gaussians = unprojection(projected_data, gaussians, device)
    else:
        reconstructed_gaussians = fix_xyz_unprojection(projected_data, gaussians, device)

    bg_color = [1, 1, 1] if lp.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    training_report(gaussians, scene, render, (pp.extract(args), background))
    # 재구성된 가우시안을 저장하려면 다음과 같이 저장합니다.
    if reconstructed_gaussians is not None:
        reconstructed_gaussians.save_ply('./output/lego/point_cloud/iteration_30000/uint8_fix.ply')

    # All done
    print("\nTraining complete.")
