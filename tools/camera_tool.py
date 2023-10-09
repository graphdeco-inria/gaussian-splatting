import numpy as np
import re
from typing import NamedTuple
import sys
import os
import torch

from pathlib import Path
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(dir_path.parent.absolute()))
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal

# function to load cameras
class ViewpointCamera:
    FoVx: np.array
    FoVy: np.array
    image_width: int
    image_height: int
    world_view_transform: np.array
    full_proj_transform: np.array
    camera_center: np.array
    R: np.array
    T: np.array

    def __init__(self, image_width=1280, image_height=720, fx = 783.5623272369992, fy = 775.0592003023257):
        self.image_width = image_width
        self.image_height = image_height
        # "fy": 775.0592003023257, "fx": 783.5623272369992
        self.FoVx = focal2fov(fx, self.image_width)
        self.FoVy = focal2fov(fy, self.image_height)

    def load_extrinsic(self, eye, target, up, znear, zfar):
        self.camera_center = eye
        RT, self.R, self.T = look_at_to_rt(eye, target, up)
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        self.world_view_transform = getWorld2View2(self.R, self.T, trans, scale)
        # self.world_view_transform = RT
        self.world_view_transform = torch.tensor(self.world_view_transform).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load_extrinsic2(self, R, T, znear, zfar):
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        self.world_view_transform = getWorld2View2(R, T, trans, scale)
        # self.world_view_transform = RT
        self.world_view_transform = torch.tensor(self.world_view_transform).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

# function for spline interpolation
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import CubicSpline

class CameraInterpolation:
    # Define a function to interpolate rotation matrices using SLERP
    def interpolate_extrinsic_matrices(Rs, Ts, inter_num):
        interpolated_matrices = []
        t_orig = np.linspace(0.0, len(Rs) - 1, num=len(Rs), endpoint=True)
        t_values = np.linspace(0.0, len(Rs) - 1, num=inter_num, endpoint=True)
        # Interpolate rotation matrices 
        rotations = Rotation.from_matrix(Rs)
        rot_spline = RotationSpline(t_orig, rotations)
        interpolated_rotations = rot_spline(t_values)
        interpolated_matrices = interpolated_rotations.as_matrix()
        # Interpolate translation matrices 
        Ts_mat = np.stack(Ts)
        spline1 = CubicSpline(t_orig, Ts_mat[:, 0])
        spline2 = CubicSpline(t_orig, Ts_mat[:, 1])
        spline3 = CubicSpline(t_orig, Ts_mat[:, 2])
        interpolated_positions = np.array([
            spline1(t_values),
            spline2(t_values),
            spline3(t_values)
        ]).T
        return interpolated_matrices, interpolated_positions

def load_params_from_file(filename):
    # Define a regular expression pattern to match floating-point numbers
    pattern = r"[-+]?\d*\.\d+|\d+"
    # init lists
    eyes = []
    targets = []
    ups = []
    fovYs = []
    clipZs = []
    with open(filename, "r") as file:
        # Read the entire file contents into a string
        key_cam_param = file.readline()
        while key_cam_param:
            # Use re.findall to extract all numbers from the string
            numbers = [float(match) for match in re.findall(pattern, key_cam_param)]
            eye = np.zeros((3,), dtype=np.float32)
            target = np.zeros((3,), dtype=np.float32)
            up = np.zeros((3,), dtype=np.float32)
            fovY = np.zeros((1,), np.float32)
            clipZ = np.zeros((2,), np.float32)
            eye[0] = numbers[0]
            eye[1] = numbers[1]
            eye[2] = numbers[2]
            target[0] = numbers[3]
            target[1] = numbers[4]
            target[2] = numbers[5]
            up[0] = numbers[6]
            up[1] = numbers[7]
            up[2] = numbers[8]
            fovY[0] = numbers[9]
            clipZ[0] = numbers[10]
            clipZ[1] = numbers[11]
            eyes.append(eye)
            targets.append(target)
            ups.append(up)
            fovYs.append(fovY)
            clipZs.append(clipZ)
            key_cam_param = file.readline()
    
    return {'eye': eyes, 'target': targets, 'up': ups, 'fovY': fovYs, 'clipZ': clipZs}

def look_at_to_rt(eye, target, up):
    # Calculate the forward, right, and up vectors.
    up = -up
    zaxis = (target - eye)
    zaxis /= np.linalg.norm(zaxis)
    
    xaxis = np.cross(up, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)

    # Create the rotation matrix (3x3).
    rotation_matrix = np.vstack((xaxis, yaxis, zaxis)).T

    # Create the translation vector (3x1).
    translation_vector = -np.dot(eye, rotation_matrix)

    # Combine the rotation and translation into an RT matrix (4x4).
    rt_matrix = np.identity(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation_vector

    rt_matrix = rt_matrix.astype(np.float32)
    rotation_matrix = rotation_matrix.astype(np.float32)
    translation_vector = translation_vector.astype(np.float32)
    
    return rt_matrix, rotation_matrix, translation_vector

def load_views_from_lookat_torch(filename):
    glcams = load_params_from_file(filename=filename)
    views = []
    for idx in range(len(glcams['eye'])):
        view = ViewpointCamera()
        view.load_extrinsic(glcams['eye'][idx], glcams['target'][idx], glcams['up'][idx], glcams['clipZ'][idx][0].item(), glcams['clipZ'][idx][1].item())
        views.append(view)
    return views

def load_views_from_lookat_torch_w_spline_interpolation(filename, inter_num = 100):
    glcams = load_params_from_file(filename=filename)
    Rs = []
    Ts = []
    for idx in range(len(glcams['eye'])):
        view = ViewpointCamera()
        RT, R, T = look_at_to_rt(glcams['eye'][idx], glcams['target'][idx], glcams['up'][idx])
        Rs.append(R)
        Ts.append(T)
    Rs_inter, Ts_inter = CameraInterpolation.interpolate_extrinsic_matrices(Rs, Ts, inter_num=inter_num)
    views = []
    for idx in range(Rs_inter.shape[0]):
        view = ViewpointCamera()
        view.load_extrinsic2(Rs_inter[idx, :, :], Ts_inter[idx, :], glcams['clipZ'][0][0].item(), glcams['clipZ'][0][1].item())
        views.append(view)

    return views

def interpolate_cameras(cameras, inter_num):
    Rs = []
    Ts = []
    for idx in range(len(cameras)):
        Rs.append(cameras[idx].R)
        Ts.append(cameras[idx].T)
    Rs_inter, Ts_inter = CameraInterpolation.interpolate_extrinsic_matrices(Rs, Ts, inter_num=inter_num)
    views = []
    for idx in range(Rs_inter.shape[0]):
        view = ViewpointCamera()
        view.load_extrinsic2(Rs_inter[idx, :, :], Ts_inter[idx, :], 0.01, 100)
        views.append(view)

    return views

def generate_LF_cameras(camera_center, cam_num, baseline):
    # views = []
    # views.append(camera_center)
    # return views
    # get rotation and translation matrix from camera
    R_center = camera_center.R
    T_center = camera_center.T
    # combine the rotation and translation into an RT matrix (4x4).
    rt_matrix = np.identity(4)
    rt_matrix[:3, :3] = R_center
    rt_matrix[:3, 3] = T_center
    lf_rt_matrices = []
    # extract the camera's up, right, and view directions
    up_direction = rt_matrix[:3, 1]  # Second column
    right_direction = rt_matrix[:3, 0]  # First column
    view_direction = -rt_matrix[:3, 2]  # Negative of third column
    # move cameras to generate LF cameras
    cam_num_half = cam_num // 2 
    for j in range(-cam_num_half, cam_num_half + 1, 1):
        for i in range(-cam_num_half, cam_num_half + 1, 1):
            shift = (up_direction * j + right_direction * i) * baseline
            trans_matrix_ = np.identity(4)
            trans_matrix_[:3, 3] = shift 
            new_camera_matrix = np.dot(rt_matrix, trans_matrix_)
            lf_rt_matrices.append(new_camera_matrix)
    # return views
    views = []
    for idx in range(len(lf_rt_matrices)):
        view = ViewpointCamera(image_width=1280, image_height=720, fx = 1100, fy = 1100)
        rt_matrix_ = lf_rt_matrices[idx]
        R_center = rt_matrix_[:3, :3]
        T_center = rt_matrix_[:3, 3]
        view.load_extrinsic2(R_center, T_center, 0.01, 100)
        views.append(view)

    return views

if __name__ == '__main__':
    a = 1
