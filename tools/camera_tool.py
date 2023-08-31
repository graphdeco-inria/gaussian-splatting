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

class ViewpointCamera:
    FoVx: np.array
    FoVy: np.array
    image_width: int
    image_height: int
    world_view_transform: np.array
    full_proj_transform: np.array
    camera_center: np.array

    def __init__(self, image_width=1280, image_height=720, fx = 783.5623272369992, fy = 775.0592003023257):
        self.image_width = image_width
        self.image_height = image_height
        # "fy": 775.0592003023257, "fx": 783.5623272369992
        self.FoVx = focal2fov(fx, self.image_width)
        self.FoVy = focal2fov(fy, self.image_height)

    def load_extrinsic(self, eye, target, up, znear, zfar):
        self.camera_center = eye
        _, R, T = look_at_to_rt(eye, target, up)
        self.world_view_transform = getWorld2View2(R, T)
        self.projection_matrix = getProjectionMatrix(znear, zfar, self.FoVx, self.FoVy)
        self.full_proj_transform = np.matmul(self.world_view_transform, self.projection_matrix)

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
    forward = np.array(target) - np.array(eye)
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)

    # Create the rotation matrix (3x3).
    rotation_matrix = np.vstack((right, new_up, -forward)).T

    # Create the translation vector (3x1).
    translation_vector = -np.dot(rotation_matrix, np.array(eye))

    # Combine the rotation and translation into an RT matrix (4x4).
    rt_matrix = np.identity(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation_vector
    
    return rt_matrix, rotation_matrix, translation_vector

def load_views_from_lookat(filename):
    glcams = load_params_from_file('./tools/cameras.lookat')
    views = []
    for idx in range(len(glcams)):
        view = ViewpointCamera()
        view.load_extrinsic(glcams['eye'][idx], glcams['target'][idx], glcams['up'][idx], glcams['clipZ'][idx][0].item(), glcams['clipZ'][idx][1].item())
        views.append(view)
    return views

def load_views_from_lookat_torch(filename):
    glcams = load_params_from_file('./tools/cameras.lookat')
    views = []
    for idx in range(len(glcams)):
        view = ViewpointCamera()
        view.load_extrinsic(glcams['eye'][idx], glcams['target'][idx], glcams['up'][idx], glcams['clipZ'][idx][0].item(), glcams['clipZ'][idx][1].item())
        view.world_view_transform = torch.tensor(view.world_view_transform).transpose(0, 1).cuda()
        view.full_proj_transform = torch.tensor(view.world_view_transform).transpose(0, 1).cuda()
        views.append(view)
    return views

# # Example usage:
# eye = [0.0, 0.0, 3.0]     # Camera position
# target = [0.0, 0.0, 0.0]  # Point the camera looks at
# up = [0.0, 1.0, 0.0]      # Up vector

# extrinsic_matrix = look_at_to_extrinsic(eye, target, up)
# print("Camera Extrinsic Matrix:")
# print(extrinsic_matrix)

if __name__ == '__main__':
    glcams = load_views_from_lookat('./tools/cameras.lookat')
    a = 1
