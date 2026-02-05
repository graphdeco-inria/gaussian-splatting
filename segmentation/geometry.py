import torch
import os
import sys
from scene.gaussian_model import GaussianModel
from utils.general_utils import build_scaling_rotation

class GaussianGeometry:
    """
    Handles the geometric aspects of 3D Gaussians for semantic segmentation
    """
    def __init__(self, model_path: str, iteration: int, sh_degree: int = 3, device: str = "cuda"):
        """
        model_path: path to the trained model directory
        iteration: the iteration number to load
        sh_degree: spherical harmonics degree
        device: device to load the tensors on
        """
        self.device = device
        self.sh_degree = sh_degree
        self.gaussians = GaussianModel(sh_degree=sh_degree, use_labels=True)
        
        # The .ply is accesible from this directories by default
        self.ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        
        if not os.path.exists(self.ply_path):
            raise FileNotFoundError(f"Could not find PLY file at {self.ply_path}")
            
        self.gaussians.load_ply(self.ply_path)

    def get_covariance_3d(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """
        Compute the full 3D covariance matrix for each Gaussian

        Returns:
            torch.Tensor: tensor containing covariance matrices, with shape (N, 3, 3) 
        """

        # L = R * S
        scaling = self.gaussians.get_scaling
        rotation = self.gaussians.get_rotation
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        
        # Eq. 6: Sigma = L*L^T
        covariance = L @ L.transpose(1, 2) # Shape: (N, 3, 3)
        return covariance

    @property
    def xyz(self):
        """Returns the positions of the Gaussians, with shape (N, 3)"""
        return self.gaussians.get_xyz

    @property
    def opacity(self):
        """Returns the opacities of the Gaussians, with shape (N, 1)"""
        return self.gaussians.get_opacity
    
    @property
    def scaling(self):
        """Returns the scaling of the Gaussians, with shape (N, 3)"""
        return self.gaussians.get_scaling

    @property
    def rotation(self):
        """Returns the rotation quaternions of the Gaussians, with shape (N, 4)"""
        return self.gaussians.get_rotation
    
    @property
    def features_dc(self):
        """Returns the DC features, with shape (N, 1, 3)"""
        return self.gaussians.get_features_dc
    
    @property
    def features_rest(self):
        """Returns the SH features, with shape (N, 15, 3)"""
        return self.gaussians.get_features_rest

    @property
    def labels(self):
        """Returns the semantic labels of the Gaussians, with shape (N,)"""
        return self.gaussians.get_labels
