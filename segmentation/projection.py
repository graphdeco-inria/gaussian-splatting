import torch
import math
from scene.cameras import Camera
from utils.graphics_utils import fov2focal, geom_transform_points

class GaussianProjector:
    """
    Handles the projection of 3D Gaussians into 2D camera space
    """
    def __init__(self, camera: Camera):
        """
        camera: the target camera for projection
        """
        self.camera = camera
        self.device = camera.data_device
        self.width = camera.image_width
        self.height = camera.image_height
        
        # Calculate focal lengths from FoV
        self.focal_x = fov2focal(camera.FoVx, self.width)
        self.focal_y = fov2focal(camera.FoVy, self.height)
        
        # World to View matrix (R, T) which corresponds to W in Eq. 5
        self.view_matrix = camera.world_view_transform
        
        # Extract R part (top-left 3x3) used in the W matrix for covariance rotation
        self.W = self.view_matrix[:3, :3] 
        self.camera_center = camera.camera_center

    def project(self, means3D: torch.Tensor, cov3D: torch.Tensor):
        """
        Project 3D Gaussians to 2D
        The center of a Gaussian distribution is the mean, that is why the 3D coordinates of the center of the Gaussian are passed as means3D. 
        The covariance matrix of the Gaussian is passed as cov3D.
        """
        N = means3D.shape[0]
        
        # Transform points to camera space
        means3D_cam = geom_transform_points(means3D, self.view_matrix)
        
        # Extract x, y, z from those points
        x, y, z = means3D_cam[:, 0], means3D_cam[:, 1], means3D_cam[:, 2]
        
        # Culling
        znear = self.camera.znear
        mask_z = z > znear
        
        # Applying the mask
        indices = torch.nonzero(mask_z).squeeze()
        x = x[indices]
        y = y[indices]
        z = z[indices]
        means3D_cam = means3D_cam[indices]
        cov3D = cov3D[indices]
        
        '''
        Project Covariance
        Implements Sigma' = J W Sigma W^T J^T (Eq. 5)
        '''
        
        # Compute W Sigma W^T, the inner part of Eq. 5
        w_matrix = self.W 
        w_sigma_wt = torch.bmm(w_matrix.T.unsqueeze(0).expand(cov3D.shape[0], -1, -1), 
                              torch.bmm(cov3D, w_matrix.unsqueeze(0).expand(cov3D.shape[0], -1, -1)))
        
        '''
        J is the Jacobian of the affine approximation of the projective transformation, pi(x, y, z) = (f_x * x / z, f_y * y / z):

            [ fx/z   0   -(fx*x)/(z*z) ]
        J = [  0    fy/z -(fy*y)/(z*z) ] (Zwicker et al, formula 34)
            [  0     0         0       ] 
        '''

        inv_z = 1.0 / z
        inv_z2 = inv_z * inv_z
        
        J = torch.zeros((x.shape[0], 2, 3), device=self.device)
        J[:, 0, 0] = self.focal_x * inv_z
        J[:, 0, 2] = -self.focal_x * x * inv_z2
        J[:, 1, 1] = self.focal_y * inv_z
        J[:, 1, 2] = -self.focal_y * y * inv_z2
        
        # Calculate Sigma' = J (W Sigma W^T) J^T (Eq. 5)
        cov2D = torch.bmm(J, torch.bmm(w_sigma_wt, J.transpose(1, 2)))
        
        # Compute projected means to 2D:
        # Perspective projection used to center the 2D splat: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        means2D = torch.stack([
            (x * self.focal_x * inv_z) + cx,
            (y * self.focal_y * inv_z) + cy
        ], dim=1)
        
        return {
            'means2D': means2D,
            'cov2D': cov2D,
            'depths': z,
            'indices': indices # Original IDs of the subset of Gaussians that satisfy z > znear.
        }
