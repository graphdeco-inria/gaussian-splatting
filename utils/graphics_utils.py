"""
Optimized Geometric Transformation Module
Version: 2.0
Author: Optimized version of the original geometry utilities
Description: Provides efficient 3D geometric transformations, camera projections, 
             and point cloud operations with GPU acceleration and type safety.
"""

from __future__ import annotations
import torch
import math
import numpy as np
from typing import NamedTuple, Optional, Union, Tuple, List
from dataclasses import dataclass
import warnings

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PointCloud:
    """
    Optimized point cloud data structure with GPU support and memory efficiency.
    
    Attributes:
        points (torch.Tensor): 3D point coordinates of shape (N, 3)
        colors (torch.Tensor): RGB colors of shape (N, 3), range [0, 1] or [0, 255]
        normals (torch.Tensor): Surface normals of shape (N, 3)
        device (torch.device): Device where tensors are stored
        dtype (torch.dtype): Data type of tensors
    """
    
    points: torch.Tensor
    colors: torch.Tensor
    normals: torch.Tensor
    
    def __post_init__(self):
        """Validate and ensure consistent device and dtype."""
        self._validate_shapes()
        self._ensure_consistent_device()
        self._ensure_consistent_dtype()
    
    def _validate_shapes(self):
        """Validate tensor shapes."""
        if not (self.points.shape == self.colors.shape == self.normals.shape):
            raise ValueError(f"Shape mismatch: points {self.points.shape}, "
                           f"colors {self.colors.shape}, normals {self.normals.shape}")
        if self.points.shape[1] != 3:
            raise ValueError(f"Expected 3D points, got shape {self.points.shape}")
    
    def _ensure_consistent_device(self):
        """Ensure all tensors are on the same device."""
        devices = {t.device for t in [self.points, self.colors, self.normals]}
        if len(devices) > 1:
            target_device = self.points.device
            self.colors = self.colors.to(target_device)
            self.normals = self.normals.to(target_device)
    
    def _ensure_consistent_dtype(self):
        """Ensure consistent data types."""
        # Points and normals typically use float32, colors use float32 or uint8
        if self.points.dtype != torch.float32:
            self.points = self.points.float()
        if self.normals.dtype != torch.float32:
            self.normals = self.normals.float()
        if self.colors.dtype not in [torch.float32, torch.uint8]:
            self.colors = self.colors.float()
    
    @property
    def device(self) -> torch.device:
        """Get the device of the point cloud."""
        return self.points.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of points tensor."""
        return self.points.dtype
    
    @property
    def num_points(self) -> int:
        """Get the number of points in the cloud."""
        return self.points.shape[0]
    
    def to(self, device: torch.device) -> PointCloud:
        """Move point cloud to specified device."""
        return PointCloud(
            points=self.points.to(device),
            colors=self.colors.to(device),
            normals=self.normals.to(device)
        )
    
    def cpu(self) -> PointCloud:
        """Move point cloud to CPU."""
        return self.to(torch.device('cpu'))
    
    def cuda(self) -> PointCloud:
        """Move point cloud to CUDA device if available."""
        if torch.cuda.is_available():
            return self.to(torch.device('cuda'))
        warnings.warn("CUDA not available, returning CPU point cloud")
        return self
    
    def numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        return (
            self.points.cpu().numpy(),
            self.colors.cpu().numpy(),
            self.normals.cpu().numpy()
        )
    
    @classmethod
    def from_numpy(cls, 
                   points: np.ndarray, 
                   colors: np.ndarray, 
                   normals: np.ndarray,
                   device: str = 'cpu') -> PointCloud:
        """Create PointCloud from numpy arrays."""
        return cls(
            points=torch.from_numpy(points).to(device),
            colors=torch.from_numpy(colors).to(device),
            normals=torch.from_numpy(normals).to(device)
        )
    
    def transform(self, transform_matrix: torch.Tensor) -> PointCloud:
        """Apply transformation matrix to points and normals."""
        # Transform points
        transformed_points = transform_points(self.points, transform_matrix)
        
        # Transform normals (rotate only, ignore translation)
        rotation_matrix = transform_matrix[:3, :3]
        transformed_normals = torch.matmul(self.normals, rotation_matrix.T)
        transformed_normals = transformed_normals / torch.norm(transformed_normals, dim=1, keepdim=True)
        
        return PointCloud(
            points=transformed_points,
            colors=self.colors,
            normals=transformed_normals
        )


class CameraParameters:
    """
    Camera intrinsic and extrinsic parameters.
    
    Attributes:
        R (torch.Tensor): Rotation matrix (3, 3)
        t (torch.Tensor): Translation vector (3,)
        fx (float): Focal length in x-direction
        fy (float): Focal length in y-direction
        cx (float): Principal point x-coordinate
        cy (float): Principal point y-coordinate
        width (int): Image width in pixels
        height (int): Image height in pixels
        znear (float): Near clipping plane
        zfar (float): Far clipping plane
    """
    
    def __init__(self, 
                 R: torch.Tensor,
                 t: torch.Tensor,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 width: int,
                 height: int,
                 znear: float = 0.01,
                 zfar: float = 100.0):
        
        self.R = R
        self.t = t
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.znear = znear
        self.zfar = zfar
        
        # Pre-compute derived parameters
        self._fov_x = focal2fov(self.fx, self.width)
        self._fov_y = focal2fov(self.fy, self.height)
        self._intrinsic_matrix = self._compute_intrinsic_matrix()
        self._extrinsic_matrix = self._compute_extrinsic_matrix()
        self._projection_matrix = self._compute_projection_matrix()
    
    def _compute_intrinsic_matrix(self) -> torch.Tensor:
        """Compute camera intrinsic matrix."""
        K = torch.eye(3, device=self.R.device, dtype=self.R.dtype)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K
    
    def _compute_extrinsic_matrix(self) -> torch.Tensor:
        """Compute camera extrinsic matrix (world to camera)."""
        return get_world_to_view_matrix(self.R, self.t)
    
    def _compute_projection_matrix(self) -> torch.Tensor:
        """Compute perspective projection matrix."""
        return get_projection_matrix(
            self.znear, self.zfar, self._fov_x, self._fov_y
        ).to(self.R.device)
    
    @property
    def fov_x(self) -> float:
        """Get horizontal field of view in radians."""
        return self._fov_x
    
    @property
    def fov_y(self) -> float:
        """Get vertical field of view in radians."""
        return self._fov_y
    
    @property
    def intrinsic_matrix(self) -> torch.Tensor:
        """Get intrinsic matrix."""
        return self._intrinsic_matrix
    
    @property
    def extrinsic_matrix(self) -> torch.Tensor:
        """Get extrinsic matrix (world to camera)."""
        return self._extrinsic_matrix
    
    @property
    def projection_matrix(self) -> torch.Tensor:
        """Get projection matrix."""
        return self._projection_matrix
    
    @property
    def camera_center(self) -> torch.Tensor:
        """Get camera center in world coordinates."""
        # Camera center = -R^T * t
        return -self.R.T @ self.t
    
    def to(self, device: torch.device) -> CameraParameters:
        """Move camera parameters to specified device."""
        return CameraParameters(
            R=self.R.to(device),
            t=self.t.to(device),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            znear=self.znear,
            zfar=self.zfar
        )


# ============================================================================
# Core Geometric Functions
# ============================================================================

def transform_points(points: torch.Tensor, 
                    transform_matrix: torch.Tensor,
                    eps: float = 1e-7) -> torch.Tensor:
    """
    Transform 3D points using a 4x4 transformation matrix.
    
    Args:
        points: Tensor of shape (N, 3) or (..., 3)
        transform_matrix: Transformation matrix of shape (4, 4) or (..., 4, 4)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Transformed points of same shape as input
    """
    # Ensure points are in homogeneous coordinates
    points_shape = points.shape
    points = points.reshape(-1, 3)
    
    # Convert to homogeneous coordinates (N, 4)
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=-1)
    
    # Apply transformation
    if transform_matrix.dim() == 2:
        # Single transformation matrix for all points
        transformed = torch.matmul(points_hom, transform_matrix.T)
    else:
        # Batch transformation
        # Reshape transform_matrix to (..., 4, 4) and points_hom to (..., N, 4)
        transformed = torch.matmul(points_hom.unsqueeze(-2), 
                                 transform_matrix.transpose(-1, -2)).squeeze(-2)
    
    # Convert back from homogeneous coordinates
    denom = transformed[..., 3:] + eps
    transformed_points = transformed[..., :3] / denom
    
    # Restore original shape
    return transformed_points.reshape(*points_shape)


def get_world_to_view_matrix(R: torch.Tensor, 
                            t: torch.Tensor) -> torch.Tensor:
    """
    Compute world-to-camera transformation matrix.
    
    Args:
        R: Rotation matrix of shape (3, 3)
        t: Translation vector of shape (3,)
    
    Returns:
        Transformation matrix of shape (4, 4)
    """
    Rt = torch.eye(4, device=R.device, dtype=R.dtype)
    Rt[:3, :3] = R.T  # R^{-1} = R^T for rotation matrices
    Rt[:3, 3] = t
    return Rt


def get_view_to_world_matrix(R: torch.Tensor, 
                            t: torch.Tensor) -> torch.Tensor:
    """
    Compute camera-to-world transformation matrix.
    
    Args:
        R: Rotation matrix of shape (3, 3)
        t: Translation vector of shape (3,)
    
    Returns:
        Transformation matrix of shape (4, 4)
    """
    Rt = torch.eye(4, device=R.device, dtype=R.dtype)
    Rt[:3, :3] = R  # Original rotation
    Rt[:3, 3] = -R @ t  # Camera center in world coordinates
    return Rt


def get_world_to_view_matrix_with_scale(R: torch.Tensor,
                                       t: torch.Tensor,
                                       translate: torch.Tensor = None,
                                       scale: float = 1.0) -> torch.Tensor:
    """
    Compute world-to-camera transformation with additional translation and scaling.
    
    Args:
        R: Rotation matrix of shape (3, 3)
        t: Translation vector of shape (3,)
        translate: Additional translation of shape (3,)
        scale: Scaling factor
    
    Returns:
        Transformation matrix of shape (4, 4)
    """
    if translate is None:
        translate = torch.zeros(3, device=R.device, dtype=R.dtype)
    
    # Get standard world-to-view matrix
    world2view = get_world_to_view_matrix(R, t)
    
    # Convert to camera-to-world
    view2world = torch.inverse(world2view)
    
    # Apply translation and scaling to camera center
    cam_center = view2world[:3, 3]
    cam_center = (cam_center + translate) * scale
    view2world[:3, 3] = cam_center
    
    # Convert back to world-to-view
    return torch.inverse(view2world)


def get_projection_matrix(znear: float,
                         zfar: float,
                         fov_x: float,
                         fov_y: float,
                         device: torch.device = None) -> torch.Tensor:
    """
    Compute perspective projection matrix (OpenGL convention).
    
    Args:
        znear: Near clipping plane distance
        zfar: Far clipping plane distance
        fov_x: Horizontal field of view in radians
        fov_y: Vertical field of view in radians
        device: Device for the output tensor
    
    Returns:
        Projection matrix of shape (4, 4)
    """
    if device is None:
        device = torch.device('cpu')
    
    tan_half_fov_y = math.tan(fov_y / 2.0)
    tan_half_fov_x = math.tan(fov_x / 2.0)
    
    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right
    
    P = torch.zeros((4, 4), device=device)
    
    # OpenGL perspective projection matrix
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    
    return P


def get_orthographic_matrix(left: float,
                           right: float,
                           bottom: float,
                           top: float,
                           znear: float,
                           zfar: float,
                           device: torch.device = None) -> torch.Tensor:
    """
    Compute orthographic projection matrix.
    
    Args:
        left, right, bottom, top: Clipping planes
        znear: Near clipping plane distance
        zfar: Far clipping plane distance
        device: Device for the output tensor
    
    Returns:
        Orthographic projection matrix of shape (4, 4)
    """
    if device is None:
        device = torch.device('cpu')
    
    P = torch.zeros((4, 4), device=device)
    
    # Orthographic projection matrix
    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[2, 2] = -2.0 / (zfar - znear)  # Negative for OpenGL convention
    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -(zfar + znear) / (zfar - znear)
    P[3, 3] = 1.0
    
    return P


# ============================================================================
# Focal Length and FOV Conversion
# ============================================================================

def focal_to_fov(focal_length: float, 
                pixels: float) -> float:
    """
    Convert focal length to field of view.
    
    Args:
        focal_length: Focal length in pixels
        pixels: Image dimension in pixels (width or height)
    
    Returns:
        Field of view in radians
    """
    return 2.0 * math.atan(pixels / (2.0 * focal_length))


def fov_to_focal(fov: float, 
                pixels: float) -> float:
    """
    Convert field of view to focal length.
    
    Args:
        fov: Field of view in radians
        pixels: Image dimension in pixels (width or height)
    
    Returns:
        Focal length in pixels
    """
    return pixels / (2.0 * math.tan(fov / 2.0))


def fov_to_focal_batch(fov: torch.Tensor, 
                      pixels: torch.Tensor) -> torch.Tensor:
    """
    Batch version of fov_to_focal for tensors.
    
    Args:
        fov: Field of view in radians, shape (N,)
        pixels: Image dimensions in pixels, shape (N,)
    
    Returns:
        Focal lengths in pixels, shape (N,)
    """
    return pixels / (2.0 * torch.tan(fov / 2.0))


def focal_to_fov_batch(focal_length: torch.Tensor, 
                      pixels: torch.Tensor) -> torch.Tensor:
    """
    Batch version of focal_to_fov for tensors.
    
    Args:
        focal_length: Focal lengths in pixels, shape (N,)
        pixels: Image dimensions in pixels, shape (N,)
    
    Returns:
        Field of view in radians, shape (N,)
    """
    return 2.0 * torch.atan(pixels / (2.0 * focal_length))


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_points(points: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize points to zero mean and unit variance.
    
    Args:
        points: Tensor of shape (N, 3)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Normalized points and the transformation parameters
    """
    # Compute mean and standard deviation
    mean = points.mean(dim=0, keepdim=True)
    std = points.std(dim=0, keepdim=True)
    
    # Avoid division by zero
    std = torch.where(std < eps, torch.ones_like(std), std)
    
    # Normalize
    normalized = (points - mean) / std
    
    return normalized, mean, std


def compute_bounding_box(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute axis-aligned bounding box of points.
    
    Args:
        points: Tensor of shape (N, 3)
    
    Returns:
        min_corner: Minimum corner point
        max_corner: Maximum corner point
    """
    min_corner = torch.min(points, dim=0)[0]
    max_corner = torch.max(points, dim=0)[0]
    return min_corner, max_corner


def compute_point_density(points: torch.Tensor,
                         radius: float = 0.1) -> torch.Tensor:
    """
    Estimate local point density using radius search.
    
    Args:
        points: Tensor of shape (N, 3)
        radius: Search radius
    
    Returns:
        Density estimates for each point
    """
    from torch.cuda.amp import autocast
    
    N = points.shape[0]
    if N > 10000:
        # Use batch processing for large point clouds
        densities = torch.zeros(N, device=points.device)
        batch_size = 1000
        
        with autocast(enabled=points.device.type == 'cuda'):
            for i in range(0, N, batch_size):
                batch_end = min(i + batch_size, N)
                dists = torch.cdist(points[i:batch_end], points)
                densities[i:batch_end] = (dists < radius).sum(dim=1)
    else:
        # Process all points at once for small clouds
        dists = torch.cdist(points, points)
        densities = (dists < radius).sum(dim=1)
    
    return densities


def project_points_to_image(points_3d: torch.Tensor,
                          intrinsic_matrix: torch.Tensor,
                          extrinsic_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: 3D points in world coordinates, shape (N, 3)
        intrinsic_matrix: Camera intrinsic matrix, shape (3, 3)
        extrinsic_matrix: World to camera matrix, shape (4, 4)
    
    Returns:
        points_2d: 2D image coordinates, shape (N, 2)
        depths: Depth values (z in camera coordinates), shape (N,)
    """
    # Transform to camera coordinates
    points_cam = transform_points(points_3d, extrinsic_matrix)
    
    # Project to 2D
    points_hom = torch.matmul(points_cam, intrinsic_matrix.T)
    points_2d = points_hom[:, :2] / points_hom[:, 2:]
    
    # Get depths
    depths = points_cam[:, 2]
    
    return points_2d, depths


# ============================================================================
# Backward Compatible Functions (for compatibility with original code)
# ============================================================================

def geom_transform_points(points: torch.Tensor, 
                         transf_matrix: torch.Tensor) -> torch.Tensor:
    """Legacy function name for backward compatibility."""
    return transform_points(points, transf_matrix)


def getWorld2View(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Legacy function name for backward compatibility."""
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    return Rt


def getWorld2View2(R: np.ndarray, 
                  t: np.ndarray, 
                  translate: np.ndarray = np.array([0.0, 0.0, 0.0]), 
                  scale: float = 1.0) -> np.ndarray:
    """Legacy function name for backward compatibility."""
    Rt = getWorld2View(R, t)
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt.astype(np.float32)


def getProjectionMatrix(znear: float, 
                       zfar: float, 
                       fovX: float, 
                       fovY: float) -> torch.Tensor:
    """Legacy function name for backward compatibility."""
    return get_projection_matrix(znear, zfar, fovX, fovY)


def fov2focal(fov: float, pixels: float) -> float:
    """Legacy function name for backward compatibility."""
    return fov_to_focal(fov, pixels)


def focal2fov(focal: float, pixels: float) -> float:
    """Legacy function name for backward compatibility."""
    return focal_to_fov(focal, pixels)


class BasicPointCloud(NamedTuple):
    """Legacy data structure for backward compatibility."""
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    
    def to_pointcloud(self, device: str = 'cpu') -> PointCloud:
        """Convert to optimized PointCloud class."""
        return PointCloud.from_numpy(
            points=self.points,
            colors=self.colors,
            normals=self.normals,
            device=device
        )


# ============================================================================
# Type Aliases for Clearer Function Signatures
# ============================================================================

Matrix3x3 = torch.Tensor  # 3x3 matrix
Matrix4x4 = torch.Tensor  # 4x4 matrix
Vector3 = torch.Tensor    # 3D vector
Point3D = torch.Tensor    # 3D point (N, 3)
Point2D = torch.Tensor    # 2D point (N, 2)


# ============================================================================
# Main Execution Guard
# ============================================================================

if __name__ == "__main__":
    # Test the optimized functions
    print("Testing geometric transformation module...")
    
    # Create test data
    points = torch.randn(100, 3)
    R = torch.eye(3)
    t = torch.zeros(3)
    
    # Test transformations
    transform_mat = get_world_to_view_matrix(R, t)
    transformed = transform_points(points, transform_mat)
    
    print(f"Original points shape: {points.shape}")
    print(f"Transformed points shape: {transformed.shape}")
    print(f"Transformation successful: {torch.allclose(points, transformed, atol=1e-6)}")
    
    # Test FOV conversions
    fov_x = math.radians(60)  # 60 degrees
    focal = fov_to_focal(fov_x, 800)  # 800 pixels width
    fov_back = focal_to_fov(focal, 800)
    
    print(f"\nFOV conversions:")
    print(f"Original FOV: {math.degrees(fov_x):.2f}°")
    print(f"Computed focal length: {focal:.2f} pixels")
    print(f"Recovered FOV: {math.degrees(fov_back):.2f}°")
