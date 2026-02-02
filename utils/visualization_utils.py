# Added by Iván Verdugo Guerra
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scene.cameras import Camera

def compute_ellipse_geometry(cov: np.ndarray):
    """
    Extracts ellipse parameters (angle, width, height) from a 2D covariance matrix
    """
    # Eigen decomposition to get ellipse parameters
    vals, vecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # angle = arctan2(y, x)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    
    # Calculate axis lengths
    width = 2 * np.sqrt(vals[0])  # Major axis length
    height = 2 * np.sqrt(vals[1]) # Minor axis length
    
    return angle, width, height

def draw_projected_gaussians(image: np.ndarray, means2D: torch.Tensor, cov2D: torch.Tensor, 
                           num_gaussians: int = 100):
    """
    image: the background image in RGB, shaped (H, W, 3)
    means2D: projected centers, shaped (M, 2)
    cov2D: projected 2D covariance matrices, with shape (M, 2, 2)
    num_gaussians: amount of gaussians to draw
    """

    vis_img = image.copy()
    means = means2D.detach().cpu().numpy() # Converting tensors to numpy arrays
    covs = cov2D.detach().cpu().numpy()
    
    count = min(num_gaussians, means.shape[0]) # In case a lot of gaussians are requested
    
    for i in range(count):
        mean = means[i]
        cov = covs[i]
        
        angle, width, height = compute_ellipse_geometry(cov)
        color = (0, 0, 255)
        
        # Draw ellipse
        try:
            center = (int(mean[0]), int(mean[1]))
            axes = (int(width), int(height))
            cv2.ellipse(vis_img, center, axes, angle, 0, 360, color, 1)
        except Exception as e:
            pass
            
    return vis_img