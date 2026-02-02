import torch
import cv2
import os
import sys
import numpy as np
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams
from scene import Scene, GaussianModel
from segmentation.geometry import GaussianGeometry
from segmentation.projection import GaussianProjector
from utils.visualization_utils import draw_projected_gaussians

def main(model_path, source_path, iteration, view_idx, num_gaussians):
    
    class PipelineParams: # Necessary to load the scene
        def __init__(self):
            self.source_path = source_path
            self.model_path = model_path
            self.images = "images"
            self.depths = ""
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = True
            self.train_test_exp = False
            self.sh_degree = 3
            
    geometry = GaussianGeometry(model_path, iteration, device="cuda")
    dataset_args = PipelineParams()
    
    try:
        scene = Scene(dataset_args, geometry.gaussians, load_iteration=iteration, shuffle=False)
    except Exception as e:
        print(f"Error loading Scene: {e}")
        return

    # Selecting a camera
    cameras = scene.getTrainCameras()
    if view_idx >= len(cameras):
        view_idx = 0
    camera = cameras[view_idx]

    # Project gaussians
    projector = GaussianProjector(camera)
    means3D = geometry.xyz
    cov3D = geometry.get_covariance_3d()
    projection_results = projector.project(means3D, cov3D)
    
    means2D = projection_results['means2D']
    cov2D = projection_results['cov2D']
    depths = projection_results['depths']
    indices = projection_results['indices']

    # Visualization
    image_tensor = camera.original_image # Camera object has original_image, but it's a tensor (C, H, W)
    image_np = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8) # Convert to numpy (H, W, C) range 0-255
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    final_image = draw_projected_gaussians(
        image_bgr, 
        means2D, 
        cov2D, 
        num_gaussians=num_gaussians
    )
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projections_images")
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, f"projection_cam{view_idx}_n{num_gaussians}.jpg")
    cv2.imwrite(output_filename, final_image)


if __name__ == "__main__":

    # Default paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(project_root, "example_data/output/truck_test")
    default_source_path = os.path.join(project_root, "example_data/data/tandt/truck")
    
    main(default_model_path, default_source_path, 30000, 0, 50)
