import torch
import os
import sys
import numpy as np
import cv2
import json
import math
from tqdm import tqdm
from argparse import ArgumentParser

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from segmentation_1view.projection import GaussianProjector
from utils.general_utils import build_rotation, build_scaling_rotation
# from utils.system_utils import searchForMaxIteration
from plyfile import PlyData, PlyElement
from geometry import GaussianGeometry

def get_covariance_3d(gaussians, scaling_modifier = 1.0) -> torch.Tensor:
        """
        Compute the full 3D covariance matrix for each Gaussian

        Returns:
            torch.Tensor: tensor containing covariance matrices, with shape (N, 3, 3) 
        """

        # L = R * S
        scaling = gaussians.get_scaling
        rotation = gaussians.get_rotation
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        
        # Eq. 6: Sigma = L*L^T
        covariance = L @ L.transpose(1, 2) # Shape: (N, 3, 3)
        return covariance

def get_target_class_id(args, classes_json_path):
    """
    Retrieves the target class ID from the classes.json file based on the provided target class name.
    
    Args:
        args: Argument parser object containing the target_class attribute.
        classes_json_path (str): Path to the classes.json file.
        
    Returns:
        int: The target class ID.
        
    Raises:
        ValueError: If the target class is not found in the classes.json file.
    """
    with open(classes_json_path, 'r') as f:
            classes_map = json.load(f)
            
    # Invert map: Name -> ID
    name_to_id = {v: int(k) for k, v in classes_map.items()}
    
    if args.target_class not in name_to_id:
        raise ValueError(f"Target class '{args.target_class}' not found in classes.json. Available: {list(name_to_id.keys())}")
        
    target_id = name_to_id[args.target_class]
    print(f"Targeting Class: '{args.target_class}' (ID: {target_id})")
    
    return target_id


def main(args):

    # Define the gaussians
    gaussians = GaussianModel(sh_degree=args.sh_degree, use_labels=True)
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)

    cov3D = get_covariance_3d(gaussians)
    # print("Covariance shape:", cov3D.shape)

    scene = Scene(args, gaussians, load_iteration=args.loaded_iter, shuffle=False)

    # Prepare global votes tensor
    total_gaussians = gaussians.get_xyz.shape[0]
    global_votes = torch.zeros((total_gaussians,), device=args.device, dtype=torch.float32)

    classes_json_path = os.path.join(args.mask_dir, "classes.json")
    target_id = get_target_class_id(args, classes_json_path)


    # Filter scene cameras to find these views
    train_cameras = scene.getTrainCameras()
    target_cameras = []
    
    for cam in train_cameras:

        basename = os.path.basename(cam.image_name)
        name_no_ext = os.path.splitext(basename)[0]
        name = f"{name_no_ext}.png"
        
        conf_full_path = os.path.join(args.mask_dir, "confidence", name)
        if os.path.exists(conf_full_path):
            target_cameras.append((cam, {
                "semantic": os.path.join("semantic", name),
                "confidence": os.path.join("confidence", name)
            }))
            
    print(f"Matched {len(target_cameras)} cameras in the scene.")

    for i, (cam, mask_info) in enumerate(target_cameras):
        if i==1:
            print(f"\n--- Processing View {i+1}/{len(target_cameras)}: {cam.image_name} ---")
            sem_path = os.path.join(args.mask_dir, mask_info["semantic"])
            semantic_img = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED) # (H, W)
            semantic_mask = torch.tensor(semantic_img, dtype=torch.long, device=args.device)
            semantic_height, semantic_width = semantic_img.shape

            # Projection
            projector = GaussianProjector(cam)
            projection_results = projector.project(gaussians.get_xyz, cov3D)
            # print(f"semantic_dimensions: {semantic_width} x {semantic_height}")
            # print(f"camera dimensions: {cam.image_width} x {cam.image_height}")
            # print(projection_results['indices'])
            gaussians.set_mask_index(projection_results['indices'])
            # gaussians.save_ply(os.path.join(args.output_dir, f"gaussians_with_masks_view_again{i+1}.ply"))

            means2D = projection_results['means2D'] # (M, 2)
            cov2D = projection_results['cov2D'] # (M, 2, 2)
            depths = projection_results['depths']
            indices = projection_results['indices'] # (M_vis,) global indices

            opacities = gaussians.get_opacity[indices] # (M_vis,)

            '''
            Equation 4 but projected in 2D
            To make the rasterization loop fast, we don't want to calculate the matrix inverse for every single pixel. We calculate it once per Gaussian and store it.
            Sigma is semidefinite positive, so simmetric (and with non negative eigenvalues)
            This is the inverse of the covariance matrix, which is used in the exponent of the Gaussian formula. We can precompute it for each Gaussian and store it as "conic" parameters (A, B, C) for the ellipse representation.
            As it's symmetric, we have only 3 unique values: [[A, B], [B, C]] where A = inv_cov2D[0,0], B = inv_cov2D[0,1], C = inv_cov2D[1,1]
            '''

            det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 0, 1]
            det_inv = 1.0 / (det + 1e-6)
            conic = torch.stack([
                cov2D[:, 1, 1] * det_inv,     
                -cov2D[:, 0, 1] * det_inv,    
                cov2D[:, 0, 0] * det_inv      
            ], dim=1)

            '''
            A Gaussian theoretically stretches to infinity, but in practice, its energy is negligible after 3 standard deviations. 
            To find the "width" of the Gaussian, we need the lengths of its major and minor axes. 
            Mathematically, these lengths are the square roots of the Eigenvalues of the covariance matrix.
            '''

            # We can compute the eigenvalues of the 2D covariance matrix using the formula for 2x2 matrices:
            trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
            sqrt_term = torch.sqrt(trace * trace - 4 * det)
            eig1 = 0.5 * (trace + sqrt_term)
            eig2 = 0.5 * (trace - sqrt_term)
            radius = torch.ceil(3.0 * torch.sqrt(torch.max(eig1, eig2))) # (M_vis,)

            # Sorting the gaussians by depth
            sort_indices = torch.argsort(depths)
            means2D = means2D[sort_indices]
            conic = conic[sort_indices]
            opacities = opacities[sort_indices]
            radius = radius[sort_indices]
            sorted_original_indices = indices[sort_indices]

            # Frustum Culling: Filter out gaussians that don't overlap with the image
            # The projection only checks z > znear, so we need to filter X/Y bounds to match the view
            # Gaussian Center must be inside the image coordinates. This removes Gaussians centered off-screen that bleed in due to large radii
            intersect_x = (means2D[:, 0] >= 0) & (means2D[:, 0] < semantic_width)
            intersect_y = (means2D[:, 1] >= 0) & (means2D[:, 1] < semantic_height)
            in_frustum = intersect_x & intersect_y

            means2D = means2D[in_frustum]
            radius = radius[in_frustum]
            sorted_original_indices = sorted_original_indices[in_frustum]
            
            


















if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="../example_data/output/truck_test")
    parser.add_argument("--source_path", default="../example_data/data/tandt/truck")
    parser.add_argument("--mask_dir", default="./data/2D_mask/02-04_23-18", help="Directory containing masks.json (e.g. data/2D_mask/timestamp)")
    parser.add_argument("--output_dir", default="./data/output/02-04_23-18", help="Directory to save outputs")
    parser.add_argument("--num_classes", type=int, default=81)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--target_class", type=str, default="truck", help="Only one object at a time can be segmented. The name must match one of the classes in the YOLO model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load tensors on")
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration number to load from the model (e.g. 1000)")
    
    args = get_combined_args(parser)
    
    with torch.no_grad():
        main(args)


    






























