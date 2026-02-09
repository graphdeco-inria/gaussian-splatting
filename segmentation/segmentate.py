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
    
    # Calculate scalar sizes (max scale axis) for all gaussians
    scales = gaussians.get_scaling
    scalar_sizes = scales.max(dim=1).values
    '''
    print(f"Gaussian sizes (max scale): Min={scalar_sizes.min().item():.5f}, Max={scalar_sizes.max().item():.5f}")

    # Statistical analysis of scales (Deciles)
    deciles = torch.quantile(scalar_sizes, torch.linspace(0.1, 0.9, 9, device=args.device))
    print("\n--- Gaussian Scale Distribution (Deciles) ---")
    for i, decile_val in enumerate(deciles):
        print(f"{(i+1)*10}th percentile: {decile_val.item():.5f}")
    print("-------------------------------------------\n")
    '''

    cov3D = get_covariance_3d(gaussians)
    # print("Covariance shape:", cov3D.shape)

    scene = Scene(args, gaussians, load_iteration=args.loaded_iter, shuffle=False)

    # Prepare global votes tensor
    total_gaussians = gaussians.get_xyz.shape[0]
    global_votes = torch.zeros((total_gaussians,), device=args.device, dtype=torch.float32)

    global_weights = torch.zeros((total_gaussians,), device=args.device, dtype=torch.float32)

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
        print(f"\n--- Processing View {i+1}/{len(target_cameras)}: {cam.image_name} ---")
        sem_path = os.path.join(args.mask_dir, mask_info["semantic"])
        semantic_img = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED) # (H, W)
        semantic_mask = torch.tensor(semantic_img, dtype=torch.long, device=args.device)
        semantic_height, semantic_width = semantic_img.shape

        # Compute Confidence for Target Class
        conf_path = os.path.join(args.mask_dir, mask_info["confidence"])
        confidence_img = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)
        if confidence_img.shape[:2] != (cam.image_height, cam.image_width):
            confidence_img = cv2.resize(confidence_img, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
        confidence_mask = torch.tensor(confidence_img, dtype=torch.float32, device=args.device)
        max_confidence = confidence_mask.max()
        if max_confidence > 1.0:
                confidence_mask /= max_confidence
        
        target_mask_view = (semantic_mask == target_id)
        if not target_mask_view.any():
            continue
        
        # Extract scalar confidence for this view
        view_confidence = confidence_mask[target_mask_view].mean()

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
        view_sizes = scalar_sizes[indices] # (M_vis,)

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
        view_sizes = view_sizes[sort_indices]

        # Frustum Culling: Filter out gaussians that don't overlap with the image
        # The projection only checks z > znear, so we need to filter X/Y bounds to match the view
        # Gaussian Center must be inside the image coordinates. This removes Gaussians centered off-screen that bleed in due to large radii
        intersect_x = (means2D[:, 0] >= 0) & (means2D[:, 0] < semantic_width)
        intersect_y = (means2D[:, 1] >= 0) & (means2D[:, 1] < semantic_height)
        in_frustum = intersect_x & intersect_y

        means2D = means2D[in_frustum]
        radius = radius[in_frustum]
        conic = conic[in_frustum]
        opacities = opacities[in_frustum]
        sorted_original_indices = sorted_original_indices[in_frustum]
        view_sizes = view_sizes[in_frustum]
        
        num_visible = means2D.shape[0]
        view_weights_sorted = torch.zeros((num_visible,), device=args.device, dtype=torch.float32)

        # Rasterization:
        BLOCK_SIZE = args.raster_block_size
        grid_columns = (semantic_width + BLOCK_SIZE - 1) // BLOCK_SIZE # It ensures that if the image width isn't perfectly divisible by 16, you still get that final partial tile at the edge
        grid_rows = (semantic_height + BLOCK_SIZE - 1) // BLOCK_SIZE 
        
        # Convert the Gaussian's 2D position and size from Pixel Coordinates to Tile Coordinates
        grid_min_x = ((means2D[:, 0] - radius).clamp(min=0) / BLOCK_SIZE).int()
        grid_min_y = ((means2D[:, 1] - radius).clamp(min=0) / BLOCK_SIZE).int()
        grid_max_x = ((means2D[:, 0] + radius).clamp(max=semantic_width-1) / BLOCK_SIZE).int()
        grid_max_y = ((means2D[:, 1] + radius).clamp(max=semantic_height-1) / BLOCK_SIZE).int()
        
        # The grid_min and grid_max tensors now represent the bounding tiles for each Gaussian in terms of tile indices
        grid_min = torch.stack([grid_min_x, grid_min_y], dim=1)
        grid_max = torch.stack([grid_max_x, grid_max_y], dim=1)

        for row_tile in range(grid_rows):
            for column_tile in range(grid_columns):
                # Find which Gaussians have bounding tiles that include this tile
                in_tile = (grid_min[:, 0] <= column_tile) & (column_tile <= grid_max[:, 0]) & (grid_min[:, 1] <= row_tile) & (row_tile <= grid_max[:, 1])
                gaussians_in_tile = torch.nonzero(in_tile).squeeze(1)

                if gaussians_in_tile.shape[0] == 0:
                    continue
                    
                # For each Gaussian that overlaps with this tile, we would calculate its contribution to the pixels in the tile
                # This is where we would apply the Gaussian formula using the precomputed "conic" parameters and opacities
                # We would also check the semantic_mask to see if the pixel belongs to the target class and accumulate votes accordingly
                # This part is complex and would involve iterating over the pixels in the tile and applying the Gaussian formula, which is why we want to minimize the number of Gaussians we check per tile
                # The output would be a per-pixel vote for the target class, which we would accumulate in the global_votes tensor using the original indices of the Gaussians

                tile_means = means2D[gaussians_in_tile]
                tile_conics = conic[gaussians_in_tile]
                tile_opacities = opacities[gaussians_in_tile]
                tile_sizes = view_sizes[gaussians_in_tile]
                
                pix_min_x = column_tile * BLOCK_SIZE
                pix_min_y = row_tile * BLOCK_SIZE
                pix_max_x = min(pix_min_x + BLOCK_SIZE, semantic_width)
                pix_max_y = min(pix_min_y + BLOCK_SIZE, semantic_height)
                
                y_range = torch.arange(pix_min_y, pix_max_y, device=args.device)
                x_range = torch.arange(pix_min_x, pix_max_x, device=args.device)
                grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
                
                flat_y = grid_y.flatten()
                flat_x = grid_x.flatten()

                # Equation 4 in the paper, but using the conic parameters and opacities, and applied to the pixels in this tile
                # For each pixel in the tile, calculate its distance to the Gaussian centers and apply the Gaussian formula using the conic parameters
                dx = flat_x.unsqueeze(0) - tile_means[:, 0].unsqueeze(1)
                dy = flat_y.unsqueeze(0) - tile_means[:, 1].unsqueeze(1)
                
                # This calculates exactly how intense the Gaussian is at those specific distances
                power = -0.5 * (tile_conics[:, 0].unsqueeze(1) * dx**2 + 
                                tile_conics[:, 2].unsqueeze(1) * dy**2) - \
                                tile_conics[:, 1].unsqueeze(1) * dx * dy
                
                # The opacity of the Gaussian modulates its contributions
                # alpha shape: (N_Gaussians, N_Pixels)
                alpha = tile_opacities.view(-1, 1) * torch.exp(power.clamp(max=0)) # Computes the raw opacity of a Gaussian at a specific pixel
                
                transmission = 1.0 - alpha 
                accummulated_transmission = torch.cumprod(transmission, dim=0) # Multiplies the transmission values down the list. Tells you how much light leaves the current layer

                # We need to know how much light reached the current layer
                ones = torch.ones((1, alpha.shape[1]), device=args.device) # The first layer receives 100% of the light, then we accumulate the transmission of the previous layers to know how much light reaches the current layer
                T = torch.cat([ones, accummulated_transmission[:-1]], dim=0)
                weights = alpha * T 

                tile_classes = semantic_mask[flat_y, flat_x] 
                unique_classes = torch.unique(tile_classes)

                if target_id not in unique_classes:
                    continue # No pixels of the target class in this tile, skip the voting
                else:
                    pixel_mask = (tile_classes == target_id)
                    
                    # Weight Calculation: alpha * T * view_confidence
                    # Weighted by inverse size to punish large gaussians
                    size_penalty_val = (tile_sizes * args.size_penalty) ** args.alpha
                    weighted_contribution = (weights * view_confidence) / size_penalty_val.view(-1, 1) # Reshape to a column vector for broadcasting
                    
                    class_votes = weighted_contribution[:, pixel_mask].sum(dim=1)          
                    view_weights_sorted[gaussians_in_tile] += class_votes


        '''
        Its possition in the loop may change
        view_max = view_weights_sorted.max()
        if view_max > 0:
        view_weights_sorted /= view_max

        '''

        global_weights[sorted_original_indices] += view_weights_sorted
        del view_weights_sorted, means2D, conic, radius, grid_min, grid_max
        torch.cuda.empty_cache()
                    
    # Threshold: hyperparameter
    threshold = len(target_cameras)*args.beta
    print(f"Applying threshold: {threshold} (Cameras: {len(target_cameras)})")
    
    pred_labels = torch.zeros((total_gaussians,), dtype=torch.long, device=args.device)
    final_mask = global_weights > threshold
    pred_labels[final_mask] = target_id
    
    print(f"Labeled {final_mask.sum()} gaussians as {target_id}")
    
    # Map to Gaussian Property
    gaussians._labels = pred_labels.unsqueeze(1).int()

    # Save the new Gaussians with the target label as a PLY file
    # gaussians.set_labels(pred_labels) # Method does not exist, setting _labels directly above
    gaussians.set_mask_index(final_mask.nonzero(as_tuple=True)[0])
    gaussians.save_ply(f"{args.output_dir}/labeled_gaussians_{target_id}.ply")

    # Save all the Gaussians with their weights as a PLY file (for visualization)
    # Custom save logic to support 'weight' property
    '''
    with torch.no_grad():
        w_path = f"{args.output_dir}/gaussians_with_weights_{target_id}.ply"
        os.makedirs(os.path.dirname(w_path), exist_ok=True)
        
        xyz = gaussians._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = gaussians._opacity.detach().cpu().numpy()
        scale = gaussians._scaling.detach().cpu().numpy()
        rotation = gaussians._rotation.detach().cpu().numpy()
        
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(f_dc.shape[1]): l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]): l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]): l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]): l.append('rot_{}'.format(i))
        
        dtype_full = [(attribute, 'f4') for attribute in l]
        dtype_full.append(('weight', 'f4'))

        weights_np = global_weights.detach().cpu().numpy().reshape(-1, 1)
        
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, weights_np), axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(w_path)

    print(f"Saved labeled gaussians and weights to {args.output_dir}")
    '''




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
    parser.add_argument("--raster_block_size", type=int, default=16, help="Block size for rasterization. Larger blocks are faster but less precise.")
    parser.add_argument("--alpha", type=float, default=2.0, help="Exponent for size punishment. Higher alpha penalizes larger Gaussians more.")
    parser.add_argument("--beta", type=float, default=0.05, help="Threshold factor for labeling Gaussians. Higher means more conservative segmentation.")
    parser.add_argument("--size_penalty", type=float, default=100.0, help="Base multiplier for size punishment. Scales the Gaussian size before exponentiation.")
    
    args = get_combined_args(parser)
    
    with torch.no_grad():
        main(args)
