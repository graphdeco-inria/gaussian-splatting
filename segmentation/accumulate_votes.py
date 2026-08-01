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
from segmentation.projection import GaussianProjector
from segmentation.threshold_labels import apply_threshold
from utils.general_utils import build_rotation, build_scaling_rotation
from plyfile import PlyData, PlyElement

def get_covariance_3d(gaussians, scaling_modifier = 1.0) -> torch.Tensor:
        """
        Compute the full 3D covariance matrix for each Gaussian

        Returns:
            torch.Tensor: tensor containing covariance matrices, with shape (N, 3, 3) 
        """

        # L = R * S
        scaling = gaussians.get_scaling # Stretch along the axes of the Gaussian. This is a diagonal matrix with the scaling factors for each axis
        rotation = gaussians.get_rotation # Rotation of the Gaussian in 3D space, represented as a quaternion (w, x, y, z)
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        
        # Eq. 6: Sigma = L*L^T
        covariance = L @ L.transpose(1, 2) # Shape: (N, 3, 3)
        return covariance

def get_target_class_id(args, classes_json_path):
    """
    Retrieves the target class ID from the classes.json file based on the provided target class name
    args contains the target_class attribute
    """

    with open(classes_json_path, 'r') as f:
            classes_map = json.load(f)
            
    # It is stores as ID -> name, so we need to invert the map to Name -> ID
    name_to_id = {v: int(k) for k, v in classes_map.items()}
    
    if args.target_class not in name_to_id:
        raise ValueError(f"Target class {args.target_class} not found in classes.json")
        
    target_id = name_to_id[args.target_class]    
    return target_id


def main(args):

    # Define the gaussians
    gaussians = GaussianModel(sh_degree=args.sh_degree, use_labels=True)
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # Calculate the size of all gaussians
    scales = gaussians.get_scaling # tensor of shape (N, 3) containing the three scaling values for every Gaussian.
    # Size measure used by the vote penalty. max punishes flat
    # panel gaussians, like tv/table surfaces, like isotropic blobs, gmean
    # ,characteristic length (s1*s2*s3)^(1/3), relieves panels, l2, sqrt of
    # sum of squares, punishes them even more.
    if args.size_measure == "gmean":
        scalar_sizes = torch.exp(scales.log().mean(dim=1))
    elif args.size_measure == "l2":
        scalar_sizes = scales.norm(dim=1)
    else:
        scalar_sizes = scales.max(dim=1).values

    '''
    Analysis of gaussian sizes to select the default size penalty

    print(f"Gaussian sizes (max scale): Min={scalar_sizes.min().item():.5f}, Max={scalar_sizes.max().item():.5f}")

    # Statistical analysis of scales (Deciles)
    deciles = torch.quantile(scalar_sizes, torch.linspace(0.1, 0.9, 9, device=args.device))
    print("\n--- Gaussian Scale Distribution (Deciles) ---")
    for i, decile_val in enumerate(deciles):
        print(f"{(i+1)*10}th percentile: {decile_val.item():.5f}")
    print("-------------------------------------------\n")
    '''

    cov3D = get_covariance_3d(gaussians) # shape (N, 3, 3), as there is one 3D covariance matrix per Gaussian. This will be used to project the 3D Gaussians into 2D for each camera view.
    # print("Covariance shape:", cov3D.shape)
    # Load the Scene with camera images on CPU: vote accumulation only needs
    # poses/intrinsics (masks are read from mask_dir), and holding e.g. 385
    # full-res DSLR images on a shared GPU (~8GB) OOMs at Scene load. Camera
    # matrices still go to the GPU explicitly (they are small).
    '''
    _orig_data_device = getattr(args, "data_device", "cuda")
    args.data_device = "cpu"
    '''
    scene = Scene(args, gaussians, load_iteration=args.loaded_iter, shuffle=False)
    '''
    args.data_device = _orig_data_device

    # Free whatever image tensors were still materialized, and move the small
    # pose/projection matrices to the GPU explicitly (they are needed by the
    # GaussianProjector and were loaded on CPU along with the images)
    for _cam in scene.getTrainCameras():
        for _attr in ("original_image", "alpha_mask", "gt_alpha_mask"):
            if hasattr(_cam, _attr):
                setattr(_cam, _attr, None)
        for _attr in ("world_view_transform", "projection_matrix",
                      "full_proj_transform", "camera_center"):
            _t = getattr(_cam, _attr, None)
            if _t is not None and torch.is_tensor(_t) and not _t.is_cuda:
                setattr(_cam, _attr, _t.cuda())
        # the GaussianProjector allocates on camera.data_device
        if getattr(_cam, "data_device", "cuda") != "cuda":
            _cam.data_device = "cuda"
    torch.cuda.empty_cache()
    '''

    # Prepare global votes tensor
    total_gaussians = gaussians.get_xyz.shape[0] # gaussians.get_xyz has shape (N, 3), where N is the total number of Gaussians in the model.

    # Initialize a tensor to accumulate votes for each Gaussian across all camera views
    global_weights = torch.zeros((total_gaussians,), device=args.device, dtype=torch.float32)

    # Path to load the classes.json file, which contains a mapping of class IDs to class names detected by YOLO across the dataset
    classes_json_path = os.path.join(args.mask_dir, "classes.json")
    target_id = get_target_class_id(args, classes_json_path) # The target class name is given in args.target_class, and we retrieve its corresponding ID from the classes.json file


    # Iterate through scene cameras to find these views
    train_cameras = scene.getTrainCameras()
    target_cameras = [] # Cameras that have a corresponding 2D mask
    
    for cam in train_cameras:
        basename = os.path.basename(cam.image_name)
        name_no_ext = os.path.splitext(basename)[0]
        name = f"{name_no_ext}.png" # The YOLO script saves all the generated 2D masks as PNG files
        
        conf_full_path = os.path.join(args.mask_dir, "confidence", name)
        if os.path.exists(conf_full_path):
            target_cameras.append((cam, {
                "semantic": os.path.join("semantic", name), # Constructs the path of the semantic mask for this camera view
                "confidence": os.path.join("confidence", name) # Constructs the path of the confidence mask for this camera view
            }))
            
    print(f"Matched {len(target_cameras)} cameras in the scene.")

    class_views = 0  # Views where the target class actually appears


    # Iterate through the matched cameras and accumulate votes for the target class
    for i, (cam, mask_info) in enumerate(target_cameras):
        print(f"\n  Processing View {i+1}/{len(target_cameras)}: {cam.image_name}")

        # Getting semantic data
        sem_path = os.path.join(args.mask_dir, mask_info["semantic"]) # Concatenating the mask directory with the semantic mask filename to get the full path of the semantic mask for this camera view
        semantic_img = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED) # (H, W)
        
        # Resize semantic mask to match camera dimensions
        if semantic_img.shape[:2] != (cam.image_height, cam.image_width):
            semantic_img = cv2.resize(semantic_img, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
            
        semantic_mask = torch.tensor(semantic_img, dtype=torch.long, device=args.device)
        semantic_height, semantic_width = semantic_img.shape

        # Getting confidence data
        conf_path = os.path.join(args.mask_dir, mask_info["confidence"])
        confidence_img = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)

        # Resize confidence mask to match camera dimensions
        if confidence_img.shape[:2] != (cam.image_height, cam.image_width):
            confidence_img = cv2.resize(confidence_img, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)

        confidence_mask = torch.tensor(confidence_img, dtype=torch.float32, device=args.device)
        
        # Normalize to [0, 1]. Since it was saved as uint8, we divide by 255.0
        if confidence_mask.max() > 1.0:
            confidence_mask /= 255.0

        # Check if the target class is present in the semantic mask for this camera view
        target_mask_view = (semantic_mask == target_id)
        if not target_mask_view.any():
            continue
        class_views += 1

        # Projection of 3D Gaussians into 2D camera space
        projector = GaussianProjector(cam)

        # projection_results is a map with means2D, cov2D, depths, and indices of the Gaussians that are visible in this camera view
        projection_results = projector.project(gaussians.get_xyz, cov3D)

        # Index to see which Gaussians are visible in this camera view. This is used to filter the global weights tensor later on
        gaussians.set_mask_index(projection_results['indices'])
        # gaussians.save_ply(os.path.join(args.output_dir, f"gaussians_with_masks_view_again{i+1}.ply")) # To save and later see the Gaussians that are visible in this camera view

        # print(f"  Visible Gaussians in this view: {projection_results['means2D'].shape[0]} / {total_gaussians}")
        means2D = projection_results['means2D'] # (M, 2) being M the amount of Gaussians that are visible in this camera view
        cov2D = projection_results['cov2D'] # (M, 2, 2)
        depths = projection_results['depths'] # (M,)
        indices = projection_results['indices'] # (M,)


        opacities = gaussians.get_opacity[indices] # (M,)
        view_sizes = scalar_sizes[indices] # (M,)
        # print(f"  Projected 2D means shape: {means2D.shape}, Covariance shape: {cov2D.shape}, Depths shape: {depths.shape}, Indices shape: {indices.shape}, opacities shape: {opacities.shape}, view_sizes shape: {view_sizes.shape}")

        '''
        Equation 4 but projected in 2D
        To make the rasterization loop fast, we don't want to calculate the inverse matrix for every pixel. We calculate it once per Gaussian and store it.
        Sigma is semidefinite positive, so simmetric (and with non negative eigenvalues)
        This is the inverse of the covariance matrix, which is used in the exponent of the Gaussian formula. We can precompute it for each Gaussian and store it as conic parameters (A, B, C) for the ellipse representation.
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
        radius = torch.ceil(3.0 * torch.sqrt(torch.max(eig1, eig2))) # (M,)

        # Sorting the gaussians by depth, before focusing on any tile
        sort_indices = torch.argsort(depths)
        means2D = means2D[sort_indices]
        conic = conic[sort_indices]
        opacities = opacities[sort_indices]
        radius = radius[sort_indices]

        # Now we save how the original indexes of the Gaussians are ordered after sorting by depth
        # With sort_indices we get the order of the Gaussians once culled and sorted by depth. Indices is the original index of the Gaussians in the global model
        sorted_original_indices = indices[sort_indices] # Save the original indexes by ordering now that we have sorted the Gaussians by depth. While we work with tiles, we will accumulate votes in a temporary depth-sorted tensor specifically for this view, and then we will add those votes to the global weights tensor using these original indices when all the tiles have been processed.
        view_sizes = view_sizes[sort_indices]

        # Frustum culling: filter out gaussians that don't overlap with the image
        # The projection only checks z > znear, so we need to filter the x and y bounds to match the view
        # Gaussian center must be inside the image coordinates. This removes Gaussians centered off-screen that bleed in due to large radius. This was a massive improvement
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
        BLOCK_SIZE = args.raster_block_size # By default, we use 16x16 pixel blocks for rasterization. This means we will check which Gaussians potentially affect each block, and then only compute the Gaussian formula for the pixels in that block and those Gaussians
        grid_columns = (semantic_width + BLOCK_SIZE - 1) // BLOCK_SIZE # It ensures that if the image width is not perfectly divisible by 16, you still get that final partial tile at the edge
        grid_rows = (semantic_height + BLOCK_SIZE - 1) // BLOCK_SIZE 
        
        # Convert the Gaussian's 2D position and size from pixel coordinates to tile coordinates
        grid_min_x = ((means2D[:, 0] - radius).clamp(min=0) / BLOCK_SIZE).int() # We use clamp to stay inside valid image/grid dimensions
        grid_min_y = ((means2D[:, 1] - radius).clamp(min=0) / BLOCK_SIZE).int()
        grid_max_x = ((means2D[:, 0] + radius).clamp(max=semantic_width-1) / BLOCK_SIZE).int()
        grid_max_y = ((means2D[:, 1] + radius).clamp(max=semantic_height-1) / BLOCK_SIZE).int()
        
        # The grid_min and grid_max tensors now represent the bounding tiles for each Gaussian in terms of tile indices
        grid_min = torch.stack([grid_min_x, grid_min_y], dim=1) # Coordinate of the lower-left corner of the bounding tile for each Gaussian. Will be something like (3, 0) for a Gaussian that starts in tile column 3, row 0
        grid_max = torch.stack([grid_max_x, grid_max_y], dim=1) # Coordinate of the upper-right corner of the bounding tile for each Gaussian. Might be something like (5, 2) for the Gaussian if its center is in tile (4, 1) and its radius extends to tiles (3, 0) and (5, 2)

        for row_tile in range(grid_rows):
            for column_tile in range(grid_columns):
                # Find which Gaussians have bounding tiles that include this tile
                in_tile = (grid_min[:, 0] <= column_tile) & (column_tile <= grid_max[:, 0]) & (grid_min[:, 1] <= row_tile) & (row_tile <= grid_max[:, 1])
                # Check which Gaussians are in this tile. This is a boolean mask of shape (M,) where M is the number of Gaussians that passed the frustum culling
                gaussians_in_tile = torch.nonzero(in_tile).squeeze(1) # Transforms the shape from (K, 1) into a tensor of shape (K,), squeezes dimension 1

                if gaussians_in_tile.shape[0] == 0:
                    continue
                    
                '''
                For each Gaussian that overlaps with this tile, calculate its contribution to the pixels in the tile.
                This applies the 2D Gaussian formula using precomputed conic parameters and opacities, 
                accumulates the pixel-wise blending weights, checks the semantic mask against the target class, 
                and accumulates votes into the global weights tensor using the original indices.
                '''

                tile_means = means2D[gaussians_in_tile]
                tile_conics = conic[gaussians_in_tile]
                tile_opacities = opacities[gaussians_in_tile]
                tile_sizes = view_sizes[gaussians_in_tile]
                
                # Obtain the boundaries of the current tile in pixel coordinates
                pix_min_x = column_tile * BLOCK_SIZE
                pix_min_y = row_tile * BLOCK_SIZE
                pix_max_x = min(pix_min_x + BLOCK_SIZE, semantic_width)
                pix_max_y = min(pix_min_y + BLOCK_SIZE, semantic_height)

                '''
                y is before x because the first dimension of the image is height, y, and the second dimension is width, x.

                If y_range is tensor([10, 11, 12]), then grid_y will be:
                tensor([[10, 10, 10],
                        [11, 11, 11],
                        [12, 12, 12]])
                And flat_y will be tensor([10, 10, 10, 11, 11, 11, 12, 12, 12])

                Similarly, if x_range is tensor([20, 21, 22]), then grid_x will be:
                tensor([[20, 21, 22],
                        [20, 21, 22],
                        [20, 21, 22]])
                And flat_x will be tensor([20, 21, 22, 20, 21, 22, 20, 21, 22])
                '''
                
                # Create a grid of pixel coordinates for the current tile
                y_range = torch.arange(pix_min_y, pix_max_y, device=args.device)
                x_range = torch.arange(pix_min_x, pix_max_x, device=args.device)

                # grid_y becomes a 2D grid where every row is identical, grid_x becomes a 2D grid where every column is identical
                grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij') # Takes two 1D ranges and expands them into a 2D coordinate grid.
                
                # Write the grid in a whole 1D array to make it easier to compute the Gaussian formula for all pixels in the tile at once
                flat_y = grid_y.flatten()
                flat_x = grid_x.flatten()

                '''
                Equation 4 in the paper, but using the conic parameters and opacities, and applied to the pixels in this tile
                For each pixel in the tile, calculate its distance to the Gaussian centers and apply the Gaussian formula using the conic parameters
                unsqueeze(0) is used to expand the dimensions of the pixel coordinates so that they can be broadcasted against the Gaussian parameters, (, N_Pixels_in_tile) -> (1, N_Pixels_in_tile)
                unsqueeze(1) is used to expand the dimensions of the Gaussian parameters so that they can be broadcasted against the pixel coordinates, (N_Gaussians_in_tile, ) -> (N_Gaussians_in_tile, 1)
                Now the shapes are compatible for broadcasting, resulting in a tensor of shape (N_Gaussians_in_tile, N_Pixels_in_tile)
                '''

                dx = flat_x.unsqueeze(0) - tile_means[:, 0].unsqueeze(1) # Calculates the horizontal distance from every Gaussian to every pixel simultaneously
                dy = flat_y.unsqueeze(0) - tile_means[:, 1].unsqueeze(1)
                
                # This calculates how intense the Gaussian is at those specific distances
                power = -0.5 * (tile_conics[:, 0].unsqueeze(1) * dx**2 + 
                                tile_conics[:, 2].unsqueeze(1) * dy**2) - \
                                tile_conics[:, 1].unsqueeze(1) * dx * dy
                
                # The opacity of the Gaussian modulates its contributions
                # alpha shape: (N_Gaussians_in_tile, N_Pixels_in_tile)
                alpha = tile_opacities.view(-1, 1) * torch.exp(power.clamp(max=0)) # Computes the opacity of a Gaussian at a specific pixel
                
                transmission = 1.0 - alpha 
                accummulated_transmission = torch.cumprod(transmission, dim=0) # Multiplies the transmission values down the list. Tells you how much light leaves the current layer

                # We need to know how much light reached the current layer
                # Ones is one row of ones, being each column a pixel in the tile
                ones = torch.ones((1, alpha.shape[1]), device=args.device) # The first layer receives 100% of the light, then we accumulate the transmission of the previous layers to know how much light reaches the current layer

                # The rest of the rows are the accumulated transmission of the previous gaussians, which tells us how much light reaches the current layer
                # Every row corresponds to a Gaussian, and every column corresponds to a pixel in the tile. We are only considering the Gaussians that are in this tile, so we can compute the contribution of each Gaussian to each pixel in the tile
                T = torch.cat([ones, accummulated_transmission[:-1]], dim=0) # (N_Gaussians_in_tile, N_Pixels_in_tile)
                weights = alpha * T # Weight from the paper. Size (N_Gaussians_in_tile, N_Pixels_in_tile). This is the usual product cell by cell

                # flat_y and flat_x represent the pixel coordinates of the current tile, and we use them to index into the semantic mask to get the class labels for those pixels
                tile_classes = semantic_mask[flat_y, flat_x] 
                unique_classes = torch.unique(tile_classes)

                if target_id not in unique_classes:
                    continue # No pixels of the target class in this tile, skip the voting
                else:
                    # Compare whether the pixel belongs to the target class and create a mask for those pixels
                    pixel_mask = (tile_classes == target_id)
                    tile_confidences = confidence_mask[flat_y, flat_x] # Pixel by pixel confidence for this tile
                    
                    '''
                    Weight Calculation: alpha * T * pixel_confidence

                    Weighted by the inverse of the size of the Gaussianto punish large gaussians

                    weights has shape (N_gaussians_in_tile, N_pixels_in_tile), each row is a Gaussian and each column is a pixel in the tile. It represents how much each Gaussian contributes to each pixel in the tile.
                    We can imagine tile_confidences as a row vector that advances through the rows of weights, affecting all Gaussians depending on the confidence of the pixel
                    tile_confidences is (N_pixels_in_tile,) so it broadcasts across gaussians (rows):
                    Shape  (K, P) * Shape (P,) results in Shape (K, P)

                    We can imagine size_penalty_val as a column vector that advances through the columns of weights, affecting all pixels depending on the size of the Gaussian
                    size_penalty_val is (N_gaussians_in_tile,) so it broadcasts across pixels (columns):
                    Shape  (K, P) / Shape (K, 1) results in Shape (K, P)
                    '''

                    size_penalty_val = (tile_sizes * args.size_penalty) ** args.alpha
                    weighted_contribution = (weights * tile_confidences) / size_penalty_val.view(-1, 1) # .view(-1, 1) transforms size_penalty_val from shape (K,) into (K, 1), allowing it to broadcast across the pixel dimension (columns) of weights.
                    
                    # Sum all the pixel contributions for the target class to get the total vote for each Gaussian in this tile
                    class_votes = weighted_contribution[:, pixel_mask].sum(dim=1)    

                    # Accumulate the votes for the Gaussians in this tile into the weights tensor of this view, where Gaussians are still sorted by depth. This is a temporary tensor that will be added to the global weights tensor after all tiles have been processed for this view.
                    view_weights_sorted[gaussians_in_tile] += class_votes

        # After processing all tiles for this view, we add the votes from this view to the global weights tensor using the original indices of the Gaussians
        # sorted_original_indices says to which original Gaussian each vote in view_weights_sorted corresponds, so we can accumulate the votes correctly into the global weights tensor
        global_weights[sorted_original_indices] += view_weights_sorted
        del view_weights_sorted, means2D, conic, radius, grid_min, grid_max
        torch.cuda.empty_cache()

    # Save the global votes and weights for later use in thresholding
    safe_class_name = args.target_class.replace(" ", "_")
    class_output_dir = os.path.join(args.output_dir, safe_class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    voting_data_path = os.path.join(class_output_dir, f"voting_data_{safe_class_name}.pt")
    
    voting_data = {
        'weights': global_weights,
        'num_cameras': len(target_cameras),
        'num_class_views': class_views,
        'target_id': target_id
    }
    
    torch.save(voting_data, voting_data_path)
    print(f"Saved voting weights to {voting_data_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="../example_data/output/truck_test")
    parser.add_argument("--mask_dir", default="./data/2D_mask/02-04_23-18", help="Directory containing masks.json (e.g. data/2D_mask/timestamp)")
    parser.add_argument("--output_dir", default="./data/output/02-04_23-18", help="Directory to save outputs")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--target_class", type=str, default="truck", help="Only one object at a time can be segmented. The name must match one of the classes in the YOLO model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load tensors on")
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration number to load from the model (e.g. 1000)")
    parser.add_argument("--raster_block_size", type=int, default=16, help="Block size for rasterization. Larger blocks are faster but less precise.")
    parser.add_argument("--alpha", type=float, default=2.0, help="Exponent for size punishment. Higher alpha penalizes larger Gaussians more.")
    parser.add_argument("--beta", type=float, default=0.05, help="Threshold factor for labeling Gaussians. Higher means more conservative segmentation.")
    parser.add_argument("--size_penalty", type=float, default=100.0, help="Base multiplier for size punishment. Scales the Gaussian size before exponentiation.")
    parser.add_argument("--size_measure", type=str, default="l2", choices=["max", "gmean", "l2"],
                        help="M1: per-gaussian size measure for the vote penalty")
    parser.add_argument("--source_path", type=str, default=None, help="Path to the source directory containing images/colmap data")

    args = get_combined_args(parser)
    
    with torch.no_grad():
        main(args)
