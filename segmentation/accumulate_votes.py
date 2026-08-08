import torch
import os
import sys
import cv2
import json
from argparse import ArgumentParser

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from arguments import get_combined_args
from segmentation.projection import GaussianProjector
from utils.general_utils import build_scaling_rotation

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
    Retrieve the stored detector-mask ID for a detector name.

    ``args.target_class`` is a detector name at this stage, not a
    main project name or a dataset name.
    """

    with open(classes_json_path, 'r') as f:
            classes_map = json.load(f)
            
    # classes.json stores detector-mask ID -> detector name, so invert it to
    # detector name -> stored detector-mask ID.
    name_to_id = {v: int(k) for k, v in classes_map.items()}
    
    if args.target_class not in name_to_id:
        raise ValueError(f"Target class {args.target_class} not found in classes.json")
        
    target_id = name_to_id[args.target_class] # Stored detector-mask ID, not a main/local ID.
    return target_id


def get_background_mask_and_confidence(detector_label_mask, confidence_mask, target_id, background_mode, background_confidence):
    """ 
    Return the selected non-target pixels and their background confidence 
    
    background_confidence is the fallback assigned to explicit background pixels
    """
    non_target = detector_label_mask != target_id

    # Only the pixels with id 0 are considered background
    if background_mode == "explicit_background":
        background_mask = detector_label_mask == 0
        background_confidence_map = torch.full_like(confidence_mask, background_confidence)

    # Everything that is not the target class is considered background, and all those pixels get the fallback confidence
    elif background_mode == "all_non_target":
        background_mask = non_target
        background_confidence_map = torch.full_like(confidence_mask, background_confidence)

    # Default mode: non-target pixels keep their detector confidence, and background pixels, i.e. id 0, get the fallback confidence
    elif background_mode == "confidence_weighted":
        background_mask = non_target
        background_confidence_map = confidence_mask.clone()
        background_confidence_map[detector_label_mask == 0] = background_confidence
    else:
        raise ValueError(f"Unknown background mode: {background_mode}")

    return background_mask, background_confidence_map.clamp(0.0, 1.0)


def main(args):
    if not 0.0 <= args.background_confidence <= 1.0:
        raise ValueError("background_confidence must be in [0, 1]")

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

    print(f"Gaussian sizes, max scale: Min={scalar_sizes.min().item():.5f}, Max={scalar_sizes.max().item():.5f}")

    # Statistical analysis of scales: deciles
    deciles = torch.quantile(scalar_sizes, torch.linspace(0.1, 0.9, 9, device=args.device))
    print("\n--- Gaussian Scale Distribution (Deciles) ---")
    for i, decile_val in enumerate(deciles):
        print(f"{(i+1)*10}th percentile: {decile_val.item():.5f}")
    '''

    cov3D = get_covariance_3d(gaussians) # shape (N, 3, 3), as there is one 3D covariance matrix per Gaussian. This will be used to project the 3D Gaussians into 2D for each camera view.
    # Vote accumulation only needs camera geometry and masks. Keep full-resolution source images on CPU, then release them after Scene has built the cameras.
    load_images_on_cpu = getattr(args, "data_device", "cuda") == "cpu"
    scene = Scene(args, gaussians, load_iteration=args.loaded_iter, shuffle=False)
    if load_images_on_cpu:
        for camera in scene.getTrainCameras():
            for attribute in ("original_image", "alpha_mask", "gt_alpha_mask"):
                if hasattr(camera, attribute):
                    setattr(camera, attribute, None)
            camera.data_device = torch.device(args.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Prepare global votes tensor
    total_gaussians = gaussians.get_xyz.shape[0] # gaussians.get_xyz has shape (N, 3), where N is the total number of Gaussians in the model.

    # Initialize a tensor to accumulate votes for each Gaussian across all camera views, and another one for background votes
    global_target_weights = torch.zeros((total_gaussians,), device=args.device, dtype=torch.float32)
    global_background_weights = torch.zeros((total_gaussians,), device=args.device, dtype=torch.float32)

    # classes.json maps stored detector-mask IDs to detector names.
    classes_json_path = os.path.join(args.mask_dir, "classes.json")
    target_id = get_target_class_id(args, classes_json_path) # Stored detector-mask ID for the requested detector name.


    # Iterate through scene cameras to find these views
    train_cameras = scene.getTrainCameras()
    masked_cameras = [] # Cameras that have a corresponding 2D mask
    
    for cam in train_cameras:
        basename = os.path.basename(cam.image_name)
        name_no_ext = os.path.splitext(basename)[0]
        name = f"{name_no_ext}.png" # The YOLO script saves all the generated 2D masks as PNG files
        
        conf_full_path = os.path.join(args.mask_dir, "confidence", name)
        if os.path.exists(conf_full_path):
            masked_cameras.append((cam, {
                "semantic": os.path.join("semantic", name), # Constructs the path of the semantic mask for this camera view
                "confidence": os.path.join("confidence", name) # Constructs the path of the confidence mask for this camera view
            }))
            
    print(f"Matched {len(masked_cameras)} cameras in the scene.")

    class_views = 0  # Views where the target class actually appears


    # Iterate through the matched cameras and accumulate votes for the target class
    for i, (cam, mask_info) in enumerate(masked_cameras):
        print(f"\n  Processing View {i+1}/{len(masked_cameras)}: {cam.image_name}")

        # Getting semantic data
        sem_path = os.path.join(args.mask_dir, mask_info["semantic"]) # Concatenating the mask directory with the semantic mask filename to get the full path of the semantic mask for this camera view
        semantic_img = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED) # (H, W)
        
        # Resize semantic mask to match camera dimensions
        if semantic_img.shape[:2] != (cam.image_height, cam.image_width):
            semantic_img = cv2.resize(semantic_img, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
            
        detector_label_mask = torch.tensor(semantic_img, dtype=torch.long, device=args.device)
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

        # Check whether the target detector-mask ID is present in this view
        target_mask_view = detector_label_mask == target_id
        has_target = bool(target_mask_view.any().item())
        if has_target:
            class_views += 1
        elif args.background_view_policy == "target_views":
            # Empty detector views provide no positive evidence and would make a large amount of uncertain background dominate the ratio
            continue

        background_mask, background_confidence = (get_background_mask_and_confidence(
                detector_label_mask,
                confidence_mask,
                target_id,
                args.background_mode,
                args.background_confidence))

        # Projection of 3D Gaussians into 2D camera space
        projector = GaussianProjector(cam)

        # projection_results is a map with means2D, cov2D, depths, and indices of the Gaussians that are visible in this camera view
        projection_results = projector.project(gaussians.get_xyz, cov3D)

        means2D = projection_results['means2D'] # (M, 2) being M the amount of Gaussians that are visible in this camera view
        cov2D = projection_results['cov2D'] # (M, 2, 2)
        depths = projection_results['depths'] # (M,)
        indices = projection_results['indices'] # (M,)


        opacities = gaussians.get_opacity[indices] # (M,)
        view_sizes = scalar_sizes[indices] # (M,)
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
        discriminant = torch.clamp(trace * trace - 4 * det, min=0.0)
        sqrt_term = torch.sqrt(discriminant)
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

        # Initialize tensors to store the accumulated weights for each visible Gaussian
        num_visible = means2D.shape[0]
        view_target_weights_sorted = torch.zeros((num_visible,), device=args.device, dtype=torch.float32)
        view_background_weights_sorted = torch.zeros((num_visible,), device=args.device, dtype=torch.float32)

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
                # Check which projected Gaussian centers overlap this tile.
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
                gaussian_exponent = -0.5 * (tile_conics[:, 0].unsqueeze(1) * dx**2 +
                                tile_conics[:, 2].unsqueeze(1) * dy**2) - \
                                tile_conics[:, 1].unsqueeze(1) * dx * dy
                
                # The opacity of the Gaussian modulates its contributions
                # alpha shape: (N_Gaussians_in_tile, N_Pixels_in_tile)
                alpha = tile_opacities.view(-1, 1) * torch.exp(gaussian_exponent.clamp(max=0)) # Computes the opacity of a Gaussian at a specific pixel
                
                transmission = 1.0 - alpha
                accumulated_transmission = torch.cumprod(transmission, dim=0) # Multiplies the transmission values down the list. Tells you how much light leaves the current layer

                # We need to know how much light reached the current layer
                # Ones is one row of ones, being each column a pixel in the tile
                ones = torch.ones((1, alpha.shape[1]), device=args.device) # The first layer receives 100% of the light, then we accumulate the transmission of the previous layers to know how much light reaches the current layer

                # The rest of the rows are the accumulated transmission of the previous gaussians, which tells us how much light reaches the current layer
                # Every row corresponds to a Gaussian, and every column corresponds to a pixel in the tile. We are only considering the Gaussians that are in this tile, so we can compute the contribution of each Gaussian to each pixel in the tile
                T = torch.cat([ones, accumulated_transmission[:-1]], dim=0) # (N_Gaussians_in_tile, N_Pixels_in_tile)

                '''
                First, alpha is multiplied by the accumulated transmission T to obtain the
                contribution of each Gaussian to each pixel:

                weights = alpha * T

                weights has shape (N_gaussians_in_tile, N_pixels_in_tile). Each row corresponds to one Gaussian and each column 
                to one pixel in the tile. It represents how much each Gaussian contributes to each pixel.
                '''

                weights = alpha * T # Weight from the paper. Size (N_Gaussians_in_tile, N_Pixels_in_tile). This is the usual product cell by cell

                # flat_y and flat_x index the semantic mask, whose pixels are stored detector-mask IDs, for the current tile
                tile_detector_labels = detector_label_mask[flat_y, flat_x]

                '''
                The target and background confidences are then applied independently:
                
                target_confidences = confidence_mask[flat_y, flat_x]
                background_confidences = background_confidence[flat_y, flat_x]

                Both confidence tensors have shape (N_pixels_in_tile,), so they broadcast
                across the Gaussian dimension:

                Shape (K, P) * Shape (P,) -> Shape (K, P)

                where K is the number of Gaussians in the tile and P is the number of pixels.
                '''

                # Pixel mask and confidence values for the target class
                target_pixel_mask = tile_detector_labels == target_id
                target_confidences = confidence_mask[flat_y, flat_x]

                # Pixel mask and confidence values for the background class
                background_pixel_mask = background_mask[flat_y, flat_x] # Not the same as tile_detector_labels != target_id, because the background mask can be defined differently depending on the background_mode argument (in default, confidence_weighted, detected non-target classes retain their detector confidence)
                background_confidences = background_confidence[flat_y, flat_x]

                '''
                The Gaussian contribution is also weighted by the inverse of its size. This
                penalizes large Gaussians so that they contribute less to the accumulated votes:

                size_penalty_val = (tile_sizes * args.size_penalty) ** args.sigma

                size_penalty_val has shape (N_gaussians_in_tile,). It is reshaped into a
                column vector with shape (K, 1), allowing it to broadcast across all pixels
                of each Gaussian:

                Shape (K, P) / Shape (K, 1) -> Shape (K, P)

                The resulting weighted_contribution is used to calculate target and background votes separately. 
                Target pixels are multiplied by target_confidences, while background pixels are multiplied by background_confidences.
                '''

                size_penalty_val = (tile_sizes * args.size_penalty) ** args.sigma
                weighted_contribution = (weights / size_penalty_val.view(-1, 1))


                # Vote Calculation: alpha * T * pixel_confidence
                # Sum all the pixel contributions for the target class and background to get the total vote for each Gaussian in this tile
                target_votes = (weighted_contribution[:, target_pixel_mask] * target_confidences[target_pixel_mask]).sum(dim=1)
                background_votes = (weighted_contribution[:, background_pixel_mask] * background_confidences[background_pixel_mask]).sum(dim=1)

                # Accumulate the votes for the Gaussians in this tile into the weights tensors of this view, where Gaussians are still sorted by depth. These are a temporary tensor that will be added to the global weights tensors after all tiles have been processed for this view.
                view_target_weights_sorted[gaussians_in_tile] += target_votes
                view_background_weights_sorted[gaussians_in_tile] += background_votes

        # After processing all tiles for this view, we add the votes from this view to the global weights tensor using the original indices of the Gaussians
        # sorted_original_indices says to which original Gaussian each vote in view_weights_sorted corresponds, so we can accumulate the votes correctly into the global weights tensor
        global_target_weights[sorted_original_indices] += view_target_weights_sorted
        global_background_weights[sorted_original_indices] += view_background_weights_sorted
        del (view_target_weights_sorted, view_background_weights_sorted, means2D, conic, radius, grid_min, grid_max)
        torch.cuda.empty_cache()

    # Save the global votes and weights for later use in thresholding
    safe_class_name = args.target_class.replace(" ", "_")
    class_output_dir = os.path.join(args.output_dir, safe_class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    voting_data_path = os.path.join(class_output_dir, f"voting_data_{safe_class_name}.pt")
    
    voting_data = {
        'target_weights': global_target_weights,
        'background_weights': global_background_weights,
        'num_cameras': len(masked_cameras),
        'num_class_views': class_views,
        'target_id': target_id,
        'background_mode': args.background_mode,
        'background_confidence': args.background_confidence,
        'background_view_policy': args.background_view_policy,
    }

    # Save statistics if requested
    if args.statistics_path:
        evidence = global_target_weights + global_background_weights
        supported = evidence > 0
        scores = torch.zeros_like(evidence)
        scores[supported] = global_target_weights[supported] / evidence[supported]

        def stats(values):
            values = values.detach().float().cpu()
            return {
                'min': float(values.min().item()) if values.numel() else None,
                'mean': float(values.mean().item()) if values.numel() else None,
                'std': float(values.std(unbiased=False).item()) if values.numel() else None,
                'max': float(values.max().item()) if values.numel() else None,
            }

        score_stats = stats(scores[supported])
        statistics = {
            'num_cameras': len(masked_cameras),
            'num_class_views': class_views,
            'num_gaussians': int(global_target_weights.numel()),
            'target_weight_sum': float(global_target_weights.sum().item()),
            'background_weight_sum': float(global_background_weights.sum().item()),
            'supported_gaussians': int(supported.sum().item()),
            'target_score_min': score_stats['min'],
            'target_score_mean': score_stats['mean'],
            'target_score_std': score_stats['std'],
            'target_score_max': score_stats['max'],
        }
        os.makedirs(os.path.dirname(args.statistics_path), exist_ok=True)
        with open(args.statistics_path, 'w') as handle:
            json.dump(statistics, handle, indent=2)
    
    torch.save(voting_data, voting_data_path)
    print(f"Saved voting weights to {voting_data_path}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model and target configuration
    parser.add_argument("--model_path", default="../example_data/output/truck_test")
    parser.add_argument("--sh_degree", type=int, default=3) # Spherical Harmonics degree for the Gaussian model
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration number to load from the model")
    parser.add_argument("--target_class", type=str, default="truck", help="Only one object at a time can be segmented. The name must match one of the classes in the YOLO model.")

    # Paths
    parser.add_argument("--statistics_path", type=str, default=None)
    parser.add_argument("--mask_dir", default="./data/2D_mask/02-04_23-18", help="Directory containing semantic and confidence masks")
    parser.add_argument("--output_dir", default="./data/output/02-04_23-18", help="Directory to save outputs")
    parser.add_argument("--source_path", type=str, default=None, help="Path to the source directory containing images/colmap data")

    # Device configuration and performance
    parser.add_argument("--device", type=str, default="cuda", help="Device to load tensors on")
    parser.add_argument("--data_device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for source images; camera matrices remain on the GPU")
    parser.add_argument("--raster_block_size", type=int, default=16, help="Block size for rasterization. Larger blocks are faster but less precise.")

    # Vote accumulation parameters
    parser.add_argument("--sigma", type=float, default=1.5, help="Exponent for size punishment. Higher sigma penalizes larger Gaussians more.")
    parser.add_argument("--size_penalty", type=float, default=100.0, help="Base multiplier for size punishment. Scales the Gaussian size before exponentiation.")
    parser.add_argument("--size_measure", type=str, default="max", choices=["max", "gmean", "l2"], help="Gaussian size measure for the vote penalty")
    
    # Background handling parameters
    parser.add_argument("--background_mode", type=str, default="confidence_weighted", choices=["all_non_target", "explicit_background", "confidence_weighted"],
        help="How to form the non-target evidence mask")
    parser.add_argument("--background_confidence", type=float, default=0.25, help="Confidence assigned to pixels with stored semantic label zero")
    parser.add_argument("--background_view_policy", type=str, default="target_views", choices=["target_views", "all_views"],
        help="Use only views containing target pixels or every matched view")

    args = get_combined_args(parser)

    if args.sigma < 0.0:
        raise ValueError("--sigma must be non-negative")
    if args.size_penalty <= 0.0:
        raise ValueError("--size_penalty must be greater than zero")
    if args.raster_block_size <= 0:
        raise ValueError("--raster_block_size must be greater than zero")
    if not 0.0 <= args.background_confidence <= 1.0:
        raise ValueError("--background_confidence must be in [0, 1]")
    
    with torch.no_grad():
        main(args)
