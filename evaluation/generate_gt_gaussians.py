import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
import argparse
import os

def generate_gt_gaussians(gaussian_path, gt_mesh_path, output_path, target_class_id, distance_threshold=0.05):
    """
    Generates a PLY of gaussians that correspond to the target class based on the ground truth mesh
    A gaussian is assigned the label of its nearest GT vertex
    """

    print(f"Loading GT Mesh from {gt_mesh_path}")
    gt_ply = PlyData.read(gt_mesh_path)
    gt_vertex = gt_ply['vertex']
    
    # Load all GT vertices
    gt_locations = np.vstack([gt_vertex['x'], gt_vertex['y'], gt_vertex['z']]).T
    gt_labels = gt_vertex['label']
    
    print(f"Loaded {len(gt_locations)} ground truth vertices.")

    gt_tree = cKDTree(gt_locations)
    
    # Load Gaussians
    gs_ply = PlyData.read(gaussian_path)
    gs_vertex = gs_ply['vertex']
    
    gs_locations = np.vstack([gs_vertex['x'], gs_vertex['y'], gs_vertex['z']]).T
    print(f"Total gaussians: {len(gs_locations)}")
    
    # For each Gaussian, find the nearest GT vertex and its label
    distances, indices = gt_tree.query(gs_locations, k=1, workers=-1, distance_upper_bound=distance_threshold)
    
    # Handling invalid indices for points outside threshold (indices == len(gt_tree.data))
    valid_mask = distances <= distance_threshold
    
    # Pre-allocate boolean target mask (False everywhere)
    is_target = np.zeros(len(gs_locations), dtype=bool)
    
    # Only access labels for valid indices to avoid out-of-bounds error
    valid_labels = gt_labels[indices[valid_mask]]
    is_target[valid_mask] = (valid_labels == target_class_id)
    
    is_gt_positive = valid_mask & is_target
    
    count = np.sum(is_gt_positive)
    print(f"Identified {count} ground truth gaussians for class {target_class_id}.")
    
    if count == 0:
        print("Finished: No gaussians found for the target class with the given threshold.")
        return

    # Filter the structured array directly
    filtered_data = gs_vertex.data[is_gt_positive]
    
    # Create new PlyElement
    new_vertex_element = PlyElement.describe(filtered_data, 'vertex')
    
    # Create new PlyData
    new_ply_data = PlyData([new_vertex_element], text=False)
    
    print(f"Saving ground truth gaussians to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_ply_data.write(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ground truth gaussians from mesh")
    parser.add_argument("--gaussian_ply", required=True, help="Input full 3DGS model")
    parser.add_argument("--gt_mesh", required=True, help="Ground truth labeled mesh")
    parser.add_argument("--output_ply", required=True, help="Output filtered PLY")
    parser.add_argument("--class_id", type=int, required=True, help="Target class ID to extract")
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    
    generate_gt_gaussians(args.gaussian_ply, args.gt_mesh, args.output_ply, args.class_id, args.threshold)
