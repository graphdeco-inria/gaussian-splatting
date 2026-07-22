import numpy as np
from scipy.spatial import cKDTree
import argparse
import os
import json
from plyfile import PlyData

def compute_iou(gt_mesh_path, pred_ply_path, full_model_path, beta, target_class_id, distance_threshold=0.05, gt_iou_max=0.0):
    """
    Computes IoU for a single class using a consistency filter.

    The consistency filter ensures we only evaluate gaussians that sit in regions
    where the round-trip (mesh -> gaussians -> mesh) is coherent:
      For each Gaussian, find the nearest GT vertex -> get label
      For each GT vertex, find the nearest Gaussian -> get round-trip label
      A GT vertex is "consistent" if its label matches the round-trip label
      Filter GT gaussians to only those near consistent GT vertices of the target class
      Compute IoU between predicted gaussians and filtered GT gaussians

    Args:
        gt_mesh_path: path to the GT labeled mesh PLY
        pred_ply_path: path to the predicted PLY (target class only)
        full_model_path: path to the full 3DGS model PLY
        beta: value used to suffix the output filename
        target_class_id: integer class ID or comma-separated string for union
        distance_threshold: max distance to consider a nearest-neighbor match
        gt_iou_max: pre-computed GT IoU for relative IoU calculation
    """

    # Load GT mesh
    plydata = PlyData.read(gt_mesh_path)
    vertex_data = plydata['vertex']
    gt_vertices = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    gt_labels = vertex_data['label'].astype(int)

    if "," in str(target_class_id):
        target_ids = [int(x) for x in str(target_class_id).split(",")]
        gt_is_target = np.isin(gt_labels, target_ids)
        print(f"Computing IoU for union of classes: {target_ids}")
    else:
        target_id = int(target_class_id)
        gt_is_target = (gt_labels == target_id)

    num_gt_positives = int(np.sum(gt_is_target))
    print(f"Loaded {len(gt_vertices)} GT vertices, {num_gt_positives} for class {target_class_id}")

    if num_gt_positives == 0:
        print("No GT vertices for this class. IoU = 0.")
        return 0.0, 0, 0, 0

    # Load full Gaussian model
    full_ply = PlyData.read(full_model_path)
    full_vertex = full_ply['vertex']
    all_gaussians = np.vstack([full_vertex['x'], full_vertex['y'], full_vertex['z']]).T
    print(f"Loaded {len(all_gaussians)} full model gaussians")

    # Load predicted gaussians
    pred_ply = PlyData.read(pred_ply_path)
    pred_vertex = pred_ply['vertex']
    pred_points = np.vstack([pred_vertex['x'], pred_vertex['y'], pred_vertex['z']]).T
    print(f"Loaded {len(pred_points)} predicted gaussians")

    if len(pred_points) == 0:
        print("No predicted points. IoU = 0.")
        return 0.0, 0, 0, 0

    # Consistency filter
    # For each Gaussian, find nearest GT vertex -> assigned label
    gt_tree = cKDTree(gt_vertices)
    distances_full, indices_full = gt_tree.query(all_gaussians, k=1, workers=-1, distance_upper_bound=distance_threshold)
    valid_gaussian = distances_full <= distance_threshold
    gaussian_labels = np.full(len(all_gaussians), -1, dtype=int)
    valid_indices = indices_full[valid_gaussian]
    gaussian_labels[valid_gaussian] = gt_labels[valid_indices]

    # For each GT vertex, find nearest Gaussian -> round-trip label
    gauss_tree = cKDTree(all_gaussians)
    _, nearest_gauss_idx = gauss_tree.query(gt_vertices, k=1, workers=-1)
    round_trip_labels = gaussian_labels[nearest_gauss_idx]

    # Consistent if GT label matches round-trip label
    consistent = (gt_labels == round_trip_labels)
    num_consistent = int(np.sum(consistent))
    print(f"Consistent GT vertices: {num_consistent}/{len(gt_vertices)} ({100*num_consistent/len(gt_vertices):.1f}%)")

    # Filter GT gaussians — must be near a consistent GT vertex of the target class
    consistent_target = consistent & gt_is_target
    consistent_target_positions = gt_vertices[consistent_target]
    num_consistent_target = len(consistent_target_positions)
    print(f"Consistent target vertices: {num_consistent_target}/{num_gt_positives}")

    if num_consistent_target == 0:
        print("No consistent target GT vertices. IoU = 0.")
        return 0.0, 0, 0, 0

    consistency_tree = cKDTree(consistent_target_positions)
    dist_to_consistent, _ = consistency_tree.query(all_gaussians, k=1, workers=-1, distance_upper_bound=distance_threshold)
    filtered_mask = dist_to_consistent <= distance_threshold
    filtered_gt_gaussians = all_gaussians[filtered_mask]
    num_filtered = len(filtered_gt_gaussians)
    print(f"Filtered GT gaussians (consistency): {num_filtered} (from {len(all_gaussians)} total)")

    if num_filtered == 0:
        print("No filtered GT gaussians remain. IoU = 0.")
        return 0.0, 0, 0, 0