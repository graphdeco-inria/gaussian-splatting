import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree # organizes points in a tree structure to make finding "nearest neighbors" extremely fast
import argparse
import os
import json
from plyfile import PlyData

def compute_iou(gt_mesh_path, pred_ply_path, beta, target_class_id, distance_threshold=0.05, gt_iou_max=0.0):
    """
    Computes IoU for a single class by mapping predictions to ground truth mesh vertices
    
    Args:
        gt_mesh_path: path to the ground truth PLY mesh
        pred_ply_path: path to the predicted PLY, which contains only points of the target class
        beta: value used to suffix the output filename
        target_class_id: the integer ID of the target class, or comma-separated string for union
        distance_threshold: distance in meters to consider that a prediction covers a ground truth vertex
        gt_iou_max: the theoretical maximum IoU achievable with ground truth gaussians
        ...
    """
    
    # Extract vertices and labels from raw data
    plydata = PlyData.read(gt_mesh_path)
    vertex_data = plydata['vertex']
    gt_vertices = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    label_key = 'label' 
    gt_labels = vertex_data[label_key]
    print(f"Loaded {len(gt_vertices)} ground truth vertices")
    
    # Identify GT Positives
    gt_labels = gt_labels.astype(int)
    
    # Boolean mask identifying which vertices belong to the class we are testing
    if "," in str(target_class_id):
        target_ids = [int(x) for x in str(target_class_id).split(",")]
        gt_is_target = np.isin(gt_labels, target_ids)
        print(f"Computing IoU for union of classes: {target_ids}")
    else:
        gt_is_target = (gt_labels == int(target_class_id))

    num_gt_positives = np.sum(gt_is_target)
    print(f"Number of GT vertices for class {target_class_id}: {num_gt_positives}")
    
    if num_gt_positives == 0:
        print("Warning: No GT vertices found for this class. IoU is either undefined or 0.")
        return 0.0, 0, 0, 0

    print(f"Loading predictions from {pred_ply_path}")
    pred_pcd = o3d.io.read_point_cloud(pred_ply_path)
    pred_points = np.asarray(pred_pcd.points)
    
    print(f"Loaded {len(pred_points)} predicted points.")
    
    if len(pred_points) == 0:
        print("No predicted points. IoU = 0.")
        return 0.0, 0, 0, 0
    
    # Build KDTree on predictions to query for each ground truth vertex
    # The logic is: for each GT vertex, how far is the closest predicted point?
    tree = cKDTree(pred_points)
    
    print(f"Querying nearest neighbors with radius={distance_threshold}")
    distances, _ = tree.query(gt_vertices, k=1, workers=-1, distance_upper_bound=distance_threshold)
    
    # Determine which GT vertices are predicted as target
    pred_mask = (distances <= distance_threshold)
    
    # Compute Intersection
    intersection_mask = gt_is_target & pred_mask
    intersection = np.sum(intersection_mask)
    
    # Compute Union. Predicted as target implies the GT vertex is spatially close to a predicted point, so we treat the spatial neighborhood of predicted points as the predicted region
    union_mask = gt_is_target | pred_mask
    union = np.sum(union_mask)
    
    iou = intersection / union if union > 0 else 0
    
    # Detailed stats
    tp = intersection
    fp = np.sum(pred_mask & (~gt_is_target)) # Predicted but not GT
    fn = np.sum(gt_is_target & (~pred_mask)) # GT but not predicted
    
    print(f"\nIoU for class {target_class_id}: {iou:.4f}")
    
    relative_iou = 0.0
    if gt_iou_max > 0:
        relative_iou = iou / gt_iou_max
        print(f"Relative IoU (against the ground truth limit {gt_iou_max:.4f}): {relative_iou:.4f}")

    # Save results to JSON for later mIoU calculation
    output_dir = os.path.dirname(pred_ply_path)
    
    beta_str = str(beta).replace('.', '_')
    result_filename = f"iou_result_beta{beta_str}.json"
        
    result_path = os.path.join(output_dir, result_filename)
    
    # Handle composite ID for JSON serialization
    try:
        serializable_id = int(target_class_id)
    except ValueError:
        serializable_id = str(target_class_id)

    results = {
        "class_id": serializable_id,
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "union": int(union),
        "threshold": float(distance_threshold),
        "gt_count": int(num_gt_positives),
        "pred_count": int(len(pred_points)),
        "gt_iou_max": float(gt_iou_max),
        "relative_iou": float(relative_iou)
    }
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved metrics to {result_path}")
    
    return iou, tp, fp, fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IoU for a 3D class")
    parser.add_argument("--gt_mesh", type=str, required=True, help="Path to GT labeled mesh")
    parser.add_argument("--pred_ply", type=str, required=True, help="Path to predicted PLY")
    parser.add_argument("--class_id", type=str, required=True, help="Class ID")
    parser.add_argument("--beta", type=str, required=True, help="Beta value for filename suffix")
    parser.add_argument("--threshold", type=float, default=0.05, help="Distance threshold")
    parser.add_argument("--gt_iou", type=float, default=0.0, help="Pre-computed IoU of GT gaussians")
    
    args = parser.parse_args()
    
    compute_iou(args.gt_mesh, args.pred_ply, args.beta, args.class_id, args.threshold, args.gt_iou)
