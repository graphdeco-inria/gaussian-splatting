import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
import argparse
import os

def assign_labels_symmetric(gs_locations, gt_locations, gt_labels, tau, min_share=0.0):
    tree = cKDTree(gt_locations)
    lists = tree.query_ball_point(gs_locations, r=tau, workers=-1)
    counts = np.fromiter(map(len, lists), dtype=np.int64, count=len(gs_locations))
    indptr = np.zeros(len(gs_locations) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    out = np.full(len(gs_locations), -1, dtype=np.int64)
    if indptr[-1] == 0:
        return out
    flat_v = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if len(l)])
    flat_g = np.repeat(np.arange(len(gs_locations)), counts)
    # distances for the weight
    d = np.linalg.norm(gs_locations[flat_g] - gt_locations[flat_v], axis=1)
    w = np.clip(1.0 / (d * d + 1e-10), 0.1, 1.0)
    lab = gt_labels[flat_v].astype(np.int64)
    valid = lab > 0  # invalid/unlabeled vertices excluded from vote and denominator
    flat_g, flat_v, w, lab = flat_g[valid], flat_v[valid], w[valid], lab[valid]
    if len(w) == 0:
        return out
    uniq = np.unique(lab)
    lpos = np.searchsorted(uniq, lab)
    order = np.lexsort((lpos, flat_g))
    g_s, l_s, w_s = flat_g[order], lpos[order], w[order]
    # segment boundaries by (gaussian, label)
    key = g_s.astype(np.int64) * len(uniq) + l_s
    new_key = np.concatenate(([True], key[1:] != key[:-1]))
    seg_sum = np.add.reduceat(w_s, np.where(new_key)[0])
    seg_g = g_s[new_key]
    seg_l = l_s[new_key]
    # totals per gaussian
    tot = np.zeros(len(gs_locations))
    np.add.at(tot, seg_g, seg_sum)
    # best label per gaussian by vote mass (ties: first in sorted order)
    best_mass = np.zeros(len(gs_locations))
    best_l = np.full(len(gs_locations), -1, dtype=np.int64)
    better = seg_sum > best_mass[seg_g]
    best_mass[seg_g[better]] = seg_sum[better]
    best_l[seg_g[better]] = seg_l[better]
    share = best_mass / np.maximum(tot, 1e-12)
    if min_share > 0.0:
        ok = (best_l >= 0) & (share >= min_share)
    else:
        ok = best_l >= 0  # plurality: always take the local majority
    out[ok] = uniq[best_l[ok]]
    return out


def generate_gt_gaussians(gaussian_path, gt_mesh_path, output_path, target_class_id, distance_threshold=0.05, method="symmetric", min_share=0.0):

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
    
    if method == "symmetric":
        g_labels = assign_labels_symmetric(gs_locations, gt_locations, gt_labels,
                                           distance_threshold, min_share=min_share)
        try:
            if "," in str(target_class_id):
                target_ids = [int(x) for x in str(target_class_id).split(",")]
            else:
                target_ids = [int(target_class_id)]
        except ValueError:
            print(f"Error parsing target class ID: {target_class_id}")
            return
        is_gt_positive = np.isin(g_labels, target_ids)
        print(f"symmetric labeling: {int(is_gt_positive.sum())} gaussians assigned "
              f"(of {int((g_labels >= 0).sum())} labeled)")
    else:
        # For each Gaussian, find the nearest GT vertex and its label
        distances, indices = gt_tree.query(gs_locations, k=1, workers=-1, distance_upper_bound=distance_threshold)
        
        # Handling invalid indices for points outside threshold (indices == len(gt_tree.data))
        valid_mask = distances <= distance_threshold
        
        # Pre-allocate boolean target mask (False everywhere)
        is_target = np.zeros(len(gs_locations), dtype=bool)
        
        # Parse target class ID(s)
        
        # Only access labels for valid indices to avoid out-of-bounds error
        valid_labels = gt_labels[indices[valid_mask]]
        
        try:
            if "," in str(target_class_id):
                target_ids = [int(x) for x in str(target_class_id).split(",")]
                # Check if label is in list
                is_target_mask = np.isin(valid_labels, target_ids)
            else:
                target_ids = [int(target_class_id)]
                is_target_mask = (valid_labels == target_ids[0])
        except ValueError:
            print(f"Error parsing target class ID: {target_class_id}")
            return

        is_target[valid_mask] = is_target_mask
        
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
    parser.add_argument("--class_id", type=str, required=True, help="Target class ID to extract")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--method", type=str, default="symmetric", choices=["legacy", "symmetric"])
    parser.add_argument("--min_share", type=float, default=0.0,
                        help="minimum vote share; 0 = plurality (default), 0.5 = strict majority")
    args = parser.parse_args()
    
    generate_gt_gaussians(args.gaussian_ply, args.gt_mesh, args.output_ply, args.class_id, args.threshold, args.method, args.min_share)
