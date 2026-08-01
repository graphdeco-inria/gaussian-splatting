import torch
import os
import sys
from argparse import ArgumentParser
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import GaussianModel


def _load_gaussians(args):
    print(f"Loading gaussian model from {args.model_path}")
    start_sh = args.sh_degree if hasattr(args, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree=start_sh, use_labels=True)
    loaded_iter = args.loaded_iter if hasattr(args, 'loaded_iter') else 30000
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    return gaussians


def _connected_components(xyz, radius):
    """Union-find connected components over cKDTree radius pairs.
    Returns int64 component labels per point (0..K-1)."""
    from scipy.spatial import cKDTree
    n = len(xyz)
    parent = np.arange(n, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def apply_threshold(args, gaussians=None, voting_data=None):
    """
    Applies the threshold to the voting weights and saves the resulting segmented PLY file
    """
    
    # Load voting data if not provided
    if voting_data is None:
        voting_data = torch.load(args.voting_data_path, map_location=args.device)

    weights = voting_data['weights']
    num_cameras = voting_data['num_cameras']
    target_id = voting_data['target_id']

    # Normalize the threshold by views where the class actually appears
    if getattr(args, 'thresh_mode', 'class_views') == 'class_views':
        n_views = voting_data.get('num_class_views', num_cameras)
        if 'num_class_views' not in voting_data:
            print("WARNING: voting_data has no 'num_class_views'; falling back to num_cameras")
    else:
        n_views = num_cameras

    # Threshold logic
    threshold = n_views * args.beta
    print(f"Applying threshold {threshold:.2f} with beta {args.beta} and {n_views} views "
          f"(mode={getattr(args, 'thresh_mode', 'class_views')}, total_cameras={num_cameras})")

    final_mask = weights > threshold

    # Hysteresis thresholding, Canny-style, on the gaussian radius graph.
    gamma = getattr(args, 'hysteresis_gamma', 0.0)
    if gamma > 0:
        if gaussians is None:
            gaussians = _load_gaussians(args)
        xyz = gaussians.get_xyz
        if torch.is_tensor(xyz):
            xyz = xyz.detach().cpu().numpy()
        w_cpu = weights.detach().cpu()
        seed = final_mask.detach().cpu()
        lo = w_cpu > (threshold * gamma)
        seed_count = int(seed.sum().item())
        if lo.sum().item() > 0 and seed_count > 0:
            comp = _connected_components(xyz[lo.numpy()], args.hysteresis_radius)
            seed_in_lo = seed[lo].numpy()
            keep_comp = np.zeros(comp.max() + 1, dtype=bool)
            np.logical_or.at(keep_comp, comp, seed_in_lo)
            kept = keep_comp[comp]
            new_mask = torch.zeros_like(seed)
            new_mask[lo] = torch.from_numpy(kept)
            n_comps = comp.max() + 1
            print(f"[hysteresis] gamma={gamma} radius={args.hysteresis_radius} | "
                  f"seeds={seed_count} lo={int(lo.sum().item())} comps={n_comps} "
                  f"kept_comps={int(keep_comp.sum())} | "
                  f"{seed_count} -> {int(new_mask.sum().item())} gaussians")
            final_mask = new_mask.to(final_mask.device)
        else:
            print("[hysteresis] degenerate set; keeping seed mask")

    # Adaptive cluster filtering for small objects
    if getattr(args, 'adaptive_cluster_filter', False):
        if gaussians is None:
            gaussians = _load_gaussians(args)
        xyz = gaussians.get_xyz
        if torch.is_tensor(xyz):
            xyz = xyz.detach().cpu().numpy()
        
        # Get indices of thresholded gaussians
        kept_indices = final_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        if len(kept_indices) > 0:
            from sklearn.cluster import DBSCAN
            
            positions = xyz[kept_indices]
            
            # Cluster spatially
            clustering = DBSCAN(eps=args.cluster_eps, min_samples=3).fit(positions)
            labels = clustering.labels_
            
            # Compute cluster sizes
            cluster_counts = {}
            for label in set(labels):
                if label == -1:
                    continue
                cluster_counts[label] = np.sum(labels == label)
            
            if cluster_counts:
                largest_cluster = max(cluster_counts, key=cluster_counts.get)
                largest_size = cluster_counts[largest_cluster]
                total_size = sum(cluster_counts.values())
                dominance = largest_size / total_size
                
                # Adaptive: if dominant, keep only largest; else keep all >= min_size
                if dominance > args.cluster_dominance_thresh:
                    keep_mask = labels == largest_cluster
                    print(f"[cluster_filter] dominant cluster ({dominance:.2f} > {args.cluster_dominance_thresh}) | "
                          f"keeping only largest cluster: {largest_size} gaussians")
                else:
                    keep_mask = np.zeros(len(positions), dtype=bool)
                    for label, size in cluster_counts.items():
                        if size >= args.cluster_min_size:
                            keep_mask |= (labels == label)
                    n_kept = keep_mask.sum()
                    print(f"[cluster_filter] non-dominant ({dominance:.2f} <= {args.cluster_dominance_thresh}) | "
                          f"keeping {len(cluster_counts)} clusters with size >= {args.cluster_min_size}: {n_kept} gaussians")
                
                # Update final_mask
                new_mask = torch.zeros_like(final_mask)
                new_mask[kept_indices[keep_mask]] = True
                final_mask = new_mask

    count = final_mask.sum().item()
    print(f"Labeled {count} gaussians as {target_id}")

    if count == 0:
        print("Warning: No Gaussians selected with this threshold.")
        return

    # If gaussians not provided, load them
    if gaussians is None:
        gaussians = _load_gaussians(args)

    # Save logic, saving the subset of gaussians
    raw_class_name = args.target_class if hasattr(args, 'target_class') else str(target_id)
    safe_class_name = raw_class_name.replace(" ", "_")
    
    # Append beta's value to the filename
    filename = f"labeled_gaussians_{safe_class_name}"
    if hasattr(args, 'beta'):
        beta_str = str(args.beta).replace('.', '_')
        filename += f"_beta{beta_str}"
    filename += ".ply"
    
    # Create class-specific output directory
    target_class_dir = os.path.join(args.output_dir, safe_class_name)
    os.makedirs(target_class_dir, exist_ok=True)
    
    output_ply = os.path.join(target_class_dir, filename)

    # set_mask_index filters the gaussians to be saved
    gaussians.set_mask_index(final_mask.nonzero(as_tuple=True)[0])
    gaussians.save_ply(output_ply)
    print(f"Saved labeled PLY to {output_ply}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--voting_data_path", type=str, required=True, help="Path to .pt file containing voting weights")
    parser.add_argument("--model_path", required=True, help="Path to trained 3DGS model output")
    parser.add_argument("--output_dir", required=True, help="Directory to save labeled PLY")
    parser.add_argument("--target_class", type=str, default="object", help="Name of target class (for filename)")
    parser.add_argument("--beta", type=float, default=0.05, help="Threshold factor")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration of model to load")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    parser.add_argument("--thresh_mode", type=str, default="class_views",
                        choices=["class_views", "cameras"],
                        help="M2a: normalize beta threshold by per-class views (default) or total cameras (legacy)")
    parser.add_argument("--hysteresis_gamma", type=float, default=0.5,
                        help="M4: low-threshold factor (0 disables hysteresis)")
    parser.add_argument("--hysteresis_radius", type=float, default=0.05,
                        help="M4: connectivity radius (m) for the bridge set")
    parser.add_argument("--adaptive_cluster_filter", action="store_true",
                        help="Enable adaptive cluster filtering for small objects")
    parser.add_argument("--cluster_eps", type=float, default=0.12,
                        help="DBSCAN eps for cluster filtering (m)")
    parser.add_argument("--cluster_dominance_thresh", type=float, default=0.4,
                        help="If largest cluster has > this fraction of gaussians, keep only it")
    parser.add_argument("--cluster_min_size", type=int, default=20,
                        help="Minimum cluster size to keep when non-dominant")

    args = parser.parse_args()
    
    with torch.no_grad():
        apply_threshold(args)
