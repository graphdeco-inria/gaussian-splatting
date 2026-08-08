import torch
import os
import sys
from argparse import ArgumentParser
import numpy as np
from scipy.spatial import cKDTree

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import GaussianModel


def _load_gaussians(args):
    """ Loads the Gaussian model from the specified path and iteration """

    print(f"Loading gaussian model from {args.model_path}")
    start_sh = args.sh_degree if hasattr(args, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree=start_sh, use_labels=True)
    loaded_iter = args.loaded_iter if hasattr(args, 'loaded_iter') else 30000
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    return gaussians


def _connected_components(xyz, radius): # Implemented using union-find with path compression
    """
    Group together points that are close enough to each other, directly or through a chain of nearby points
    
    Returns one component ID per point
    """

    # Initialize union-find structure
    n = len(xyz)
    parent = np.arange(n, dtype=np.int64) # Each point is initially its own parent (root of its own tree)

    # Union-find "find" function with path compression
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]] # Path compression: make the parent of a point point to its grandparent, flattening the tree
            a = parent[a]
        return a

    # Use cKDTree to find all pairs of points whose distance is as most radius
    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")

    # Union the pairs of points that are close enough
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # After all unions, find the root of each point to determine its component label
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def apply_threshold(args, gaussians=None, voting_data=None):
    """ Applies the threshold to the voting weights and saves the resulting segmented PLY file """
    if voting_data is None:
        voting_data = torch.load(args.voting_data_path, map_location=args.device)

    # Each tensor contains one accumulated evidence value per Gaussian
    target_weights = voting_data['target_weights']
    background_weights = voting_data['background_weights']

    # Stored ID of the target class represented by the selected Gaussians
    target_id = voting_data['target_id']

    # beta is the minimum target fraction of the total evidence
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("target evidence beta must be in [0, 1]")

    # Combine both types of evidence. Unsupported Gaussians have zero evidence
    evidence = target_weights + background_weights
    score = torch.zeros_like(target_weights)
    supported = evidence > 0

    # Compute the fraction of supported evidence assigned to the target class. Background competes here
    score[supported] = target_weights[supported] / evidence[supported]

    # Report the threshold, background mode and number of supported Gaussians.
    print(f"Applying target/background threshold beta={args.beta:.3f} "
          f"(mode={voting_data.get('background_mode', 'unknown')}, "
          f"supported={int(supported.sum().item())})")

    # Keep supported Gaussians whose target evidence ratio reaches beta
    final_mask = supported & (score >= args.beta)

    # Optionally expand high confidence seeds through nearby lower score Gaussians using Canny-style hysteresis on the radius graph
    gamma = getattr(args, 'hysteresis_gamma', 0.0)
    if gamma > 0:

        # Hysteresis needs Gaussian positions, so load the model if necessary
        if gaussians is None:
            gaussians = _load_gaussians(args)
        xyz = gaussians.get_xyz
        if torch.is_tensor(xyz):
            xyz = xyz.detach().cpu().numpy()

        # Work on CPU copies while preserving the original tensors
        score_cpu = score.detach().cpu()
        seed = final_mask.detach().cpu()

        # The low threshold defines candidate bridge Gaussians around the seeds
        low_threshold_mask = supported.detach().cpu() & (score_cpu >= args.beta * gamma)
        seed_count = int(seed.sum().item())

        # Hysteresis requires both a non-empty bridge set and at least one seed
        if low_threshold_mask.sum().item() > 0 and seed_count > 0:

            # Group bridge Gaussians into spatially connected components
            component_labels = _connected_components(xyz[low_threshold_mask.numpy()], args.hysteresis_radius) # CC of the low-threshold Gaussians only

            # Keep only components containing at least one high-threshold seed
            seed_in_low_threshold_mask = seed[low_threshold_mask].numpy()
            keep_component = np.zeros(component_labels.max() + 1, dtype=bool)

            # Mark components that contain at least one seed.
            np.logical_or.at(keep_component, component_labels, seed_in_low_threshold_mask)
            kept = keep_component[component_labels] # Expand the component decisions to every low-threshold Gaussian

            # Reconstruct the full Gaussian mask from the retained components
            new_mask = torch.zeros_like(seed)
            new_mask[low_threshold_mask] = torch.from_numpy(kept)
            n_comps = component_labels.max() + 1

            # Report the hysteresis expansion and the retained components
            print(f"hysteresis phase: gamma={gamma} radius={args.hysteresis_radius} | "
                  f"seeds={seed_count} low_threshold_mask={int(low_threshold_mask.sum().item())} comps={n_comps} "
                  f"kept_comps={int(keep_component.sum())} | "
                  f"{seed_count} -> {int(new_mask.sum().item())} gaussians")

            # Return the final mask to the original device
            final_mask = new_mask.to(final_mask.device)
        else:
            # Keep the beta mask when hysteresis has no valid seed or bridge set
            print("hysteresis phase: degenerate set; keeping seed mask")

    # Count the Gaussians selected after thresholding and optional hysteresis
    count = final_mask.sum().item()
    print(f"Labeled {count} gaussians as {target_id}")

    # An empty selection is valid and is still saved as an empty PLY
    if count == 0:
        print("Warning: No Gaussians selected with this threshold; saving an empty PLY.")

    # Load the model if it was not already loaded for hysteresis
    if gaussians is None:
        gaussians = _load_gaussians(args)

    # Build a safe filesystem class name for the output path
    raw_class_name = args.target_class if hasattr(args, 'target_class') else str(target_id)
    safe_class_name = raw_class_name.replace(" ", "_")
    
    # Include beta so outputs from different thresholds can be distinguished
    filename = f"labeled_gaussians_{safe_class_name}"
    if hasattr(args, 'beta'):
        beta_str = str(args.beta).replace('.', '_')
        filename += f"_beta{beta_str}"
    filename += ".ply"
    
    # Store each target class in its own output directory
    target_class_dir = os.path.join(args.output_dir, safe_class_name)
    os.makedirs(target_class_dir, exist_ok=True)
    
    output_ply = os.path.join(target_class_dir, filename)

    # Select and save only the Gaussians contained in the final mask
    gaussians.set_mask_index(final_mask.nonzero(as_tuple=True)[0])
    gaussians.save_ply(output_ply)
    print(f"Saved labeled PLY to {output_ply}")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model and target configuration
    parser.add_argument("--model_path", required=True, help="Path to trained 3DGS model output")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    parser.add_argument("--loaded_iter", type=int, default=30000, help="Iteration of model to load")
    parser.add_argument("--target_class", type=str, default="object", help="Name of target class, for filename")

    # Input and output paths
    parser.add_argument("--voting_data_path", type=str, required=True, help="Path to .pt file containing voting weights")
    parser.add_argument("--output_dir", required=True, help="Directory to save labeled PLY")

    # Target selection
    parser.add_argument("--beta", type=float, default=0.5, help="Minimum target evidence ratio in [0, 1]")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda", help="Device, either cuda or cpu")
    
    # Hysteresis expansion
    parser.add_argument("--hysteresis_gamma", type=float, default=0.8, help="Low-threshold factor. 0 disables hysteresis")
    parser.add_argument("--hysteresis_radius", type=float, default=0.05, help="Connectivity radius in meters for the bridge set")
    args = parser.parse_args()
    if args.hysteresis_gamma < 0.0:
        raise ValueError("--hysteresis_gamma must be non-negative")
    if args.hysteresis_radius <= 0.0:
        raise ValueError("--hysteresis_radius must be greater than zero")
    
    with torch.no_grad():
        apply_threshold(args)
