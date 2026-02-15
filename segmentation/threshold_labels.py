import torch
import os
import sys
from argparse import ArgumentParser
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import GaussianModel

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

    # Threshold logic
    threshold = num_cameras * args.beta
    print(f"Applying threshold {threshold:.2f} with beta {args.beta} and {num_cameras} cameras")

    final_mask = weights > threshold
    count = final_mask.sum().item()
    print(f"Labeled {count} gaussians as {target_id}")

    if count == 0:
        print("Warning: No Gaussians selected with this threshold.")
        return

    # If gaussians not provided, load them
    if gaussians is None:
        print(f"Loading gaussian model from {args.model_path}")
        start_sh = 3
        if hasattr(args, 'sh_degree'):
            start_sh = args.sh_degree
        
        gaussians = GaussianModel(sh_degree=start_sh, use_labels=True)
        
        # Construct ply path
        loaded_iter = args.loaded_iter if hasattr(args, 'loaded_iter') else 30000
        ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
        gaussians.load_ply(ply_path)

    # Save logic, saving the subset of gaussians
    raw_class_name = args.target_class if hasattr(args, 'target_class') else str(target_id)
    safe_class_name = raw_class_name.replace(" ", "_")
    
    # Check if beta is non-default (assuming 0.05 is default reference, but using the actual arg default if available)
    filename = f"labeled_gaussians_{safe_class_name}"
    if hasattr(args, 'beta'): # In a previous version of this code, 0.05 was default and not included in filename (in case no beta is stated in filename)
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
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree of model")

    args = parser.parse_args()
    
    with torch.no_grad():
        apply_threshold(args)
