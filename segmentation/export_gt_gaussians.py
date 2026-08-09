# Export one ground-truth Gaussian PLY for each requested class

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import GaussianModel


def _load_gaussians(model_path, loaded_iter):
    """ Load the full Gaussian model used by the evaluation run """
    gaussians = GaussianModel(sh_degree=3, use_labels=True)
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    return gaussians


def export(args):
    """ Write the Ground Truth-transferred Gaussians for each class """
    labels = np.load(args.gt_labels_path)["labels"]

    gaussians = _load_gaussians(args.model_path, args.loaded_iter)
    if len(labels) != len(gaussians.get_xyz):
        raise ValueError("GT labels do not match the Gaussian model")

    # Export one PLY per class
    for class_spec in args.class_spec:
        class_name, raw_id = class_spec.rsplit(":", 1)
        class_id = int(raw_id)

        # Select the Gaussian indices corresponding to this class
        selected = np.flatnonzero(labels == class_id)
        output_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save the selected Gaussians to a PLY file
        output_path = os.path.join(output_dir, "ground_truth_gaussians.ply")
        gaussians.set_mask_index(torch.as_tensor(selected, dtype=torch.long, device=gaussians.get_xyz.device))
        gaussians.save_ply(output_path)
        print(f"Saved {len(selected)} GT gaussians to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gt_labels_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--loaded_iter", type=int, default=30000)
    parser.add_argument("--class_spec", action="append", required=True, help="class directory and local ID in the form name:id")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
