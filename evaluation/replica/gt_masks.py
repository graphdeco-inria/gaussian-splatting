# CLI entry point for Replica GT masks to be used inside the lifting container

import argparse
from pathlib import Path

from .scene import ReplicaScene


def main():
    """ Generate Replica GT masks from the CLI arguments """
    parser = argparse.ArgumentParser()

    # Identify the Replica sequence and the directory where masks will be written
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sequence_name", default="Sequence_2")
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--vertex_label_min_share", type=float, required=True)
    parser.add_argument("--visibility_slop", type=float, required=True)
    parser.add_argument("--output_dir", required=True, type=Path)

    # Recreate masks even when the output directory already contains metadata
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Use the same scene thresholds as the unified evaluation driver
    scene = ReplicaScene(
        args.data_root, args.scene, args.sequence_name, args.frame_step,
        seed=3,
        vertex_label_min_share=args.vertex_label_min_share,
        visibility_slop=args.visibility_slop,
    )
    
    # Generate the semantic and confidence masks for the selected frames
    scene.generate_gt_masks(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()