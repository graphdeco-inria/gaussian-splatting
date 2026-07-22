import os
import sys
import glob
import zipfile
import subprocess
import argparse

def run_cmd(cmd):
    print(f"\n[EXEC] Running command:\n{' '.join(cmd)}\n")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[WARNING] Command failed with return code {res.returncode}")

def create_submission_zip(submission_dir, zip_name="submission.zip"):
    print(f"\nCreating submission zip archive: {zip_name} from {submission_dir}")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, submission_dir)
                    zipf.write(full_path, rel_path)
    print(f"Submission zip successfully created at: {os.path.abspath(zip_name)}")

def main():
    parser = argparse.ArgumentParser(description="End-to-end multi-scene training and rendering pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset scenes (e.g. ../phase1/public_set)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save trained model checkpoints")
    parser.add_argument("--submission_dir", type=str, default="./submission", help="Directory to save rendered output PNGs")
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations per scene (default: 30000, local test: 2000-7000)")
    parser.add_argument("--zip_name", type=str, default="submission.zip", help="Output zip filename")
    parser.add_argument("--scenes", nargs="+", type=str, default=None, help="Filter specific scene directory names (e.g. --scenes HCM0181)")
    parser.add_argument("--max_scenes", type=int, default=None, help="Limit number of scenes to process for fast local testing")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree (default: 3, local test: 1 or 2)")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step if model already exists")
    
    args = parser.parse_args()

    # Find scenes
    all_scenes = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    all_scenes = sorted(all_scenes)

    if args.scenes:
        filtered_scenes = [s for s in all_scenes if s in args.scenes or s.lower() in [x.lower() for x in args.scenes]]
    else:
        filtered_scenes = all_scenes

    if args.max_scenes is not None:
        filtered_scenes = filtered_scenes[:args.max_scenes]

    scene_paths = [os.path.join(args.data_dir, s) for s in filtered_scenes]

    print(f"\n=======================================================")
    print(f"Targeting {len(scene_paths)} scene(s) in {args.data_dir}:")
    for sp in scene_paths:
        print(f" - {os.path.basename(sp)}")
    print(f"Iterations per scene: {args.iterations}")
    print(f"SH Degree: {args.sh_degree}")
    print(f"=======================================================\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.submission_dir, exist_ok=True)

    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        model_path = os.path.join(args.output_dir, scene_name)
        csv_path = os.path.join(scene_path, "test", "test_poses.csv")
        scene_out_dir = os.path.join(args.submission_dir, scene_name)

        if not os.path.exists(csv_path):
            print(f"[SKIP] {scene_name}: test_poses.csv not found at {csv_path}")
            continue

        print(f"\n=======================================================")
        print(f"Processing Scene: {scene_name}")
        print(f"=======================================================")

        # Step 1: Train
        if not args.skip_train:
            train_cmd = [
                sys.executable, "train.py",
                "-s", scene_path,
                "-m", model_path,
                "--sh_degree", str(args.sh_degree),
                "--iterations", str(args.iterations),
                "--checkpoint_iterations", str(args.iterations)
            ]
            run_cmd(train_cmd)

        # Step 2: Render
        render_cmd = [
            sys.executable, "render_competition.py",
            "--model_path", model_path,
            "--csv_path", csv_path,
            "--output_dir", scene_out_dir,
            "--iteration", str(args.iterations)
        ]
        run_cmd(render_cmd)

    # Step 3: Zip submission
    create_submission_zip(args.submission_dir, args.zip_name)

if __name__ == "__main__":
    main()
