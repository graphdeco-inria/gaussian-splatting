import os
import sys
import subprocess
import json
import random
import argparse
import shutil
from datetime import datetime

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES_DIR = os.path.join(ROOT_DIR, "data", "scannetpp", "validation_data")
METADATA_FILE = os.path.join(ROOT_DIR, "data", "scannetpp", "metadata", "semantic_classes.txt")
TRAIN_SCRIPT = os.path.join(ROOT_DIR, "train.py")
VOTE_SCRIPT = os.path.join(ROOT_DIR, "segmentation", "accumulate_votes.py")
GEN_GT_SCRIPT = os.path.join(ROOT_DIR, "evaluation", "generate_gt_gaussians.py")
IOU_SCRIPT = os.path.join(ROOT_DIR, "evaluation", "compute_iou.py")
THRESH_SCRIPT = os.path.join(ROOT_DIR, "segmentation", "threshold_labels.py")

# Environments
# gaussian_splatting is the environment for training
PYTHON_TRAIN = "/home/ivanverdugo/.conda/envs/gaussian_splatting/bin/python" 
# fusion environment is used for all non-training steps to avoid dependency conflicts with COLMAP and gaussian splatting
PYTHON_FUSION = sys.executable # default environment from which the script has been called
# colmap_env is the environment for COLMAP undistortion to avoid conflicts with gaussian_splatting dependencies
COLMAP_BIN = "/home/ivanverdugo/.conda/envs/colmap_env/bin/colmap"

def get_scene_gt_labels(scene_id):
    """
    Reads segments_anno.json from the scannet dataset
    Extracts exactly the "label" field that appear in that specific scene
    """
    segments_anno_path = os.path.join(SCENES_DIR, scene_id, "scans", "segments_anno.json")

    if not os.path.exists(segments_anno_path):
        print(f"Annotation file not found at {segments_anno_path}")
        return set()

    try:
        with open(segments_anno_path, 'r') as f:
            data = json.load(f)
        
        labels = set()
        if "segGroups" in data:
            for group in data["segGroups"]:
                if "label" in group:
                    labels.add(group["label"].lower().strip())
        return labels

    except Exception as e:
        print("Error reading annotation file")
        return set()


def get_possible_gt_ids(class_name, valid_scene_labels):
    """
    Returns list of potential IDs using word intersection
    """
    
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return []
    
    candidates = []
    target_words = set(class_name.lower().replace("_", " ").split())
    if not target_words:
        return []

    # Create mapping from label name to ID, the line number
    with open(METADATA_FILE, 'r') as f:
        label_to_id = {line.strip().lower(): i for i, line in enumerate(f.readlines())}

    # Iterate through valid scene labels
    for scene_label in valid_scene_labels:
        scene_label_clean = scene_label.lower().strip()
        
        # Check word intersection
        scene_words = set(scene_label_clean.split())
        if target_words.intersection(scene_words):
            if scene_label_clean in label_to_id:
                candidates.append(label_to_id[scene_label_clean])

    return list(set(candidates))


def prepare_scene(scene_id):
    """
    Prepares the scene for 3DGS training by running COLMAP image_undistorter
    """

    dslr_path = os.path.join(SCENES_DIR, scene_id, "dslr")
    input_colmap = os.path.join(dslr_path, "colmap")
    
    # Check for images folder at resized_images, or images if not found
    input_images = os.path.join(dslr_path, "resized_images")
    if not os.path.exists(input_images):
        input_images = os.path.join(dslr_path, "images")
        
    output_path = os.path.join(dslr_path, "undistorted_colmap")
    
    bin_path = os.path.join(output_path, "sparse", "0", "cameras.bin")
    txt_path = os.path.join(output_path, "sparse", "0", "cameras.txt")
    
    # Check if already the scene has already been prepared by checking either bin or txt files exist and are non-empty
    if (os.path.exists(bin_path) and os.path.getsize(bin_path) > 0) or (os.path.exists(txt_path) and os.path.getsize(txt_path) > 0):
        print(f"Scene {scene_id} already prepared in {output_path}")
        return

    # If not prepared, run COLMAP undistortion
    print(f"Preparing scene {scene_id} to undistort images")
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare COLMAP image_undistorter in its own environment to avoid conflicts
    cmd = [
        COLMAP_BIN, "image_undistorter",
        "--image_path", input_images,
        "--input_path", input_colmap,
        "--output_path", output_path,
        "--output_type", "COLMAP",
        "--max_image_size", "1600"
    ]
    
    try:
        subprocess.run(cmd, check=True) # If this fails, raises CalledProcessError
    except subprocess.CalledProcessError as e:
        print(f"COLMAP undistortion failed for {scene_id}: {e}")
        raise

    # Fix folder structure: gaussian splatting needs sparse/0/ instead of sparse/
    sparse_path = os.path.join(output_path, "sparse")
    sparse_0_path = os.path.join(sparse_path, "0")
    
    # Check if model files are in sparse/ and not in sparse/0/, if so move them to sparse/0/
    if os.path.exists(sparse_path) and not os.path.exists(sparse_0_path):
        model_files = ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]
        files_to_move = [f for f in os.listdir(sparse_path) if f in model_files or f.endswith(".bin") or f.endswith(".txt")]
        
        if len(files_to_move) > 0:
            print("Adjusting folder structure")
            os.makedirs(sparse_0_path, exist_ok=True)
            for f in files_to_move:
                shutil.move(os.path.join(sparse_path, f), os.path.join(sparse_0_path, f))


def run_training(scene_id):
    """
    Execute Gaussian splatting optimization
    """

    source_path = os.path.join(SCENES_DIR, scene_id, "dslr", "undistorted_colmap")
    output_path = os.path.join(ROOT_DIR, "output", scene_id)
    
    if os.path.exists(os.path.join(output_path, "point_cloud", "iteration_30000", "point_cloud.ply")):
        print(f"Training already completed for {scene_id}")
        return

    print(f"Starting training for scene {scene_id}")
    cmd = [
        PYTHON_TRAIN, TRAIN_SCRIPT,
        "-s", source_path,
        "-m", output_path,
        "-r", "2"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {scene_id}: {e}")
        raise

def run_2d_segmentation(scene_id):
    """
    Run 2D segmentation on all images and generate classes.json. It uses separate script segmentation/generate_mask.py
    """

    seg_script = os.path.join(ROOT_DIR, "segmentation", "generate_mask.py")
    source_images = os.path.join(SCENES_DIR, scene_id, "dslr", "undistorted_colmap", "images")
    output_mask_dir = os.path.join(ROOT_DIR, "data", "2D_masks", scene_id)
    
    classes_file = os.path.join(output_mask_dir, "classes.json")

    # If I want to recalculate masks, I can simply delete the classes.json file for that scene and re-run the workflow, it will trigger the segmentation step again
    if os.path.exists(classes_file):
         print(f"2D masks already generated for {scene_id}")
         return

    print(f"Running 2D segmentation for {scene_id}")
    
    cmd = [
        PYTHON_FUSION, seg_script,
        "--images_dir", source_images,
        "--output_root", output_mask_dir,
        "--model", os.path.join(ROOT_DIR, "yolo26x-seg.pt"),
        "--conf", "0.75"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Segmentation failed for {scene_id}: {e}")
        raise


def compute_iou_for_beta(beta, scene_id, class_name, gt_id, output_base, class_output_dir, gt_mesh):
    """
    Helper to run thresholding and IoU computation for a specific beta
    """

    # In directory names we don't want either spaces or dots, so we sanitize them
    safe_class_name = class_name.replace(" ", "_")
    beta_str = str(beta).replace('.', '_')
    ply_filename = f"labeled_gaussians_{safe_class_name}_beta{beta_str}.ply"
    ply_path = os.path.join(class_output_dir, ply_filename)
    
    # Run threshold_labels.py
    cmd_thresh = [
        PYTHON_FUSION, THRESH_SCRIPT,
        "--voting_data_path", os.path.join(class_output_dir, f"voting_data_{safe_class_name}.pt"),
        "--model_path", output_base,
        "--output_dir", os.path.dirname(class_output_dir), 
        "--target_class", class_name,
        "--beta", str(beta)
    ]
    
    subprocess.run(cmd_thresh, check=True, capture_output=True)
    
    if not os.path.exists(ply_path):
        return 0.0

    # Compute IoU
    cmd_iou = [
        PYTHON_FUSION, IOU_SCRIPT,
        "--gt_mesh", gt_mesh,
        "--pred_ply", ply_path,
        "--class_id", str(gt_id),
        "--beta", str(beta)
    ]
    
    try:
        subprocess.run(cmd_iou, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return 0.0
    
    # Read result
    iou_json = os.path.join(class_output_dir, f"iou_result_beta{beta_str}.json")
    
    iou = 0.0
    if os.path.exists(iou_json):
        with open(iou_json, 'r') as f:
            res = json.load(f)
            iou = res.get("iou", 0.0)
    return iou

def evaluate_object(scene_id, class_name, yolo_id, available_labels):
    """
    Runs the evaluation workflow for a single object
    Returns best IoU found among betas [0.01, 0.02, 0.03] if initial seed > 0.1
    """

    # Sanitize for output folder
    safe_class_name = class_name.replace(" ", "_")
    output_base = os.path.join(ROOT_DIR, "output", scene_id)
    seg_output_dir = os.path.join(output_base, "segmentation") # Base for accum output
    class_output_dir = os.path.join(seg_output_dir, safe_class_name) # Folder for this class
    
    # Check if class exists in scene ground truth labels before processing
    candidates = get_possible_gt_ids(safe_class_name, available_labels)
    if not candidates:
        print(f"{class_name}: No matching ground truth label in scene.")
        return 0.0

    # 1. Accumulate Votes with beta 0.03
    cmd_vote = [
        PYTHON_FUSION, VOTE_SCRIPT,
        "--model_path", output_base,
        "--mask_dir", os.path.join(ROOT_DIR, "data", "2D_masks", scene_id),
        "--output_dir", seg_output_dir,
        "--target_class", class_name,
        "--beta", "0.03"
    ]
    
    subprocess.run(cmd_vote, check=True)
    
    ply_path_03 = os.path.join(class_output_dir, f"labeled_gaussians_{safe_class_name}_beta0_03.ply")
    if not os.path.exists(ply_path_03):
        print(f"Failed to generate labeled PLY for {class_name}")
        return 0.0

    # 2. Find ground truth Correspondence
    best_overall_iou = 0.0
    
    # Attempt available candidates
    for i, gt_id in enumerate(candidates):
        print(f"Testing ground truth label candidate {i+1}/{len(candidates)}: ID {gt_id} for {class_name}")
        
        gt_ply_out = os.path.join(class_output_dir, f"gt_{safe_class_name}_id{gt_id}.ply")
        gt_mesh = os.path.join(SCENES_DIR, scene_id, "scans", "mesh_aligned_0.05_semantic.ply")
        gaussian_ply = os.path.join(output_base, "point_cloud", "iteration_30000", "point_cloud.ply")
        
        # Generator for GT
        cmd_gen_gt = [
            PYTHON_FUSION, GEN_GT_SCRIPT,
            "--gaussian_ply", gaussian_ply,
            "--gt_mesh", gt_mesh,
            "--output_ply", gt_ply_out,
            "--class_id", str(gt_id)
        ]
        
        try:
            subprocess.run(cmd_gen_gt, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            pass
            
        # Compute first IoU with beta 0.03 as starting point
        iou_03 = compute_iou_for_beta(0.03, scene_id, class_name, gt_id, output_base, class_output_dir, gt_mesh)

        current_candidate_best = iou_03
        
        # o.1 of IoU is the least to consider that labels correspond to each other, otherwise we consider that the class is mislabeled or simply not present in the scene, and we discard it
        if iou_03 > 0.1:
            print(f"First IoU is greater than 0.1, so we consider this label valid. Testing refinement betas 0.01 and 0.02")
            
            iou_01 = compute_iou_for_beta(0.01, scene_id, class_name, gt_id, output_base, class_output_dir, gt_mesh)
            iou_02 = compute_iou_for_beta(0.02, scene_id, class_name, gt_id, output_base, class_output_dir, gt_mesh)
            
            # Find max of refined search
            current_candidate_best = max(iou_01, iou_03, iou_02)
            print(f"Refinement results: beta 0.01: {iou_01:.4f}, beta 0.02: {iou_02:.4f}, beta 0.03: {iou_03:.4f}. Greatest: {current_candidate_best:.4f}")
            
        if current_candidate_best > best_overall_iou:
            best_overall_iou = current_candidate_best
            
    return best_overall_iou


def process_scene(scene_id):

    # Check for existing results first to avoid redundant processing
    miou_file = os.path.join(ROOT_DIR, "output", scene_id, "segmentation", "miou.txt")
    if os.path.exists(miou_file):
        try:
            with open(miou_file, 'r') as f:
                val = float(f.read().strip())
            print(f"Scene {scene_id} already processed, with mIoU {val}")
            return val
        except:
            print(f"Corrupt miou.txt for {scene_id}, reprocessing.")


    print(f" Processing scene {scene_id}")
    
    # Prepare data
    prepare_scene(scene_id)
    
    # Train
    run_training(scene_id)
    
    # 2D segmentation and mask generation
    run_2d_segmentation(scene_id)
    
    # Pre-fetch scene labels
    scene_labels = get_scene_gt_labels(scene_id)
    
    # Load 2D segmentation classes from this scene
    classes_file = os.path.join(ROOT_DIR, "data", "2D_masks", scene_id, "classes.json")
        
    with open(classes_file, 'r') as f:
        classes_raw = json.load(f)
        
    # Create reverse mapping from class name to ID for easier lookup
    classes = {}
    for k, v in classes_raw.items():
        classes[v] = int(k)
    
    ious = []
    
    for class_name, yolo_id in classes.items():
        print(f"Evaluating Class: {class_name} (YOLO ID: {yolo_id})...")
        iou = evaluate_object(scene_id, class_name, yolo_id, available_labels=scene_labels)
        
        if iou >= 0.1:
            ious.append(iou)
        else:
            print(f"Class {class_name} discarded (Mislabeled or no match).")
            
    mIoU = sum(ious) / len(ious) if ious else 0.0

    # Save mIoU to file
    try:
        os.makedirs(os.path.dirname(miou_file), exist_ok=True)
        with open(miou_file, "w") as f:
            f.write(f"{mIoU:.6f}")
        print(f"Saved mIoU to {miou_file}")
    except Exception as e:
        print(f"Failed to save mIoU file: {e}")

    print(f"Scene {scene_id} finished with mIoU: {mIoU}")
    
    # Write to markdowns/results.md for easy tracking of results so we have a historical log of all runs and their mIoUs. We also include the scene ID for reference.
    results_md_path = os.path.join(ROOT_DIR, "markdowns", "results.md")
    os.makedirs(os.path.dirname(results_md_path), exist_ok=True)
    
    # Format entry
    if not os.path.exists(results_md_path):
        header = " Scene ID and mIoU: \n"
        with open(results_md_path, 'w') as f:
            f.write(header)
            
    with open(results_md_path, 'a') as f:
        f.write(f"Scene id {scene_id} got mIoU of {mIoU:.4f} \n")
    return mIoU
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="Number of scenes to process")
    parser.add_argument("--seed", type=int, default=36, help="Random seed")
    parser.add_argument("--scene_id", type=str, default=None, help="Process a specific scene ID only")
    args = parser.parse_args()

    # Set a random seed for reproducibility
    random.seed(args.seed)
    
    # Get scenes
    if not os.path.exists(SCENES_DIR):
        print(f"Scenes dir {SCENES_DIR} does not exist.")
        sys.exit(1)

    all_scenes = [d for d in os.listdir(SCENES_DIR) if os.path.isdir(os.path.join(SCENES_DIR, d))]
    
    # Filter valid scenes
    valid_scenes = []
    
    # Explicit exclusion list
    # excluded_scenes = ["c4c04e6d6c"]
    excluded_scenes = []
    
    for s in all_scenes:
        if s in excluded_scenes: # Scene where the workflow was developed, not used for final evaluation
            continue
            
        mesh_path = os.path.join(SCENES_DIR, s, "scans", "mesh_aligned_0.05_semantic.ply")
        if os.path.exists(mesh_path):
            valid_scenes.append(s)
            
    if not valid_scenes:
        print("No valid scenes found")
        sys.exit(1)

    # If user specified a scene ID, use it, otherwise shuffle and select
    if args.scene_id:
        if args.scene_id in valid_scenes:
            selected_scenes = [args.scene_id]
            print(f"User selected scene: {args.scene_id}")
        else:
            print(f"Scene {args.scene_id} not found in valid scenes list.")
            sys.exit(1)
    else:
        # Shuffle and select
        selected_scenes = random.sample(valid_scenes, min(len(valid_scenes), args.limit))
        print(f"Selected scenes: {selected_scenes}")
    
    scene_mious = []
    results_md_path = os.path.join(ROOT_DIR, "markdowns", "results.md")
    
    if not os.path.exists(results_md_path):
        os.makedirs(os.path.dirname(results_md_path), exist_ok=True)
        with open(results_md_path, "w") as f:
            f.write("Evaluation results: \n")

    with open(results_md_path, "a") as f:
        f.write(f"\n## Workflow executed - {datetime.now()}\n")
    
    for s in selected_scenes:
        miou = process_scene(s)
        scene_mious.append(miou)
        
    final_mean = sum(scene_mious) / len(scene_mious) if scene_mious else 0.0
    
    print(f"\nFinal workflow result, with seed {args.seed} and {args.limit} scenes)")
    print(f"Scenes: {selected_scenes}")
    print(f"mIoUs: {scene_mious}")
    print(f"Average mIoU: {final_mean}\n")
    
    # Save Results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(results_md_path, "a") as f:
        f.write(f"\n## Workflow done - {timestamp}\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write(f"- Scenes: {', '.join(selected_scenes)}\n")
        f.write(f"- Scene mIoUs: {scene_mious}\n")
        f.write(f"- Average mIoU: {final_mean:.4f}\n")
