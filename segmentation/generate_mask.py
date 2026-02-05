import os
import sys
import argparse
import numpy as np
import cv2
import json
from datetime import datetime
from glob import glob
import torch
import numpy as np
from ultralytics import YOLO

# Add project root to path to ensure imports work if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_segmentation_masks(image_path="./000122.jpg", model_path='./yolo11l-seg.pt', conf=0.5, debug=True, output_dir="debug_vis"):
    """
    Runs YOLO 2D segmentation on a single image file.

    Args:
        image_path (str): Path to a .jpg image file.
        model_path (str): Path to the .pt model file.
        conf (float): Confidence threshold.
        debug (bool): Whether to save a debug visualization of the generated mask.
        output_dir (str): Directory to save the debug image.

    Returns:
        tuple: (semantic_mask, confidence_mask, names_map)
    """
    
    # 1. Input Validation
    if not isinstance(image_path, str):
         raise ValueError(f"Input must be a string path. Got {type(image_path)}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    # Load Model
    model = YOLO(model_path)
    
    # Read Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    orig_H, orig_W = img_bgr.shape[:2]

    # 2. Inference
    # Note: retina_masks=True usually improves mask quality but uses more VRAM.
    # We stick to default behavior or previous script behavior for consistency unless requested.
    # Previous script: results = model(img_for_inference, verbose=False, conf=conf, save=False)
    results = model(img_bgr, verbose=False, conf=conf, save=False)
    result = results[0]
        
    # 3. Process Result
    semantic_mask = np.zeros((orig_H, orig_W), dtype=np.int32)
    confidence_mask = np.zeros((orig_H, orig_W), dtype=np.float32)
    names_map = {}
    
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy() # (N, H, W) - often reduced resolution
        boxes = result.boxes
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        
        # Sort by confidence (low to high) so higher confidence overwrites lower
        sort_idx = np.argsort(confidences)
        
        for idx in sort_idx:
            mask_raw = masks[idx]
            cls_id = int(class_ids[idx])
            conf_val = confidences[idx]
            
            stored_id = cls_id + 1
            class_name = result.names[cls_id]
            names_map[stored_id] = class_name
            
            # Resize mask if necessary
            if mask_raw.shape != (orig_H, orig_W):
                mask_resized = cv2.resize(mask_raw, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask_resized > 0.5
            else:
                mask_bool = mask_raw > 0.5
            
            # Write to masks
            semantic_mask[mask_bool] = stored_id
            confidence_mask[mask_bool] = conf_val

    # 4. Debug Visualization
    if debug:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        vis_img = img_bgr.copy()
        overlay = np.zeros_like(vis_img)
        
        # Random colors for visualization
        unique_ids = np.unique(semantic_mask)
        np.random.seed(42)
        colors = {uid: np.random.randint(0, 255, 3).tolist() for uid in unique_ids if uid != 0}
        
        for uid in unique_ids:
            if uid == 0: continue
            
            mask_cls = (semantic_mask == uid)
            color = colors[uid]
            
            # Draw Mask
            overlay[mask_cls] = color
            
            # Draw Boundaries (Contours)
            contours, _ = cv2.findContours(mask_cls.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color, 1)
            
            # Draw Label Name
            ys, xs = np.nonzero(mask_cls)
            if len(ys) > 0:
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                # Ensure text is inside image
                cy = np.clip(cy, 10, orig_H - 10)
                cx = np.clip(cx, 10, orig_W - 10)
                
                label_name = names_map.get(uid, f"Class {uid}")
                text = f"{uid}: {label_name}"
                
                # White text with black outline for visibility
                cv2.putText(vis_img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(vis_img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Blend overlay
        alpha = 0.4
        mask_all = semantic_mask > 0
        if mask_all.any():
            vis_img[mask_all] = cv2.addWeighted(vis_img[mask_all], 1-alpha, overlay[mask_all], alpha, 0)
        
        fname = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"debug_{fname}")
        cv2.imwrite(save_path, vis_img)
        print(f"Debug segmentation saved to {save_path}")

    return semantic_mask, confidence_mask, names_map

def save_batch_masks(mask_data_list, names_map, data_root):
    """
    Saves a batch of segmentation results into a single timestamped folder.
    
    Args:
        mask_data_list (list): List of dicts {'key': str, 'semantic': ndarray, 'confidence': ndarray}
        names_map (dict): Class ID to name mapping.
        data_root (str): Root directory for data.
    """
    
    # Create timestamped directory
    now = datetime.now()
    tx_name = now.strftime("%m-%d_%H-%M")
    output_dir = os.path.join(data_root, tx_name)
    os.makedirs(output_dir, exist_ok=True)
    
    sem_dir = os.path.join(output_dir, "semantic")
    conf_dir = os.path.join(output_dir, "confidence")
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    json_index = {}
    
    for item in mask_data_list:
        key = item['key']
        semantic = item['semantic']
        confidence = item['confidence']
        
        # Save Semantic
        sem_path = os.path.join(sem_dir, f"{key}.png")
        cv2.imwrite(sem_path, semantic.astype(np.uint8))
        
        # Save Confidence
        conf_path = os.path.join(conf_dir, f"{key}.png")
        conf_uint8 = (confidence * 255).astype(np.uint8)
        cv2.imwrite(conf_path, conf_uint8)
        
        # Update Index
        json_index[key] = {
            "semantic": f"semantic/{key}.png",
            "confidence": f"confidence/{key}.png"
        }

    # Save masks.json
    index_path = os.path.join(output_dir, "masks.json")
    with open(index_path, 'w') as f:
        json.dump(json_index, f, indent=4)

    # Save classes.json
    classes_path = os.path.join(output_dir, "classes.json")
    serializable_map = {str(k): v for k, v in names_map.items()}
    with open(classes_path, 'w') as f:
        json.dump(serializable_map, f, indent=4)
        
    print(f"Saved {len(mask_data_list)} masks to {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Optional input image. If not provided, we look for default set.
    parser.add_argument("--image", help="Path to single input image", default=None)
    parser.add_argument("--images_dir", help="Directory of images", default="example_data/data/tandt/truck/images")
    parser.add_argument("--model", default="./yolo26x-seg.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Define data root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "data", "2D_mask")
    
    # TARGET IMAGES
    target_ids = ["00005", "00060", "00095", "00116", "00169", "00212", "00231", 
                    "000005", "000060", "000095", "000116", "000212", "000231"]
    
    images_to_process = []
    
    if args.image:
        images_to_process.append(args.image)
    else:
        # Process specific images
        all_possible = glob(os.path.join(args.images_dir, "*.jpg")) + glob(os.path.join(args.images_dir, "*.png"))
        
        for img_p in all_possible:
            fname = os.path.splitext(os.path.basename(img_p))[0]
            if fname in target_ids:
                images_to_process.append(img_p)
        
        images_to_process.sort()
        
        if not images_to_process:
            print(f"No target images found in {args.images_dir}")

    results = []
    global_names = {}

    for img_path in images_to_process:
        print(f"Processing {img_path}...")
        sem, conf, names = get_segmentation_masks(img_path, model_path=args.model, conf=args.conf, debug=True)
        
        # Merge names (assumes consistency across YOLO class IDs)
        global_names.update(names)
        
        key = os.path.splitext(os.path.basename(img_path))[0]
        results.append({
            'key': key,
            'semantic': sem,
            'confidence': conf
        })

    if results:
        save_batch_masks(results, global_names, data_root)
