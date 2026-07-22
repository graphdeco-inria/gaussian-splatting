import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as tf

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr

def evaluate_scene(renders_dir, gt_dir, psnr_max=50.0):
    renders = []
    gts = []
    image_names = []
    
    for fname in os.listdir(renders_dir):
        if not (fname.endswith('.png') or fname.endswith('.jpg')):
            continue
        gt_file = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_file):
            continue
            
        render = Image.open(os.path.join(renders_dir, fname)).convert('RGB')
        gt = Image.open(gt_file).convert('RGB')
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
        
    if len(renders) == 0:
        return None
        
    ssims = []
    psnrs = []
    lpipss = []
    
    for idx in range(len(renders)):
        ssims.append(ssim(renders[idx], gts[idx]).item())
        psnrs.append(psnr(renders[idx], gts[idx]).item())
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg').item())
        
    mean_ssim = float(np.mean(ssims))
    mean_psnr = float(np.mean(psnrs))
    mean_lpips = float(np.mean(lpipss))
    
    psnr_norm = min(max(mean_psnr / psnr_max, 0.0), 1.0)
    score = 0.4 * (1.0 - mean_lpips) + 0.3 * mean_ssim + 0.3 * psnr_norm
    
    return {
        "SSIM": mean_ssim,
        "PSNR": mean_psnr,
        "PSNR_norm": psnr_norm,
        "LPIPS": mean_lpips,
        "Score": score,
        "num_images": len(renders)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Competition Score according to exact metric formula")
    parser.add_argument("--renders_dir", type=str, required=True, help="Directory containing rendered scene folders (e.g. ./submission)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth dataset (e.g. ../phase1/public_set)")
    parser.add_argument("--psnr_max", type=float, default=50.0, help="PSNR max normalization constant (default: 50.0)")
    
    args = parser.parse_args()
    
    scene_names = [d for d in os.listdir(args.renders_dir) if os.path.isdir(os.path.join(args.renders_dir, d))]
    scene_names = sorted(scene_names)
    
    total_scores = []
    print(f"\n=======================================================")
    print(f"Evaluating {len(scene_names)} scenes...")
    print(f"=======================================================")
    
    for scene in scene_names:
        r_dir = os.path.join(args.renders_dir, scene)
        gt_scene_dir = os.path.join(args.gt_dir, scene, "test", "images")
        if not os.path.exists(gt_scene_dir):
            gt_scene_dir = os.path.join(args.gt_dir, scene, "test")
            
        res = evaluate_scene(r_dir, gt_scene_dir, args.psnr_max)
        if res is None:
            print(f"Scene {scene}: No matching ground truth images found at {gt_scene_dir}")
            continue
            
        print(f"Scene [{scene}] ({res['num_images']} images):")
        print(f"  SSIM     : {res['SSIM']:.4f}")
        print(f"  PSNR     : {res['PSNR']:.2f} dB (Norm: {res['PSNR_norm']:.4f})")
        print(f"  LPIPS    : {res['LPIPS']:.4f}")
        print(f"  --> SCORE: {res['Score']:.4f}\n")
        
        total_scores.append(res['Score'])
        
    if total_scores:
        overall = np.mean(total_scores)
        print(f"=======================================================")
        print(f"OVERALL COMPETITION SCORE: {overall:.6f}")
        print(f"=======================================================")

if __name__ == "__main__":
    main()
