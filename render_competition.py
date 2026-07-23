import os
import csv
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from gaussian_renderer import render
from scene import GaussianModel
from utils.camera_utils import Camera
import math

def qvec2rotmat(qvec):
    """Chuyển đổi quaternion (qw, qx, qy, qz) từ COLMAP sang ma trận quay 3x3"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def main(args):
    # Khởi tạo mô hình và load trọng số đã train
    gaussians = GaussianModel(sh_degree=3)
    
    chkpnt_path = os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    
    if os.path.exists(chkpnt_path):
        print(f"Loading checkpoint: {chkpnt_path}")
        scene_info = torch.load(chkpnt_path, weights_only=False)
        from arguments import OptimizationParams
        opt = OptimizationParams(ArgumentParser())
        
        # Viettel codebase bug: _exposure is not saved in checkpoints but expected by training_setup
        gaussians.pretrained_exposures = None
        mock_exposure = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32, device="cuda")
        gaussians._exposure = torch.nn.Parameter(mock_exposure.requires_grad_(False))
        
        gaussians.restore(scene_info[0], opt)
    elif os.path.exists(ply_path):
        print(f"Loading PLY model: {ply_path}")
        gaussians.load_ply(ply_path)
    else:
        # Fallback to search any chkpnt or ply in model_path
        print(f"Searching for model files in {args.model_path}...")
        found = False
        for root, dirs, files in os.walk(args.model_path):
            for file in files:
                if file.endswith(".ply") and "point_cloud" in file:
                    p = os.path.join(root, file)
                    print(f"Found fallback PLY: {p}")
                    gaussians.load_ply(p)
                    found = True
                    break
            if found: break
        if not found:
            raise FileNotFoundError(f"Could not find model checkpoint or PLY in {args.model_path}")
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading CSV file: {args.csv_path}")
    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            image_name = row['image_name']
            width = int(row['width'])
            height = int(row['height'])
            
            # Read Quaternion and Translation (World-to-Camera COLMAP)
            qvec = np.array([float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])])
            tvec = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])
            
            R = qvec2rotmat(qvec)
            
            # Calculate FOV from Focal Length
            fx = float(row['fx'])
            fy = float(row['fy'])
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            
            # Configure Camera for 3DGS
            custom_cam = Camera(
                resolution=(width, height),
                colmap_id=idx, 
                R=R.transpose(), # 3DGS requires transposed matrix
                T=tvec, 
                FoVx=FovX, 
                FoVy=FovY, 
                depth_params=None,
                image=Image.new("RGB", (width, height)), # Dummy PIL image
                invdepthmap=None,
                image_name=image_name, 
                uid=idx,
                data_device="cuda"
            )
            
            # Rendering
            with torch.no_grad():
                render_pkg = render(custom_cam, gaussians, args.pipeline, background)
                rendered_image = render_pkg["render"]
            
            # Save image
            rendered_image = rendered_image.cpu().permute(1, 2, 0).numpy()
            rendered_image = (np.clip(rendered_image, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(rendered_image).save(os.path.join(args.output_dir, image_name))
            
            print(f"Generated image: {image_name}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render test poses for BTS Digital Twin")
    parser.add_argument("--model_path", type=str, required=True, help="Model directory path")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number to load (default: 30000)")
    from arguments import PipelineParams
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    args.pipeline = pipeline.extract(args)
    main(args)