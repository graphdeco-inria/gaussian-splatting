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
    scene_info = torch.load(os.path.join(args.model_path, "chkpnt30000.pth")) # Load model gốc
    gaussians.restore(scene_info[0], args.model_path)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Đang đọc file CSV: {args.csv_path}")
    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            image_name = row['image_name']
            width = int(row['width'])
            height = int(row['height'])
            
            # Đọc Quaternion và Translation (World-to-Camera theo chuẩn COLMAP)
            qvec = np.array([float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])])
            tvec = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])
            
            R = qvec2rotmat(qvec)
            
            # Tính toán góc nhìn (FOV) từ Focal Length
            fx = float(row['fx'])
            fy = float(row['fy'])
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            
            # Cấu hình Camera cho 3DGS
            custom_cam = Camera(
                colmap_id=idx, 
                R=R.transpose(), # 3DGS yêu cầu ma trận chuyển vị
                T=tvec, 
                FoVx=FovX, 
                FoVy=FovY, 
                image=torch.zeros((3, height, width)), # Dummy image
                gt_alpha_mask=None,
                image_name=image_name, 
                uid=idx,
                data_device="cuda"
            )
            
            # Sinh ảnh (Rendering)
            with torch.no_grad():
                render_pkg = render(custom_cam, gaussians, args.pipeline, background)
                rendered_image = render_pkg["render"]
            
            # Lưu ảnh
            rendered_image = rendered_image.cpu().permute(1, 2, 0).numpy()
            rendered_image = (np.clip(rendered_image, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(rendered_image).save(os.path.join(args.output_dir, image_name))
            
            print(f"Đã sinh ảnh: {image_name}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render test poses for BTS Digital Twin")
    parser.add_argument("--model_path", type=str, required=True, help="Đường dẫn tới thư mục model đã train (chứa point_cloud)")
    parser.add_argument("--csv_path", type=str, required=True, help="Đường dẫn tới test_poses.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục lưu ảnh đầu ra (vd: submission/scene_001)")
    
    # 3DGS pipeline params (giữ nguyên mặc định)
    class PipelineParams:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    
    args = parser.parse_args()
    args.pipeline = PipelineParams()
    main(args)