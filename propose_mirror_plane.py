import numpy as np
import os
import json
from argparse import ArgumentParser
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from scene.colmap_loader import read_points3D_binary

def propose_plane_robust(points_bin_path: str, output_json_path: str):
    print(f"Analyzing point cloud from: {points_bin_path}")
    try:
        xyz, _, _ = read_points3D_binary(points_bin_path)
        if xyz.shape[0] == 0:
            raise ValueError("Point cloud is empty.")

        y_coords = xyz[:, 1]
        
        # 모든 가우시안의 y값의 하위 5%를 실질적 바닥으로 둠
        y_percentile_5 = np.percentile(y_coords, 5)
        y_mean = np.mean(y_coords)

        print(f"Point cloud Y-axis 5th percentile: {y_percentile_5:.3f}")
        print(f"Point cloud Y-axis mean: {y_mean:.3f}")

        # 바닥에서 y값 평균 ~ 바닥 거리 만큼 뺌 -> 바닥 아래에 반사 평면 위치시킴
        plane_y_value = y_percentile_5 - (y_mean - y_percentile_5)

        # 법선벡터는 (0, 1, 0)으로 고정
        plane_params = {
            "a": 0.0,
            "b": 1.0,
            "c": 0.0,
            "d": -plane_y_value
        }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(plane_params, f, indent=4)
        
        print(f"\nSuccessfully proposed robust mirror plane: y = {plane_y_value:.3f}")
        print(f"Plane parameters saved to: {output_json_path}")

    except Exception as e:
        print(f"Error: Failed to propose mirror plane. {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Propose a robust mirror plane from a COLMAP sparse point cloud.")
    parser.add_argument("-s", "--source_path", required=True, type=str, help="Path to the COLMAP dataset directory")
    parser.add_argument("-m", "--model_path", required=True, type=str, help="Path to the output model directory where the plane file will be saved")
    args = parser.parse_args()

    points_3d_bin = os.path.join(args.source_path, "sparse/0/points3D.bin")
    output_json = os.path.join(args.model_path, "mirror_plane.json")

    propose_plane_robust(points_3d_bin, output_json)
