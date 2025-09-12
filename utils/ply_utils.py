import open3d as o3d
import numpy as np
import torch

def write_lines(start_points, end_points, filename="out_lines.ply"):
    
    # Stack all vertices
    all_points = np.vstack([start_points, end_points])
    all_colors = np.vstack([np.ones_like(start_points), np.zeros_like(end_points)]) * 255  # Start points are white, end points are black
    all_colors = all_colors.astype(np.uint8)

    P = start_points.shape[0]

    # Edges: start i -> end i
    edges = [[i, i + P] for i in range(P)]

    # Write ASCII PLY
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {2*P}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {P}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        # Write vertices
        for p, c in zip(all_points, all_colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

        # Write edges
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")

def write_gaussians_to_ply(gaussians, update_step=None, filename="out.ply"):
    means = gaussians._xyz.detach().cpu().numpy()
    colors = gaussians._features_dc
    colors = torch.sigmoid(colors)
    colors = colors.detach().cpu().numpy().squeeze()


    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(means)
    points.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, points)

    if update_step is not None:
        update = update_step.xyz_grad.detach().cpu().numpy()
        ends = means + update

        write_lines(means, ends, filename=filename.replace(".ply", "_lines.ply"))



