import numpy as np
import torch
import cv2
import os

def generate_depth_map_from_tensor(image_tensor, model, transform, device, image_name):
    """
    Generate depth map with values between 0 and 1 (where 0 is the furthest) from 
    a PyTorch tensor image.
    """
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction_resized = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(img.shape[0], img.shape[1]),
            mode='bicubic',
            align_corners=False,
        ).squeeze(0).squeeze(0)

    depth_map = prediction_resized.cpu()

    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

    visualize_depth_map(depth_map_normalized, image_name)

    return depth_map_normalized  # [H, W]

def visualize_depth_map(depth_map_normalized, image_name, percentile = 0.2):
    """
    Visualize and save the depth map, coloring the most distant points in red and
    the closest in green.
    """
    depth_map_vis = (depth_map_normalized * 255).numpy().astype("uint8")

    depth_map_color = cv2.cvtColor(depth_map_vis, cv2.COLOR_GRAY2BGR)
    depth_array = depth_map_normalized.numpy()

    high_depth_mask = depth_array > 1 - percentile
    low_depth_mask = depth_array < percentile
    depth_map_color[high_depth_mask] = [0, 255, 0]  # green
    depth_map_color[low_depth_mask] = [0, 0, 255]   # red

    os.makedirs('depth_maps', exist_ok=True)
    cv2.imwrite(f"depth_maps/{image_name}_depth_map.png", depth_map_color)
