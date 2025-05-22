import os
import glob
import math
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def get_theta(x, y):
    if y < 0:
        return -np.arctan2(y, x)
    else:
        return 2 * math.pi - np.arctan2(y, x)


def create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z):
    map_x = np.zeros((output_sqr, output_sqr), dtype=np.float32)
    map_y = np.zeros((output_sqr, output_sqr), dtype=np.float32)
    for row in tqdm(range(output_sqr), desc="Bottom/Top mapping"):
        for col in range(output_sqr):
            x = row - output_sqr / 2.0
            y = col - output_sqr / 2.0

            rho = np.sqrt(x * x + y * y + z * z)
            norm_theta = get_theta(x, y) / (2 * math.pi)
            norm_phi = (math.pi - np.arccos(z / rho)) / math.pi
            ix = norm_theta * input_w
            iy = norm_phi * input_h

            # Wrap-around
            ix = ix % input_w
            iy = iy % input_h

            map_x[row, col] = ix
            map_y[row, col] = iy

    return map_x, map_y


def create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x):
    map_x = np.zeros((output_sqr, output_sqr), dtype=np.float32)
    map_y = np.zeros((output_sqr, output_sqr), dtype=np.float32)
    for row in tqdm(range(output_sqr), desc="Front/Back mapping"):
        for col in range(output_sqr):
            z = row - output_sqr / 2.0
            y = col - output_sqr / 2.0

            rho = np.sqrt(x * x + y * y + z * z)
            norm_theta = get_theta(x, y) / (2 * math.pi)
            norm_phi = (math.pi - np.arccos(z / rho)) / math.pi
            ix = norm_theta * input_w
            iy = norm_phi * input_h

            ix = ix % input_w
            iy = iy % input_h

            map_x[row, col] = ix
            map_y[row, col] = iy

    return map_x, map_y


def create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y):
    map_x = np.zeros((output_sqr, output_sqr), dtype=np.float32)
    map_y = np.zeros((output_sqr, output_sqr), dtype=np.float32)
    for row in tqdm(range(output_sqr), desc="Left/Right mapping"):
        for col in range(output_sqr):
            z = row - output_sqr / 2.0
            x = col - output_sqr / 2.0

            rho = np.sqrt(x * x + y * y + z * z)
            norm_theta = get_theta(x, y) / (2 * math.pi)
            norm_phi = (math.pi - np.arccos(z / rho)) / math.pi
            ix = norm_theta * input_w
            iy = norm_phi * input_h

            ix = ix % input_w
            iy = iy % input_h

            map_x[row, col] = ix
            map_y[row, col] = iy

    return map_x, map_y


def create_cube_map(bottom_img, top_img, front_img, back_img, left_img, right_img, output_sqr):
    output_w = output_sqr * 4
    output_h = output_sqr * 3
    cube_map_img = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    # Layout net:
    #       [   top   ]
    # [left][front][right][back]
    #       [ bottom ]
    cube_map_img[0:output_sqr, output_sqr:output_sqr*2] = top_img
    cube_map_img[output_sqr:output_sqr*2, 0:output_sqr] = left_img
    cube_map_img[output_sqr:output_sqr*2, output_sqr:output_sqr*2] = front_img
    cube_map_img[output_sqr:output_sqr*2, output_sqr*2:output_sqr*3] = right_img
    cube_map_img[output_sqr:output_sqr*2, output_sqr*3:output_sqr*4] = back_img
    cube_map_img[output_sqr*2:output_sqr*3, output_sqr:output_sqr*2] = bottom_img

    return cube_map_img


def process_image(image_path, output_dir, output_sqr, normalized_f):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    input_h, input_w = img.shape[:2]
    base = os.path.splitext(os.path.basename(image_path))[0]

    # Bottom face
    z = output_sqr / (2.0 * normalized_f)
    bm_x, bm_y = create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z)
    bottom = cv2.remap(img, bm_x, bm_y, cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, f"{base}_bottom.png"), bottom)

    # Top face
    z = -z
    tm_x, tm_y = create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z)
    top = cv2.remap(img, tm_x, tm_y, cv2.INTER_CUBIC)
    top = cv2.flip(top, 0)
    cv2.imwrite(os.path.join(output_dir, f"{base}_top.png"), top)

    # Front face
    x = -output_sqr / (2.0 * normalized_f)
    fm_x, fm_y = create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x)
    front = cv2.remap(img, fm_x, fm_y, cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, f"{base}_front.png"), front)

    # Back face
    x = -x
    bm_x2, bm_y2 = create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x)
    back = cv2.remap(img, bm_x2, bm_y2, cv2.INTER_CUBIC)
    back = cv2.flip(back, 1)
    cv2.imwrite(os.path.join(output_dir, f"{base}_back.png"), back)

    # Left face
    y = -output_sqr / (2.0 * normalized_f)
    lm_x, lm_y = create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y)
    left = cv2.remap(img, lm_x, lm_y, cv2.INTER_CUBIC)
    left = cv2.flip(left, 1)
    cv2.imwrite(os.path.join(output_dir, f"{base}_left.png"), left)

    # Right face
    y = -y
    rm_x, rm_y = create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y)
    right = cv2.remap(img, rm_x, rm_y, cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, f"{base}_right.png"), right)

    # # Cube map
    # cube = create_cube_map(bottom, top, front, back, left, right, output_sqr)
    # cv2.imwrite(os.path.join(output_dir, f"{base}_cube_map.png"), cube)


def main():
    parser = argparse.ArgumentParser(description="Convert all equirectangular images in a folder to cube faces and cube map.")
    parser.add_argument("--input_dir", required=True, help="Path to folder containing source images")
    parser.add_argument("--output_dir", required=True, help="Path to folder for saving outputs")
    parser.add_argument("--size", type=int, default=800, help="Size (pixels) of each cube face, default=800")
    parser.add_argument("--f", type=float, default=1.0, help="Normalized focal length, default=1.0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, pat)))

    for img_path in sorted(image_paths):
        print(f"Processing {img_path}...")
        process_image(img_path, args.output_dir, args.size, args.f)

if __name__ == "__main__":
    main()
