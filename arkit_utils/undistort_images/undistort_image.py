import cv2
import numpy as np
import json
import os
import argparse
from concurrent.futures import ProcessPoolExecutor

class Data:
    def __init__(self, intrinsic_matrix, intrinsic_matrix_reference_dimensions, lens_distortion_center, inverse_lens_distortion_lookup_table, lens_distortion_lookup_table):
        self.intrinsic_matrix = intrinsic_matrix
        self.intrinsic_matrix_reference_dimensions = intrinsic_matrix_reference_dimensions
        self.lens_distortion_center = lens_distortion_center
        self.inverse_lens_distortion_lookup_table = inverse_lens_distortion_lookup_table
        self.lens_distortion_lookup_table = lens_distortion_lookup_table

def readCalibrationJson(path):
    # Open the JSON file
    with open(path, "r") as f:
        # Read the contents of the file
        data = json.load(f)

    # Access specific data from the dictionary
    pixel_size = data["calibration_data"]["pixel_size"]
    intrinsic_matrix = data["calibration_data"]["intrinsic_matrix"]
    intrinsic_matrix_reference_dimensions = data["calibration_data"]["intrinsic_matrix_reference_dimensions"]
    lens_distortion_center = data["calibration_data"]["lens_distortion_center"]
    # Access specific elements from lists within the dictionary
    inverse_lut = data["calibration_data"]["inverse_lens_distortion_lookup_table"]
    lut = data["calibration_data"]["lens_distortion_lookup_table"]

    data = Data(intrinsic_matrix, intrinsic_matrix_reference_dimensions, lens_distortion_center, inverse_lut, lut)
    # # Print some of the data for verification
    # print(f"Pixel size: {pixel_size}")
    # print(f"Intrinsic matrix:\n {intrinsic_matrix}")
    # print(f"Lens distortion center: {lens_distortion_center}")
    # print(f"Inverse lookup table length: {len(inverse_lut)}")
    return data

def get_lens_distortion_point(point, lookup_table, distortion_center, image_size):
    radius_max_x = min(distortion_center[0], image_size[0] - distortion_center[0])
    radius_max_y = min(distortion_center[1], image_size[1] - distortion_center[1])
    radius_max = np.sqrt(radius_max_x**2 + radius_max_y**2)

    radius_point = np.sqrt(np.square(point[0] - distortion_center[0]) + np.square(point[1] - distortion_center[1]))

    magnification = lookup_table[-1]
    if radius_point < radius_max:
        relative_position = radius_point / radius_max * (len(lookup_table) - 1)
        frac = relative_position - np.floor(relative_position)
        lower_lookup = lookup_table[int(np.floor(relative_position))]
        upper_lookup = lookup_table[int(np.ceil(relative_position))]
        magnification = lower_lookup * (1.0 - frac) + upper_lookup * frac

    mapped_point = np.array([distortion_center[0] + (point[0] - distortion_center[0]) * (1.0 + magnification),
                              distortion_center[1] + (point[1] - distortion_center[1]) * (1.0 + magnification)])
    return mapped_point

def rectify_single_image(image_path, output_path, distortion_param_json_path, crop_x, crop_y):
    """Processes a single image with distortion correction."""
    image = cv2.imread(image_path)
    height, width, channel = image.shape
    rectified_image = np.zeros((height, width, channel), dtype=image.dtype)

    # read calibration data
    data = readCalibrationJson(distortion_param_json_path)
    lookup_table = data.inverse_lens_distortion_lookup_table# data.lens_distortion_lookup_table
    distortion_center = data.lens_distortion_center
    reference_dimensions = data.intrinsic_matrix_reference_dimensions
    ratio_x = width / reference_dimensions[0]
    ratio_y = height / reference_dimensions[1]
    distortion_center[0] = distortion_center[0] * ratio_x
    distortion_center[1] = distortion_center[1] * ratio_y

    for i in range(width):
        for j in range(height):
            rectified_index = np.array([i, j])
            original_index = get_lens_distortion_point(
                rectified_index, lookup_table, distortion_center, [width, height])

            if (original_index[0] < 0 or original_index[0] >= width or original_index[1] < 0 or original_index[1] >= height):
                continue

            rectified_image[j, i] = image[int(original_index[1]), int(original_index[0])]

    # crop image
    u_shift = crop_x
    v_shift = crop_y
    crop_image = rectified_image[v_shift:height-v_shift, u_shift:width-u_shift]
    cv2.imwrite(output_path, crop_image)
    print(f"finish process {image_path}")

def rectify_all_image(image_folder_path, distortion_param_json_path, output_image_folder_path, crop_x, crop_y):
    with ProcessPoolExecutor() as executor:
        for filename in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, filename)
            output_path = os.path.join(output_image_folder_path, filename)
            executor.submit(
                rectify_single_image, image_path, output_path, distortion_param_json_path, crop_x, crop_y)


def rectified_intrinsic(input_path, output_path, crop_x, crop_y):
    num_intrinsic = 0
    with open(input_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_intrinsic += 1
    print(num_intrinsic)

    camera_ids = np.empty((num_intrinsic, 1))
    widths = np.empty((num_intrinsic, 1))
    heights = np.empty((num_intrinsic, 1))
    paramss = np.empty((num_intrinsic, 4))

    count = 0
    with open(input_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])-crop_x*2
                height = int(elems[3])-crop_y*2
                params = np.array(tuple(map(float, elems[4:])))
                params[2] = params[2] - crop_x
                params[3] = params[3] - crop_y

                camera_ids[count] = camera_id
                widths[count] = width
                heights[count] = height
                paramss[count] = params

                count = count+1

    with open(output_path, "w") as f:
        for i in range(num_intrinsic):
            line = str(int(camera_ids[i])) + " " + "PINHOLE" + " " + str(int(widths[i]))+ " " + str(int(heights[i]))+ " " + str(paramss[i][0]) + " " + str(paramss[i][1])+ " " + str(paramss[i][2])+ " " +  str(paramss[i][3])
            f.write(line  + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="undistort ARKit image using distortion params get from AVfoundation")
    parser.add_argument("--input_base", type=str)
    parser.add_argument("--crop_x", type=int, default=10)
    parser.add_argument("--crop_y", type=int, default=8)


    args = parser.parse_args()
    base_folder_path = args.input_base
    input_image_folder_path = base_folder_path + "/distort_images"
    distortion_param_json_path = base_folder_path + "/sparse/0/calibration.json"
    output_image_folder_path = base_folder_path + "/post/images/"
    crop_x = args.crop_x
    crop_y = args.crop_y
    input_camera = base_folder_path + "/sparse/0/distort_cameras.txt"
    output_camera = base_folder_path + "/post/sparse/online/cameras.txt"


    if not os.path.exists(output_image_folder_path):
        os.makedirs(output_image_folder_path)

    rectify_all_image(input_image_folder_path, distortion_param_json_path, output_image_folder_path, crop_x, crop_y)
    rectified_intrinsic(input_camera, output_camera, crop_x, crop_y)

    