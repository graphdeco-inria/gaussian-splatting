import cv2
import numpy as np
import json
import os
import argparse
from pycuda import driver, gpuarray

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
    return data


# Function to check for CUDA error
def check_cuda_error(err):
    if err != 0:
        driver.Context.synchronize()  # Synchronize to ensure proper error handling
        print("CUDA error:", driver.Error(err))
        exit(1)
        
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

def process_image(image_path, output_path, distortion_param_json_path):
    """Processes a single image with distortion correction."""
    image = cv2.imread(image_path)
    height, width, channel = image.shape
    rectified_image = np.zeros((height, width, channel), dtype=image.dtype)

    # read calibration data
    data = readCalibrationJson(distortion_param_json_path)
    lookup_table = data.lens_distortion_lookup_table
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

    cv2.imwrite(output_path, rectified_image)
    print(f"finish process {image_path}")


def process_image_cuda(image_path, output_path, distortion_param_json_path):
    # Load image data on host (CPU)
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    # read calibration data
    data = readCalibrationJson(distortion_param_json_path)
    lookup_table = data.inverse_lens_distortion_lookup_table
    distortion_center = data.lens_distortion_center
    reference_dimensions = data.intrinsic_matrix_reference_dimensions
    ratio_x = width / reference_dimensions[0]
    ratio_y = height / reference_dimensions[1]
    distortion_center[0] = distortion_center[0] * ratio_x
    distortion_center[1] = distortion_center[1] * ratio_y

    # Allocate memory on GPU for image data, lookup table, and rectified image
    image_gpu = gpuarray.to_gpu(image.astype(np.float32))
    lookup_table_gpu = gpuarray.to_gpu(lookup_table.astype(np.float32))
    rectified_image_gpu = gpuarray.empty((height, width, channel), np.float32)

    # Prepare CUDA kernel code (replace with your actual kernel implementation)
    kernel_code = """
    __global__ void process_image(float* image, float* lookup_table, float* distortion_center, float* rectified_image, int width, int height) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= width * height) {
            return;
        }

        int y = idx / width;
        int x = idx % width;

        // Replace with your actual distortion correction logic using `get_lens_distortion_point`
        float rectified_x, rectified_y;
        get_lens_distortion_point(x, y, lookup_table, distortion_center, &rectified_x, &rectified_y);

        if (rectified_x < 0 || rectified_x >= width || rectified_y < 0 || rectified_y >= height) {
            return;
        }

        int rectified_idx = (int)rectified_y * width + (int)rectified_x;
        rectified_image[rectified_idx] = image[idx];
    }
    """

    # Compile the kernel
    mod = driver.SourceModule(kernel_code)
    process_image = mod.get_function("process_image")

    # Set kernel parameters and launch
    threads_per_block = (16, 16)  # Adjust block size as needed
    grid_size = (width // threads_per_block[0] + 1, height // threads_per_block[1] + 1)
    check_cuda_error(process_image(
        image_gpu, lookup_table_gpu, driver.In(distortion_center), rectified_image_gpu, np.int32(width), np.int32(height), block=threads_per_block, grid=grid_size
    ))

    # Transfer rectified image back to host and convert to uint8 (assuming original image format)
    rectified_image = rectified_image_gpu.get_array().astype(np.uint8)

    # Save the processed image
    cv2.imwrite(output_path, rectified_image)

    # Free GPU memory
    image_gpu.free()
    lookup_table_gpu.free()
    rectified_image_gpu.free()

def rectify_image(image_folder_path, distortion_param_json_path, output_image_folder_path):
    for filename in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, filename)
        output_path = os.path.join(output_image_folder_path, filename)
        process_image_cuda(image_path, output_path, distortion_param_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="undistort ARKit image using distortion params get from AVfoundation")
    parser.add_argument("--input", type=str)
    parser.add_argument("--json", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    input_image_folder_path = args.input
    distortion_param_json_path = args.json
    output_image_folder_path = args.output

    rectify_image(input_image_folder_path, distortion_param_json_path, output_image_folder_path)