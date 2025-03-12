import os
import numpy as np
from sklearn.neighbors import BallTree
from colmap.scripts.python.read_write_model import *
import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA
import rtree
from shapely.geometry import Point, box
from collections import defaultdict
from sklearn.decomposition import PCA
from colmap_utils import compute_extrinsics, get_colmap_data
from matplotlib import pyplot as plt

class Node:
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.children = []
        self.unoccupied = True
        self.sampled_point_uv = None
        self.sampled_point_rgb = None
        self.sampled_point_depth = None
        self.depth_interpolated = None
        self.sampled_point_neighbours_indices = None
        self.inference_count = 0
        self.rejection_count = 0
        self.sampled_point_world = None
        self.matching_log = {}
        self.points3d_indices = []
        self.points3d_depths = []
        self.points3d_rgb = []
        self.sampled_point_neighbours_uv = None

    def get_error(self, img):
        # Calculate the standard deviation of the region as an error metric
        region = img[self.y0:self.y0+self.height, self.x0:self.x0+self.width]
        return np.std(region)

def recursive_subdivide(node, threshold, min_pixel_size, img):
    if node.get_error(img) <= threshold:
        return

    w_1 = node.width // 2
    w_2 = node.width - w_1
    h_1 = node.height // 2
    h_2 = node.height - h_1

    if w_1 <= min_pixel_size or h_1 <= min_pixel_size:
        return

    # Create four children nodes
    x1 = Node(node.x0, node.y0, w_1, h_1)  # top left
    recursive_subdivide(x1, threshold, min_pixel_size, img)
    
    x2 = Node(node.x0, node.y0 + h_1, w_1, h_2)  # bottom left
    recursive_subdivide(x2, threshold, min_pixel_size, img)
    
    x3 = Node(node.x0 + w_1, node.y0, w_2, h_1)  # top right
    recursive_subdivide(x3, threshold, min_pixel_size, img)
    
    x4 = Node(node.x0 + w_1, node.y0 + h_1, w_2, h_2)  # bottom right
    recursive_subdivide(x4, threshold, min_pixel_size, img)

    node.children = [x1, x2, x3, x4]

def quadtree_decomposition(img, threshold, min_pixel_size):
    root = Node(0, 0, img.shape[1], img.shape[0])
    recursive_subdivide(root, threshold, min_pixel_size, img)
    return root

def gather_leaf_nodes(node, leaf_nodes):
    if not node.children:
        leaf_nodes.append(node)
    else:
        for child in node.children:
            gather_leaf_nodes(child, leaf_nodes)

def find_leaf_node(root, pixel_x, pixel_y):
    if not (root.x0 <= pixel_x < root.x0 + root.width and
            root.y0 <= pixel_y < root.y0 + root.height):
        return None  # 픽셀이 루트 노드의 범위를 벗어난 경우

    current = root
    while current.children:
        for child in current.children:
            if (child.x0 <= pixel_x < child.x0 + child.width and
                child.y0 <= pixel_y < child.y0 + child.height):
                current = child
                break
        else:
            # 적절한 자식 노드를 찾지 못한 경우
            return current
    
    return current

def draw_quadtree(node, ax):
    if not node.children:
        rect = plt.Rectangle((node.x0, node.y0), node.width, node.height, fill=False, edgecolor='red')
        ax.add_patch(rect)
    else:
        for child in node.children:
            draw_quadtree(child, ax)

def pixel_to_3d(point, depth, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x, y = point
    z = depth
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    return np.array([x_3d, y_3d, z])

def project_3d_to_2d(points3d, intrinsics, extrinsics):
    """
    Project a 3D point to 2D using camera intrinsics and extrinsics.
    
    Args:
    - point_3d: 3D point as a numpy array (x, y, z).
    - intrinsics: Camera intrinsics matrix (3x3).
    - extrinsics: Camera extrinsics matrix (4x4).
    
    Returns:
    - 2D point as a tuple (x, y) in pixel coordinates.
    """
    point_3d_homogeneous = np.hstack((points3d, np.ones((points3d.shape[0], 1)))) # Convert to homogeneous coordinates
    point_camera = extrinsics @ point_3d_homogeneous.T
    
    point_image_homogeneous = intrinsics @ point_camera
    point_2d = point_image_homogeneous[:2] / point_image_homogeneous[2]

    return point_2d.T, point_image_homogeneous[2:].T

def check_points_in_quadtree(points3d_pix, points3d_depth, points3d_rgb, leaf_nodes, near_culled_indices):
    rect_index = rtree.index.Index()
    rectangles = [((node.x0, node.y0, node.x0 + node.width, node.y0 + node.height), i) for i, node in enumerate(leaf_nodes)]
    for rect, i in rectangles:
        rect_index.insert(i, rect)
    rectangle_indices = []
    for i, (x, y) in enumerate(points3d_pix):
        if near_culled_indices[i]:
            continue
        point = Point(x, y)
        matches = list(rect_index.intersection((x, y, x, y)))
        for match in matches:
            if box(*rectangles[match][0]).contains(point):
                rectangle_indices.append([match, i])

    for index, point3d_idx in rectangle_indices:
        leaf_nodes[index].unoccupied = False
        leaf_nodes[index].points3d_indices.append(point3d_idx)
        leaf_nodes[index].points3d_depths.append(points3d_depth[point3d_idx])
        leaf_nodes[index].points3d_rgb.append(points3d_rgb[point3d_idx])
    return leaf_nodes

def compute_normal_vector(points3d_cameraframe, points3d_depth, indices):
    points3d_cameraframe = np.concatenate((points3d_cameraframe[:,:2]*points3d_depth.reshape(-1, 1), points3d_depth.reshape(-1, 1)), axis=1)
    p1 = points3d_cameraframe[indices[:, 0]]
    p2 = points3d_cameraframe[indices[:, 1]]
    p3 = points3d_cameraframe[indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    a, b, c = normal[:, 0], normal[:, 1], normal[:, 2]
    d = -a * p1[:, 0] - b * p1[:, 1] - c * p1[:, 2]

    return a, b, c, d

def compute_depth(sample_points_cameraframe, a, b, c, d, cosine_threshold=0.01):
    direction_vectors = sample_points_cameraframe
    t = -d.reshape(-1,1) / np.sum(np.concatenate((a.reshape(-1,1), b.reshape(-1,1), c.reshape(-1,1)), axis=1) * direction_vectors, axis=1).reshape(-1,1)
    normal_vectors = np.concatenate((a.reshape(-1,1), b.reshape(-1,1), c.reshape(-1,1)), axis=1)
    cosine_similarity = np.abs(np.sum(normal_vectors * direction_vectors, axis=1)).reshape(-1,1) / np.linalg.norm(normal_vectors, axis=1).reshape(-1,1) / np.linalg.norm(direction_vectors, axis=1).reshape(-1,1)
    rejected_indices = cosine_similarity < cosine_threshold
    depth = t.reshape(-1,1)*direction_vectors[:, 2:]
    return depth, rejected_indices.reshape(-1)

def find_perpendicular_triangle_indices(a, b, c, cosine_threshold=0.01):
    normal_vectors = np.concatenate((a.reshape(-1,1), b.reshape(-1,1), c.reshape(-1,1)), axis=1)
    camera_normal = np.array([0, 0, 1]).reshape(1, 3)
    cosine_similarity = np.dot(normal_vectors, camera_normal.T) / np.linalg.norm(normal_vectors, axis=1).reshape(-1,1)
    cosine_culled_indices = -cosine_threshold < cosine_similarity < cosine_threshold
    return cosine_culled_indices


def find_depth_from_nn(image, leaf_nodes, points3d_pix, points3d_depth, near_culled_indices, intrinsics_camera, rotations_image, translations_image, depth_cutoff = 2.):
    sampled_points = []
    for node in leaf_nodes:
        if node.unoccupied:
            sampled_points.append(node.sampled_point_uv)
    sampled_points = np.array(sampled_points)

    # Later we should store the 3D points' original indices
    # This is because the 3D points cannot be masked during NN search.
    # Near_culled_indices indicates the indices of the 3D points that are outside the camera frustum or have negative depth.
    original_indices = np.arange(points3d_pix.shape[0])
    original_indices = original_indices[~near_culled_indices]

    # Only infrustum 3D points are used for depth interpolation.
    points3d_pix = points3d_pix[~near_culled_indices]
    points3d_depth = points3d_depth[~near_culled_indices]
    # Construct a BallTree for nearest neighbor search.
    tree = BallTree(points3d_pix, leaf_size=40)
    # Query the nearest 3 neighbors for each sampled point.
    distances, indices = tree.query(sampled_points, k=3)
    # Inverse the camera intrinsic matrix for generating the camera rays.
    inverse_intrinsics = np.linalg.inv(intrinsics_camera)
    # Generate the camera rays in camera coordinate system.
    sampled_points_homogeneous = np.concatenate((sampled_points, np.ones((sampled_points.shape[0], 1))), axis=1)
    sampled_points_cameraframe = (inverse_intrinsics @ sampled_points_homogeneous.T).T
    # Transform the 3D points to the camera coordinate system with precomputed depths.
    points3d_pix_homogeneous = np.concatenate((points3d_pix, np.ones((points3d_pix.shape[0], 1))), axis=1)
    points3d_cameraframe = (inverse_intrinsics @ points3d_pix_homogeneous.T).T
    # Compute the normal vector and the distance of the triangle formed by the nearest 3 neighbors.
    a, b, c, d = compute_normal_vector(points3d_cameraframe, points3d_depth, indices)
    # Compute the depth of the sampled points from the normal vector.
    sampled_points_depth, cosine_culled_indices = compute_depth(sampled_points_cameraframe, a, b, c, d)
    # Reject the points that have negative depth, are NaN, or the plane constructed by the nearest 3 neighbors is parallel to the camera ray.
    depth_rejected_indices = (sampled_points_depth < depth_cutoff).reshape(-1) | np.isnan(sampled_points_cameraframe).any(axis=1) | cosine_culled_indices.reshape(-1)
    # Transform the sampled points to the world coordinate system.
    sampled_points_world = transform_camera_to_world(np.concatenate((sampled_points_cameraframe[:,:2]*sampled_points_depth.reshape(-1,1), sampled_points_depth.reshape(-1,1)), axis=1), rotations_image, translations_image)
    sampled_points_world = sampled_points_world[:, :3]

    augmented_count = 0
    depth_index = 0
    for node in leaf_nodes:
        if node.unoccupied:
            if depth_rejected_indices[depth_index]:
                node.depth_interpolated = False
            else:
                node.sampled_point_depth = sampled_points_depth[depth_index]
                node.sampled_point_world = sampled_points_world[depth_index]
                node.depth_interpolated = True
                node.sampled_point_neighbours_indices = original_indices[indices[depth_index]]
                node.sampled_point_neighbours_uv = points3d_pix[indices[depth_index]]
                augmented_count += 1
            depth_index += 1

    return leaf_nodes, augmented_count

def transform_camera_to_world(sampled_points_cameraframe, rotations_image, translations_image):
    extrinsics_image = np.concatenate((np.array(rotations_image), np.array(translations_image).reshape(3,1)), axis=1)
    extrinsics_4x4 = np.concatenate((extrinsics_image, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    sampled_points_cameraframe_homogeneous = np.concatenate((sampled_points_cameraframe, np.ones((sampled_points_cameraframe.shape[0], 1))), axis=1)
    sampled_points_worldframe = np.linalg.inv(extrinsics_4x4) @ sampled_points_cameraframe_homogeneous.T

    return sampled_points_worldframe.T

def pixelwise_rgb_diff(point_a, point_b, threshold=0.3):
    return np.linalg.norm(point_a - point_b) / 255. > threshold

def image_quadtree_augmentation(image_key, 
                                image_dir, 
                                colmap_cameras, 
                                colmap_images, 
                                colmap_points3D,
                                points3d, points3d_rgb, 
                                intrinsics_camera, 
                                rotations_image, 
                                translations_image, 
                                quadtree_std_threshold=7, 
                                quadtree_min_pixel_size=5, 
                                visibility_aware_culling=False):
    image_name = colmap_images[image_key].name
    image_path = os.path.join(image_dir, image_name)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    quadtree_root = quadtree_decomposition(image_grayscale, quadtree_std_threshold, quadtree_min_pixel_size)

    leaf_nodes = []
    gather_leaf_nodes(quadtree_root, leaf_nodes)
    # Project 3D points onto the image plane.
    points3d_pix, points3d_depth = project_3d_to_2d(points3d, intrinsics_camera[colmap_images[image_key].camera_id], np.concatenate((np.array(rotations_image[image_key]), np.array(translations_image[image_key]).reshape(3,1)), axis=1))
    # Cull the points that are outside the camera frustum and have negative depth.
    near_culled_indices = (points3d_pix[:, 0] < 0) | (points3d_pix[:, 0] >= image.shape[1]) | (points3d_pix[:, 1] < 0) | (points3d_pix[:, 1] >= image.shape[0]) | (points3d_depth.reshape(-1) < 0) | np.isnan(points3d_pix).any(axis=1)
    points3d_pix_rgb_diff = np.zeros(points3d_pix.shape[0], dtype=np.bool_)
    if visibility_aware_culling:
        for i, point in enumerate(points3d_pix):
            if near_culled_indices[i]:
                continue
            point_a = image[int(point[1]), int(point[0])]
            point_b = points3d_rgb[i]
            points3d_pix_rgb_diff[i] = pixelwise_rgb_diff(point_a, point_b)
        near_culled_indices = near_culled_indices | points3d_pix_rgb_diff.reshape(-1)
    # Check every node in the quadtree that contains projected 3D points.
    leaf_nodes = check_points_in_quadtree(points3d_pix, points3d_depth, points3d_rgb, leaf_nodes, near_culled_indices)
    # Sample points from the leaf nodes that are not occupied by projected 3D points.
    sampled_points = []
    sampled_points_rgb = []
    for node in leaf_nodes:
        if node.unoccupied:
            node.sampled_point_uv = np.array([node.x0, node.y0]) + np.random.sample(2) * np.array([node.width, node.height])
            node.sampled_point_rgb = image[int(node.sampled_point_uv[1]), int(node.sampled_point_uv[0])]
            sampled_points.append(node.sampled_point_uv)
            sampled_points_rgb.append(node.sampled_point_rgb)
    sampled_points = np.array(sampled_points)
    sampled_points_rgb = np.array(sampled_points_rgb)
    # Interpolate the depth of the sampled points from the nearest 3D points.
    leaf_nodes, augmented_count = find_depth_from_nn(image, leaf_nodes, points3d_pix, points3d_depth, near_culled_indices, intrinsics_camera[colmap_images[image_key].camera_id], rotations_image[image_key], translations_image[image_key])

    return quadtree_root, augmented_count

def transform_sample_3d(image_key, root, colmap_images, colmap_cameras, intrinsics_camera, rotations_image, translations_image):
    # Gather leaf nodes and transform the sampled 2D points to the 3D world coordinates.
    leaf_nodes = []
    gather_leaf_nodes(root, leaf_nodes)
    sample_points_imageframe = []
    sample_points_depth = []
    sample_points_rgb = []
    for node in leaf_nodes:
        if node.unoccupied:
            if node.depth_interpolated:
                sample_points_imageframe.append(node.sampled_point_uv)
                sample_points_depth.append(node.sampled_point_depth)
                sample_points_rgb.append(node.sampled_point_rgb)
    sample_points_imageframe = np.array(sample_points_imageframe)
    sample_points_depth = np.array(sample_points_depth)
    sample_points_rgb = np.array(sample_points_rgb)
    sample_points_cameraframe = (np.linalg.inv(intrinsics_camera[colmap_images[image_key].camera_id]) @ np.concatenate((sample_points_imageframe, np.ones((sample_points_imageframe.shape[0], 1))), axis=1).T).T
    sample_points_cameraframe = np.concatenate((sample_points_cameraframe[:,:2]*sample_points_depth.reshape(-1,1), sample_points_depth.reshape(-1, 1)), axis=1)
    sample_points_worldframe = transform_camera_to_world(sample_points_cameraframe, rotations_image[image_key], translations_image[image_key])
    return sample_points_worldframe[:,:-1], sample_points_rgb

def write_points3D_colmap_binary(points3D, xyz, rgb, file_path):
    with open(file_path, "wb") as fid:
        write_next_bytes(fid, len(points3D) + len(xyz), "Q")
        for j, (_, pt) in enumerate(points3D.items()):
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")
        id = points3D[max(points3D.keys())].id + 1
        print("starts from id=", id)
        for i in range(len(xyz)):
            write_next_bytes(fid, id + i, "Q")
            write_next_bytes(fid, xyz[i].tolist(), "ddd")
            write_next_bytes(fid, rgb[i].tolist(), "BBB")
            write_next_bytes(fid, 0.0, "d")
            write_next_bytes(fid, 1, "Q")
            write_next_bytes(fid, [0, 0], "ii")
    return i + id

def compare_local_texture(patch1, patch2):
    """
    Compare two patches using a Gaussian kernel.
    If the patch size is not 3x3, apply zero padding.
    """
    # Define a 3x3 Gaussian kernel (sigma=1.0)
    gaussian_kernel = np.array([
        [0.077847, 0.123317, 0.077847],
        [0.123317, 0.195346, 0.123317],
        [0.077847, 0.123317, 0.077847]
    ])

    # Check the patch size and apply zero padding if it is not 3x3.
    def pad_to_3x3(patch):
        if patch.shape[:2] != (3,3):
            padded = np.zeros((3,3,3) if len(patch.shape)==3 else (3,3))
            h, w = patch.shape[:2]
            y_start = (3-h)//2
            x_start = (3-w)//2
            padded[y_start:y_start+h, x_start:x_start+w] = patch
            return padded
        return patch

    patch1 = pad_to_3x3(patch1)
    patch2 = pad_to_3x3(patch2)

    if len(patch1.shape) == 3:  # RGB image
        # Apply weights to each channel and calculate the difference.
        weighted_diff = np.sum([
            np.sum(gaussian_kernel * (patch1[:,:,c] - patch2[:,:,c])**2)
            for c in range(3)
        ])
    else:  # Grayscale image
        weighted_diff = np.sum(gaussian_kernel * (patch1 - patch2)**2)

    return np.sqrt(weighted_diff)/255.