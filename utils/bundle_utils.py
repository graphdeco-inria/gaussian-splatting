import random
import os
from colmap.scripts.python.read_write_model import *
import numpy as np
from collections import defaultdict
from colmap_utils import compute_extrinsics, compute_intrinsics, get_colmap_data
import cv2
from sklearn.decomposition import PCA



def build_covisibility_matrix(images, points3D):
    n_images = len(images)
    covisibility_matrix = np.zeros((n_images, n_images))
    
    id_to_idx = {img_id: idx for idx, img_id in enumerate(images.keys())}
    idx_to_id = {idx: img_id for img_id, idx in id_to_idx.items()}

    for point3D in points3D.values():
        image_ids = point3D.image_ids
        for i in range(len(image_ids)):
            for j in range(i+1, len(image_ids)):
                id1, id2 = image_ids[i], image_ids[j]
                idx1, idx2 = id_to_idx[id1], id_to_idx[id2]
                covisibility_matrix[idx1, idx2] += 1
                covisibility_matrix[idx2, idx1] += 1

    return covisibility_matrix, id_to_idx, idx_to_id

def create_covisibility_graph(covisibility_matrix, idx_to_id):
    graph = defaultdict(dict)
    for i in range(len(covisibility_matrix)):
        for j in range(len(covisibility_matrix)):
            if i != j and covisibility_matrix[i,j] > 0:
                id1 = idx_to_id[i]
                id2 = idx_to_id[j]
                graph[id1][id2] = covisibility_matrix[i,j]

    return graph

def create_sequence_from_covisibility_graph(covisibility_graph, min_covisibility=20):
    if not covisibility_graph:
        print("No covisibility graph found")
        return []
    
    start_node = max(covisibility_graph.keys(),
                     key=lambda k: sum(1 for v in covisibility_graph[k].values() if v >= min_covisibility))
    
    visited = set([start_node])
    sequence = [start_node]
    current = start_node

    while len(sequence) < len(covisibility_graph):
        next_node = None
        max_covisibility = -1

        for neighbor, covisibility in covisibility_graph[current].items():
            if neighbor not in visited and covisibility > max_covisibility and covisibility >= min_covisibility:
                next_node = neighbor
                max_covisibility = covisibility

        if next_node is None:
            for node in covisibility_graph:
                if node not in visited:
                    for seq_node in sequence:
                        covisibility = covisibility_graph[node].get(seq_node, 0)
                        if covisibility > max_covisibility and covisibility >= min_covisibility:
                            max_covisibility = covisibility
                            next_node = node

        if next_node is None:
            next_node = min(set(covisibility_graph.keys()) - visited)

        current = next_node
        visited.add(current)
        sequence.append(current)

    return sequence

def cluster_cameras(model_path, camera_order):
    colmap_path = os.path.join(model_path, 'sparse/0')
    colmap_images, colmap_points3D, colmap_cameras = get_colmap_data(colmap_path)
    if camera_order == 'covisibility':
        covisibility_matrix, id_to_idx, idx_to_id = build_covisibility_matrix(colmap_images, colmap_points3D)
        covisibility_graph = create_covisibility_graph(covisibility_matrix, idx_to_id)
        covisibility_sequence = create_sequence_from_covisibility_graph(covisibility_graph)

        image_id_name = [[colmap_images[key].id, colmap_images[key].name] for key in colmap_images.keys()]
        image_id_name_sorted = sorted(image_id_name, key=lambda x: x[1])
        test_id = []
        train_id = []
        count = 0
        train_only_idx = []
        for i, (id, name) in enumerate(image_id_name_sorted):
            if i % 8 == 0:
                test_id.append(id)
            else:
                train_id.append(id)
                train_only_idx.append(count)
                count+=1
       
        rotations_image, translations_image = compute_extrinsics(colmap_images)

        train_only_visibility_idx = []
        for id in covisibility_sequence:
            if id in train_id:
                train_only_visibility_idx.append(id)
        train_only_visibility_idx = np.array(train_only_visibility_idx)
        sorted_keys = train_only_visibility_idx  # sorted_indices 대신 sorted_keys로 할당

    elif camera_order == 'PCA':
        image_idx_name = [[colmap_images[key].id, colmap_images[key].name] for key in colmap_images.keys()]
        image_idx_name_sorted = sorted(image_idx_name, key=lambda x: x[1])
        test_idx = []
        train_idx = []
        for i, (idx, name) in enumerate(image_idx_name_sorted):
            if i % 8 == 0:
                test_idx.append(idx)
            else:
                train_idx.append(idx)

        rotations_image, translations_image = compute_extrinsics(colmap_images)

        cam_center = []
        key = []
        for idx in train_idx:
            cam_center.append((-rotations_image[idx].T @ translations_image[idx].reshape(3,1)))
            key.append(idx)

        cam_center = np.array(cam_center)[:,:,0]
        pca = PCA(n_components=2)
        cam_center_2d = pca.fit_transform(cam_center)

        center_cam_center = np.mean(cam_center_2d, axis=0)
        centered_cam_center = cam_center_2d - center_cam_center
        angles = np.arctan2(centered_cam_center[:, 1], centered_cam_center[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_cam_centers = cam_center_2d[sorted_indices]
        sorted_keys = np.array(key)[sorted_indices]

    return sorted_keys

def bundle_start_index_generator(sorted_keys, initial_interval):
    start_indices = []
    cluster_sizes = []
    for i in range(200):
        start_indices.append((initial_interval*i)%len(sorted_keys))
        cluster_sizes.append(initial_interval)

    return start_indices, cluster_sizes

def adaptive_cluster(start_idx, sorted_keys, cluster_size = 40, offset = 0):
    idx = start_idx
    indices = [sorted_keys[index % len(sorted_keys)] for index in range(idx, idx + cluster_size)]
    random_index = random.choice(indices)
    return random_index
