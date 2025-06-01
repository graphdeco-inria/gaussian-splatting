import sys, os
sys.path.append("/mnt/disk2/auggs/gaussian-splatting")
from utils.bundle_utils import *
from utils.colmap_utils import *
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def cluster_from_eigenvector(eigenvector, n_clusters, return_score=False):
    U = eigenvector[:, :n_clusters]
    U_norm = normalize(U, norm='l2')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(U_norm)
    if return_score:
        return clusters, silhouette_score(U_norm, clusters)
    else:
        return clusters

def cluster_score(covisibility_matrix):
    W = np.array(covisibility_matrix)
    W = (W - W.min()) / (W.max() - W.min())
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
    L_sym = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    sscore = []
    for i in range(2, 100):
        clusters, score = cluster_from_eigenvector(eigenvectors, i, return_score=True)
        sscore.append(score)
    return sscore

def intra_cluster_ordering(covisibility_matrix, clusters):
    W = np.array(covisibility_matrix)
    W = (W - W.min()) / (W.max() - W.min())
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
    L_sym = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    cluster_ids = np.unique(clusters)
    cluster_orderings = {}
    for cid in cluster_ids:
        indices = np.where(clusters == cid)[0]
        sub_adj = W[np.ix_(indices, indices)]
        degrees = sub_adj.sum(axis=1)
        start_node_idx = np.argmax(degrees)
        ordered_indices = [start_node_idx]
        remaining_indices = set(range(len(indices)))
        remaining_indices.remove(start_node_idx)
        current_idx = start_node_idx
        while remaining_indices:
            adjacencies = sub_adj[current_idx, list(remaining_indices)]
            next_idx = list(remaining_indices)[np.argmax(adjacencies)]
            ordered_indices.append(next_idx)
            remaining_indices.remove(next_idx)
            current_idx = next_idx
        cluster_orderings[cid] = indices[ordered_indices]
    return cluster_orderings

def cluster_ordering(covisibility_matrix, clusters, cluster_orderings):
    cluster_adj_matrix = np.zeros((len(cluster_orderings), len(cluster_orderings)))
    W = np.array(covisibility_matrix)
    W = (W - W.min()) / (W.max() - W.min())
    cluster_ids = np.unique(clusters)
    for i, ci in enumerate(cluster_ids):
        indices_i = np.where(clusters == ci)[0]
        for j, cj in enumerate(cluster_ids):
            if i >= j:
                continue
            indices_j = np.where(clusters == cj)[0]
            inter_adj = W[np.ix_(indices_i, indices_j)]
            cluster_adj_matrix[i, j] = cluster_adj_matrix[j, i] = inter_adj.sum()

    ordered_cluster_ids = []
    remaining_cluster_ids = set(range(len(cluster_ids)))

    cluster_degrees = cluster_adj_matrix.sum(axis=1)
    current_cluster = np.argmax(cluster_degrees)

    ordered_cluster_ids.append(current_cluster)
    remaining_cluster_ids.remove(current_cluster)

    # Greedy하게 다음 클러스터 선택
    while remaining_cluster_ids:
        adjacencies = cluster_adj_matrix[current_cluster, list(remaining_cluster_ids)]
        
        if adjacencies.max() == 0:
            # 연결된 클러스터가 없으면 남은 것 중 degree가 높은 것을 선택
            next_cluster = max(remaining_cluster_ids, key=lambda x: cluster_degrees[x])
        else:
            next_cluster = list(remaining_cluster_ids)[np.argmax(adjacencies)]
        
        ordered_cluster_ids.append(next_cluster)
        remaining_cluster_ids.remove(next_cluster)
        current_cluster = next_cluster

    # 실제 클러스터 ID로 변환
    ordered_clusters = [cluster_ids[idx] for idx in ordered_cluster_ids]
    return ordered_clusters

def order_indices_covisibility_matrix(covisibility_matrix, scene):
    W = np.array(covisibility_matrix)
    W = (W - W.min()) / (W.max() - W.min())
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
    L_sym = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    score = cluster_score(covisibility_matrix)
    
    # 실루엣 스코어 시각화를 파일로 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, len(score)+2), score)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score vs number of clusters')
    plt.savefig(f'covis_vis/silhouette_score_{scene}.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

    # 스코어 출력
    print("\nSilhouette scores for different numbers of clusters:")
    for i, s in enumerate(score, start=2):
        print(f"Clusters: {i}, Score: {s:.3f}")
    
    # 최적의 클러스터 수 추천
    best_n_clusters = np.argmax(score) + 2
    print(f"\nRecommended number of clusters (highest score): {best_n_clusters}")
    print(f"Score: {score[best_n_clusters-2]:.3f}")
    
    n_clusters = int(input(f"\nEnter the number of clusters (check silhouette_score_{scene}.png for the plot): "))
    
    covisibility_matrix_ordered = np.zeros_like(covisibility_matrix)
    sorted_indices = []
    clusters = cluster_from_eigenvector(eigenvectors, n_clusters)
    intra_cluster_orderings = intra_cluster_ordering(covisibility_matrix, clusters)
    ordered_clusters = cluster_ordering(covisibility_matrix, clusters, intra_cluster_orderings)
    for i in ordered_clusters:
        sorted_indices.extend(intra_cluster_orderings[i])
    for i, seq_id_i in enumerate(sorted_indices):
        for j, seq_id_j in enumerate(sorted_indices):
            covisibility_matrix_ordered[i,j] = covisibility_matrix[seq_id_i, seq_id_j]
    return covisibility_matrix_ordered, n_clusters, sorted_indices

def visualize_covisibility_matrix(covisibility_matrix, scene):
    covisibility_matrix_ordered, n_clusters, sorted_indices = order_indices_covisibility_matrix(covisibility_matrix, scene)
    ax, fig = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.log(covisibility_matrix+1))
    ax[0].set_title(f"covisibility_matrix_{scene}")
    ax[0].set_ylabel("log(# covisible points)")
    ax[1].imshow(np.log(covisibility_matrix_ordered+1))
    ax[1].set_title(f"covisibility_matrix_ordered_{n_clusters}_{scene}")
    ax[1].set_ylabel("log(# covisible points)")
    plt.savefig(f"covis_vis/covisibility_matrix_ordered_{n_clusters}_{scene}.png")
    plt.close()
    return sorted_indices

def visualize_covisibility_matrix_trajectory(dataset_path, scene):
    scene_path = os.path.join(dataset_path,scene,"sparse/0")
    images, points3D, cameras = get_colmap_data(scene_path)
    covisibility_matrix, id_to_idx, idx_to_id = build_covisibility_matrix(images, points3D)

    # 카메라 위치 계산
    rotations_image, translations_image = compute_extrinsics(images)
    cam_center = []
    print("id_to_idx keys:", sorted(id_to_idx.keys()))
    print("images keys:", sorted(images.keys()))
    
    # id_to_idx의 값들을 기준으로 정렬된 순서로 카메라 센터 계산
    for image_id in sorted(id_to_idx.keys()):
        if image_id in rotations_image and image_id in translations_image:
            cam_center.append((- rotations_image[image_id].T @ translations_image[image_id].reshape(3,1)))
        else:
            print(f"Warning: Image ID {image_id} not found in rotations/translations")
    
    print(f"Number of cameras processed: {len(cam_center)}")
    cam_center = np.array(cam_center)[:,:,0]
    
    # PCA로 2D 투영
    pca = PCA(n_components=2)
    cam_center_2d = pca.fit_transform(cam_center)
    center_cam_center = np.mean(cam_center_2d, axis=0)
    centered_cam_center = cam_center_2d - center_cam_center

    # Covisibility matrix 시각화
    print("Visualizing covisibility matrix")
    plt.figure(figsize=(20, 8))
    
    plt.subplot(121)
    plt.imshow(np.log(covisibility_matrix+1))
    plt.colorbar(label='log(# covisible points)')
    plt.title('Original Covisibility Matrix')
    
    # order_indices_covisibility_matrix 함수를 사용하여 정렬된 인덱스와 클러스터 정보 얻기
    covisibility_matrix_ordered, n_clusters, sorted_indices = order_indices_covisibility_matrix(covisibility_matrix, scene)
    
    plt.subplot(122)
    plt.imshow(np.log(covisibility_matrix_ordered+1))
    plt.colorbar(label='log(# covisible points)')
    plt.title(f'Ordered Covisibility Matrix (n_clusters={n_clusters})')
    
    plt.savefig(f'covis_vis/covisibility_matrices_{scene}_cluster_{n_clusters}.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

    sorted_cam_centers = centered_cam_center[sorted_indices]

    print("Creating camera position visualizations")
    # 카메라 위치 시각화
    plt.figure(figsize=(36, 8))
    
    # 1. 원래 순서 시각화
    plt.subplot(131)
    n_cameras = len(centered_cam_center)
    scatter = plt.scatter(centered_cam_center[:, 0], 
                         centered_cam_center[:, 1],
                         c=np.arange(n_cameras), 
                         cmap='hsv',
                         s=100)
    for i, (x, y) in enumerate(centered_cam_center):
        plt.annotate(f'{i}', (x, y), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8)
    plt.colorbar(scatter, label='Original indices')
    plt.title('Original Sequence')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # 2. 클러스터링 시각화
    plt.subplot(132)
    # 클러스터 정보 다시 계산
    W = np.array(covisibility_matrix)
    W = (W - W.min()) / (W.max() - W.min())
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
    L_sym = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    clusters = cluster_from_eigenvector(eigenvectors, n_clusters)
    
    unique_clusters = np.unique(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        plt.scatter(centered_cam_center[mask, 0], 
                   centered_cam_center[mask, 1],
                   c=[colors[i]], 
                   label=f'Cluster {cluster_id}',
                   s=100)
    
    for i, (x, y) in enumerate(centered_cam_center):
        plt.annotate(f'{i}', (x, y), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8)
    plt.title(f'Clustered Cameras (n={n_clusters})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # 3. 새로운 순서 시각화
    plt.subplot(133)
    scatter = plt.scatter(sorted_cam_centers[:, 0], 
                         sorted_cam_centers[:, 1],
                         c=np.arange(n_cameras), 
                         cmap='hsv',
                         s=100)
    
    for i, (x, y) in enumerate(sorted_cam_centers):
        plt.annotate(f'{i}', (x, y), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8)
    
    plt.plot(sorted_cam_centers[:, 0], 
             sorted_cam_centers[:, 1], 
             'k--', 
             alpha=0.3)
    
    plt.scatter(sorted_cam_centers[0, 0], 
                sorted_cam_centers[0, 1], 
                c='green', 
                s=200, 
                label='start', 
                alpha=0.5)
    plt.scatter(sorted_cam_centers[-1, 0], 
                sorted_cam_centers[-1, 1], 
                c='red', 
                s=200, 
                label='end', 
                alpha=0.5)
    
    plt.colorbar(scatter, label='New sequence indices')
    plt.title('Reordered Sequence')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    print("Saving camera position figure")
    plt.savefig(f'covis_vis/camera_positions_{scene}_cluster_{n_clusters}.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()
    return sorted_indices

if __name__ == "__main__":
    dataset_path = "/mnt/disk2/360"
    scene = input("Enter the scene name: ")
    visualize_covisibility_matrix_trajectory(dataset_path, scene)
    
