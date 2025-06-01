import numpy as np
from random import randint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import sys, os
sys.path.append("/mnt/disk2/auggs/gaussian-splatting")
from utils.bundle_utils import *
from utils.colmap_utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class GroupScheduler:
    """
    Gaussian Splatting Scheduler
    Given a set of grouped indices, the scheduler will return a uid for each iteration.
    Also, the scheduler will trigger a densify_and_prune or reset_opacity at the appropriate steps.
    
    The training schedule in the training of 3D Gaussian splatting is composed of three stages:
    (1) Warmup stage: From the start until 500 iterations, randomly access to cameras.
    (2) Densification stage: From the end of warmup stage until mid-training,
        multiple ADC intervals are applied.
    (3) Finetuning stage: From the end of densification stage until the maximum iterations,
        randomly access to cameras.
    Each ADC interval is consisted of an ADC batch,
    and there are three types of ADC batches: sequential, grouped random, and random.
    (a) Sequential: A single batch of camera sequence.
        The size of the batch is the total number of cameras.
    (b) Grouped Random: A single batch is consisted of single group,
        and each batch is consisted of random cameras from the group.
        The size of the batch is determined by the number of cameras in the group.
    (c) Random: A single batch is consisted of random cameras.
        The size of the batch is hard-coded as 100.

    The scheduler takes ordered groups as a dictionary,
    the keys are the group indices and the values are the ordered camera names within the group.
    """
    def __init__(self, 
                 cameras,
                 grouped_names: dict, 
                 densify_until_iter: int, 
                 densify_from_iter: int, 
                 debug: bool=False,
                 ):
        self.debug = debug
        if self.debug and grouped_names is None:
            self.grouped_names = {0: [i for i in range(42)], 1: [i for i in range(42, 86)], 2: [i for i in range(86, 120)],
                                  3: [i for i in range(120, 150)], 4: [i for i in range(150, 200)]}
        else:
            self.grouped_names = grouped_names
        self.n_groups = len(self.grouped_names)
        self.densify_until_iter = densify_until_iter
        self.densify_from_iter = densify_from_iter
        self.generate_dict_name_to_uid(cameras)
        self.generate_ordered_uids()
        self.uid_stack = None
        self.sequential_count = 0
        self.group_idx = 0
        self.densify_and_prune_flag = False
        self.reset_opacity_flag = False
        self.random_group_uid_stack = None

    def generate_dict_name_to_uid(self, cameras):
        """
        Generate a dictionary that maps the image name to the uid.
        """
        if self.debug:
            self.name_to_uid = {}
            for group_name_list in self.grouped_names.values():
                for name in group_name_list:
                    self.name_to_uid[name] = len(self.name_to_uid)
        else:   
            self.name_to_uid = {cam.image_name: cam.uid for cam in cameras}

    def generate_ordered_uids(self):
        """
        Generate a list of uids that are ordered by the group indices.
        """
        ordered_uids = []
        for group_index in self.grouped_names.keys():
            ordered_uids.extend(self.name_to_uid[name] for name in self.grouped_names[group_index])
        print(ordered_uids)
        self.ordered_uids = ordered_uids

    def scheduled_training_index(self, iteration: int,):
        """
        For given iteration, return the uid of the camera to be accessed.
        Also, the scheduler will trigger a densify_and_prune or reset_opacity at the appropriate steps.
        The iteration range starts from 1 to the maximum iterations(30000).
        """
        if iteration <= self.densify_from_iter or iteration > self.densify_until_iter:
            # Warmup stage
            if not self.uid_stack:
                self.uid_stack = list(self.name_to_uid.values())
            rand_idx = randint(0, len(self.uid_stack) - 1)
            vind = self.uid_stack.pop(rand_idx)
            print(f"Iteration: {iteration}, Warmup stage: {vind}")
            return vind
        elif iteration > self.densify_from_iter and iteration <= self.densify_until_iter:
            if iteration <= self.densify_from_iter + len(self.ordered_uids)*5:
                if (iteration - self.densify_from_iter) % len(self.ordered_uids) == 1:
                    interval = ((iteration - self.densify_from_iter)//len(self.ordered_uids))*(len(self.ordered_uids)//5)
                    first_part = self.ordered_uids.copy()[interval:]
                    second_part = self.ordered_uids.copy()[:interval]
                    self.uid_sequence = first_part + second_part
                    if self.sequential_count % 2 != 0:
                        self.uid_sequence = self.uid_sequence[::-1]
                    self.sequential_count += 1
                if (iteration - self.densify_from_iter) % len(self.ordered_uids) == 0:
                    self.densify_and_prune_flag = True
                if iteration == self.densify_from_iter + len(self.ordered_uids)*5:
                    self.reset_opacity_flag = True
                vind = self.uid_sequence[(iteration - self.densify_from_iter-1) % len(self.uid_sequence)]
                print(f"Iteration: {iteration}, Sequential stage: {vind}, densification_flag = {self.densify_and_prune_flag}, reset_opacity_flag = {self.reset_opacity_flag}")
                return vind
            elif iteration > self.densify_from_iter + len(self.ordered_uids)*5 \
                and iteration <= self.densify_from_iter + len(self.ordered_uids)*25:
                if not self.random_group_uid_stack:
                    self.group_names = self.grouped_names[self.group_idx]
                    self.group_uids = [self.name_to_uid[name] for name in self.group_names]
                    self.random_group_uid_stack = self.group_uids.copy()
                if len(self.random_group_uid_stack) == 1:
                    self.densify_and_prune_flag = True
                    self.group_idx = randint(0, len(self.grouped_names) - 1)
                if iteration == self.densify_from_iter + len(self.ordered_uids)*25:
                    self.reset_opacity_flag = True
                rand_idx = randint(0, len(self.random_group_uid_stack) - 1)
                vind = self.random_group_uid_stack.pop(rand_idx)
                print(f"Iteration: {iteration}, Grouped Random stage: {vind}, densification_flag = {self.densify_and_prune_flag}, reset_opacity_flag = {self.reset_opacity_flag}")
                return vind
            else:
                if not self.uid_stack:
                    self.uid_stack = list(self.name_to_uid.values())
                if (iteration - self.densify_from_iter - len(self.ordered_uids)*25) % 100 == 0:
                    self.densify_and_prune_flag = True
                if (iteration - self.densify_from_iter - len(self.ordered_uids)*25) % 3000 == 0:
                    self.reset_opacity_flag = True
                rand_idx = randint(0, len(self.uid_stack) - 1)
                vind = self.uid_stack.pop(rand_idx)
                print(f"Iteration: {iteration}, Random densification stage: {vind}, densification_flag = {self.densify_and_prune_flag}, reset_opacity_flag = {self.reset_opacity_flag}")
                return vind
        else:
            if not self.uid_stack:
                self.uid_stack = list(self.name_to_uid.values())
            rand_idx = randint(0, len(self.uid_stack) - 1)
            vind = self.uid_stack.pop(rand_idx)
            print(f"Iteration: {iteration}, Random stage: {vind}, densification_flag = {self.densify_and_prune_flag}, reset_opacity_flag = {self.reset_opacity_flag}")
            return vind
        
class ImageClustering:
    def __init__(self,
                 dataset_path,):
        self.dataset_path = dataset_path
        self.images, self.points3D, self.cameras = get_colmap_data(self.dataset_path)
        self.split_train_test()
        self.create_affinity_matrix()
        self.cluster_images()
        self.select_n_clusters()
        self.intra_cluster_ordering()
        self.cluster_ordering()

    def split_train_test(self):
        image_id_name = [[self.images[key].id, self.images[key].name] for key in self.images.keys()]
        image_id_name_sorted = sorted(image_id_name, key=lambda x: x[1])
        self.train_ids = []
        self.test_ids = []
        for i, (id, name) in enumerate(image_id_name_sorted):
            if i % 8 == 0:
                self.test_ids.append(id)
            else:
                self.train_ids.append(id)
        self.train_images = {k: v for k, v in self.images.items() if v.id in self.train_ids}
    
    def create_affinity_matrix(self):
        self.affinity_matrix, self.id_to_idx, self.idx_to_id = build_covisibility_matrix(self.train_images, self.points3D)
    
    def cluster_images(self):
        W = np.array(self.affinity_matrix)
        self.W = (W - W.min()) / (W.max() - W.min())
        D = np.diag(self.W.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(self.W.sum(axis=1), 1e-10)))
        L_sym = np.eye(len(self.W)) - D_inv_sqrt @ self.W @ D_inv_sqrt
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(L_sym)

    def select_n_clusters(self):
        self.score = []
        for i in range(2, 50):
            n_clusters = i
            U = self.eigenvectors[:, :n_clusters]
            U_norm = normalize(U, norm='l2')
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(U_norm)
            sscore = silhouette_score(U_norm, clusters)
            self.score.append(sscore)
            print(f"{i} clusters, silhouette score: {sscore}")
        self.n_clusters = int(input("select number of clusters: "))
        U = self.eigenvectors[:, :self.n_clusters]
        U_norm = normalize(U, norm='l2')
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(U_norm)
    
    def intra_cluster_ordering(self):
        self.cluster_ids = np.unique(self.clusters)
        self.intra_cluster_ordering = {}

        for cid in self.cluster_ids:
            indices = np.where(self.clusters==cid)[0]
            sub_adj = self.W[np.ix_(indices, indices)]
            degrees = sub_adj.sum(axis=1)
            start_node_idx = np.argmax(degrees)

            ordered_indices = [start_node_idx]
            remaining_indices = set(range(len(indices)))
            remaining_indices.remove(start_node_idx)

            current_idx = start_node_idx
            while remaining_indices:
                adjacencies = sub_adj[current_idx, list(remaining_indices)]

                if adjacencies.max() == 0:
                    next_idx = max(remaining_indices, key=lambda x: degrees[x])
                else:
                    next_idx = list(remaining_indices)[np.argmax(adjacencies)]

                ordered_indices.append(next_idx)
                remaining_indices.remove(next_idx)
                current_idx = next_idx
            sorted_node_indices = indices[ordered_indices]
            self.intra_cluster_ordering[cid] = sorted_node_indices.tolist()
        for cid, ordering in self.intra_cluster_ordering.items():
            print(f"Cluster {cid} #views: {len(ordering)}")
    
    def cluster_ordering(self):
        cluster_adj_matrix = np.zeros((self.n_clusters, self.n_clusters))
        for i, ci in enumerate(self.cluster_ids):
            indices_i = np.where(self.clusters == ci)[0]
            for j, cj in enumerate(self.cluster_ids):
                if i >= j:
                    continue
                indices_j = np.where(self.clusters == cj)[0]
                inter_adj = self.W[np.ix_(indices_i, indices_j)]
                cluster_adj_matrix[i, j] = cluster_adj_matrix[j, i] = inter_adj.sum()

        self.ordered_cluster_ids = []
        remaining_cluster_ids = set(range(self.n_clusters))
        cluster_degrees = cluster_adj_matrix.sum(axis=1)
        current_cluster = np.argmax(cluster_degrees)
        self.ordered_cluster_ids.append(current_cluster)
        remaining_cluster_ids.remove(current_cluster)

        while remaining_cluster_ids:
            adjacencies = cluster_adj_matrix[current_cluster, list(remaining_cluster_ids)]
            if adjacencies.max() == 0:
                next_cluster = max(remaining_cluster_ids, key=lambda x: cluster_degrees[x])
            else:
                next_cluster = list(remaining_cluster_ids)[np.argmax(adjacencies)]

            self.ordered_cluster_ids.append(next_cluster)
            remaining_cluster_ids.remove(next_cluster)
            current_cluster = next_cluster

        self.ordered_cluster_ids = [self.cluster_ids[i] for i in self.ordered_cluster_ids]
        print("Ordered cluster ids: ", self.ordered_cluster_ids)
        self.ordered_clusters = [self.intra_cluster_ordering[i] for i in self.ordered_cluster_ids]
        print("Ordered clusters: ", self.ordered_clusters)
        self.ordered_colmap_ids = {}
        for i, cluster in enumerate(self.ordered_clusters):
            self.ordered_colmap_ids[i] = [self.idx_to_id[idx] for idx in cluster]
        print("Ordered colmap ids: ", self.ordered_colmap_ids)        
        self.ordered_cluster_names = {}
        for i in range(self.n_clusters):
            self.ordered_cluster_names[i] = [self.images[id].name for id in self.ordered_colmap_ids[i]]
        print("Ordered cluster names: ", self.ordered_cluster_names)