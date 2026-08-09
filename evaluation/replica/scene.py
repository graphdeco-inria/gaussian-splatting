# Replica scene loading, dataset conversion, visibility and GT masks

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

from ..common import SceneData, TargetClassInfo, ensure_dir


# Future possible improvement: use embeddings to match detector names with the corresponding Replica dataset names.

CLASSES = [
    # Fields are: main project name, detector name, and stored detector-mask ID. The stored ID is the detector model ID plus one.
    TargetClassInfo("chair", "chair", 57),
    TargetClassInfo("sofa", "couch", 58),
    TargetClassInfo("table", "dining table", 61),
    TargetClassInfo("tv", "tv", 63),
    TargetClassInfo("plant", "potted plant", 59),
    TargetClassInfo("clock", "clock", 75),
]

REPLICA_CLASS_NAMES = {
    # Map each main project name to the class name used by the Replica dataset in info_semantic.json
    "chair": "chair",
    "sofa": "sofa",
    "table": "table",
    "tv": "tv-screen",
    "plant": "indoor-plant",
    "clock": "clock",
}

# State the reference for this function, as it is a standard conversion from rotation matrix to quaternion
def _rotmat_to_qvec(rotation):
    """ 
    Converts a rotation matrix to COLMAP's quaternion convention using a symmetric matrix, 
    its eigenvalues and eigenvectors, and the eigenvector of the largest eigenvalue. 
    It flips the sign if the first component is negative to keep a stable convention. 
    """
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.array([
        [rxx - ryy - rzz, 0, 0, 0],
        [ryx + rxy, ryy - rxx - rzz, 0, 0],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
    ]) / 3.0
    values, vectors = np.linalg.eigh(matrix)
    qvec = vectors[[3, 0, 1, 2], np.argmax(values)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class ReplicaScene:
    """ Load Replica data and convert it to the common evaluation format """

    def __init__(self, data_root, scene, sequence_name, frame_step, seed,
                 vertex_label_min_share, visibility_slop):
        """ 
        Store the scene paths and thresholds used by Replica processing 
        
        - data_root: the root directory of the Replica dataset
        - scene: the name of the scene to process
        - sequence_name: the name of the sequence to process
        - frame_step: the step size for selecting frames from the sequence
        - seed: the random seed for sampling points
        - vertex_label_min_share: the minimum share of face labels required for a vertex to be annotated
        - visibility_slop: the maximum allowed depth difference for a vertex to be considered visible
        """
        self.data_root = Path(data_root)
        self.scene = scene
        self.scene_root = self.data_root / scene
        self.sequence = self.scene_root / scene / sequence_name
        self.frame_step = frame_step
        self.seed = seed
        self.vertex_label_min_share = vertex_label_min_share
        self.visibility_slop = visibility_slop

    def selected_frames(self):
        """ Return the frame indices selected using the configured step """
        count = sum(1 for _ in open(self.sequence / "traj_w_c.txt")) # traj_w_c.txt contains one line per frame, so counting lines gives the total number of frames.
        return list(range(0, count, self.frame_step)) # Creates a list of frame indices from 0 to count with the given step size.

    def _load_mesh(self):
        """ Load vertices, faces and Replica dataset object IDs """
        # Replica stores semantic dataset IDs on mesh square faces
        ply = PlyData.read(str(self.scene_root / "mesh_semantic.ply"))
        vertex = ply["vertex"]

        # Load vertex coordinates
        vertices = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)

        # Load face vertex indices (numpy returned an error if list was not used)
        faces = np.asarray([list(item) for item in ply["face"].data["vertex_indices"]], dtype=np.int64)

        # Load their corresponding Replica dataset object IDs, as every face has an object_id attribute, not the vertices
        face_instances_ids = np.asarray(ply["face"].data["object_id"], dtype=np.int64)
        return vertices, faces, face_instances_ids

    def _load_info(self):
        """ 
        Load the semantic class metadata for the current scene

        It is something like this:

        {
            "classes": [{
                        "children": [],
                        "id": 1,
                        "name": "backpack",
                        "parents": []
                        },

                        {
                        "children": [],
                        "id": 2,
                        "name": "base-cabinet",
                        "parents": []
                        }],
            ...
        }

        """
        # This file can give a map from Replica dataset semantic IDs to Replica dataset names
        return json.loads((self.scene_root / "info_semantic.json").read_text())

    def _dataset_ids_to_local_ids(self, info):
        """ Map Replica dataset semantic IDs to SceneData local IDs """
        # Map Replica names to Replica IDs
        name_to_id = {item["name"]: int(item["id"]) for item in info["classes"]} # Info is something like the example from above

        # Map main name to their corresponding Replica dataset IDs. -1 means that the dataset does not contain that main class
        # Uses the REPLICA_CLASS_NAMES dictionary that maps main class names to Replica dataset names
        dataset_names = {name: name_to_id.get(dataset_name, -1)
                     for name, dataset_name in REPLICA_CLASS_NAMES.items()}

        # Map Replica dataset IDs to SceneData local main IDs
        return {dataset_id: localID for localID, item in enumerate(CLASSES)
                for dataset_id in [dataset_names[item.name]] if dataset_id >= 0}

    @staticmethod
    def _vertex_majority(n_vertices, faces, face_labels, minimum):
        """ Assign each vertex its most common face label when it reaches the threshold """
        # Face labels are converted to vertex labels because metrics operate on vertices. The segmentation IDs are the dataset ones
        votes = {}

        # Count the number of votes for each label at each vertex, based on the labels of the faces that include that vertex
        for face_index, face in enumerate(faces): # A face contains the indices of its vertices
            label = int(face_labels[face_index])
            for vertex_index in np.unique(face):
                values = votes.setdefault(int(vertex_index), {})
                values[label] = values.get(label, 0) + 1

        # Vertices below the majority threshold remain invalid and are not annotated
        labels = np.full(n_vertices, -1, dtype=np.int64)
        for vertex_index, values in votes.items():
            label, count = max(values.items(), key=lambda item: item[1])
            if count / sum(values.values()) >= minimum and label >= 0:
                labels[vertex_index] = label
        return labels

    @staticmethod
    def _world_to_camera(pose): # Get the reference for this function, as it is a standard conversion from camera-to-world to world-to-camera (this repo has to already use it)
        """ Invert a camera-to-world pose for world-to-camera projection """
        rotation = pose[:3, :3]
        output = np.eye(4)
        output[:3, :3] = rotation.T

        # Gets the last column of the pose, which is the translation vector, and applies the inverted rotation to it. This is because when inverting a transformation, the translation needs to be transformed by the inverse of the rotation.
        output[:3, 3] = -rotation.T @ pose[:3, 3] # The translation is inverted and rotated to match the camera-space convention expected by the depth images
        return output

    def load_data(self):
        """Load mesh labels and visibility as common scene data."""
        # Load Replica geometry and convert every Replica dataset ID to a SceneData local ID in the main project vocabulary
        vertices, faces, face_instances_ids = self._load_mesh()
        info = self._load_info()
        dataset_ids_to_local_ids = self._dataset_ids_to_local_ids(info)

        # Converts the Replica instance id to the Replica semantic class id
        id_to_label = np.asarray(info["id_to_label"], dtype=np.int64)

        # Map Replica face instance IDs to Replica face dataset semantic IDs
        face_dataset = np.where((face_instances_ids >= 0) & (face_instances_ids < len(id_to_label)),
                            id_to_label[np.clip(face_instances_ids, 0, len(id_to_label) - 1)], # Clip sets lower and upper bounds
                            -1)

        # Convert Replica face dataset IDs to main local IDs for voting
        face_labels = np.asarray([dataset_ids_to_local_ids.get(int(value), -1) # If it is not in the mapping, it returns -1
                                  for value in face_dataset], dtype=np.int64)
        
        # Uses face_dataset, Replica dataset IDs to identify vertices with a source annotation, independently of the main local labels
        vertex_dataset = self._vertex_majority(
            len(vertices), faces, face_dataset, self.vertex_label_min_share,
        )

        # Convert main local face labels to main local vertex labels for metrics
        # Uses face_labels, which are already converted to main local IDs, to identify vertices with a main local label
        semantic = self._vertex_majority(
            len(vertices), faces, face_labels, self.vertex_label_min_share,
        )

        # Visibility is derived from the RGB-D semantic frames
        visible = self._visibility(vertices)
        return SceneData(
            dataset="replica",
            scene=self.scene,
            vertices=vertices,
            semantic_labels=semantic, # Answers which category is the vertex
            annotated=(vertex_dataset >= 0),
            visible=visible, # Answers which vertices are visible in the selected frames
            classes=CLASSES,
        )

    def _load_trajectory(self):
        """ Load the sequence camera poses in world coordinates """
        # Each row contains one flattened 4x4 camera-to-world transform.
        return np.loadtxt(self.sequence / "traj_w_c.txt", dtype=np.float64).reshape(-1, 4, 4) # Converts the flat array into a 3D array where each slice along the first axis is a 4x4 matrix representing a camera pose.

    def _load_depth(self, index):
        """ Load one depth image and convert its values to meters """
        # Replica stores depth in millimeters, so we convert it to meters here
        return np.asarray(Image.open(self.sequence / "depth" / f"depth_{index}.png"),
                          dtype=np.float64) * 0.001

    def _load_semantic_image(self, index):
        """ Load one Replica semantic image """
            # The semantic image uses Replica dataset IDs before conversion to the local ID space
        return np.asarray(Image.open(
            self.sequence / "semantic_class" / f"semantic_class_{index}.png"),
            dtype=np.int64,
        )

    def _visibility(self, vertices):
        """ 
        Calculate visible vertices supported by 2D labels 
        
        A vertex is considered visible if:
            - It is projected into the camera's view frustum
            - It is in front of the camera
            - Its depth matches the observed depth within a certain tolerance. 
            
        """

        # Load the camera trajectory and initialize a visibility mask for all vertices
        trajectory = self._load_trajectory() # Load the camera poses for the sequence, size (num_frames, 4, 4)
        visible = np.zeros(len(vertices), dtype=bool)
        frame_indices = self.selected_frames() # Get the indices of the selected frames, as not all are used always

        # Intrinsics of Replica's pinhole camera
        height, width = 480, 640
        fx = fy = 320.0
        cx, cy = 320.0, 240.0

        for index in frame_indices:

            # Represent mesh vertices in camera coordinates for the current frame
            pose = self._world_to_camera(trajectory[index]) # trajectory[index] is a 4x4 camera-to-world matrix, and _world_to_camera converts it to a world-to-camera matrix
            camera_points = (pose[:3, :3] @ vertices.T).T + pose[:3, 3] # vertices is (N, 3) so we rotate its transpose and transpose back: (N, 3) and translate it so that they get to the camera coordinate system
            z = camera_points[:, 2] # Depth values of the projected points in camera space. z > 0 is in front of the camera

            # Project the 3D camera coordinates to 2D pixel coordinates using the pinhole camera model. The projection equations are u = fx * x / z + cx and v = fy * y / z + cy, where (x, y, z) are the camera coordinates of the vertex, and (u, v) are the pixel coordinates in the image plane
            # The np.errstate context manager is used to ignore division by zero and invalid value warnings that may occur when z is zero or negative
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx * camera_points[:, 0] / z + cx
                v = fy * camera_points[:, 1] / z + cy

            # Rounding the pixel coordinates to integer
            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)

            # Determine which vertices are projected inside the image boundaries and in front of the camera
            inside = ((z > 0) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height))
            candidates = np.where(inside)[0] # Indices of the vertices that are projected inside the image and in front of the camera
            if len(candidates) == 0:
                continue

            # Compare projected depth with the observed depth to reject occluded vertices
            depth = self._load_depth(index)
            image_depth = depth[vi[candidates], ui[candidates]] # (v, u) because the image is indexed by (height, width) and the pixel coordinates are (u, v)

            # Compare the projected depth with the observed depth, allowing for a small tolerance defined by visibility_slop
            hit = ((image_depth > 0) & (np.abs(z[candidates] - image_depth) <= self.visibility_slop))
            selected = candidates[hit]
            visible[selected] = True

        return visible

    def prepare_dataset(self, output_dir, max_points=250000, frame_stride=10, pixel_stride=4, max_depth_m=10.0):
        """ 
        Prepare Replica images and COLMAP model for training 
        
        COLMAP is a standard that consists of:
            - intrinsics in sparse/0/cameras.txt
            - extrinsics in sparse/0/images.txt
            - a sparse point cloud in sparse/0/points3D.txt
        """

        # Training expects an images directory and a COLMAP model
        images_dir = output_dir / "images"
        sparse_dir = output_dir / "sparse" / "0"
        required = [images_dir, sparse_dir]

        # Reuse the prepared dataset when all three COLMAP text files exist
        if all((output_dir / item).exists() for item in
               ["sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt"]):
            return output_dir
        for path in required:
            ensure_dir(path)

        # Link or copy the selected RGB frames into the training directory
        trajectory = self._load_trajectory()
        frames = self.selected_frames() # Index of the selected frames to be used for training, based on the frame_step parameter
        for index in frames:
            source = self.sequence / "rgb" / f"rgb_{index}.png"
            target = images_dir / f"rgb_{index}.png"
            if not target.exists():
                try:
                    os.symlink(os.path.relpath(source, target.parent), target) # Computes the relative path from the target's parent directory to the source file and creates a symbolic link at the target location pointing to the source file
                except OSError:
                    target.write_bytes(source.read_bytes())

        # cameras.txt: Write the fixed Replica camera intrinsics used by the sequence
        (sparse_dir / "cameras.txt").write_text("# Camera list\n1 PINHOLE 640 480 320.0 320.0 320.0 240.0\n")

        # images.txt: Convert each selected camera pose into the COLMAP text format
        image_lines = ["# Image list\n"]
        for image_id, index in enumerate(frames, start=1): # In COLMAP, image IDs start from 1
            pose = self._world_to_camera(trajectory[index]) # Replica has camera-to-world poses, but COLMAP expects world-to-camera poses, so we convert them
            qvec = _rotmat_to_qvec(pose[:3, :3]) # Convert the rotation matrix to a quaternion representation, which is the format expected by COLMAP for camera orientations
            translation = pose[:3, 3]

            # Write the image line
            image_lines.append(
                f"{image_id} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} "
                f"{qvec[3]:.12f} {translation[0]:.12f} {translation[1]:.12f} "
                f"{translation[2]:.12f} 1 rgb_{index}.png\n\n"
            )
        (sparse_dir / "images.txt").write_text("".join(image_lines))

        # COLMAP needs an initial sparse point cloud to start the reconstruction
        # points3D.txt: As we know the extrinsics, intrinsics and depth, we can sample a point cloud from the RGB-D images and save it in COLMAP's format
        rng = np.random.default_rng(self.seed)
        points, colors = [], []

        # We iterate every frame_stride frames, sample does not need to be enormous
        for index in frames[::frame_stride]:

            # For every selected frame, load depth and rgb
            depth = self._load_depth(index)
            rgb = np.asarray(Image.open(self.sequence / "rgb" / f"rgb_{index}.png"))

            # We sample one out of pixel_stride pixels in each axis
            ys, xs = np.meshgrid(np.arange(0, 480, pixel_stride), np.arange(0, 640, pixel_stride), indexing="ij")
            z = depth[ys, xs].reshape(-1) # We sample their depth
            valid = (z > 0.01) & (z < max_depth_m) # Filter some invalid values

            # Inverse formula of the pinhole projection: now we get the 3D coordinates from the pixel coordinates and depth
            x = (xs.reshape(-1) - 320.0) * z / 320.0
            y = (ys.reshape(-1) - 240.0) * z / 320.0
            camera_points = np.stack([x, y, z], axis=1)[valid]

            # Convert the camera coordinates to world coordinates using the camera pose
            world_points = (trajectory[index][:3, :3] @ camera_points.T).T + trajectory[index][:3, 3]

            # Save the sampled world-space points and their RGB colors for COLMAP. We only keep the valid points that are within the specified depth range
            colors.append(rgb[ys.reshape(-1)[valid], xs.reshape(-1)[valid]])
            points.append(world_points)

        # Concatenate all sampled points and colors from the selected frames into single arrays
        points = np.concatenate(points) # Appends arrays maintaining its shape
        colors = np.concatenate(colors)
        if len(points) > max_points:
            selected = rng.choice(len(points), max_points, replace=False)
            points, colors = points[selected], colors[selected]

        # points3D.txt: Save the sampled world-space points and their RGB colors for COLMAP
        with open(sparse_dir / "points3D.txt", "w") as output:
            output.write("# Point list\n")
            for point_id, (point, color) in enumerate(zip(points, colors), start=1):
                output.write(
                    f"{point_id} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} 1.0\n"
                )
        return output_dir

    def generate_gt_masks(self, output_dir, force=False):
        """
        Generate or reuse binary 2D GT masks from Replica semantic images

        force regenerates the masks when enabled
        """

        # The classes file marks whether it already exists, so we can skip the generation if it is present and force is not enabled
        if (output_dir / "classes.json").exists() and not force:
            return output_dir
        ensure_dir(output_dir / "semantic")
        ensure_dir(output_dir / "confidence")

        # Convert Replica dataset semantic IDs into the stored detector-mask IDs expected by the mask pipeline
        info = self._load_info()
        dataset_ids_to_local_ids = self._dataset_ids_to_local_ids(info)

        # Convert SceneData local IDs to stored detector-mask IDs
        local_to_detector_stored = {index: item.detector_stored_id for index, item in enumerate(CLASSES)}
        dataset_semantic_to_detector_stored = {dataset_id: local_to_detector_stored[local_id] for dataset_id, local_id in dataset_ids_to_local_ids.items()}
        
        # Save one semantic and confidence pair per selected frame
        for frame in self.selected_frames():

            # Loads the Replica semantic 2D image for the current frame, which contains the dataset 2D GT semantic IDs for each pixel
            dataset = self._load_semantic_image(frame)
            mapped = np.zeros(dataset.shape, dtype=np.uint8)

            # dataset_id is the Replica dataset ID. Stored_id is the detector mask ID written to the PNG (detector model ID + 1). The mapping is done so that the masks are compatible with the detector's expected IDs
            for dataset_id, stored_id in dataset_semantic_to_detector_stored.items():
                mapped[dataset == dataset_id] = stored_id # Just replace the pixels in the semantic image that match the dataset_id with the corresponding stored_id, effectively creating a new image where each pixel's value corresponds to the detector's expected mask ID for that class
            name = f"rgb_{frame}"
            cv2.imwrite(str(output_dir / "semantic" / f"{name}.png"), mapped)
            cv2.imwrite(str(output_dir / "confidence" / f"{name}.png"),
                        (mapped > 0).astype(np.uint8) * 255)
            
        # Store detector names keyed by stored detector-mask IDs so run.py can select the main classes present in the masks.
        classes = {str(item.detector_stored_id): item.name_by_detector for item in CLASSES}
        (output_dir / "classes.json").write_text(json.dumps(classes, indent=2))
        return output_dir
