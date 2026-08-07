"""Replica scene loading, dataset conversion, visibility and GT masks."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

from ..common import SceneData, TargetClassInfo, ensure_dir


# Future possible improvement: have embeddings for each class, and use them to match the classes in the detector with the classes in Replica.

CLASSES = [
    # The first one is the main name, the second one is the COCO name, and the third one is the COCO ID.
    TargetClassInfo("chair", "chair", 57),
    TargetClassInfo("sofa", "couch", 58),
    TargetClassInfo("table", "dining table", 61),
    TargetClassInfo("tv", "tv", 63),
    TargetClassInfo("plant", "potted plant", 59),
    TargetClassInfo("clock", "clock", 75),
]

REPLICA_CLASS_NAMES = {
    # Maps each main name to the name that appears in info_semantic.json
    "chair": "chair",
    "sofa": "sofa",
    "table": "table",
    "tv": "tv-screen",
    "plant": "indoor-plant",
    "clock": "clock",
}

# State the reference for this function, as it is a standard conversion from rotation matrix to quaternion.
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
        with open(self.sequence / "traj_w_c.txt") as trajectory:
            count = sum(1 for _ in trajectory)
        return list(range(0, count, self.frame_step))

    def _load_mesh(self):
        """ Load Replica vertex positions, faces and dataset object identifiers """
        # Replica stores semantic and instance information on mesh square faces
        ply = PlyData.read(str(self.scene_root / "mesh_semantic.ply"))
        vertex = ply["vertex"]

        # Load vertex coordinates
        vertices = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)

        # Load face indices and their corresponding object identifiers (every face has an object_id attribute, not the vertices)
        faces = np.asarray([list(item) for item in ply["face"].data["vertex_indices"]],
                           dtype=np.int64)
        face_objects = np.asarray(ply["face"].data["object_id"], dtype=np.int64)
        return vertices, faces, face_objects

    def _load_info(self):
        """ 
        Load the semantic class metadata for the current scene

        Something like this:

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
        # This file maps the Replica semantic IDs to their dataset names
        return json.loads((self.scene_root / "info_semantic.json").read_text())

    def _dataset_ids_to_main_ids(self, info):
        """ Build a mapping from Replica semantic IDs to target class indices """
        # First map Replica names to dataset IDs, then map those IDs to CLASSES positions.
        name_to_id = {item["name"]: int(item["id"]) for item in info["classes"]} # Info is something like the example from above

        # Map the names in REPLICA_CLASS_NAMES to their corresponding dataset IDs, using -1 for any missing names
        dataset_names = {name: name_to_id.get(dataset_name, -1)
                     for name, dataset_name in REPLICA_CLASS_NAMES.items()}

        # Map the dataset IDs to the main class indices in CLASSES, using -1 for any missing IDs
        return {dataset_id: index for index, item in enumerate(CLASSES)
                for dataset_id in [dataset_names[item.name]] if dataset_id >= 0}

    @staticmethod
    def _vertex_majority(n_vertices, faces, face_labels, minimum):
        """ Assign each vertex its most common face label when it reaches the threshold """
        # Face labels are converted to vertex labels because metrics operate on vertices
        votes = {}

        # Count the number of votes for each label at each vertex, based on the labels of the faces that include that vertex
        for face_index, face in enumerate(faces):
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
    def _vertex_instances(n_vertices, faces, face_objects):
        """ 
        Assign each vertex the most common object identifier of its faces 
        
        The same as _vertex_majority, but without a threshold
        """
        # Object IDs do not use a threshold because every valid vertex needs an instance
        votes = {}

        # Count the number of votes for each label at each vertex, based on the labels of the faces that include that vertex
        for face_index, face in enumerate(faces):
            object_id = int(face_objects[face_index])
            for vertex_index in np.unique(face):
                values = votes.setdefault(int(vertex_index), {})
                values[object_id] = values.get(object_id, 0) + 1

        # Assign each vertex the object ID with the most votes
        instances = np.full(n_vertices, -1, dtype=np.int64)
        for vertex_index, values in votes.items():
            instances[vertex_index] = max(values.items(), key=lambda item: item[1])[0]
        return instances

    @staticmethod
    def _world_to_camera(pose): # Get the reference for this function, as it is a standard conversion from camera-to-world to world-to-camera (this repo has to already use it)
        """ Invert a camera-to-world pose for world-to-camera projection """
        # Projection uses the camera-space convention expected by the depth images
        rotation = pose[:3, :3]
        output = np.eye(4)
        output[:3, :3] = rotation.T
        output[:3, 3] = -rotation.T @ pose[:3, 3]
        return output

    def load_data(self):
        """ Load mesh labels, instances and visibility as common scene data """
        # Load dataset geometry and convert every dataset label to the common taxonomy.
        vertices, faces, face_objects = self._load_mesh()
        info = self._load_info()
        dataset_ids_to_main_ids = self._dataset_ids_to_main_ids(info)
        id_to_label = np.asarray(info["id_to_label"], dtype=np.int64)

        # Map face dataset IDs into main class indices for voting, using -1 for any missing target classes
        face_dataset = np.where((face_objects >= 0) & (face_objects < len(id_to_label)),
                            id_to_label[np.clip(face_objects, 0, len(id_to_label) - 1)],
                            -1)

        # Convert face dataset IDs into main class indices for voting
        face_labels = np.asarray([dataset_ids_to_main_ids.get(int(value), -1)
                                  for value in face_dataset], dtype=np.int64)
        
        # Keep dataset labels to identify vertices with a source annotation
        vertex_dataset = self._vertex_majority(
            len(vertices), faces, face_dataset, self.vertex_label_min_share,
        )

        # Convert main face labels to main vertex labels for metrics
        semantic = self._vertex_majority(
            len(vertices), faces, face_labels, self.vertex_label_min_share,
        )

        # Assign each vertex the most common object identifier of its faces
        instances = self._vertex_instances(len(vertices), faces, face_objects)

        # Visibility is derived from the selected RGB-D semantic frames
        visible, instances_seen_by_2D_masks = self._visibility(
            vertices, semantic, instances,
        )
        return SceneData(
            dataset="replica",
            scene=self.scene,
            vertices=vertices,
            semantic_labels=semantic, # Answers which category is the vertex
            instance_labels=instances, # Answers which object does the vertex belong to
            annotated=(vertex_dataset >= 0),
            visible=visible, # Answers which vertices are visible in the selected frames
            instances_seen_by_2D_masks=instances_seen_by_2D_masks, # Answers which instances are seen in the selected frames
            classes=CLASSES,
        )

    def _load_trajectory(self):
        """Load the sequence camera poses in world coordinates."""
        # Each row contains one flattened 4x4 camera-to-world transform.
        return np.loadtxt(self.sequence / "traj_w_c.txt", dtype=np.float64).reshape(-1, 4, 4)

    def _load_depth(self, index):
        """Load one depth image and convert its values to meters."""
        # Replica stores depth in millimeters, so convert it to meters here.
        return np.asarray(Image.open(self.sequence / "depth" / f"depth_{index}.png"),
                          dtype=np.float64) * 0.001

    def _load_semantic_image(self, index):
        """Load one Replica semantic image."""
        # The dataset semantic image uses Replica IDs before taxonomy conversion.
        return np.asarray(Image.open(
            self.sequence / "semantic_class" / f"semantic_class_{index}.png"),
            dtype=np.int64,
        )

    def _visibility(self, vertices, semantic, instances):
        """Calculate visible vertices and instances supported by 2D labels."""
        # Start with no visible vertices or instances seen by 2D masks.
        trajectory = self._load_trajectory()
        visible = np.zeros(len(vertices), dtype=bool)
        instances_seen_by_2D_masks = set()
        frame_indices = self.selected_frames()
        # Replica sequences use a fixed 640x480 pinhole camera.
        height, width = 480, 640
        fx = fy = 320.0
        cx, cy = 320.0, 240.0
        dataset_ids_to_main_ids = self._dataset_ids_to_main_ids(self._load_info())
        for index in frame_indices:
            # Project mesh vertices into the current camera and keep points in front of it.
            pose = self._world_to_camera(trajectory[index])
            camera_points = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
            z = camera_points[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx * camera_points[:, 0] / z + cx
                v = fy * camera_points[:, 1] / z + cy
            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)
            inside = ((z > 0) & (ui >= 0) & (ui < width) &
                      (vi >= 0) & (vi < height))
            candidates = np.where(inside)[0]
            if len(candidates) == 0:
                continue
            # Compare projected depth with the observed depth to reject occluded vertices.
            depth = self._load_depth(index)
            semantic_image = self._load_semantic_image(index)
            image_depth = depth[vi[candidates], ui[candidates]]
            hit = ((image_depth > 0) &
                   (np.abs(z[candidates] - image_depth) <= self.visibility_slop))
            selected = candidates[hit]
            visible[selected] = True
            # An instance is supported only when its visible pixels agree with its label.
            dataset_pixels = semantic_image[vi[selected], ui[selected]]
            mapped_pixels = np.asarray([dataset_ids_to_main_ids.get(int(value), -1)
                                        for value in dataset_pixels])
            selected_semantic = semantic[selected]
            matched = ((mapped_pixels == selected_semantic) &
                       (selected_semantic >= 0))
            instances_seen_by_2D_masks.update(
                int(value) for value in instances[selected[matched]]
                if value >= 0
            )
        return visible, instances_seen_by_2D_masks

    def prepare_dataset(self, output_dir, max_points=250000,
                        frame_stride=10, pixel_stride=4,
                        max_depth_m=10.0):
        """Prepare Replica images and a textual COLMAP model for training."""
        # Training expects an images directory and a COLMAP sparse model.
        images_dir = output_dir / "images"
        sparse_dir = output_dir / "sparse" / "0"
        required = [images_dir, sparse_dir]
        if all((output_dir / item).exists() for item in
               ["sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt"]):
            # Reuse the prepared dataset when all three COLMAP text files exist.
            return output_dir
        for path in required:
            ensure_dir(path)
        # Link or copy the selected RGB frames into the training directory.
        trajectory = self._load_trajectory()
        frames = self.selected_frames()
        for index in frames:
            source = self.sequence / "rgb" / f"rgb_{index}.png"
            target = images_dir / f"rgb_{index}.png"
            if not target.exists():
                try:
                    os.symlink(os.path.relpath(source, target.parent), target)
                except OSError:
                    target.write_bytes(source.read_bytes())

        # Write the fixed Replica camera intrinsics used by the sequence.
        (sparse_dir / "cameras.txt").write_text(
            "# Camera list\n1 PINHOLE 640 480 320.0 320.0 320.0 240.0\n"
        )
        # Convert each selected camera pose into the COLMAP text format.
        image_lines = ["# Image list\n"]
        for image_id, index in enumerate(frames, start=1):
            pose = self._world_to_camera(trajectory[index])
            qvec = _rotmat_to_qvec(pose[:3, :3])
            translation = pose[:3, 3]
            image_lines.append(
                f"{image_id} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} "
                f"{qvec[3]:.12f} {translation[0]:.12f} {translation[1]:.12f} "
                f"{translation[2]:.12f} 1 rgb_{index}.png\n\n"
            )
        (sparse_dir / "images.txt").write_text("".join(image_lines))

        # Create a bounded point cloud from sampled RGB-D pixels for initialization.
        rng = np.random.default_rng(self.seed)
        points, colors = [], []
        for index in frames[::frame_stride]:
            depth = self._load_depth(index)
            rgb = np.asarray(Image.open(self.sequence / "rgb" / f"rgb_{index}.png"))
            ys, xs = np.meshgrid(np.arange(0, 480, pixel_stride),
                                 np.arange(0, 640, pixel_stride), indexing="ij")
            z = depth[ys, xs].reshape(-1)
            valid = (z > 0.01) & (z < max_depth_m)
            x = (xs.reshape(-1) - 320.0) * z / 320.0
            y = (ys.reshape(-1) - 240.0) * z / 320.0
            camera_points = np.stack([x, y, z], axis=1)[valid]
            world_points = (trajectory[index][:3, :3] @ camera_points.T).T + trajectory[index][:3, 3]
            colors.append(rgb[ys.reshape(-1)[valid], xs.reshape(-1)[valid]])
            points.append(world_points)
        points = np.concatenate(points)
        colors = np.concatenate(colors)
        if len(points) > max_points:
            selected = rng.choice(len(points), max_points, replace=False)
            points, colors = points[selected], colors[selected]
        # Save the sampled world-space points and their RGB colors for COLMAP.
        with open(sparse_dir / "points3D.txt", "w") as output:
            output.write("# Point list\n")
            for point_id, (point, color) in enumerate(zip(points, colors), start=1):
                output.write(
                    f"{point_id} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} 1.0\n"
                )
        return output_dir

    def generate_gt_masks(self, output_dir, force=False):
        """Generate or reuse binary GT masks from Replica semantic images.

        ``force`` is a boolean flag that regenerates the masks when enabled.
        """
        # The classes file marks a complete mask generation pass.
        if (output_dir / "classes.json").exists() and not force:
            return output_dir
        ensure_dir(output_dir / "semantic")
        ensure_dir(output_dir / "confidence")
        # Convert dataset semantic IDs into the detector IDs expected by the mask pipeline.
        info = self._load_info()
        dataset_ids_to_main_ids = self._dataset_ids_to_main_ids(info)
        canonical_to_stored = {index: item.detector_stored_id
                               for index, item in enumerate(CLASSES)}
        semantic_to_stored = {dataset_id: canonical_to_stored[canonical_id]
                              for dataset_id, canonical_id in dataset_ids_to_main_ids.items()}
        # Save one semantic/confidence pair per selected frame.
        for frame in self.selected_frames():
            dataset = self._load_semantic_image(frame)
            mapped = np.zeros(dataset.shape, dtype=np.uint8)
            for dataset_id, stored_id in semantic_to_stored.items():
                mapped[dataset == dataset_id] = stored_id
            name = f"rgb_{frame}"
            cv2.imwrite(str(output_dir / "semantic" / f"{name}.png"), mapped)
            cv2.imwrite(str(output_dir / "confidence" / f"{name}.png"),
                        (mapped > 0).astype(np.uint8) * 255)
        # Store the detector vocabulary so ``run.py`` can select present classes.
        classes = {str(item.detector_stored_id): item.name_by_detector for item in CLASSES}
        (output_dir / "classes.json").write_text(json.dumps(classes, indent=2))
        return output_dir
