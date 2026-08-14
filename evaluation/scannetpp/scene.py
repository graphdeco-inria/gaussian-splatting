# Scannet++ scene loading, taxonomy conversion, COLMAP preparation and GT masks

import json
import shutil
from pathlib import Path
from plyfile import PlyData

import numpy as np

from ..common import SceneData, TargetClassInfo


CLASSES = [
    # Fields are: main project name, detector name, and stored detector-mask ID. The stored ID is the detector model ID plus one, because 0 is reserved for the background class in the mask images.
    TargetClassInfo("bench", "bench", 14),
    TargetClassInfo("chair", "chair", 57),
    TargetClassInfo("table", "dining table", 61),
    TargetClassInfo("tv", "tv", 63),
    TargetClassInfo("laptop", "laptop", 64),
    TargetClassInfo("sink", "sink", 72),
    TargetClassInfo("clock", "clock", 75),
]

DATASET_LABELS = {
    # Map each main project name to all Scannet++ datasetcname spellings that represent it.
    "bench": {"bench", "experiment bench", "laboratory bench", "work bench",
               "window bench", "wood bench"},
    "chair": {
        "chair", "office chair", "armchair", "arm chair", "dining chair",
        "folding chair", "office visitor chair", "rolling chair", "lounge chair",
        "sofa chair", "deck chair", "papasan chair",
    },
    "table": {"table", "dining table", "office table", "conference table",
              "joined tables"},
    "tv": {"tv", "television", "tv screen"},
    "laptop": {"laptop"},
    "sink": {"sink", "kitchen sink", "bathroom sink", "washbasin", "wash basin"},
    "clock": {"clock", "wall clock", "table clock", "alarm clock"},
}

# Increment when the mesh-to-mask conversion or output contract changes
MASKS_CACHE_VERSION = 3


class ScannetScene:
    """ Load Scannet++ data and convert it to the common evaluation format """

    def __init__(self, data_root, scene, support_dir):
        """
        Store the scene paths and the directory containing generated GT

        - data_root: the root directory of the Scannet++ dataset
        - scene: the name of the scene to process
        - support_dir: the directory containing rasterized masks and visible vertices
        """
        self.data_root = Path(data_root)
        self.scene = scene
        self.scene_root = self.data_root / "validation_data" / scene
        self.scans = self.scene_root / "scans"
        self.support_dir = Path(support_dir)

    @property
    def metadata_path(self):
        """ Return the semantic class metadata file for Scannet++ """
        # Each line position in this file is the dataset semantic ID used by the mesh labels
        return self.data_root / "metadata" / "semantic_classes.txt"

    @property
    def prepared_dir(self):
        """ Return the directory containing the prepared COLMAP model """
        return self.scene_root / "dslr" / "undistorted_colmap"

    def _load_mesh(self):
        """ Load vertex positions and Scannet++ dataset IDs """

        # Scannet++ stores one semantic dataset ID per mesh vertex, unlike Replica which stores labels on faces
        ply = PlyData.read(str(self.scans / "mesh_aligned_0.05_semantic.ply"))
        vertex = ply["vertex"]

        # Load vertex coordinates
        vertices = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)

        # Load the Scannet++ dataset ID assigned to each vertex
        labels = np.asarray(vertex["label"], dtype=np.int64)
        return vertices, labels

    def _dataset_ids_to_local_ids(self, names):
        """ Map Scannet++ dataset IDs to SceneData local IDs """

        # Convert Scannet++ dataset names into the local class ordering used by metrics
        mapping = {}
        for local_id, item in enumerate(CLASSES):

            # A main class can correspond to several Scannet++ names
            for name in DATASET_LABELS[item.name]:
                if name in names:
                    mapping[names.index(name)] = local_id
        return mapping

    def load_data(self):
        """ Load mesh labels and generated visibility as common scene data """

        # Load Scannet++ dataset IDs before converting them to local IDs
        vertices, dataset_labels = self._load_mesh()

        # The metadata order defines which integer ID corresponds to each Scannet++ class name
        dataset_names = [line.strip().lower() for line in self.metadata_path.read_text().splitlines()]
        dataset_ids_to_local_ids = self._dataset_ids_to_local_ids(dataset_names)

        # Unknown Scannet++ dataset IDs remain -1
        semantic = np.asarray([dataset_ids_to_local_ids.get(int(label), -1) for label in dataset_labels], dtype=np.int64)

        # The GT mask stage records the vertices observed by the rendered camera views
        support_path = self.support_dir / "support.npz"
        cache_info_path = self.support_dir / "render_metadata.json"
        intrinsics_path = self.support_dir / "camera_intrinsics.json"

        if not support_path.exists():
            raise FileNotFoundError(
                f"Scannet++ GT support is missing: {support_path}. "
                "Generate the GT 2D masks before loading the scene."
            )
        
        if not cache_info_path.exists():
            raise FileNotFoundError(
                f"Scannet++ GT support metadata is missing: {cache_info_path}. "
                "Regenerate the GT 2D masks with the current pipeline."
            )

        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"Scannet++ camera intrinsics are missing: {intrinsics_path}. "
                "Regenerate the GT 2D masks with the current pipeline."
            )

        cache_info = json.loads(cache_info_path.read_text())
        if cache_info.get("version") != MASKS_CACHE_VERSION:
            raise ValueError("Scannet++ GT support was generated by an incompatible pipeline version")

        # Load the visibility support data, which indicates which vertices are visible in the rendered views
        support = np.load(support_path)
        camera_intrinsics = json.loads(intrinsics_path.read_text())

        # Visibility is computed by nvdiffrast from the rendered triangle IDs and stored per mesh vertex
        visible = support["visible_vertices"].astype(bool)
        if visible.shape != (len(vertices),):
            raise ValueError("Scannet++ GT support and semantic mesh use different vertex counts")
        
        return SceneData(
            dataset="scannetpp",
            scene=self.scene,
            vertices=vertices,
            semantic_labels=semantic,
            annotated=((dataset_labels >= 0) & (dataset_labels < len(dataset_names))), # A vertex is annotated when its dataset label points to a valid metadata entry, even if it is not an evaluated class
            visible=visible, # Answers which vertices are visible in the selected rendered views
            classes=CLASSES,
            num_images=len(camera_intrinsics),
            camera_intrinsics=camera_intrinsics,
        )

    def prepare_dataset(self, runtime, max_image_size=1600):
        """
        Prepare Scannet++ DSLR (which have distortion, and we want undistorted ones) images and reuse or create a COLMAP model
        """

        # Reuse a prepared model when COLMAP already produced either binary or text files
        output = self.prepared_dir
        if ((output / "sparse" / "0" / "cameras.bin").exists() or (output / "sparse" / "0" / "cameras.txt").exists()):
            return output
        
        # Prefer the dataset resized images, but if not available, use the original images
        images = self.scene_root / "dslr" / "resized_images"
        if not images.exists():
            images = self.scene_root / "dslr" / "images"

        # Undistort the images and write an output directory ready for COLMAP using the existing COLMAP reconstruction
        runtime.run_colmap([
            "image_undistorter",
            "--image_path", str(images),
            "--input_path", str(self.scene_root / "dslr" / "colmap"),
            "--output_path", str(output),
            "--output_type", "COLMAP",
            "--max_image_size", str(max_image_size),
        ])

        # Normalize COLMAP sparse output so we can always use sparse/0
        sparse = output / "sparse"
        sparse_zero = sparse / "0"
        if sparse.exists() and not sparse_zero.exists():

            # COLMAP can place its files directly in sparse, but the rest of the project expects sparse/0
            sparse_zero.mkdir(parents=True, exist_ok=True)
            for item in sparse.iterdir():
                if item.is_file() and item.suffix in {".bin", ".txt"}:
                    shutil.move(str(item), str(sparse_zero / item.name)) # shutil.move can move across filesystems
        return output

    def generate_gt_masks(self, runtime, output_dir, bands=4, viz=0,
                          force=False):
        """
        Generate or reuse rasterized Scannet++ GT masks and visibility support

        - bands: number of horizontal image bands used to reduce GPU memory
        - viz: number of optional visualizations to write, to see what the rasterization looks like
        - force: regenerate masks and support data even when completion files exist
        """

        # Reuse masks and support data when both completion markers exist
        cache_info_path = output_dir / "render_metadata.json"
        cache_info = None

        if cache_info_path.exists():
            cache_info = json.loads(cache_info_path.read_text())

        if ((output_dir / "classes.json").exists() and (output_dir / "support.npz").exists() and
                (output_dir / "camera_intrinsics.json").exists() and cache_info is not None and
                cache_info.get("version") == MASKS_CACHE_VERSION and not force):
            return output_dir
        
        # Rasterize the mesh inside the lifting container because nvdiffrast requires CUDA
        runtime.run_lifting_module(
            "evaluation.scannetpp.gt_masks",
            [
                "--scene_root", str(self.scene_root),
                "--repo_root", str(runtime.repo_root),
                "--metadata", str(self.metadata_path),
                "--output_dir", str(output_dir),
                "--bands", str(bands),
                "--viz", str(viz),
            ] + (["--force"] if force else []),
        )
        return output_dir