"""ScanNet++ mesh, taxonomy, instance and prepared-scene handling."""

import json
import shutil
from pathlib import Path
from plyfile import PlyData

import numpy as np

from ..common import SceneData, TargetClassInfo


CLASSES = [
    # ``name`` is the canonical ScanNet++ class; detector fields match mask files.
    TargetClassInfo("bench", "bench", 14),
    TargetClassInfo("chair", "chair", 57),
    TargetClassInfo("table", "dining table", 61),
    TargetClassInfo("tv", "tv", 63),
    TargetClassInfo("laptop", "laptop", 64),
    TargetClassInfo("sink", "sink", 72),
    TargetClassInfo("clock", "clock", 75),
]

DATASET_LABELS = {
    # Map the many ScanNet++ taxonomy spellings into the target classes above.
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

# Keep the renderer and scene loader on the same taxonomy table.
RAW_LABELS = DATASET_LABELS


class ScanNetScene:
    """Load ScanNet++ mesh data and its generated visibility support."""

    def __init__(self, data_root, scene, support_dir):
        """Store the scene root and the directory containing GT support data."""
        self.data_root = Path(data_root)
        self.scene = scene
        self.scene_root = self.data_root / "validation_data" / scene
        self.scans = self.scene_root / "scans"
        self.support_dir = Path(support_dir)

    @property
    def metadata_path(self):
        """Return the semantic class metadata file for ScanNet++."""
        # The metadata file gives the dataset integer label at each line position.
        return self.data_root / "metadata" / "semantic_classes.txt"

    @property
    def prepared_dir(self):
        """Return the directory containing the prepared COLMAP model."""
        # COLMAP output is kept beside the ScanNet++ DSLR images for reuse.
        return self.scene_root / "dslr" / "undistorted_colmap"

    def _load_mesh(self):
        """Load vertex positions and semantic labels."""
        # ScanNet++ stores one semantic label per mesh vertex.
        ply = PlyData.read(str(self.scans / "mesh_aligned_0.05_semantic.ply"))
        vertex = ply["vertex"]
        vertices = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
        labels = np.asarray(vertex["label"], dtype=np.int64)
        return vertices, labels

    def _dataset_ids_to_main_ids(self):
        """Map ScanNet++ dataset class IDs to the target class indices."""
        # Convert dataset label names into the compact class ordering used by metrics.
        names = [line.strip().lower() for line in self.metadata_path.read_text().splitlines()]
        mapping = {}
        for canonical_id, item in enumerate(CLASSES):
            for name in DATASET_LABELS[item.name]:
                if name in names:
                    mapping[names.index(name)] = canonical_id
        return mapping

    def _load_instances(self, count):
        """Map mesh segment indices to ScanNet++ object identifiers."""
        # Segment annotations group mesh segments into object instances.
        segments = json.loads((self.scans / "segments.json").read_text())
        annotations = json.loads((self.scans / "segments_anno.json").read_text())
        segment_to_instance = {}
        for group in annotations.get("segGroups", []):
            object_id = int(group["objectId"])
            for segment in group.get("segments", []):
                segment_to_instance[int(segment)] = object_id
        # Expand the segment-to-instance mapping into one ID per mesh vertex.
        output = np.asarray([
            segment_to_instance.get(int(segment), -1)
            for segment in segments["segIndices"]
        ], dtype=np.int64)
        if len(output) != count:
            raise ValueError("ScanNet++ segment index count does not match mesh vertices")
        return output

    def load_data(self):
        """Load the common scene representation and generated support data."""
        # Load dataset mesh labels before converting them to the common taxonomy.
        vertices, dataset_labels = self._load_mesh()
        dataset_names = [line.strip().lower()
                         for line in self.metadata_path.read_text().splitlines()]
        dataset_ids_to_main_ids = self._dataset_ids_to_main_ids()
        # Unknown dataset labels remain -1 and are treated as non-target labels.
        semantic = np.asarray([dataset_ids_to_main_ids.get(int(label), -1)
                               for label in dataset_labels], dtype=np.int64)
        instances = self._load_instances(len(vertices))
        # The rasterized GT stage records visibility and supported instances separately.
        support_path = self.support_dir / "support.npz"
        if not support_path.exists():
            raise FileNotFoundError(
                f"ScanNet++ GT support is missing: {support_path}. "
                "Generate the GT 2D masks before loading the scene."
            )
        support = np.load(support_path)
        visible = support["visible_vertices"].astype(bool)
        instances_seen_by_2D_masks = set(
            int(value) for value in support["instances_seen_by_2D_masks"]
        )
        return SceneData(
            dataset="scannetpp",
            scene=self.scene,
            vertices=vertices,
            semantic_labels=semantic,
            instance_labels=instances,
            annotated=((dataset_labels >= 0) & (dataset_labels < len(dataset_names))),
            visible=visible,
            instances_seen_by_2D_masks=instances_seen_by_2D_masks,
            classes=CLASSES,
        )

    def prepare_dataset(self, runtime, max_image_size=1600):
        """Prepare the ScanNet++ DSLR images with COLMAP in Docker."""
        # Reuse a prepared model when COLMAP already produced either file format.
        output = self.prepared_dir
        if ((output / "sparse" / "0" / "cameras.bin").exists() or
                (output / "sparse" / "0" / "cameras.txt").exists()):
            return output
        # Prefer the dataset's resized images and fall back to the original images.
        images = self.scene_root / "dslr" / "resized_images"
        if not images.exists():
            images = self.scene_root / "dslr" / "images"
        # Undistort the images and write a COLMAP-compatible output directory.
        runtime.run_colmap([
            "image_undistorter",
            "--image_path", str(images),
            "--input_path", str(self.scene_root / "dslr" / "colmap"),
            "--output_path", str(output),
            "--output_type", "COLMAP",
            "--max_image_size", str(max_image_size),
        ])
        # Normalize COLMAP's sparse output so later code always uses ``sparse/0``.
        sparse = output / "sparse"
        sparse_zero = sparse / "0"
        if sparse.exists() and not sparse_zero.exists():
            sparse_zero.mkdir(parents=True, exist_ok=True)
            for item in sparse.iterdir():
                if item.is_file() and item.suffix in {".bin", ".txt"}:
                    shutil.move(str(item), str(sparse_zero / item.name))
        return output

    def generate_gt_masks(self, runtime, output_dir, bands=4, viz=0,
                          force=False):
        """Generate or reuse rasterized ScanNet++ GT masks.

        ``force`` is a boolean flag that regenerates masks and support data.
        ``bands`` controls the vertical rasterization split, while ``viz``
        controls how many visualizations are written.
        """
        # Reuse masks and support data when both completion markers exist.
        if ((output_dir / "classes.json").exists() and
                (output_dir / "support.npz").exists() and not force):
            return output_dir
        # Rasterize the mesh inside the fusion container because nvdiffrast needs CUDA.
        runtime.run_fusion_module(
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
