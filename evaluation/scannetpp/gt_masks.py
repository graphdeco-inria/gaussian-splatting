"""Generate ScanNet++ GT masks and the visible-instance support cache."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from ..common import ensure_dir
from .scene import CLASSES, RAW_LABELS
from plyfile import PlyData


# Convert OpenCV camera coordinates to the OpenGL convention used by nvdiffrast.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _load_colmap_loader(repo_root):
    """Load the repository COLMAP reader without importing the full package."""
    # Import only the reader module because the full training package has extra dependencies.
    path = repo_root / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("unified_colmap_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _projection(fx, fy, cx, cy, width, height, near, far):
    """Build the OpenGL projection matrix for one image band."""
    # Build the matrix directly from the pinhole intrinsics and clipping planes.
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = 2.0 * fx / width
    matrix[0, 2] = 1.0 - 2.0 * cx / width
    matrix[1, 1] = 2.0 * fy / height
    matrix[1, 2] = 2.0 * cy / height - 1.0
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -2.0 * far * near / (far - near)
    matrix[3, 2] = -1.0
    return matrix


def _load_cameras(repo_root, sparse_dir):
    """Load pinhole camera records from the prepared COLMAP model.

    The returned collection contains image names, image sizes, intrinsic
    matrices and world-to-camera transforms.
    """
    # Read binary COLMAP files when available and otherwise use their text versions.
    loader = _load_colmap_loader(repo_root)
    if (sparse_dir / "cameras.bin").exists():
        cameras = loader.read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        images = loader.read_extrinsics_binary(str(sparse_dir / "images.bin"))
    else:
        cameras = loader.read_intrinsics_text(str(sparse_dir / "cameras.txt"))
        images = loader.read_extrinsics_text(str(sparse_dir / "images.txt"))
    # Normalize both COLMAP formats into the camera records used by rendering.
    result = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras[image.camera_id]
        # Support the two pinhole models produced by the prepared datasets.
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx = fy = camera.params[0]
            cx, cy = camera.params[1], camera.params[2]
        else:
            raise ValueError("the prepared COLMAP model must use a pinhole camera")
        # Store the world-to-camera transform in homogeneous coordinates.
        transform = np.eye(4)
        transform[:3, :3] = loader.qvec2rotmat(image.qvec)
        transform[:3, 3] = np.asarray(image.tvec)
        result.append({
            "name": image.name,
            "width": int(camera.width),
            "height": int(camera.height),
            "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            "world_to_camera": transform,
        })
    return result


def _load_mesh(scene_root, metadata_path):
    """Load ScanNet++ geometry, rendered class IDs and instance IDs."""

    # Load mesh geometry and the raw semantic labels used to build triangle labels.
    ply = PlyData.read(str(scene_root / "scans" / "mesh_aligned_0.05_semantic.ply"))
    vertex = ply["vertex"]
    vertices = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    raw_labels = np.asarray(vertex["label"], dtype=np.int64)
    faces = np.asarray(list(ply["face"].data["vertex_indices"]), dtype=np.int32)

    # Map raw ScanNet++ labels into the compact target-class ordering.
    names = [line.strip().lower() for line in metadata_path.read_text().splitlines()]
    dataset_ids_to_main_ids = {}
    for canonical_id, item in enumerate(CLASSES):
        for raw_name in RAW_LABELS[item.name]:
            if raw_name in names:
                dataset_ids_to_main_ids[raw_name] = canonical_id
    canonical = np.asarray([dataset_ids_to_main_ids.get(int(label), -1)
                            for label in raw_labels], dtype=np.int64)

    # Convert segment annotations into one instance ID per mesh vertex.
    scans = scene_root / "scans"
    segments = json.loads((scans / "segments.json").read_text())
    annotations = json.loads((scans / "segments_anno.json").read_text())
    segment_to_instance = {}
    for group in annotations.get("segGroups", []):
        for segment in group.get("segments", []):
            segment_to_instance[int(segment)] = int(group["objectId"])
    instances = np.asarray([
        segment_to_instance.get(int(segment), -1)
        for segment in segments["segIndices"]
    ], dtype=np.int64)
    # Rasterization uses one semantic and instance label per triangle.
    tri_canonical = canonical[faces[:, 0]]
    tri_stored = np.zeros(len(faces), dtype=np.uint8)
    for index, item in enumerate(CLASSES):
        tri_stored[tri_canonical == index] = item.detector_stored_id
    tri_instances = instances[faces[:, 0]]
    return vertices, faces, tri_stored, tri_instances


def generate(scene_root, repo_root, metadata_path, output_dir, bands=4, viz=0,
             force=False):
    """Render GT masks and save visible-vertex support data.

    ``force`` is a boolean flag that ignores existing masks and support data.
    ``bands`` controls the vertical rendering split and ``viz`` controls the
    number of optional overlay images. The generated files include semantic
    masks, confidence masks, class metadata and support data.
    """
    if bands < 1:
        raise ValueError("bands must be positive")
    # Reuse the completed output unless the caller explicitly requests a rebuild.
    if ((output_dir / "classes.json").exists() and
            (output_dir / "support.npz").exists() and not force):
        return output_dir
    # nvdiffrast renders these masks on CUDA rather than through the host CPU.
    if not torch.cuda.is_available():
        raise RuntimeError("ScanNet++ GT rendering requires a CUDA device")
    import nvdiffrast.torch as dr

    # Use the normalized COLMAP model generated by the scene instance.
    sparse_dir = scene_root / "dslr" / "undistorted_colmap" / "sparse" / "0"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"prepared COLMAP model not found: {sparse_dir}")
    # The image root is only needed when optional visualizations are requested.
    image_root = scene_root / "dslr" / "undistorted_colmap" / "images"
    vertices, faces, tri_stored, tri_instances = _load_mesh(
        scene_root, metadata_path,
    )
    cameras = _load_cameras(repo_root, sparse_dir)
    # Prepare output folders before rendering the camera views.
    ensure_dir(output_dir / "semantic")
    ensure_dir(output_dir / "confidence")
    device = torch.device("cuda")
    # Homogeneous vertices allow one matrix multiplication per camera and band.
    vertices_h = torch.from_numpy(
        np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    ).to(device)
    faces_t = torch.from_numpy(faces).to(device).contiguous()
    labels_t = torch.from_numpy(tri_stored.astype(np.int64)).to(device)
    context = dr.RasterizeCudaContext()
    # Choose clipping planes from the scene size so the complete mesh is renderable.
    diagonal = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    near = max(1e-3, diagonal * 1e-3)
    far = max(near + 1.0, diagonal * 10.0)
    visible_vertices = np.zeros(len(vertices), dtype=bool)
    instances_seen_by_2D_masks = set()
    viz_left = int(viz)

    for camera in cameras:
        # Render each image in horizontal bands to limit peak GPU memory usage.
        width, height = camera["width"], camera["height"]
        transform = CV_TO_GL @ camera["world_to_camera"]
        edges = np.linspace(0, height, bands + 1).astype(int)
        rendered_bands = []
        for band in range(bands):
            # Shift the principal point for the current vertical image band.
            y0, y1 = int(edges[band]), int(edges[band + 1])
            band_height = y1 - y0
            projection = _projection(
                camera["K"][0, 0], camera["K"][1, 1], camera["K"][0, 2],
                camera["K"][1, 2] - y0, width, band_height, near, far,
            )
            matrix = torch.from_numpy((projection @ transform).T.astype(np.float32)).to(device)
            clip_vertices = (vertices_h @ matrix).unsqueeze(0).contiguous()
            # Rasterize the mesh and recover the visible triangle IDs.
            raster, _ = dr.rasterize(
                context, clip_vertices, faces_t, resolution=[band_height, width],
            )
            face_ids = raster[0, :, :, 3].round().long() - 1
            hit = face_ids >= 0
            visible_face_ids = torch.unique(face_ids[hit]).cpu().numpy()
            if len(visible_face_ids):
                visible_faces = faces[visible_face_ids].reshape(-1)
                visible_vertices[visible_faces] = True
                visible_instances = tri_instances[visible_face_ids]
                visible_stored = tri_stored[visible_face_ids]
                instances_seen_by_2D_masks.update(
                    int(instance) for instance, stored in
                    zip(visible_instances, visible_stored)
                    if instance >= 0 and stored > 0
                )
            # Convert visible triangle IDs into the stored detector class labels.
            band_labels = torch.zeros((band_height, width), dtype=torch.int64, device=device)
            band_labels[hit] = labels_t[face_ids[hit]]
            rendered_bands.append(torch.flip(band_labels, dims=(0,)).to(torch.uint8))

        # Join the vertically rendered bands back into one image-sized mask.
        semantic = torch.cat(rendered_bands, dim=0).cpu().numpy()
        confidence = (semantic > 0).astype(np.uint8) * 255
        stem = Path(camera["name"]).stem
        cv2.imwrite(str(output_dir / "semantic" / f"{stem}.png"), semantic)
        cv2.imwrite(str(output_dir / "confidence" / f"{stem}.png"), confidence)
        # Only write the requested number of optional overlay visualizations.
        if viz_left > 0:
            source = image_root / camera["name"]
            image = cv2.imread(str(source))
            if image is not None:
                colors = np.zeros_like(image)
                palette = {
                    14: (255, 0, 255), 57: (0, 0, 255), 61: (0, 255, 0),
                    63: (255, 0, 0), 64: (0, 165, 255), 72: (255, 255, 0),
                    75: (0, 255, 255),
                }
                for stored_id, color in palette.items():
                    colors[semantic == stored_id] = color
                overlay = image.copy()
                selected = semantic > 0
                overlay[selected] = (0.4 * image[selected] +
                                     0.6 * colors[selected]).astype(np.uint8)
                ensure_dir(output_dir / "viz")
                cv2.imwrite(str(output_dir / "viz" / f"{stem}.jpg"), overlay)
            viz_left -= 1

    # Save the detector vocabulary consumed by the unified vote stage.
    classes = {str(item.detector_stored_id): item.name_by_detector for item in CLASSES}
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2))
    # Save visibility and supported-instance information for SceneData.load_data.
    np.savez_compressed(
        output_dir / "support.npz",
        visible_vertices=visible_vertices,
        instances_seen_by_2D_masks=np.asarray(
            sorted(instances_seen_by_2D_masks), dtype=np.int64,
        ),
    )
    return output_dir


def main():
    """Generate ScanNet++ GT masks from the command-line arguments."""
    parser = argparse.ArgumentParser()

    # Identify the scene, repository reader and output directory for rendering.
    parser.add_argument("--scene_root", required=True, type=Path)
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--viz", type=int, default=0)

    # Re-render masks and support data even when completion files already exist.
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Run the renderer with the selected band and visualization settings.
    generate(args.scene_root, args.repo_root, args.metadata, args.output_dir,
             args.bands, args.viz, args.force)


if __name__ == "__main__":
    main()
