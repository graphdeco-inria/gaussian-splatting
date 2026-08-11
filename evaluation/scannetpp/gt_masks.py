# CLI entry point for Scannet++ GT masks and visible-vertex support to be used inside the fusion container

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr

from ..common import ensure_dir
from .scene import CLASSES, DATASET_LABELS, MASKS_CACHE_VERSION
from plyfile import PlyData

'''
Convert OpenCV camera coordinates to the OpenGL convention used by nvdiffrast

OpenCV uses x right, y down and z forward, while the OpenGL projection used by nvdiffrast uses x right, y up and 
the opposite z direction before the perspective divide. The matrix is applied before the projection matrix below.
'''
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _load_colmap_loader(repo_root):
    """
    Load the repository COLMAP reader
    """

    # Import only the reader module because the full training package has extra dependencies
    path = repo_root / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("unified_colmap_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _projection(fx, fy, cx, cy, width, height, near, far):
    """
    Build the OpenGL projection matrix for one image band

    The matrix converts pinhole camera coordinates into clip coordinates. The principal point is expressed 
    relative to the current band, not the full image, because each band has its own vertical coordinate origin.
    """

    # Build the matrix directly from the pinhole intrinsics and clipping planes
    matrix = np.zeros((4, 4), dtype=np.float64)

    # Convert the horizontal focal length from pixel units into normalized device coordinates
    matrix[0, 0] = 2.0 * fx / width

    # Move the horizontal principal point from pixel coordinates to the [-1, 1] OpenGL range
    matrix[0, 2] = 1.0 - 2.0 * cx / width

    # Convert the vertical focal length and band-relative principal point to OpenGL coordinates
    matrix[1, 1] = 2.0 * fy / height
    matrix[1, 2] = 2.0 * cy / height - 1.0

    # Map camera-space depth between the near and far clipping planes into clip-space depth
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -2.0 * far * near / (far - near)

    # Set w so the rasterizer performs the perspective divide using camera-space depth
    matrix[3, 2] = -1.0
    return matrix


def _load_cameras(repo_root, sparse_dir):
    """
    Load pinhole camera records from the prepared COLMAP model

    The returned records contain image names, image sizes, intrinsic matrices and world-to-camera transforms
    """

    # Read binary COLMAP files when available and otherwise use their text versions
    # Both files describe the same sparse model: cameras provide intrinsics,
    # while images provide the pose and image filename for every rendered view.
    loader = _load_colmap_loader(repo_root)
    if (sparse_dir / "cameras.bin").exists():
        cameras = loader.read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        images = loader.read_extrinsics_binary(str(sparse_dir / "images.bin"))
    else:
        cameras = loader.read_intrinsics_text(str(sparse_dir / "cameras.txt"))
        images = loader.read_extrinsics_text(str(sparse_dir / "images.txt"))

    # Normalize both COLMAP formats into the camera records used by nvdiffrast
    result = []
    for image in sorted(images.values(), key=lambda item: item.name): # Sorting by image name makes the generated filenames deterministic and does not change the camera pose associated with any image
        camera = cameras[image.camera_id]

        # Support the two pinhole models produced by the prepared Scannet++ datasets
        # The original DSLR model is fisheye, but COLMAP undistorter writes a pinhole model in undistorted_colmap, which is the model rendered
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx = fy = camera.params[0]
            cx, cy = camera.params[1], camera.params[2]
        else:
            raise ValueError("the prepared COLMAP model must use a pinhole camera")
        
        # COLMAP stores a world-to-camera quaternion and translation. Convert them to a homogeneous transform so vertices can be multiplied by one matrix
        transform = np.eye(4)
        transform[:3, :3] = loader.qvec2rotmat(image.qvec)
        transform[:3, 3] = np.asarray(image.tvec)
        result.append({
            "name": image.name,
            "width": int(camera.width),
            "height": int(camera.height),

            # K maps camera coordinates (x, y, z) to pixels using the pinhole model
            "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            "world_to_camera": transform,
        })
    return result


def _load_mesh(scene_root, metadata_path):
    """
    Load geometry and convert official Scannet++ object annotations to triangle mask IDs

    The official 2D pipeline rasterizes the unlabelled mesh, maps visible faces to
    vertex object IDs from segments.json and segments_anno.json, and then maps
    object labels to the detector IDs expected by this evaluator.

    We are literally following the official Scannet++ 2D pipeline, so the output is compatible with the released 2D annotations
    That is why the semantic mesh is not used here, even though it contains semantic information. 
    The 2D object annotation format is defined over the official geometric mesh and its segment indices
    """

    # Load the same geometric mesh used by the official Scannet++ 2D rasterization pipeline
    scans_dir = scene_root / "scans"
    ply = PlyData.read(str(scans_dir / "mesh_aligned_0.05.ply"))
    vertex = ply["vertex"]

    # Load aligned world-space vertex coordinates in the order indexed by
    # segments.json; segments_anno.json refers to the segment IDs themselves.
    vertices = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    # Load triangle topology because rasterization returns the face covering each pixel
    # and the official 2D helper transfers vertex properties through face indices
    faces = np.asarray(list(ply["face"].data["vertex_indices"]), dtype=np.int32)

    # segments.json assigns one segment ID to every mesh vertex.
    # Its length proves that the annotations and mesh share the same vertex domain.
    segments = json.loads((scans_dir / "segments.json").read_text())
    segment_indices = np.asarray(segments["segIndices"], dtype=np.int64)
    if len(segment_indices) != len(vertices):
        raise ValueError("segments.json and the rasterized mesh use different vertex counts")

    # segments_anno.json stores object labels and the segment IDs belonging to each object
    annotations = json.loads((scans_dir / "segments_anno.json").read_text())

    # Validate that object labels come from the same released Scannet++ taxonomy
    # This catches mixing annotations from another scene or taxonomy release before converting those objects to background.
    metadata_names = {
        line.strip().lower()
        for line in metadata_path.read_text().splitlines()
        if line.strip()
    }

    # Convert every supported Scannet++ spelling into the local target class index
    # Several raw labels can represent one evaluated class, for example office chair and armchair both map to the local class chair
    label_to_local_id = {
        name.lower(): canonical_id
        for canonical_id, item in enumerate(CLASSES)
        for name in DATASET_LABELS[item.name]
    }
    object_to_local_id = {}

    # Build the mapping from segment IDs to annotated object IDs before
    # projecting those object IDs onto the vertices that contain each segment.
    segment_to_object_id = {}
    for group in annotations["segGroups"]:
        raw_object_id = group.get("objectId", group.get("id"))

        if raw_object_id is None:
            raise ValueError("an annotation group has no object ID")
        object_id = int(raw_object_id)

        if object_id < 0:
            raise ValueError(f"object {object_id} is invalid")
        
        label = str(group.get("label", "")).strip().lower()

        if label and label not in metadata_names:
            raise ValueError(f"object {object_id} uses unknown Scannet++ label: {label}")
        
        # Keep the object mapping separate so one object label can cover all of
        # the segments assigned to that object.
        object_to_local_id[object_id] = label_to_local_id.get(label, -1)

        # segments_anno.json stores segment IDs, not vertex indices. The
        # per-vertex segment_indices array performs the segment-to-vertex map.
        segment_ids = np.asarray(group.get("segments", []), dtype=np.int64)
        for segment_id in segment_ids:
            segment_to_object_id[int(segment_id)] = object_id

    if np.any((faces < 0) | (faces >= len(vertices))):
        raise ValueError("the mesh contains a face with an invalid vertex index")

    # Convert the segment-to-object mapping into one object ID per mesh vertex.
    vertex_object_ids = np.asarray(
        [segment_to_object_id.get(int(segment_id), -1)
         for segment_id in segment_indices],
        dtype=np.int64,
    )

    # Convert annotated object IDs into local class IDs.
    # This is the same semantic representation used by get_sem_ids_on_2d() in the official code before this project applies detector IDs
    vertex_local_ids = np.asarray([
        object_to_local_id.get(int(object_id), -1)
        for object_id in vertex_object_ids
    ], dtype=np.int64)

    '''
    Important decision:
    The official helper get_vtx_prop_on_2d() assigns a face property using the first vertex of each face. 
    Keeping this rule makes the output comparable with the official Scannet++ 2D annotations.

    This differs from the Replica GT mask generation, which uses the majority vertex label of each face.
    As Replica does not have official 2D annotations, the majority rule is a reasonable choice for that dataset.
    '''

    tri_canonical = vertex_local_ids[faces[:, 0]]

    # Rasterization uses one stored detector-mask ID per triangle, preserving the global YOLO+1 vocabulary
    # Zero remains background and ignored objects are not confused with any target class
    tri_stored = np.zeros(len(faces), dtype=np.uint8)
    for index, item in enumerate(CLASSES):
        tri_stored[tri_canonical == index] = item.detector_stored_id
    return vertices, faces, tri_stored


def generate(scene_root, repo_root, metadata_path, output_dir, bands=4, viz=0,
             force=False):
    """
    Render GT masks and save visible-vertex support data

    - bands: number of horizontal bands used to limit GPU memory
    - viz: number of optional overlay images to generate
    - force: ignore existing masks and support data when enabled

    Each output semantic PNG contains detector stored IDs, not local class IDs.
    This is required because the global evaluator consumes the same mask format for YOLO predictions and dataset ground truth.
    """
    if bands < 1:
        raise ValueError("bands must be at least 1")
    if viz < 0:
        raise ValueError("viz cannot be negative")
    
    # Reuse the completed output unless the caller requests a rebuild
    # A metadata marker is required so masks generated by the previous semantic-mesh implementation cannot be mistaken for current outputs.
    cache_info_path = output_dir / "render_metadata.json"
    cache_info = None

    if cache_info_path.exists():
        cache_info = json.loads(cache_info_path.read_text())

    if ((output_dir / "classes.json").exists() and (output_dir / "support.npz").exists() and cache_info is not None and
            cache_info.get("version") == MASKS_CACHE_VERSION and not force):
        return output_dir
    
    # nvdiffrast renders these masks on CUDA rather than through the host CPU
    if not torch.cuda.is_available():
        raise RuntimeError("Scannet++ GT rendering requires a CUDA device")

    # Use the normalized COLMAP model generated by ScannetScene.prepare_dataset
    # The images and sparse model must come from the same undistortion run
    sparse_dir = scene_root / "dslr" / "undistorted_colmap" / "sparse" / "0"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"prepared COLMAP model not found: {sparse_dir}")
    
    # The image root is only needed when optional visualizations are requested
    image_root = scene_root / "dslr" / "undistorted_colmap" / "images"
    vertices, faces, tri_stored = _load_mesh(scene_root, metadata_path)
    cameras = _load_cameras(repo_root, sparse_dir)

    # Prepare output folders before rendering the camera views
    ensure_dir(output_dir / "semantic")
    ensure_dir(output_dir / "confidence")
    device = torch.device("cuda")

    # Homogeneous vertices allow one matrix multiplication per camera and band
    # Appending one makes the affine translation part of the matrix product
    vertices_h = torch.from_numpy(np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)).to(device)
    # Faces are uploaded once and reused for every camera and image band.

    faces_t = torch.from_numpy(faces).to(device).contiguous()

    # Labels are indexed by the triangle IDs returned by nvdiffrast
    labels_t = torch.from_numpy(tri_stored.astype(np.int64)).to(device)
    context = dr.RasterizeCudaContext()
    visible_vertices = np.zeros(len(vertices), dtype=bool)
    viz_left = int(viz)

    for camera in cameras:

        # Render each image in horizontal bands to limit GPU memory usage
        width, height = camera["width"], camera["height"]

        # COLMAP uses OpenCV camera coordinates, while nvdiffrast expects OpenGL camera coordinates
        transform = CV_TO_GL @ camera["world_to_camera"]

        # Derive clipping planes from the actual camera-space mesh depths instead of a scene-wide heuristic
        # COLMAP's z coordinate is positive in front of the camera
        # Vertices behind the camera are discarded by the projection/rasterizer.
        camera_points = (camera["world_to_camera"][:3, :3] @ vertices.T).T + camera["world_to_camera"][:3, 3]
        positive_depth = camera_points[:, 2][camera_points[:, 2] > 1e-4]

        if len(positive_depth) == 0:
            raise ValueError(f"camera {camera['name']} cannot see any mesh vertex")
        
        near = max(1e-3, float(positive_depth.min()))
        far = max(near + 1.0, float(positive_depth.max()))

        # Split the image height into equally sized bands so intermediate CUDA buffers stay bounded
        edges = np.linspace(0, height, bands + 1).astype(int)
        rendered_bands = []
        for band in range(bands):

            # Shift the principal point for the current vertical image band because the band has its own image origin
            y0, y1 = int(edges[band]), int(edges[band + 1])
            band_height = y1 - y0
            projection = _projection(
                camera["K"][0, 0], camera["K"][1, 1], camera["K"][0, 2],
                camera["K"][1, 2] - y0, width, band_height, near, far,
            )

            # nvdiffrast expects the combined projection and world-to-camera matrix transposed for row-vector multiplication
            # Vertices are stored as row vectors here, hence the transpose.
            matrix = torch.from_numpy((projection @ transform).T.astype(np.float32)).to(device)
            clip_vertices = (vertices_h @ matrix).unsqueeze(0).contiguous()

            # Rasterize the mesh and recover nvdiffrast triangle indices, not semantic class IDs
            # The rasterizer performs the depth test, so the surviving face is the triangle that is more at the front at each pixel.
            raster, _ = dr.rasterize(
                context, clip_vertices, faces_t, resolution=[band_height, width],
            )

            # The fourth raster channel stores the one-based triangle index;
            # background is zero, so convert it to -1 for Python indexing.
            face_ids = raster[0, :, :, 3].round().long() - 1
            hit = face_ids >= 0

            # Any triangle contributing to a pixel makes all of its vertices visible in at least one camera view
            # This matches the official support approximation; it is not a per-vertex occlusion test.
            visible_face_ids = torch.unique(face_ids[hit]).cpu().numpy()
            if len(visible_face_ids):
                visible_faces = faces[visible_face_ids].reshape(-1)
                visible_vertices[visible_faces] = True

            # Convert visible triangle indices into stored detector-mask class IDs
            # Unknown objects already have label zero and remain background.
            band_labels = torch.zeros((band_height, width), dtype=torch.int64, device=device)
            band_labels[hit] = labels_t[face_ids[hit]]

            # Restore the image row order after rendering the band in OpenGL coordinates
            rendered_bands.append(torch.flip(band_labels, dims=(0,)).to(torch.uint8))

        # Join the vertically rendered bands back into one image mask size
        # Bands are appended from top to bottom after each individual OpenGL row flip, reconstructing the original image coordinate order.
        semantic = torch.cat(rendered_bands, dim=0).cpu().numpy()

        # Confidence follows the existing mask contract: every non-background
        # detector ID is a confident GT pixel, represented by 255.
        confidence = (semantic > 0).astype(np.uint8) * 255
        stem = Path(camera["name"]).stem
        cv2.imwrite(str(output_dir / "semantic" / f"{stem}.png"), semantic)
        cv2.imwrite(str(output_dir / "confidence" / f"{stem}.png"), confidence)

        # Only write the requested number of optional overlay visualizations
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

    # Save detector names keyed by stored detector-mask IDs so run.py and the vote stage can identify the classes
    classes = {str(item.detector_stored_id): item.name_by_detector for item in CLASSES}
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2))

    # Preserve the actual COLMAP intrinsics used to render the masks. This also
    # gives analytics a reproducible camera summary for each validation scene.
    (output_dir / "camera_intrinsics.json").write_text(json.dumps([
        {
            "name": camera["name"],
            "width": camera["width"],
            "height": camera["height"],
            "fx": float(camera["K"][0, 0]),
            "fy": float(camera["K"][1, 1]),
            "cx": float(camera["K"][0, 2]),
            "cy": float(camera["K"][1, 2]),
        }
        for camera in cameras
    ], indent=2))

    # Save visibility information for ScannetScene.load_data. This is indexed in the same vertex order as the mesh loaded by scene.py.
    np.savez_compressed(
        output_dir / "support.npz",
        visible_vertices=visible_vertices,
    )

    # Record the conversion contract so later runs can reject stale outputs.
    cache_info_path.write_text(json.dumps({
        "version": MASKS_CACHE_VERSION,
        "mesh": "mesh_aligned_0.05.ply",
        "annotations": ["segments.json", "segments_anno.json"],
        "bands": int(bands),
    }, indent=2))
    return output_dir


def main():
    """
    Generate Scannet++ GT masks from the command-line arguments

    The CLI is called by run.py inside the fusion container, which can use CUDA
    The metadata path is retained as an explicit input because it validates that
    segment annotation labels belong to the released Scannet++ taxonomy.
    """
    parser = argparse.ArgumentParser()

    # Identify the scene, repository reader, metadata and output directory for rendering
    parser.add_argument("--scene_root", required=True, type=Path)
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--viz", type=int, default=0)

    # Re-render masks and support data even when completion files already exist
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Run the renderer with the selected band and visualization settings
    generate(args.scene_root, args.repo_root, args.metadata, args.output_dir,
             args.bands, args.viz, args.force)


if __name__ == "__main__":
    main()
