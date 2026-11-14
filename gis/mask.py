import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyproj


WGS84_A = 6378137.0
WGS84_B = 6356752.314245


def lla_from_ecef(x, y, z):
    """Compute latitude, longitude and altitude from ECEF XYZ (WGS84)."""
    a = WGS84_A
    b = WGS84_B
    ea = np.sqrt((a**2 - b**2) / a**2)
    eb = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + eb**2 * b * np.sin(theta) ** 3,
        p - ea**2 * a * np.cos(theta) ** 3,
    )
    n = a / np.sqrt(1 - ea**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def ecef_from_lla(lat, lon, alt):
    """Compute ECEF XYZ from latitude, longitude and altitude (WGS84)."""
    a2 = WGS84_A**2
    b2 = WGS84_B**2
    lat = np.radians(lat)
    lon = np.radians(lon)
    l = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    x = (a2 * l + alt) * np.cos(lat) * np.cos(lon)
    y = (a2 * l + alt) * np.cos(lat) * np.sin(lon)
    z = (b2 * l + alt) * np.sin(lat)
    return x, y, z


def ecef_from_topocentric_transform(lat, lon, alt):
    """Transformation from topocentric frame to ECEF."""
    x, y, z = ecef_from_lla(lat, lon, alt)
    sa = np.sin(np.radians(lat))
    ca = np.cos(np.radians(lat))
    so = np.sin(np.radians(lon))
    co = np.cos(np.radians(lon))
    return np.array(
        [
            [-so, -sa * co, ca * co, x],
            [co, -sa * so, ca * so, y],
            [0, ca, sa, z],
            [0, 0, 0, 1],
        ]
    )


def lla_from_topocentric(x, y, z, reflat, reflon, refalt):
    """Transform topocentric XYZ to latitude, longitude and altitude."""
    transform = ecef_from_topocentric_transform(reflat, reflon, refalt)
    ex = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2] * z + transform[0, 3]
    ey = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2] * z + transform[1, 3]
    ez = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2] * z + transform[2, 3]
    return lla_from_ecef(ex, ey, ez)


def topocentric_from_lla(lat, lon, alt, reflat, reflon, refalt):
    """Transform latitude, longitude and altitude to topocentric XYZ."""
    transform = np.linalg.inv(ecef_from_topocentric_transform(reflat, reflon, refalt))
    x, y, z = ecef_from_lla(lat, lon, alt)
    tx = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2] * z + transform[0, 3]
    ty = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2] * z + transform[1, 3]
    tz = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2] * z + transform[2, 3]
    return tx, ty, tz


class TopocentricConverter:
    """Convert to and from a topocentric reference frame."""

    def __init__(self, reflat, reflon, refalt):
        self.lat = reflat
        self.lon = reflon
        self.alt = refalt

    def to_topocentric(self, lat, lon, alt):
        return topocentric_from_lla(lat, lon, alt, self.lat, self.lon, self.alt)

    def to_lla(self, x, y, z):
        return lla_from_topocentric(x, y, z, self.lat, self.lon, self.alt)



def _transform(point, reference, projection):
    """Transform one local point to projected coordinates."""
    lat, lon, altitude = reference.to_lla(point[0], point[1], point[2])
    easting, northing = projection(lon, lat)
    return [easting, northing, altitude]


def _get_transformation(reference, projection):
    """Linear transform from OpenSfM local coordinates to projected coordinates."""
    basis_points = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    projected_points = [_transform(point, reference, projection) for point in basis_points]

    return np.array(
        [
            [
                projected_points[0][0] - projected_points[3][0],
                projected_points[1][0] - projected_points[3][0],
                projected_points[2][0] - projected_points[3][0],
                projected_points[3][0],
            ],
            [
                projected_points[0][1] - projected_points[3][1],
                projected_points[1][1] - projected_points[3][1],
                projected_points[2][1] - projected_points[3][1],
                projected_points[3][1],
            ],
            [
                projected_points[0][2] - projected_points[3][2],
                projected_points[1][2] - projected_points[3][2],
                projected_points[2][2] - projected_points[3][2],
                projected_points[3][2],
            ],
            [0, 0, 0, 1],
        ]
    )


def get_transformation_matrix(reference_lla, projection):
    reference = TopocentricConverter(
        reference_lla["latitude"],
        reference_lla["longitude"],
        reference_lla["altitude"],
    )
    pyproj_projection = pyproj.Proj(projection)
    transform = _get_transformation(reference, pyproj_projection)
    transform_inv = np.linalg.inv(transform)
    return transform, transform_inv


def load_json(json_path):
    return json.loads(Path(json_path).read_text())


def load_reference_lla(reference_lla_path):
    reference = load_json(reference_lla_path)
    required = {"latitude", "longitude", "altitude"}
    if not required.issubset(reference.keys()):
        raise ValueError("reference_lla JSON must contain latitude, longitude, altitude")
    return reference


def load_camera_parameters(camera_params_path):
    camera = load_json(camera_params_path)
    required = {"focal_x", "focal_y", "c_x", "c_y"}
    if not required.issubset(camera.keys()):
        raise ValueError("camera params JSON must contain focal_x, focal_y, c_x, c_y")
    return camera


def load_reconstruction_shots(reconstruction_path):
    reconstruction = load_json(reconstruction_path)

    if isinstance(reconstruction, list):
        if not reconstruction:
            raise ValueError("reconstruction file is an empty list")
        reconstruction_root = reconstruction[0]
    elif isinstance(reconstruction, dict):
        reconstruction_root = reconstruction
    else:
        raise ValueError("Unsupported reconstruction JSON format")

    shots = reconstruction_root.get("shots", {})
    if not isinstance(shots, dict) or not shots:
        raise ValueError("No shots found in reconstruction file")

    return shots


def geojson_crs_name(geojson):
    """Return an explicit legacy GeoJSON CRS, or the RFC 7946 WGS84 default."""
    crs = geojson.get("crs") or {}
    name = (crs.get("properties") or {}).get("name")
    if not name:
        return "EPSG:4326"
    if "EPSG::" in name.upper():
        return "EPSG:" + name.rsplit("::", 1)[-1]
    return name


def load_geojson_outer_ring(geojson_path, return_crs=False):
    """Load the first polygon outer ring and, optionally, its source CRS."""
    geojson = load_json(geojson_path)
    features = geojson.get("features", [])
    if not features:
        raise ValueError("GeoJSON has no features")

    geometry = features[0].get("geometry", {})
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "Polygon":
        ring = coordinates[0]
    elif geometry_type == "MultiPolygon":
        ring = coordinates[0][0]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

    ring_array = np.array(ring, dtype=np.float64)
    if ring_array.shape[0] >= 2 and np.allclose(ring_array[0], ring_array[-1]):
        ring_array = ring_array[:-1]

    if ring_array.shape[0] < 3:
        raise ValueError("Polygon ring must have at least 3 unique points")

    if return_crs:
        return ring_array, geojson_crs_name(geojson)
    return ring_array


def transform_geo_points_to_local(reference_lla, projection, coordinate_points, z_value,
                                  source_crs="EPSG:4326"):
    """Convert boundary coordinates from ``source_crs`` to local OpenSfM XYZ.

    The geographic footprint is always transformed at projected height zero,
    exactly like ``transform_extent_to_local``.  The local Z is then assigned
    separately; it must not be passed through the projected-coordinate matrix.
    """
    coordinate_transform = pyproj.Transformer.from_crs(
        source_crs, projection, always_xy=True
    )
    _, transform_inv = get_transformation_matrix(reference_lla, projection)

    local_points = []
    for x, y in coordinate_points[:, :2]:
        easting, northing = coordinate_transform.transform(x, y)
        local = transform_inv @ np.array([easting, northing, 0.0, 1.0], dtype=np.float64)
        local[2] = z_value
        local_points.append(local[:3])

    return np.array(local_points, dtype=np.float32)


def project_local_points(local_points, rotation, translation, camera):
    """Project local XYZ points to image pixel coordinates."""
    camera_matrix = np.array(
        [
            [camera["focal_x"], 0.0, camera["c_x"]],
            [0.0, camera["focal_y"], camera["c_y"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.array(
        [
            camera.get("k1", 0.0),
            camera.get("k2", 0.0),
            camera.get("p1", 0.0),
            camera.get("p2", 0.0),
            camera.get("k3", 0.0),
        ],
        dtype=np.float32,
    )

    rotation_matrix, _ = cv2.Rodrigues(np.array(rotation, dtype=np.float32))
    camera_space = (rotation_matrix @ local_points.T).T + np.array(translation, dtype=np.float32)
    in_front = camera_space[:, 2] > 0.0

    points_2d, _ = cv2.projectPoints(
        local_points,
        np.array(rotation, dtype=np.float32),
        np.array(translation, dtype=np.float32),
        camera_matrix,
        dist_coeffs,
    )
    points_2d = points_2d.reshape(-1, 2)

    return points_2d, in_front


def rasterize_footprint_mask(image_shape, points_2d):
    """Rasterize the original projected footprint; do not expand it to a hull."""
    height, width = image_shape[:2]
    mask_img = np.zeros((height, width), dtype=np.uint8)

    if not np.isfinite(points_2d).all():
        return mask_img
    # OpenCV clips the polygon against the image. Clipping vertices one by one
    # would move exterior corners onto the frame and can create false wedges.
    polygon = np.round(points_2d).astype(np.int32)

    if len(np.unique(polygon, axis=0)) >= 3:
        cv2.fillPoly(mask_img, [polygon], 255)

    return mask_img


def create_debug_overlay(image, mask_img):
    """Create a debug visualization with mask tint and red contour on the image."""
    debug = image.copy()

    green_tint = np.zeros_like(debug)
    green_tint[:, :] = (0, 255, 0)
    blended = cv2.addWeighted(debug, 0.7, green_tint, 0.3, 0)
    masked_pixels = mask_img > 0
    debug[masked_pixels] = blended[masked_pixels]

    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, contours, contourIdx=-1, color=(0, 0, 255), thickness=3)
    return debug


def find_shot_for_image(shots, image_name):
    """Match image file name to shot key in reconstruction."""
    candidates = [
        image_name,
        image_name.lower(),
        image_name.upper(),
        image_name.replace(".jpeg", ".JPG"),
        image_name.replace(".jpg", ".JPG"),
        image_name.replace(".JPG", ".jpg"),
    ]

    for key in candidates:
        if key in shots:
            return shots[key]

    stem = Path(image_name).stem
    for key, shot in shots.items():
        if Path(key).stem == stem:
            return shot

    return None


def ensure_output_dirs(output_folder, mask_folder, debug_folder):
    base_output = Path(output_folder) if output_folder else Path(".")

    mask_dir = Path(mask_folder)
    if not mask_dir.is_absolute():
        mask_dir = base_output / mask_dir

    debug_dir = Path(debug_folder)
    if not debug_dir.is_absolute():
        debug_dir = base_output / debug_dir

    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    return mask_dir, debug_dir


def generate_masks(
    images_folder,
    reconstruction_file,
    projection,
    geojson_file,
    output_folder,
    mask_folder,
    debug_folder,
    camera_params_file,
    reference_lla_file,
    footprint_z,
    geojson_crs=None,
):
    reference_lla = load_reference_lla(reference_lla_file)
    camera = load_camera_parameters(camera_params_file)
    shots = load_reconstruction_shots(reconstruction_file)
    coordinate_ring, detected_crs = load_geojson_outer_ring(geojson_file, return_crs=True)
    source_crs = geojson_crs or detected_crs
    print("GeoJSON source CRS: {}".format(source_crs))
    print("OpenSfM metric projection: {}".format(projection))

    local_footprint_points = transform_geo_points_to_local(
        reference_lla=reference_lla,
        projection=projection,
        coordinate_points=coordinate_ring,
        z_value=footprint_z,
        source_crs=source_crs,
    )

    mask_dir, debug_dir = ensure_output_dirs(output_folder, mask_folder, debug_folder)

    images_dir = Path(images_folder)
    image_files = sorted(
        [
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.JPG"),
            *images_dir.glob("*.jpeg"),
            *images_dir.glob("*.JPEG"),
            *images_dir.glob("*.png"),
            *images_dir.glob("*.PNG"),
        ]
    )

    if not image_files:
        raise ValueError(f"No images found in folder: {images_folder}")

    written = 0
    skipped = 0

    for image_path in image_files:
        shot = find_shot_for_image(shots, image_path.name)
        if shot is None:
            print(f"Skipping {image_path.name}: shot not found in reconstruction")
            skipped += 1
            continue

        rotation = shot.get("rotation")
        translation = shot.get("translation")
        if rotation is None or translation is None:
            print(f"Skipping {image_path.name}: shot missing rotation/translation")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping {image_path.name}: could not read image")
            skipped += 1
            continue

        points_2d, in_front = project_local_points(local_footprint_points, rotation, translation, camera)
        if not np.all(in_front):
            print(f"Skipping {image_path.name}: part of footprint is behind the camera")
            skipped += 1
            continue

        mask_img = rasterize_footprint_mask(image.shape, points_2d)
        debug_overlay = create_debug_overlay(image, mask_img)

        mask_output_path = mask_dir / f"{image_path.stem}.png"
        debug_output_path = debug_dir / f"{image_path.stem}_debug_overlay.jpg"
        cv2.imwrite(str(mask_output_path), mask_img)
        cv2.imwrite(str(debug_output_path), debug_overlay)
        written += 1

        print(f"Mask written: {mask_output_path}")
        print(f"Debug overlay written: {debug_output_path}")

    print(f"Completed. Written: {written}, Skipped: {skipped}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate GeoJSON-based masks for OpenSfM shots")
    parser.add_argument("--images-folder", required=True, help="Folder containing input images")
    parser.add_argument(
        "--reconstruction-file",
        required=True,
        help="OpenSfM reconstruction JSON file containing shots",
    )
    parser.add_argument("--projection", required=True, help="Projection, for example EPSG:32755")
    parser.add_argument("--geojson-file", required=True, help="Boundary GeoJSON file path")
    parser.add_argument(
        "--geojson-crs",
        help="source CRS override, e.g. EPSG:32755; otherwise read GeoJSON crs or use EPSG:4326",
    )
    parser.add_argument(
        "--output-folder",
        default=".",
        help="Base output folder used for relative mask/debug folders",
    )
    parser.add_argument(
        "--mask-folder",
        required=True,
        help="Mask output folder (absolute or relative to output-folder)",
    )
    parser.add_argument(
        "--debug-folder",
        required=True,
        help="Debug overlay output folder (absolute or relative to output-folder)",
    )
    parser.add_argument(
        "--camera-params-file",
        required=True,
        help="JSON file with camera intrinsics and distortion params",
    )
    parser.add_argument(
        "--reference-lla-file",
        required=True,
        help="JSON file with latitude, longitude, altitude",
    )
    parser.add_argument("--footprint-z", type=float, required=True,
                        help="Single local elevation used to project the stockpile footprint")
    return parser


def main():
    args = build_arg_parser().parse_args()

    generate_masks(
        images_folder=args.images_folder,
        reconstruction_file=args.reconstruction_file,
        projection=args.projection,
        geojson_file=args.geojson_file,
        output_folder=args.output_folder,
        mask_folder=args.mask_folder,
        debug_folder=args.debug_folder,
        camera_params_file=args.camera_params_file,
        reference_lla_file=args.reference_lla_file,
        footprint_z=args.footprint_z,
        geojson_crs=args.geojson_crs,
    )


if __name__ == "__main__":
    main()
