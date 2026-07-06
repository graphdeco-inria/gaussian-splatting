#
# Validate and zip VAR/Viettel AI Race rendered submissions.
#

import csv
import zipfile
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image


def expected_output_name(csv_image_name, keep_csv_extension):
    image_name = Path(csv_image_name).name
    if keep_csv_extension:
        return image_name
    return str(Path(image_name).with_suffix(".png"))


def iter_scene_dirs(phase_dir, set_name):
    root = Path(phase_dir) / set_name
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        csv_path = scene_dir / "test" / "test_poses.csv"
        if csv_path.exists():
            yield scene_dir.name, csv_path


def validate_scene(submission_dir, scene_name, csv_path, keep_csv_extension):
    rendered_scene_dir = Path(submission_dir) / scene_name
    if not rendered_scene_dir.exists():
        raise FileNotFoundError(f"Missing rendered scene folder: {rendered_scene_dir}")

    missing = []
    wrong_size = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        out_name = expected_output_name(row["image_name"], keep_csv_extension)
        out_path = rendered_scene_dir / out_name
        if not out_path.exists():
            missing.append(out_name)
            continue
        expected_size = (int(float(row["width"])), int(float(row["height"])))
        with Image.open(out_path) as image:
            if image.size != expected_size:
                wrong_size.append((out_name, image.size, expected_size))

    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{scene_name}: missing {len(missing)} files, e.g. {preview}")
    if wrong_size:
        preview = ", ".join(f"{name}: {got}!={want}" for name, got, want in wrong_size[:5])
        raise RuntimeError(f"{scene_name}: wrong image sizes, e.g. {preview}")

    return len(rows)


def zip_submission(submission_dir, zip_path):
    submission_dir = Path(submission_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in submission_dir.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(submission_dir).as_posix())


if __name__ == "__main__":
    parser = ArgumentParser(description="Validate and zip VAR/VAI NVS submission")
    parser.add_argument("--phase_dir", required=True, help="Path to phase1 directory")
    parser.add_argument("--set_name", default="private_set1")
    parser.add_argument("--submission_dir", required=True)
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--keep_csv_extension", action="store_true")
    args = parser.parse_args()

    total = 0
    for scene_name, csv_path in iter_scene_dirs(args.phase_dir, args.set_name):
        count = validate_scene(args.submission_dir, scene_name, csv_path, args.keep_csv_extension)
        print(f"{scene_name}: {count} images OK")
        total += count

    zip_submission(args.submission_dir, args.zip_path)
    print(f"Packed {total} images -> {args.zip_path}")
