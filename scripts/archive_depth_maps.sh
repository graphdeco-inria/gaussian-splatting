#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/archive_depth_maps.sh [ROOT ...]

Default roots: /mnt/DATA1/dongjk/navdp_data /mnt/DATA2/dongjk
For each dataset directory under each root, this script:
  1) Moves *_depth.png into a staging folder inside the dataset
     (so we stay on the same disk), preserving the scene/path structure.
  2) Zips the staging folder into <dataset>/<dataset>_depths.zip.
  3) Deletes the staging folder after the zip succeeds.
This avoids creating a second copy of depth PNGs.
Default scene folders are matched by ^[0-9]{4}_*.
Use SKIP_DATASETS to skip datasets, e.g. SKIP_DATASETS="10w_fpv".
EOF
  exit 0
fi

if [ "$#" -gt 0 ]; then
  ROOTS=("$@")
else
  ROOTS=(/mnt/DATA1/dongjk/navdp_data /mnt/DATA2/dongjk)
fi

SKIP_DATASETS=${SKIP_DATASETS:-"10w_fpv"}

for root in "${ROOTS[@]}"; do
  if [ ! -d "$root" ]; then
    echo "[WARN] Root not found: $root"
    continue
  fi
  echo "[SCAN] Root: $root"
  found_dataset=false
  while IFS= read -r -d '' dataset; do
    found_dataset=true
    dataset_name="$(basename "$dataset")"
    if [[ ",${SKIP_DATASETS}," == *",${dataset_name},"* ]]; then
      echo "[SKIP] Dataset excluded: $dataset_name"
      continue
    fi
    echo "[SCAN] Dataset: $dataset"
    if ! find "$dataset" -type f -name "*_depth.png" -print -quit | grep -q .; then
      echo "[SKIP] No depth maps in $dataset"
      continue
    fi

    staging="${dataset}/__depth_maps"
    zip_path="${dataset}/${dataset_name}_depths.zip"
    if [ -e "$staging" ]; then
      echo "[WARN] Staging path already exists, skipping: $staging"
      continue
    fi

    mkdir -p "$staging"
    moved_count=0
    while IFS= read -r -d '' scene_dir; do
      while IFS= read -r -d '' file; do
        rel="${file#$dataset/}"
        mkdir -p "$staging/$(dirname "$rel")"
        echo "[MOVE] $file -> $staging/$rel"
        mv "$file" "$staging/$rel"
        moved_count=$((moved_count + 1))
      done < <(find "$scene_dir" -type f -name "*_depth.png" -print0)
    done < <(find "$dataset" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9]_*' -print0)

    if [ "$moved_count" -eq 0 ]; then
      echo "[SKIP] No depth maps moved for $dataset"
      rm -rf "$staging"
      continue
    fi

    rm -f "$zip_path"
    if ! command -v zip >/dev/null 2>&1; then
      echo "[ERROR] zip not found. Install zip or set ARCHIVE_ROOT and use tar."
      exit 1
    fi
    (
      cd "$dataset"
      zip -r -q "$(basename "$zip_path")" "$(basename "$staging")"
    )

    rm -rf "$staging"
    echo "[OK] Archived depth maps -> $zip_path"
  done < <(find -L "$root" -mindepth 1 -maxdepth 1 -type d -print0)
  if [ "$found_dataset" = false ]; then
    echo "[WARN] No dataset directories found under: $root"
  fi
done
