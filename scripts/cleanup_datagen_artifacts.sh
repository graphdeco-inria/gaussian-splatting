#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/cleanup_datagen_artifacts.sh [--dry-run]

Deletes:
  - *.plt files
  - scene/path-level *.ply files
  - per-path *_BEV.png files

Use --dry-run to only print what would be deleted.
EOF
  exit 0
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

targets=("/mnt/DATA1/dongjk/navdp_data" "/mnt/DATA2/dongjk")

for t in "${targets[@]}"; do
  if [[ -d "$t" ]]; then
    echo "Removing scene/path-level .ply files under: $t"
    while IFS= read -r -d '' file; do
      echo "[DEL] $file"
      if [ "$DRY_RUN" = false ]; then
        rm -f "$file"
      fi
    done < <(find "$t" -mindepth 3 -maxdepth 4 -type f -name "*.ply" -print0)

    echo "Removing per-path BEV PNGs under: $t"
    while IFS= read -r -d '' file; do
      echo "[DEL] $file"
      if [ "$DRY_RUN" = false ]; then
        rm -f "$file"
      fi
    done < <(find "$t" -mindepth 3 -maxdepth 4 -type f -name "*_BEV.png" -print0)
  else
    echo "Skip missing dir: $t"
  fi
done
