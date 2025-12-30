#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

workers=${WORKERS:-4}
verbose=${VERBOSE:-true}

for base in "${ROOT_DIR}/data1" "${ROOT_DIR}/data2"; do
  if [ ! -d "${base}" ]; then
    continue
  fi
  dataset_dirs=()
  while IFS= read -r -d '' entry; do
    dataset_dirs+=("${entry}")
  done < <(find -L "${base}" -mindepth 1 -maxdepth 1 -type d -print0)

  if [ "${#dataset_dirs[@]}" -eq 0 ]; then
    echo "[WARN] No dataset folders found under ${base}" >&2
    continue
  fi

  printf '%s\0' "${dataset_dirs[@]}" \
    | xargs -0 -P "${workers}" -I {} bash -c '
        if [ "$0" = "true" ]; then
          echo "[RUN] $2" >&2
          python3 "$1" "$2" --verbose --clean
        else
          python3 "$1" "$2" --clean
        fi
      ' "${verbose}" "${ROOT_DIR}/scripts/export_frame_actions.py" {} \
    || exit 1
done
