#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/storage_targets.sh"

# ===== Storage toggles =====
ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE:-true}
ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE:-true}
ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE:-false}
CLEAR_LOCAL_OUTPUT_DIR=${CLEAR_LOCAL_OUTPUT_DIR:-true}

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
  && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
  && ! storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  echo "[STORAGE] ERROR: At least one of ENABLE_LOCAL_STORAGE, ENABLE_NAS_STORAGE, ENABLE_REMOTE_STORAGE must be true." >&2
  exit 1
fi

# ===== Paths and defaults =====
SCENE=${SCENE:-test_scene}
CONDA_ENV=${CONDA_ENV:-cuda121}
LOCAL_OUTPUT_DIR=${LOCAL_OUTPUT_DIR:-${SCRIPT_DIR}/data/path_video_frames_Jiankun_test}
ACTOR_SEQ_DIR=${ACTOR_SEQ_DIR:-/media/dongjk/walk_45/}
OFFLOAD_NAS_DIR=${OFFLOAD_NAS_DIR:-/mnt/nas/jiankundong/path_video_frames_Jiankun_test}
OFFLOAD_MIN_FREE_GB=${OFFLOAD_MIN_FREE_GB:-0.5}
REMOTE_STORAGE_ROOT=${REMOTE_STORAGE_ROOT:-${REMOTE_OUTPUT_DIR:-/srv/navdp}}
REMOTE_SSH_TARGET=${REMOTE_SSH_TARGET:-user@other-training-pc}
LOCAL_OUTPUT_BASENAME="$(basename "$LOCAL_OUTPUT_DIR")"
REMOTE_TARGET_DIR="${REMOTE_STORAGE_ROOT%/}/${LOCAL_OUTPUT_BASENAME}"

echo "[CONFIG] ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE}"
echo "[CONFIG] ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE}"
echo "[CONFIG] ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE}"

if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  NAS_TEST_DIR="${OFFLOAD_NAS_DIR}/${SCENE}"
  if mkdir -p "${NAS_TEST_DIR}" \
    && : > "${NAS_TEST_DIR}/__touch_test__" \
    && rm -f "${NAS_TEST_DIR}/__touch_test__"; then
    echo "[CHECK] NAS reachable at ${OFFLOAD_NAS_DIR}"
  else
    echo "[CHECK] ERROR: cannot write to NAS ${OFFLOAD_NAS_DIR}" >&2
    exit 1
  fi
fi

prepare_local_output_dir() {
  local target="$1"
  mkdir -p "$target"
  if storage_bool_true "$CLEAR_LOCAL_OUTPUT_DIR"; then
    echo "[CLEAN] Clearing previous contents under ${target}"
    find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

prepare_local_output_dir "$LOCAL_OUTPUT_DIR"

BASE_ARGS=(
  --overwrite
  --stabilize
  --gpu-only
  --show-BEV
  --actor-seq-dir "$ACTOR_SEQ_DIR"
  --height-offset -0.098
  --minimal-frames 90
  --output-dir "$LOCAL_OUTPUT_DIR"
)

if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  BASE_ARGS+=(--offload-nas-dir "$OFFLOAD_NAS_DIR" --offload-min-free-gb "$OFFLOAD_MIN_FREE_GB")
fi

echo "[RUN] conda env=${CONDA_ENV}"
conda run --no-capture-output -n "$CONDA_ENV" python render_label_paths.py \
  "${BASE_ARGS[@]}" \
  "$@"

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  storage_sync_remote "$LOCAL_OUTPUT_DIR" "$REMOTE_TARGET_DIR"
fi

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE"; then
  if [ -d "$LOCAL_OUTPUT_DIR" ]; then
    echo "[STORAGE] Removing local outputs at ${LOCAL_OUTPUT_DIR}"
    rm -rf "$LOCAL_OUTPUT_DIR"
  fi
fi
