#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/storage_targets.sh"

# Ensure we have a Python interpreter available (needed for path resolution helper below).

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] python3 is required but was not found in PATH." >&2
    exit 1
  fi
fi

# Tiny helper for consistent usage errors so RESUME mode is easy to discover.
show_usage_and_exit() {
  echo "Usage: $(basename "$0") [RESUME <log-file>]" >&2
  exit 1
}

# CLI parsing: optional leading "RESUME <log>" pair switches to resume mode and
# consumes the following logfile argument so the remainder of the script can lean
# on env vars only.
RESUME_MODE=false
RESUME_LOG_PATH="./65k.log"
if [ $# -gt 0 ]; then
  if [ "$1" = "RESUME" ]; then
    RESUME_MODE=true
    shift
    if [ $# -lt 1 ]; then
      echo "[RESUME] ERROR: log file path missing." >&2
      show_usage_and_exit
    fi
    RESUME_LOG_PATH="$1"
    shift
  else
    echo "[ERROR] Unknown argument: $1" >&2
    show_usage_and_exit
  fi
fi
if [ $# -gt 0 ]; then
  echo "[ERROR] Unexpected arguments: $*" >&2
  show_usage_and_exit
fi

# Convenience wrapper so we can expand relative paths and keep the script POSIX-ish.
abspath() {
  "$PYTHON_BIN" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

# -----------------------------------------------------------------------------
# User-configurable defaults (override via env vars before invoking the script).
# Collect everything editable here so deployments only need to tweak this block.
# -----------------------------------------------------------------------------
# Storage toggles so the same runner can ship data to different targets.
ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE:-true}
ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE:-false}
ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE:-false}
CLEAR_LOCAL_OUTPUT_DIR=${CLEAR_LOCAL_OUTPUT_DIR:-false}

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
  && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
  && ! storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  echo "[STORAGE] ERROR: At least one of ENABLE_LOCAL_STORAGE, ENABLE_NAS_STORAGE, ENABLE_REMOTE_STORAGE must be true." >&2
  exit 1
fi

REMOTE_ONLY_STORAGE=false
if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
  && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
  && storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  REMOTE_ONLY_STORAGE=true
fi
: "${REMOTE_SYNC_REMOVE_SOURCE_FILES:=false}"
if [ "$REMOTE_ONLY_STORAGE" = true ]; then
  : "${REMOTE_SYNC_REMOVE_SOURCE_FILES:=true}"
fi
ENABLE_REMOTE_STORAGE_GUARD=${ENABLE_REMOTE_STORAGE_GUARD:-true}
REMOTE_STORAGE_LIMIT_GB=${REMOTE_STORAGE_LIMIT_GB:-500}
REMOTE_STORAGE_RESUME_GB=${REMOTE_STORAGE_RESUME_GB:-200}
REMOTE_STORAGE_GUARD_INTERVAL_SECS=${REMOTE_STORAGE_GUARD_INTERVAL_SECS:-60}
REMOTE_STORAGE_GUARD_ENABLED=false
if [ "$REMOTE_ONLY_STORAGE" = true ] && storage_bool_true "$ENABLE_REMOTE_STORAGE_GUARD"; then
  REMOTE_STORAGE_GUARD_ENABLED=true
fi
SETSID_BIN=""
if command -v setsid >/dev/null 2>&1; then
  SETSID_BIN=$(command -v setsid)
fi
if [ "$REMOTE_STORAGE_GUARD_ENABLED" = true ] && [ -z "$SETSID_BIN" ]; then
  echo "[STORAGE] WARN: setsid is unavailable; disabling remote storage guard throttling." >&2
  REMOTE_STORAGE_GUARD_ENABLED=false
fi

# Core configuration for assignment planning + rendering. Most callers just tweak
# DATA roots or seeds via environment variables.
SEED=${SEED:-65}
CONDA_ENV=${CONDA_ENV:-cuda121}
ACTOR_ROOT=${ACTOR_ROOT:-./data/human_gs_source}
BAN_LIST=${BAN_LIST:-${ACTOR_ROOT}/BanList.txt}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-./data/actor_assignments_w_ban_65k.json}
SCENES_DIR=${SCENES_DIR:-./data/scenes}
TASKS_DIR=${TASKS_DIR:-./data/selected_65k}
OUTPUT_DIR=${OUTPUT_DIR:-./data/path_video_frames_random_humans_65k}
OFFLOAD_NAS_DIR=${OFFLOAD_NAS_DIR:-/mnt/nas/jiankundong/random_human_dataset_w_ban_65k}
OFFLOAD_MIN_FREE_GB=${OFFLOAD_MIN_FREE_GB:-0.5}
PROGRESS_JSON=${PROGRESS_JSON:-./analysis/random_human_progress.json}
PER_JOB_METRICS_DIR=${PER_JOB_METRICS_DIR:-./analysis/random_human_metrics}
REMOTE_STORAGE_ROOT=${REMOTE_STORAGE_ROOT:-${REMOTE_OUTPUT_DIR:-/mnt/DATA/navdp_data_65k}}
REMOTE_SSH_TARGET=${REMOTE_SSH_TARGET:-lenovo@192.168.151.40}
LOCAL_OUTPUT_BASENAME="$(basename "$OUTPUT_DIR")"
REMOTE_TARGET_DIR="${REMOTE_STORAGE_ROOT%/}/${LOCAL_OUTPUT_BASENAME}"
REMOTE_SYNC_INTERVAL_SECS=${REMOTE_SYNC_INTERVAL_SECS:-120}
WORKERS=${WORKERS:-12}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-38}
# vram reserve function
RESERVE_VRAM_GB=${RESERVE_VRAM_GB:-0}
RESERVE_VRAM_HEADROOM_GB=${RESERVE_VRAM_HEADROOM_GB:-1}
# set files types we want as output
ENABLE_BEV_IMAGES=${ENABLE_BEV_IMAGES:-true}
ENABLE_VIDEO_OUTPUT=${ENABLE_VIDEO_OUTPUT:-true}
ENABLE_RGB_FRAMES=${ENABLE_RGB_FRAMES:-false}
ENABLE_DEPTH_OUTPUT=${ENABLE_DEPTH_OUTPUT:-true}
ENABLE_FOLLOW_METADATA=${ENABLE_FOLLOW_METADATA:-true}

# Default render_label_paths.py snippets appended to every worker invocation.
render_extra_args="--overwrite --stabilize --gpu-only --navdp-ply-per-scene"
if storage_bool_true "$ENABLE_BEV_IMAGES"; then
  render_extra_args+=' --show-BEV'
else
  render_extra_args+=' --no-show-BEV'
fi
if storage_bool_true "$ENABLE_VIDEO_OUTPUT"; then
  render_extra_args+=' --video'
else
  render_extra_args+=' --no-video'
fi
if storage_bool_true "$ENABLE_RGB_FRAMES"; then
  render_extra_args+=' --rgb-frames'
else
  render_extra_args+=' --no-rgb-frames'
fi
if storage_bool_true "$ENABLE_DEPTH_OUTPUT"; then
  render_extra_args+=' --save-depth-maps'
else
  render_extra_args+=' --no-save-depth-maps'
fi
if storage_bool_true "$ENABLE_FOLLOW_METADATA"; then
  render_extra_args+=' --save-follow-metadata'
else
  render_extra_args+=' --no-save-follow-metadata'
fi
render_extra_snippets=("$render_extra_args")
if ! [[ "$RESERVE_VRAM_GB" =~ ^[0-9]+$ ]]; then
  echo "[VRAM] ERROR: RESERVE_VRAM_GB must be an integer value (received '$RESERVE_VRAM_GB')." >&2
  exit 1
fi
if ! [[ "$RESERVE_VRAM_HEADROOM_GB" =~ ^[0-9]+$ ]]; then
  echo "[VRAM] ERROR: RESERVE_VRAM_HEADROOM_GB must be an integer value (received '$RESERVE_VRAM_HEADROOM_GB')." >&2
  exit 1
fi
if ! [[ "$REMOTE_STORAGE_LIMIT_GB" =~ ^[0-9]+$ ]]; then
  echo "[STORAGE] ERROR: REMOTE_STORAGE_LIMIT_GB must be an integer value (received '$REMOTE_STORAGE_LIMIT_GB')." >&2
  exit 1
fi
if ! [[ "$REMOTE_STORAGE_RESUME_GB" =~ ^[0-9]+$ ]]; then
  echo "[STORAGE] ERROR: REMOTE_STORAGE_RESUME_GB must be an integer value (received '$REMOTE_STORAGE_RESUME_GB')." >&2
  exit 1
fi
if [ "$REMOTE_STORAGE_RESUME_GB" -ge "$REMOTE_STORAGE_LIMIT_GB" ]; then
  echo "[STORAGE] ERROR: REMOTE_STORAGE_RESUME_GB (${REMOTE_STORAGE_RESUME_GB}GB) must be less than REMOTE_STORAGE_LIMIT_GB (${REMOTE_STORAGE_LIMIT_GB}GB)." >&2
  exit 1
fi
if ! [[ "$REMOTE_STORAGE_GUARD_INTERVAL_SECS" =~ ^[0-9]+$ ]]; then
  echo "[STORAGE] ERROR: REMOTE_STORAGE_GUARD_INTERVAL_SECS must be an integer value (received '$REMOTE_STORAGE_GUARD_INTERVAL_SECS')." >&2
  exit 1
fi

# VRAM throttling: spawn a small Python helper that grabs a configurable chunk of
# GPU memory so other users do not over-subscribe the card while this run is in
# flight. The tensor holder stays alive until we exit (trap below).
VRAM_RESERVATION_PID=""
reserve_vram() {
  local reserve_gb="$1"
  if [ -z "$reserve_gb" ]; then
    return
  fi
  local bytes=$((reserve_gb * 1024 * 1024 * 1024))
  if [ "$bytes" -le 0 ]; then
    return
  fi
  local headroom_bytes=$((RESERVE_VRAM_HEADROOM_GB * 1024 * 1024 * 1024))
  echo "[VRAM] Guarding ${reserve_gb} GiB (headroom ${RESERVE_VRAM_HEADROOM_GB} GiB) to discourage other jobs."
  RESERVE_VRAM_TARGET_BYTES="$bytes" \
  RESERVE_VRAM_HEADROOM_BYTES="$headroom_bytes" \
    conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY' &
import os
import sys
import time

try:
    import torch
except Exception as exc:  # pylint: disable=broad-except
    print(f"[VRAM] ERROR: Unable to import torch: {exc}", file=sys.stderr, flush=True)
    sys.exit(1)

target_bytes = int(os.environ.get("RESERVE_VRAM_TARGET_BYTES", "0"))
headroom_bytes = int(os.environ.get("RESERVE_VRAM_HEADROOM_BYTES", str(512 * 1024 * 1024)))
if target_bytes <= 0:
    sys.exit(0)
device = torch.device("cuda:0")
torch.cuda.set_device(device)
dev_index = torch.cuda.current_device()
dev_name = torch.cuda.get_device_name(dev_index)

CHUNK_BYTES = 256 * 1024 * 1024
tensors = []

def reserved_bytes() -> int:
    return sum(t.element_size() * t.numel() for t in tensors)

def grow(target_delta: int) -> None:
    remaining = target_delta
    while remaining > 0:
        chunk = min(remaining, CHUNK_BYTES)
        if chunk >= 4:
            tensors.append(torch.empty((chunk // 4,), dtype=torch.float32, device=device))
            chunk = (chunk // 4) * 4
        else:
            tensors.append(torch.empty((chunk,), dtype=torch.uint8, device=device))
        remaining -= chunk

def shrink(target_delta: int) -> None:
    remaining = target_delta
    while tensors and remaining > 0:
        tensor = tensors.pop()
        size = tensor.element_size() * tensor.numel()
        remaining -= size
        del tensor
    torch.cuda.empty_cache()

def refresh_reservation() -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    max_hold = max(0, free_bytes - headroom_bytes)
    desired = min(target_bytes, max_hold)
    current = reserved_bytes()
    delta = desired - current
    if abs(delta) < (32 * 1024 * 1024):
        return
    if delta > 0:
        grow(delta)
    else:
        shrink(-delta)
    new_total = reserved_bytes()
    print(
        f"[VRAM] Adjusted guard tensors to {new_total / (1024 ** 3):.2f} GiB (free {free_bytes / (1024 ** 3):.2f} / total {total_bytes / (1024 ** 3):.2f} GiB).",
        flush=True,
    )

print(
    f"[VRAM] Dynamic guard active on cuda:{dev_index} ({dev_name}), target {target_bytes / (1024 ** 3):.2f} GiB, headroom {headroom_bytes / (1024 ** 3):.2f} GiB.",
    flush=True,
)
try:
    while True:
        refresh_reservation()
        time.sleep(5)
except KeyboardInterrupt:
    pass
PY
  VRAM_RESERVATION_PID=$!
  sleep 1
  if ! kill -0 "$VRAM_RESERVATION_PID" >/dev/null 2>&1; then
    echo "[VRAM] ERROR: Failed to start reservation helper." >&2
    exit 1
  fi
}

release_vram() {
  if [ -n "$VRAM_RESERVATION_PID" ]; then
    kill "$VRAM_RESERVATION_PID" >/dev/null 2>&1 || true
    wait "$VRAM_RESERVATION_PID" >/dev/null 2>&1 || true
    echo "[VRAM] Released reserved GPU memory."
    VRAM_RESERVATION_PID=""
  fi
}

REMOTE_SYNC_WORKER_PID=""
REMOTE_SYNC_DONE_FILE=""
REMOTE_STORAGE_UNAVAILABLE=false
PARALLEL_PID=""
STOP_REQUESTED=false

handle_remote_storage_unavailable() {
  if [ "$REMOTE_STORAGE_UNAVAILABLE" = true ]; then
    return
  fi
  REMOTE_STORAGE_UNAVAILABLE=true
  echo "[STORAGE] Remote destination unavailable; pausing generation to avoid data loss." >&2
  if [ -n "$PARALLEL_PID" ]; then
    kill "$PARALLEL_PID" >/dev/null 2>&1 || true
  fi
}

remote_sync_worker_loop() {
  local source_dir="$1"
  local remote_dir="$2"
  local done_flag="$3"
  local interval_secs="$4"
  local abort_on_failure="${5:-false}"
  local parent_pid="${6:-}"
  if [ -z "$interval_secs" ] || [ "$interval_secs" -le 0 ]; then
    interval_secs=60
  fi
  echo "[STORAGE] Remote sync worker started for ${source_dir} -> ${REMOTE_SSH_TARGET:-?}:${remote_dir} (interval ${interval_secs}s)"
  local iteration=0
  while true; do
    iteration=$((iteration + 1))
    storage_sync_remote "$source_dir" "$remote_dir"
    local sync_status=$?
    if [ $sync_status -ne 0 ]; then
      echo "[STORAGE] WARN: Remote sync worker pass ${iteration} failed with status ${sync_status}." >&2
      if [ "$abort_on_failure" = true ]; then
        echo "[STORAGE] Remote sync worker detected unreachable destination; notifying renderer to pause." >&2
        if [ -n "$parent_pid" ]; then
          kill -s USR1 "$parent_pid" >/dev/null 2>&1 || true
        fi
        break
      fi
    fi
    if [ -f "$done_flag" ] && [ $sync_status -eq 0 ]; then
      echo "[STORAGE] Remote sync worker confirmed final sync after ${iteration} pass(es)."
      break
    fi
    sleep "$interval_secs"
  done
  echo "[STORAGE] Remote sync worker exiting."
}

start_remote_sync_worker() {
  local source_dir="$1"
  local remote_dir="$2"
  local interval_secs="$3"
  local abort_on_failure="${4:-false}"
  local parent_pid="${5:-}"
  REMOTE_SYNC_DONE_FILE=$(mktemp "${TMPDIR:-/tmp}/remote_sync_done.XXXXXX") || return 1
  rm -f "$REMOTE_SYNC_DONE_FILE"
  remote_sync_worker_loop "$source_dir" "$remote_dir" "$REMOTE_SYNC_DONE_FILE" "$interval_secs" "$abort_on_failure" "$parent_pid" &
  REMOTE_SYNC_WORKER_PID=$!
}

signal_remote_sync_completion() {
  if [ -n "$REMOTE_SYNC_DONE_FILE" ]; then
    : > "$REMOTE_SYNC_DONE_FILE"
  fi
}

wait_remote_sync_worker() {
  if [ -n "$REMOTE_SYNC_WORKER_PID" ]; then
    wait "$REMOTE_SYNC_WORKER_PID" || true
    REMOTE_SYNC_WORKER_PID=""
  fi
  if [ -n "$REMOTE_SYNC_DONE_FILE" ]; then
    rm -f "$REMOTE_SYNC_DONE_FILE"
    REMOTE_SYNC_DONE_FILE=""
  fi
}

REMOTE_STORAGE_GUARD_PID=""
REMOTE_STORAGE_GUARD_STOP_FILE=""
PARALLEL_GROUP_ID=""

storage_guard_loop() {
  local watch_dir="$1"
  local pgid="$2"
  local limit_bytes="$3"
  local resume_bytes="$4"
  local interval_secs="$5"
  local stop_flag="$6"
  local limit_gb="$7"
  local resume_gb="$8"
  local paused=false
  if [ -z "$interval_secs" ] || [ "$interval_secs" -le 0 ]; then
    interval_secs=60
  fi
  while true; do
    if [ -n "$stop_flag" ] && [ -f "$stop_flag" ]; then
      break
    fi
    if [ -z "$pgid" ] || ! kill -0 "$pgid" >/dev/null 2>&1; then
      break
    fi
    local usage_bytes
    usage_bytes=$(du -sb "$watch_dir" 2>/dev/null | awk '{print $1}')
    if [ -z "$usage_bytes" ]; then
      usage_bytes=0
    fi
    if [ "$usage_bytes" -ge "$limit_bytes" ]; then
      if [ "$paused" = false ]; then
        local usage_gb
        usage_gb=$(awk -v b="$usage_bytes" 'BEGIN { printf("%.2f", b / (1024*1024*1024)) }')
        echo "[STORAGE] Local usage ${usage_gb}GB reached limit ${limit_gb}GB; pausing generation until below ${resume_gb}GB."
        kill -STOP -- "-$pgid" >/dev/null 2>&1 || true
        paused=true
      fi
    elif [ "$paused" = true ] && [ "$usage_bytes" -le "$resume_bytes" ]; then
      echo "[STORAGE] Local usage dropped below ${resume_gb}GB; resuming generation."
      kill -CONT -- "-$pgid" >/dev/null 2>&1 || true
      paused=false
    fi
    sleep "$interval_secs"
  done
  if [ "$paused" = true ] && [ -n "$pgid" ]; then
    kill -CONT -- "-$pgid" >/dev/null 2>&1 || true
  fi
}

start_storage_guard() {
  local watch_dir="$1"
  local pgid="$2"
  local limit_gb="$3"
  local resume_gb="$4"
  local interval_secs="$5"
  local limit_bytes=$((limit_gb * 1024 * 1024 * 1024))
  local resume_bytes=$((resume_gb * 1024 * 1024 * 1024))
  REMOTE_STORAGE_GUARD_STOP_FILE=$(mktemp "${TMPDIR:-/tmp}/storage_guard_stop.XXXXXX") || return 1
  rm -f "$REMOTE_STORAGE_GUARD_STOP_FILE"
  storage_guard_loop "$watch_dir" "$pgid" "$limit_bytes" "$resume_bytes" "$interval_secs" "$REMOTE_STORAGE_GUARD_STOP_FILE" "$limit_gb" "$resume_gb" &
  REMOTE_STORAGE_GUARD_PID=$!
}

stop_storage_guard() {
  if [ -n "$REMOTE_STORAGE_GUARD_STOP_FILE" ]; then
    : > "$REMOTE_STORAGE_GUARD_STOP_FILE"
  fi
  if [ -n "$REMOTE_STORAGE_GUARD_PID" ]; then
    wait "$REMOTE_STORAGE_GUARD_PID" || true
    REMOTE_STORAGE_GUARD_PID=""
  fi
  if [ -n "$REMOTE_STORAGE_GUARD_STOP_FILE" ]; then
    rm -f "$REMOTE_STORAGE_GUARD_STOP_FILE"
    REMOTE_STORAGE_GUARD_STOP_FILE=""
  fi
}

cleanup_run() {
  release_vram
  wait_remote_sync_worker
  stop_storage_guard
}

# Always tear down helpers even on errors.
trap cleanup_run EXIT
trap handle_remote_storage_unavailable USR1

handle_interrupt() {
  if [ "$STOP_REQUESTED" = true ]; then
    return
  fi
  STOP_REQUESTED=true
  echo "[RUN] Interrupt received; stopping workers..." >&2
  trap - EXIT
  if [ -n "$PARALLEL_PID" ]; then
    kill -SIGINT "$PARALLEL_PID" >/dev/null 2>&1 || true
    wait "$PARALLEL_PID" >/dev/null 2>&1 || true
    PARALLEL_PID=""
  fi
  if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
    signal_remote_sync_completion
  fi
  cleanup_run
  exit 130
}
trap handle_interrupt INT TERM

# Assignment manifest generation helper.
generate_assignment_manifest() {
  local manifest_dir
  manifest_dir="$(dirname "$ASSIGNMENTS_OUT")"
  mkdir -p "$manifest_dir"
  echo "[RUN] Random human data generation, seed ${SEED}"
  conda run --no-capture-output -n "$CONDA_ENV" python random_actor_assignments.py \
    --actor-root "${ACTOR_ROOT}" \
    --ban-list "${BAN_LIST}" \
    --assignments-out "${ASSIGNMENTS_OUT}" \
    --scenes-dir "${SCENES_DIR}" \
    --tasks-dir "${TASKS_DIR}" \
    --seed "${SEED}"
  echo "Assignment manifest generated at ${ASSIGNMENTS_OUT}"
}

RESUME_LOG_ABS=""
# Resume bookkeeping: we reuse the assignment manifest referenced inside the
# provided log file and disable destructive cleanup so partially-generated data is
# preserved.
if $RESUME_MODE; then
  RESUME_LOG_ABS=$(abspath "$RESUME_LOG_PATH")
  if [ ! -f "$RESUME_LOG_ABS" ]; then
    echo "[RESUME] ERROR: Log file not found: $RESUME_LOG_PATH" >&2
    exit 1
  fi
  manifest_line=$(grep -m1 "Assignment manifest generated at" "$RESUME_LOG_ABS" || true)
  if [ -z "$manifest_line" ]; then
    echo "[RESUME] ERROR: Could not locate assignment manifest line in $RESUME_LOG_PATH" >&2
    exit 1
  fi
  resume_manifest_path="${manifest_line#*Assignment manifest generated at }"
  resume_manifest_path="${resume_manifest_path%% *}"
  if [ -z "$resume_manifest_path" ]; then
    echo "[RESUME] ERROR: Failed to parse assignment manifest path from log." >&2
    exit 1
  fi
  if [[ "$resume_manifest_path" != /* ]]; then
    resume_manifest_path="${resume_manifest_path#./}"
    resume_manifest_path="${SCRIPT_DIR}/${resume_manifest_path}"
  fi
  ASSIGNMENTS_OUT=$(abspath "$resume_manifest_path")
  if [ ! -f "$ASSIGNMENTS_OUT" ]; then
    echo "[RESUME] WARN: Assignment manifest missing at $ASSIGNMENTS_OUT; regenerating."
    generate_assignment_manifest
    if [ ! -f "$ASSIGNMENTS_OUT" ]; then
      echo "[RESUME] ERROR: Failed to regenerate assignment manifest at $ASSIGNMENTS_OUT." >&2
      exit 1
    fi
  fi
  echo "[RESUME] Using manifest $ASSIGNMENTS_OUT (derived from $RESUME_LOG_PATH)"
  CLEAR_LOCAL_OUTPUT_DIR=false
fi

# Utility: guarantee output dir exists + is writable before we drop a ton of
# frames in there.
ensure_writable_dir() {
  local target="$1"
  if [ ! -d "$target" ]; then
    mkdir -p "$target"
  fi
  if [ ! -w "$target" ]; then
    chmod 777 "$target"
  fi
  if [ ! -w "$target" ]; then
    echo "ERROR: Output directory $target is not writable." >&2
    exit 1
  fi
}

# Optionally wipe previous contents to keep runs deterministic unless resume
# mode disabled the cleanup step earlier.
prepare_local_output_dir() {
  local target="$1"
  ensure_writable_dir "$target"
  if storage_bool_true "$CLEAR_LOCAL_OUTPUT_DIR"; then
    echo "[CLEAN] Clearing previous contents under ${target}"
    find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

prepare_local_output_dir "$OUTPUT_DIR"

# Connectivity sanity check so we fail early if the NAS is unreachable before any
# heavy compute starts.
if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  NAS_TEST_DIR="${OFFLOAD_NAS_DIR}/__connectivity_check__"
  if mkdir -p "${NAS_TEST_DIR}" \
    && : > "${NAS_TEST_DIR}/.touch" \
    && rm -f "${NAS_TEST_DIR}/.touch"; then
    echo "[CHECK] NAS reachable at ${OFFLOAD_NAS_DIR}"
  else
    echo "[CHECK] ERROR: cannot write to NAS ${OFFLOAD_NAS_DIR}" >&2
    exit 1
  fi
fi

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  if ! storage_test_remote_connection "$REMOTE_TARGET_DIR"; then
    if [ "$REMOTE_ONLY_STORAGE" = true ]; then
      echo "[CHECK] ERROR: remote destination ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR} is unreachable and no alternate storage is configured." >&2
      exit 1
    else
      echo "[CHECK] WARN: remote destination ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR} is unreachable; continuing with other storage backends." >&2
    fi
  else
    echo "[CHECK] Remote destination reachable at ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR}"
  fi
fi

echo "[CONFIG] ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE}"
echo "[CONFIG] ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE}"
echo "[CONFIG] ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE}"

if [ "$RESERVE_VRAM_GB" -gt 0 ]; then
  reserve_vram "$RESERVE_VRAM_GB"
fi

# Assignment planning is deterministic: in resume mode we skip generation and
# reuse the previous manifest so scene/actor pairings stay stable.
if $RESUME_MODE; then
  echo "[RESUME] Skipping assignment generation and reusing ${ASSIGNMENTS_OUT}"
else
  generate_assignment_manifest
fi

# Rendering CLI snippets are composed here so storage flags can extend/override
# behavior (NAS uploads, BEV toggles, etc.) without duplicating the Python call.
if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  render_extra_snippets+=("--offload-nas-dir ${OFFLOAD_NAS_DIR} --offload-min-free-gb ${OFFLOAD_MIN_FREE_GB}")
fi

parallel_cmd=(
  conda run --no-capture-output -n "$CONDA_ENV" python parallel_render_paths.py
  --assignment-manifest "${ASSIGNMENTS_OUT}"
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --workers "${WORKERS}"
  --minimal-frames "${MINIMAL_FRAMES}"
  --output-dir "${OUTPUT_DIR}"
  --progress-json "${PROGRESS_JSON}"
  --per-job-metrics-dir "${PER_JOB_METRICS_DIR}"
)
# Thread the resume log into the renderer so it can skip completed scene/actor
# pairs. Remaining CLI snippets (overwrite/offload/etc.) are appended below.
if [ -n "$RESUME_LOG_ABS" ]; then
  parallel_cmd+=(--skip-completed-log "$RESUME_LOG_ABS")
fi
for snippet in "${render_extra_snippets[@]}"; do
  parallel_cmd+=(--render-extra-args "$snippet")
done

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  remote_abort_flag="false"
  if [ "$REMOTE_ONLY_STORAGE" = true ]; then
    remote_abort_flag="true"
  fi
  start_remote_sync_worker "$OUTPUT_DIR" "$REMOTE_TARGET_DIR" "$REMOTE_SYNC_INTERVAL_SECS" "$remote_abort_flag" "$$"
fi

render_status=0
set +e
if [ "$REMOTE_STORAGE_GUARD_ENABLED" = true ]; then
  "$SETSID_BIN" "${parallel_cmd[@]}" &
else
  "${parallel_cmd[@]}" &
fi
PARALLEL_PID=$!
if [ "$REMOTE_STORAGE_GUARD_ENABLED" = true ]; then
  PARALLEL_GROUP_ID="$PARALLEL_PID"
  if ! start_storage_guard "$OUTPUT_DIR" "$PARALLEL_GROUP_ID" "$REMOTE_STORAGE_LIMIT_GB" "$REMOTE_STORAGE_RESUME_GB" "$REMOTE_STORAGE_GUARD_INTERVAL_SECS"; then
    echo "[STORAGE] WARN: Failed to start storage guard; continuing without throttling." >&2
    REMOTE_STORAGE_GUARD_ENABLED=false
    stop_storage_guard
  fi
fi
wait "$PARALLEL_PID"
render_status=$?
PARALLEL_PID=""
PARALLEL_GROUP_ID=""
set -e
if [ "$REMOTE_STORAGE_UNAVAILABLE" = true ]; then
  render_status=99
  echo "[STORAGE] Remote destination unavailable; generation paused. Resume once storage is back online." >&2
elif [ $render_status -ne 0 ]; then
  echo "[WARN] parallel_render_paths.py exited with status ${render_status}, continuing per request."
fi

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  if [ "$REMOTE_STORAGE_UNAVAILABLE" = false ]; then
    signal_remote_sync_completion
  fi
  wait_remote_sync_worker
fi

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" && [ "$REMOTE_STORAGE_UNAVAILABLE" = false ]; then
  # When purely offloading to NAS/remote, purge local outputs to conserve disk.
  if [ -d "$OUTPUT_DIR" ]; then
    echo "[STORAGE] Removing local outputs at ${OUTPUT_DIR}"
    rm -rf "$OUTPUT_DIR"
  fi
fi

exit $render_status
