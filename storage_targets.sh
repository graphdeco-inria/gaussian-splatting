#!/usr/bin/env bash
# Helper utilities shared by data-generation runners for handling storage toggles.
# shellcheck shell=bash

_storage_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_storage_config_loaded="${_storage_config_loaded:-false}"

if ! $_storage_config_loaded; then
  STORAGE_TARGETS_CONFIG_FILE=${STORAGE_TARGETS_CONFIG_FILE:-${_storage_dir}/storage_remote_config.sh}
  if [ -f "$STORAGE_TARGETS_CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STORAGE_TARGETS_CONFIG_FILE"
  fi
  _storage_config_loaded=true
fi

# Return 0 when the provided value looks like "true".
storage_bool_true() {
  local val="${1:-}"
  if [ -z "$val" ]; then
    return 1
  fi
  case "${val,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_storage_join_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    if [ -z "$out" ]; then
      out=$(printf '%q' "$arg")
    else
      out+=" $(printf '%q' "$arg")"
    fi
  done
  printf '%s' "$out"
}

storage_make_ssh_cmd() {
  local -n _dest=$1
  local ssh_target="${REMOTE_SSH_TARGET:-}"
  if [ -z "$ssh_target" ]; then
    echo "[STORAGE] ERROR: REMOTE_SSH_TARGET is not set." >&2
    return 1
  fi
  local ssh_port="${REMOTE_SSH_PORT:-22}"
  local ssh_opts=(-p "$ssh_port")
  if [ -n "${REMOTE_SSH_KEY:-}" ]; then
    ssh_opts+=(-i "$REMOTE_SSH_KEY")
  fi
  if storage_bool_true "${REMOTE_SSH_STRICT_HOST_KEY_CHECKING:-true}"; then
    ssh_opts+=(-o StrictHostKeyChecking=accept-new)
  else
    ssh_opts+=(-o StrictHostKeyChecking=no)
  fi
  _dest=(ssh "${ssh_opts[@]}" "$ssh_target")
}

# Sync a directory to a remote Linux PC via rsync over SSH.
# Requires REMOTE_SSH_TARGET and REMOTE_OUTPUT_DIR to be set by the caller.
storage_sync_remote() {
  local source_dir="$1"
  local remote_dir="$2"

  if [ ! -d "$source_dir" ]; then
    echo "[STORAGE] WARN: local directory missing, skipping remote sync: ${source_dir}" >&2
    return 1
  fi

  if ! command -v rsync >/dev/null 2>&1; then
    echo "[STORAGE] ERROR: rsync is required for remote sync but was not found in PATH." >&2
    return 1
  fi
  if ! command -v ssh >/dev/null 2>&1; then
    echo "[STORAGE] ERROR: ssh is required for remote sync but was not found in PATH." >&2
    return 1
  fi

  local remote_host="${REMOTE_SSH_TARGET:-}"
  if [ -z "$remote_host" ]; then
    echo "[STORAGE] ERROR: REMOTE_SSH_TARGET is not set." >&2
    return 1
  fi
  local ssh_cmd=()
  storage_make_ssh_cmd ssh_cmd || return 1
  local ssh_base_cmd=("${ssh_cmd[@]:0:${#ssh_cmd[@]}-1}")
  local ssh_base_str
  ssh_base_str=$(_storage_join_cmd "${ssh_base_cmd[@]}")
  local remote_path_cmd="mkdir -p $(printf '%q' "$remote_dir") && rsync"

  echo "[STORAGE] Syncing ${source_dir} -> ${remote_host}:${remote_dir} via rsync ..."
  local info_flags="${REMOTE_RSYNC_INFO:-stats1}"
  if storage_bool_true "${REMOTE_RSYNC_PROGRESS:-false}"; then
    if [ -n "$info_flags" ]; then
      info_flags+=",progress2"
    else
      info_flags="progress2"
    fi
  fi
  local remove_sources="${REMOTE_SYNC_REMOVE_SOURCE_FILES:-false}"
  local -a rsync_cmd=(-az --partial --inplace -e "$ssh_base_str")
  if [ -n "$info_flags" ]; then
    rsync_cmd+=(--info="${info_flags}")
  fi
  if [ -n "${REMOTE_RSYNC_EXTRA_OPTS:-}" ]; then
    local -a extra_opts=(${REMOTE_RSYNC_EXTRA_OPTS})
    rsync_cmd+=("${extra_opts[@]}")
  fi
  if storage_bool_true "$remove_sources"; then
    rsync_cmd+=(--remove-source-files)
  fi
  rsync_cmd+=(
    --rsync-path "$remote_path_cmd"
    "${source_dir}/"
    "${remote_host}:${remote_dir}/"
  )
  if rsync "${rsync_cmd[@]}"; then
    if storage_bool_true "$remove_sources"; then
      # Clean up any empty directories that remain after rsync removes the files.
      find "$source_dir" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    fi
    return 0
  fi
  return 1
}

storage_test_remote_connection() {
  local remote_dir="$1"
  if ! command -v ssh >/dev/null 2>&1; then
    echo "[STORAGE] ERROR: ssh is required to test remote connectivity but was not found in PATH." >&2
    return 1
  fi
  local remote_host="${REMOTE_SSH_TARGET:-}"
  if [ -z "$remote_host" ]; then
    echo "[STORAGE] ERROR: REMOTE_SSH_TARGET is not set." >&2
    return 1
  fi
  local ssh_cmd=()
  storage_make_ssh_cmd ssh_cmd || return 1
  echo "[STORAGE] Probing remote ${remote_host}:${remote_dir} ..."
  "${ssh_cmd[@]}" "mkdir -p $(printf '%q' "$remote_dir") && echo '[STORAGE] Remote path ready: ${remote_dir}'"
}

storage_measure_transfer_speed() {
  local direction="$1"
  local remote_dir="$2"
  local size_mb="${3:-100}"
  local block_size=$((1024 * 1024))
  local tmp_file
  tmp_file=$(mktemp "${TMPDIR:-/tmp}/storage_speed.XXXXXX")
  dd if=/dev/urandom of="$tmp_file" bs=$block_size count="$size_mb" status=none

  local remote_host="${REMOTE_SSH_TARGET:-}"
  if [ -z "$remote_host" ]; then
    echo "[STORAGE] ERROR: REMOTE_SSH_TARGET is not set." >&2
    rm -f "$tmp_file"
    return 1
  fi

  local ssh_cmd=()
  storage_make_ssh_cmd ssh_cmd || { rm -f "$tmp_file"; return 1; }
  local ssh_base_cmd=("${ssh_cmd[@]:0:${#ssh_cmd[@]}-1}")
  local ssh_base_str
  ssh_base_str=$(_storage_join_cmd "${ssh_base_cmd[@]}")
  local remote_file="${remote_dir%/}/storage_speed_test.bin"

  local start end duration speed
  if [[ "$direction" == "upload" ]]; then
    echo "[STORAGE] Measuring upload speed to ${remote_host}:${remote_file}"
    "${ssh_cmd[@]}" "mkdir -p $(printf '%q' "${remote_dir}")"
    start=$(date +%s.%N)
    rsync -az --info=stats2 -e "$ssh_base_str" "$tmp_file" "${remote_host}:${remote_file}"
    end=$(date +%s.%N)
  else
    echo "[STORAGE] Measuring download speed from ${remote_host}:${remote_file}"
    "${ssh_cmd[@]}" "mkdir -p $(printf '%q' "${remote_dir}") && dd if=/dev/urandom of=$(printf '%q' "${remote_file}") bs=$block_size count=$size_mb status=none"
    local local_copy="${tmp_file}.dl"
    start=$(date +%s.%N)
    rsync -az --info=stats2 -e "$ssh_base_str" "${remote_host}:${remote_file}" "$local_copy"
    end=$(date +%s.%N)
    rm -f "$local_copy"
  fi

  duration=$(python3 -c "import sys; start=float(sys.argv[1]); end=float(sys.argv[2]); print(max(1e-7, end-start))" "$start" "$end")
  speed=$(python3 -c "import sys; size=float(sys.argv[1]); duration=float(sys.argv[2]); print(size/duration)" "$size_mb" "$duration")
  echo "[STORAGE] ${direction} speed: ${size_mb}MB in ${duration}s -> ${speed} MB/s"
  "${ssh_cmd[@]}" "rm -f $(printf '%q' "${remote_file}")" || true
  rm -f "$tmp_file"
}
