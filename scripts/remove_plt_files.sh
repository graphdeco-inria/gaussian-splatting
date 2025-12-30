#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

targets=("$root_dir/data1" "$root_dir/data2")

for t in "${targets[@]}"; do
  if [[ -d "$t" ]]; then
    echo "Removing .plt files under: $t"
    find "$t" -type f -name "*.plt" -print -delete
  else
    echo "Skip missing dir: $t"
  fi
done
