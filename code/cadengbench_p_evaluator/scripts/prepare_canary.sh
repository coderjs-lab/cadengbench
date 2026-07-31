#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REPOSITORY_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

repo_root="$(cd "$1" && pwd)"
output_root="$2"
package_root="$(cd "$(dirname "$0")/.." && pwd)"
items_root="$repo_root/data/interim/cadengbench_p_r25_combined_hpc_canary_30/dataset/items"

cd "$package_root"
uv run cadengbench-p-eval prepare-evalset \
  --items-root "$items_root" \
  --profile "$package_root/configs/profiles/dev.json" \
  --output "$output_root"

