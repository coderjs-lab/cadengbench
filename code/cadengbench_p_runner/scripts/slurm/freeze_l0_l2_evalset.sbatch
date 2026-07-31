#!/usr/bin/env bash
#SBATCH --job-name=cbp-freeze-l02
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/cbp-freeze-l02-%j.out
#SBATCH --error=logs/cbp-freeze-l02-%j.err

set -euo pipefail

ROOT="${CADENGBENCH_ROOT:?set CADENGBENCH_ROOT to the extracted supplement root}"
DATASET_ROOT="${CADENGBENCH_P_DATASET_ROOT:-${ROOT}/benchmarks/cadengbench-p/dataset/cadengbench-p-300-v1-two-source}"
EVALSET_ROOT="${CADENGBENCH_P_EVALSET_ROOT:-${ROOT}/benchmarks/cadengbench-p/evalsets/l0-l2-v2}"
EVALUATOR="${ROOT}/cadengbench_p_evaluator/.venv/bin/cadengbench-p-eval"
PROFILE="${ROOT}/cadengbench_p_evaluator/configs/profiles/paper.json"

mkdir -p "${ROOT}/cadengbench_p_evaluator/logs"
cd "${ROOT}/cadengbench_p_evaluator"
exec "${EVALUATOR}" freeze-l0-l2-release \
  --dataset-root "${DATASET_ROOT}" \
  --profile "${PROFILE}" \
  --output "${EVALSET_ROOT}"
