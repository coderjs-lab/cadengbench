#!/usr/bin/env bash
#SBATCH --job-name=cbp-benchmark
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/cbp-benchmark-%j.out
#SBATCH --error=logs/cbp-benchmark-%j.err

set -euo pipefail

: "${MODEL_SPEC:?Set MODEL_SPEC to a frozen model configuration JSON}"
: "${RUN_ID:?Set RUN_ID to a stable, filesystem-safe benchmark run ID}"

ROOT="${CADENGBENCH_ROOT:?set CADENGBENCH_ROOT to the extracted supplement root}"
EVALSET_ROOT="${CADENGBENCH_P_EVALSET_ROOT:-${ROOT}/benchmarks/cadengbench-p/evalsets/l0-l2-v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/cadengbench-p/${RUN_ID}}"
INPUT_PROFILE="${INPUT_PROFILE:-text_iso}"
WORKERS="${WORKERS:-8}"
RUNNER="${ROOT}/cadengbench_p_runner/.venv/bin/cadengbench-p-run"
EVALUATOR="${ROOT}/cadengbench_p_evaluator/.venv/bin/cadengbench-p-eval"
ENV_FILE="${CADENGBENCH_ENV_FILE:-${ROOT}/cadengbench_p_runner/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

mkdir -p "${ROOT}/cadengbench_p_runner/logs"
cd "${ROOT}/cadengbench_p_runner"
exec "${RUNNER}" benchmark \
  --evalset "${EVALSET_ROOT}" \
  --model-spec "${MODEL_SPEC}" \
  --output "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --evaluator-cli "${EVALUATOR}" \
  --repo-root "${ROOT}" \
  --input-profile "${INPUT_PROFILE}" \
  --workers "${WORKERS}"
