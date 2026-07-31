#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  source "${ROOT}/.env"
  set +a
fi

PROVIDER="${CADENGBENCH_PROVIDER:-openai}"
MODEL="${CADENGBENCH_MODEL:-gpt-5.2-2025-12-11}"
RUN_ID="${CADENGBENCH_RUN_ID:-reviewer-sample}"
WORKERS="${CADENGBENCH_WORKERS:-2}"
MODEL_SPEC="${CADENGBENCH_P_MODEL_SPEC:-${ROOT}/configs/models/openai_gpt_5_2_sample.json}"
OUT="${ROOT}/outputs/${RUN_ID}"
P_DATA="${ROOT}/datasets/cadengbench_p_sample10_v1"
A_DATA="${ROOT}/datasets/cadengbench_a_sample7_v2"
P_EVAL="${ROOT}/.venv-p/bin/cadengbench-p-eval"
P_RUN="${ROOT}/.venv-p/bin/cadengbench-p-run"

mkdir -p "${OUT}/p/prepared" "${OUT}/p/generation" "${OUT}/p/scoring"

if [[ ! -f "${OUT}/p/prepared/zero_to_cad/manifest.json" ]]; then
  "${P_EVAL}" prepare-evalset \
    --items-root "${P_DATA}/items" \
    --profile "${ROOT}/code/cadengbench_p_evaluator/configs/profiles/paper.json" \
    --protocol zero_to_cad \
    --output "${OUT}/p/prepared/zero_to_cad"
fi

if [[ ! -f "${OUT}/p/prepared/functional_edit/manifest.json" ]]; then
  "${P_EVAL}" prepare-evalset \
    --items-root "${P_DATA}/items" \
    --profile "${ROOT}/code/cadengbench_p_evaluator/configs/profiles/paper.json" \
    --protocol functional_edit \
    --output "${OUT}/p/prepared/functional_edit"
fi

for PROTOCOL in zero_to_cad functional_edit; do
  GEN="${OUT}/p/generation/${PROTOCOL}"
  SCORE="${OUT}/p/scoring/${PROTOCOL}"
  if [[ ! -f "${GEN}/run_manifest.json" ]]; then
    "${P_RUN}" plan \
      --items-root "${P_DATA}/items" \
      --model-spec "${MODEL_SPEC}" \
      --output "${GEN}" \
      --run-id "${RUN_ID}.${PROTOCOL}" \
      --task-kind "${PROTOCOL}" \
      --input-profile text_iso
  fi
  "${P_RUN}" resume --run "${GEN}"
  "${P_RUN}" package --run "${GEN}"
  if [[ "${PROTOCOL}" == "functional_edit" ]]; then
    "${P_EVAL}" evaluate \
      --evalset "${OUT}/p/prepared/${PROTOCOL}" \
      --submissions "${GEN}/artifacts" \
      --repo-root "${ROOT}" \
      --output "${SCORE}" \
      --model-id "${MODEL}" \
      --workers "${WORKERS}" \
      --resume
  else
    "${P_EVAL}" evaluate \
      --evalset "${OUT}/p/prepared/${PROTOCOL}" \
      --submissions "${GEN}/artifacts" \
      --output "${SCORE}" \
      --model-id "${MODEL}" \
      --workers "${WORKERS}" \
      --resume
  fi
  "${P_EVAL}" aggregate --run "${SCORE}"
done

A_ARGS=(
  --dataset-root "${A_DATA}"
  --labels-root "${A_DATA}/labels"
  --provider "${PROVIDER}"
  --model "${MODEL}"
  --reasoning-effort none
  --max-output-tokens 16384
  --output-dir "${OUT}/a"
)
if [[ -f "${OUT}/a/run_manifest.json" ]]; then
  A_ARGS+=(--resume)
fi
"${ROOT}/.venv-a/bin/python" \
  "${ROOT}/code/cadengbench_a/scripts/run_cadengbench_a_benchmark.py" \
  "${A_ARGS[@]}"

if "${ROOT}/scripts/resolve_ccx.sh" >/dev/null 2>&1; then
  "${ROOT}/scripts/run_l3_sample.sh" "${OUT}"
else
  printf '%s\n' "CalculiX not found; run scripts/install_system_dependencies.sh." >&2
  exit 2
fi

printf '%s\n' "Sample benchmark completed under ${OUT}."
