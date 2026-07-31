#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ITEMS_ROOT="${ROOT}/datasets/cadengbench_p_sample10_v1/items"
OUTPUT_ROOT="${1:-${ROOT}/outputs/controls/p_l3_reference}"
CCX="$("${ROOT}/scripts/resolve_ccx.sh")"
export CADENGBENCH_L3_RUNTIME_LOCK="${ROOT}/configs/l3/cadengbench_p_l3_runtime_lock_v1.json"

for ITEM_ROOT in "${ITEMS_ROOT}"/p_*; do
  ITEM="$(basename "${ITEM_ROOT}")"
  if [[ "${ITEM}" == p_benchcad_* ]]; then
    REFERENCE="${ITEM_ROOT}/source/generated_from_cadquery.step"
  else
    REFERENCE="${ITEM_ROOT}/source/fusion360_occ_reconstruction.step"
  fi
  mkdir -p "${OUTPUT_ROOT}"
  "${ROOT}/.venv-p/bin/cadengbench-p-l3-response" evaluate-pair \
    --contract "${ITEM_ROOT}/l3/response_contract.json" \
    --card "${ITEM_ROOT}/l3/physics_card.solver_qualified.json" \
    --reference-step "${REFERENCE}" \
    --candidate-step "${REFERENCE}" \
    --variant-id nominal-reference-control \
    --ccx "${CCX}" \
    --output "${OUTPUT_ROOT}/${ITEM}" \
    > "${OUTPUT_ROOT}/${ITEM}.stdout.json"
done

printf 'L3 reference controls completed under %s.\n' "${OUTPUT_ROOT}"
