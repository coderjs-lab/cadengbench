#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:?pass the sample output root}"
ITEMS_ROOT="${ROOT}/datasets/cadengbench_p_sample10_v1/items"
CCX="$("${ROOT}/scripts/resolve_ccx.sh")"
export CADENGBENCH_L3_RUNTIME_LOCK="${ROOT}/configs/l3/cadengbench_p_l3_runtime_lock_v1.json"

for ITEM_ROOT in "${ITEMS_ROOT}"/p_*; do
  ITEM="$(basename "${ITEM_ROOT}")"
  CARD="${ITEM_ROOT}/l3/physics_card.solver_qualified.json"
  CONTRACT="${ITEM_ROOT}/l3/response_contract.json"
  if [[ "${ITEM}" == p_benchcad_* ]]; then
    REFERENCE="${ITEM_ROOT}/source/generated_from_cadquery.step"
  else
    REFERENCE="${ITEM_ROOT}/source/fusion360_occ_reconstruction.step"
  fi
  CANDIDATE="${OUT_ROOT}/p/scoring/zero_to_cad/items/${ITEM}/l0/candidate.step"
  L3_OUT="${OUT_ROOT}/p/l3/${ITEM}/nominal"

  for REQUIRED in "${CARD}" "${CONTRACT}" "${REFERENCE}"; do
    if [[ ! -f "${REQUIRED}" ]]; then
      printf 'Missing inline L3 input: %s\n' "${REQUIRED}" >&2
      exit 2
    fi
  done

  if [[ ! -f "${CANDIDATE}" ]]; then
    mkdir -p "${L3_OUT}"
    printf '{"item_id":"%s","status":"not_reached","reason":"candidate_l0_step_missing"}\n' "${ITEM}" \
      > "${L3_OUT}/not_reached.json"
    continue
  fi

  "${ROOT}/.venv-p/bin/cadengbench-p-l3-response" evaluate-pair \
    --contract "${CONTRACT}" \
    --card "${CARD}" \
    --reference-step "${REFERENCE}" \
    --candidate-step "${CANDIDATE}" \
    --variant-id nominal \
    --ccx "${CCX}" \
    --output "${L3_OUT}"
done
