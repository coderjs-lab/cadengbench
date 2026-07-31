#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/outputs/controls"
mkdir -p "${OUT}"

"${ROOT}/scripts/test_calculix.sh"
"${ROOT}/scripts/run_l3_reference_controls.sh" "${OUT}/p_l3_reference"
"${ROOT}/.venv-p/bin/cadengbench-p-eval" doctor
for PROTOCOL in zero_to_cad functional_edit; do
  "${ROOT}/.venv-p/bin/cadengbench-p-eval" prepare-evalset \
    --items-root "${ROOT}/datasets/cadengbench_p_sample10_v1/items" \
    --profile "${ROOT}/code/cadengbench_p_evaluator/configs/profiles/paper.json" \
    --protocol "${PROTOCOL}" \
    --output "${OUT}/p_prepared/${PROTOCOL}"
done
"${ROOT}/.venv-p/bin/cadengbench-p-eval" evaluate \
  --evalset "${OUT}/p_prepared/zero_to_cad" \
  --submissions "${ROOT}/datasets/cadengbench_p_sample10_v1/submissions/positive" \
  --output "${OUT}/p_positive" \
  --model-id evaluator-positive-control.v2 \
  --workers 4
"${ROOT}/.venv-p/bin/cadengbench-p-eval" aggregate \
  --run "${OUT}/p_positive"
"${ROOT}/.venv-p/bin/cadengbench-p-eval" evaluate \
  --evalset "${OUT}/p_prepared/zero_to_cad" \
  --submissions "${ROOT}/datasets/cadengbench_p_sample10_v1/submissions/mutant_parameter_scale" \
  --output "${OUT}/p_mutant" \
  --model-id evaluator-semantic-mutant.v2 \
  --workers 4
"${ROOT}/.venv-p/bin/cadengbench-p-eval" aggregate \
  --run "${OUT}/p_mutant"
"${ROOT}/.venv-p/bin/cadengbench-p-eval" check-dev-slice \
  --manifest "${ROOT}/datasets/cadengbench_p_sample10_v1/development_slice_manifest.json" \
  --positive-run "${OUT}/p_positive" \
  --mutant-run "${OUT}/p_mutant" \
  --output "${OUT}/p_calibration_report.json"
"${ROOT}/.venv-a/bin/python" \
  "${ROOT}/code/cadengbench_a/scripts/verify_cadengbench_a_release.py" \
  "${ROOT}/datasets/cadengbench_a_sample7_v2"
"${ROOT}/.venv-a/bin/python" \
  "${ROOT}/code/cadengbench_a/scripts/run_cadengbench_a_a2_fixture_smoke.py" \
  --output-dir "${OUT}/a2_fixture" \
  > "${OUT}/a2_fixture_smoke.json"

if [[ ! -f "${OUT}/a_dry_run/run_manifest.json" ]]; then
  "${ROOT}/.venv-a/bin/python" \
    "${ROOT}/code/cadengbench_a/scripts/run_cadengbench_a_benchmark.py" \
    --dataset-root "${ROOT}/datasets/cadengbench_a_sample7_v2" \
    --labels-root "${ROOT}/datasets/cadengbench_a_sample7_v2/labels" \
    --provider openai \
    --model dry-run-model \
    --dry-run \
    --output-dir "${OUT}/a_dry_run"
fi

printf '%s\n' "Provider-free controls completed under ${OUT}."
