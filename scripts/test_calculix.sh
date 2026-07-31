#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CCX="$("${ROOT}/scripts/resolve_ccx.sh")"
WORK="${ROOT}/outputs/calculix_smoke"

set +e
VERSION_OUTPUT="$("${CCX}" -v 2>&1)"
VERSION_EXIT=$?
set -e
if [[ "${VERSION_EXIT}" -ne 0 && "${VERSION_EXIT}" -ne 201 ]]; then
  printf '%s\n' "CalculiX version probe failed with exit ${VERSION_EXIT}." >&2
  exit 1
fi
if [[ "${VERSION_OUTPUT}" != *"Version 2.21"* ]]; then
  printf '%s\n' \
    "CalculiX 2.21 is required by the frozen L3 runtime lock." \
    "Detected: ${VERSION_OUTPUT}" >&2
  exit 1
fi

mkdir -p "${WORK}"
cp "${ROOT}/configs/l3/calculix_smoke.inp" "${WORK}/calculix_smoke.inp"
: > "${WORK}/calculix_smoke.frd"
(
  cd "${WORK}"
  "${CCX}" -i calculix_smoke > calculix_smoke.stdout.log 2>&1
)

if [[ ! -s "${WORK}/calculix_smoke.frd" ]]; then
  printf '%s\n' "CalculiX smoke test did not produce an FRD result." >&2
  exit 1
fi

printf '%s\n' "CalculiX smoke test passed: ${WORK}/calculix_smoke.frd"
