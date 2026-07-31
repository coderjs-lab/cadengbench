#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_command() {
  local candidate="$1"
  if [[ -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  if command -v "${candidate}" >/dev/null 2>&1; then
    command -v "${candidate}"
    return 0
  fi
  return 1
}

if [[ -n "${CADENGBENCH_CCX:-}" ]] && resolve_command "${CADENGBENCH_CCX}"; then
  exit 0
fi

for candidate in \
  "${ROOT}/.solver/bin/ccx" \
  "${ROOT}/.venv-a/bin/ccx" \
  "${ROOT}/.solver/Library/bin/ccx.exe" \
  ccx; do
  if resolve_command "${candidate}"; then
    exit 0
  fi
done

printf '%s\n' \
  "CalculiX executable not found. Run scripts/install_system_dependencies.sh." >&2
exit 1
