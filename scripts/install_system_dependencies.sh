#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALCULIX_VERSION="2.21"

if "${ROOT}/scripts/resolve_ccx.sh" >/dev/null 2>&1; then
  printf '%s\n' "CalculiX is already available."
  exit 0
fi

if command -v micromamba >/dev/null 2>&1; then
  micromamba create -y -p "${ROOT}/.solver" -c conda-forge \
    "calculix=${CALCULIX_VERSION}"
elif command -v conda >/dev/null 2>&1; then
  conda create -y -p "${ROOT}/.solver" -c conda-forge \
    "calculix=${CALCULIX_VERSION}"
elif [[ "$(uname -s)" == "Linux" ]] && command -v apt-get >/dev/null 2>&1; then
  if [[ "${EUID}" -eq 0 ]]; then
    APT=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    APT=(sudo apt-get)
  else
    printf '%s\n' \
      "Installing calculix-ccx requires root/sudo, conda, or micromamba." >&2
    exit 2
  fi
  "${APT[@]}" update
  "${APT[@]}" install -y calculix-ccx
else
  printf '%s\n' \
    "No supported CalculiX installer was found." \
    "Install conda/micromamba or install CalculiX 2.21 as ccx, then rerun setup." >&2
  exit 2
fi

CCX="$("${ROOT}/scripts/resolve_ccx.sh")"
printf '%s\n' "CalculiX ${CALCULIX_VERSION} is available at ${CCX}."
