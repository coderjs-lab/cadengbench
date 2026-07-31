#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"${PYTHON_BIN}" -m venv "${ROOT}/.venv-p"
"${ROOT}/.venv-p/bin/python" -m pip install --upgrade pip
"${ROOT}/.venv-p/bin/python" -m pip install \
  -e "${ROOT}/code/cadengbench_p_evaluator[l3]" \
  -e "${ROOT}/code/cadengbench_p_runner" \
  -e "${ROOT}/code/cadengbench_p_l3_response_evaluator" \
  -e "${ROOT}/code/cadengbench_p_l3_extended_evaluator"

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    ENV_MANAGER=(micromamba create -y -p "${ROOT}/.venv-a")
  elif command -v conda >/dev/null 2>&1; then
    ENV_MANAGER=(conda create -y -p "${ROOT}/.venv-a")
  else
    printf '%s\n' \
      "Apple Silicon requires micromamba or conda because PyPI has no PyBullet wheel." \
      "Install micromamba, then rerun this script." >&2
    exit 2
  fi
  "${ENV_MANAGER[@]}" -c conda-forge \
    python=3.11 cadquery=2.8.0 pybullet=3.25 matplotlib pillow jsonschema pytest pyyaml z3-solver
  "${ROOT}/.venv-a/bin/python" -m pip install \
    "SALib==1.5.2" \
    "openai==2.30.0" \
    "anthropic>=0.52,<1" \
    "google-genai>=1.14,<2"
else
  "${PYTHON_BIN}" -m venv "${ROOT}/.venv-a"
  "${ROOT}/.venv-a/bin/python" -m pip install --upgrade pip
  "${ROOT}/.venv-a/bin/python" -m pip install \
    -r "${ROOT}/code/cadengbench_a/requirements-benchmark.txt" \
    "jsonschema>=4.23,<5" \
    "pillow>=11,<13" \
    "pybullet==3.2.5"
fi

if [[ "${CADENGBENCH_SKIP_SYSTEM_DEPS:-0}" != "1" ]]; then
  "${ROOT}/scripts/install_system_dependencies.sh"
  "${ROOT}/scripts/test_calculix.sh"
fi

printf '%s\n' "Python environments and benchmark dependencies are ready."
