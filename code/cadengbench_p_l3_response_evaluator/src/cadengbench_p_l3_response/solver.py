"""Isolated adapter around the frozen CADEngBench-P L3 solver."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from cadengbench_p_l3_response.io import read_json, sha256_file, write_json


class SolverAdapterError(RuntimeError):
    """The isolated frozen-L3 worker did not produce a valid result."""


def prepare_output(path: Path, *, force: bool = False) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    if path.is_symlink() or not path.is_dir():
        raise SolverAdapterError(f"output is not a normal directory: {path}")
    if any(path.iterdir()):
        if not force:
            raise SolverAdapterError(f"output is nonempty: {path}")
        shutil.rmtree(path)
        path.mkdir(parents=True)


def force_controlled_card(
    base_card: dict[str, Any],
    *,
    target_force_n: list[float],
    loaded_area_mm2: float,
    maximum_equilibrium_relative_error: float,
    role: str,
    variant_id: str,
    contract_sha256: str,
) -> dict[str, Any]:
    if loaded_area_mm2 <= 0:
        raise SolverAdapterError("loaded area must be positive")
    if maximum_equilibrium_relative_error <= 0:
        raise SolverAdapterError("equilibrium tolerance must be positive")
    card = deepcopy(base_card)
    card.pop("qualification", None)
    card["status"] = "geometry_qualified"
    safe_variant = "".join(
        value if value.isalnum() or value in "_.:-" else "_" for value in variant_id
    )[:48]
    card["card_id"] = (
        f"{card['item_id']}:response:{role}:{safe_variant}:{contract_sha256[:12]}"
    )
    card["load"]["traction_vector_mpa"] = [
        float(value) / float(loaded_area_mm2) for value in target_force_n
    ]
    card["analysis"]["max_equilibrium_relative_error"] = float(
        maximum_equilibrium_relative_error
    )
    return card


def run_solver_job(
    *,
    card_path: Path,
    step_path: Path,
    output_dir: Path,
    ccx_path: Path,
    timeout_seconds: int,
    render: bool,
) -> dict[str, Any]:
    if not card_path.is_file() or not step_path.is_file() or not ccx_path.is_file():
        raise SolverAdapterError("card, STEP, or CalculiX input is missing")
    prepare_output(output_dir)
    command = [
        sys.executable,
        "-m",
        "cadengbench_p_l3_response.solver_worker",
        "--card",
        str(card_path.resolve()),
        "--step",
        str(step_path.resolve()),
        "--output",
        str(output_dir.resolve()),
        "--ccx",
        str(ccx_path.resolve()),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if render:
        command.append("--render")
    started = time.monotonic()
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds + 600,
            env=environment,
            check=False,
        )
        report = {
            "schema_version": "cadengbench-p.l3-response-worker-report.v1",
            "command": command,
            "returncode": completed.returncode,
            "runtime_seconds": time.monotonic() - started,
            "card_sha256": sha256_file(card_path),
            "step_sha256": sha256_file(step_path),
        }
        (output_dir / "worker.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output_dir / "worker.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
    except subprocess.TimeoutExpired as exc:
        report = {
            "schema_version": "cadengbench-p.l3-response-worker-report.v1",
            "command": command,
            "returncode": None,
            "runtime_seconds": time.monotonic() - started,
            "timeout": True,
            "message": str(exc),
            "card_sha256": sha256_file(card_path),
            "step_sha256": sha256_file(step_path),
        }
    write_json(output_dir / "response_worker_report.json", report)
    result_path = output_dir / "result.json"
    if report.get("returncode") != 0 or not result_path.is_file():
        raise SolverAdapterError(
            "isolated L3 worker failed: " + json.dumps(report, sort_keys=True)
        )
    return read_json(result_path)
