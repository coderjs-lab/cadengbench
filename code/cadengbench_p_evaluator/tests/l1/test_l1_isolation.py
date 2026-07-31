from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cadquery as cq
import pytest

from cadengbench_p_eval.case import LoadedCase
from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.io import sha256_file
from cadengbench_p_eval.layers import l1_isolated
from cadengbench_p_eval.layers.l1_isolated import (
    L1_WORKER_REPORT_SCHEMA,
    evaluate_l1_isolated,
)

pytestmark = pytest.mark.integration


def _export_box(path: Path) -> Path:
    cq.exporters.export(cq.Workplane("XY").box(10, 8, 2), str(path))
    return path


def _case_data(root: Path, target: Path) -> dict:
    return {
        "source_item_root": str(root),
        "artifacts": {
            "target_step": {
                "item_relative_path": target.name,
                "sha256": sha256_file(target),
            }
        },
        "task_contract": {
            "task_objective": "engineering_specification",
            "fidelity_profile": "requirements_only",
        },
        "requirements": [
            {
                "requirement_id": "R-001",
                "criticality": "required",
                "value_and_tolerance": {"body_count": 1},
                "check_ids": ["hidden.check.solid_count.v1"],
            }
        ],
        "dfm": {"families": []},
        "evaluator_profile": {
            "geometry": {
                "linear_tessellation_tolerance_mm": 0.2,
                "surface_sample_count": 64,
                "surface_tolerance_mm": 0.2,
                "voxel_resolution": 6,
                "voxel_margin_fraction": 0.05,
                "seed": 7,
                "hard_gates": {
                    "voxel_iou_min": None,
                    "surface_f1_min": None,
                },
            },
            "dfm": {
                "wall_sample_count": 32,
                "wall_neighbor_count": 8,
                "opposed_normal_cosine_min": 0.9,
                "hole_axis_tolerance_degrees": 1.0,
            },
        },
    }


@pytest.mark.parametrize("use_loaded_case", [False, True])
def test_isolated_l1_passes_and_atomically_reports(
    tmp_path: Path, use_loaded_case: bool
) -> None:
    target = _export_box(tmp_path / "target.step")
    data = _case_data(tmp_path, target)
    case = (
        LoadedCase(path=tmp_path / "case.json", data=data) if use_loaded_case else data
    )
    work_dir = tmp_path / f"worker-{use_loaded_case}"

    result = evaluate_l1_isolated(
        case,
        target,
        target_step=target,
        timeout_seconds=60,
        work_dir=work_dir,
    )

    assert result.status is Status.PASSED
    assert result.metrics["reference_geometry_diagnostics"]["status"] == "skipped"
    assert (
        result.metrics["reference_geometry_diagnostics"]["score_bearing"] is False
    )
    report = json.loads((work_dir / "worker.report.json").read_text())
    assert report["schema_version"] == L1_WORKER_REPORT_SCHEMA
    assert report["outcome"] == "completed"
    assert not list(work_dir.glob("*.tmp"))


def test_isolated_l1_forced_timeout_is_evaluator_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def hanging_command(_job_path: Path, _report_path: Path) -> list[str]:
        return [sys.executable, "-I", "-c", "import time; time.sleep(60)"]

    monkeypatch.setattr(l1_isolated, "_worker_command", hanging_command)
    started = time.monotonic()
    result = evaluate_l1_isolated(
        {},
        tmp_path / "unused.step",
        timeout_seconds=0.1,
        work_dir=tmp_path / "timeout-worker",
    )

    assert time.monotonic() - started < 5.0
    assert result.status is Status.EVALUATOR_ERROR
    assert result.checks[0].observed["error_code"] == "l1_worker_timeout"
    assert result.checks[0].observed["details"]["process_group_reaped"] is True


def test_isolated_l1_parameter_variant_uses_candidate_local_contract(
    tmp_path: Path,
) -> None:
    target = _export_box(tmp_path / "variant-target.step")
    result = evaluate_l1_isolated(
        _case_data(tmp_path, target),
        target,
        is_parameter_variant=True,
        timeout_seconds=30,
    )

    assert result.status is Status.PASSED
    diagnostics = result.metrics["reference_geometry_diagnostics"]
    assert diagnostics["status"] == "not_applicable"
    assert diagnostics["reason"] == "parameter_variant"
    assert all("reference" not in check.check_id for check in result.checks)


def test_isolated_l1_crash_is_evaluator_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crashing_command(_job_path: Path, _report_path: Path) -> list[str]:
        return [sys.executable, "-I", "-c", "raise SystemExit(7)"]

    monkeypatch.setattr(l1_isolated, "_worker_command", crashing_command)
    result = evaluate_l1_isolated(
        {},
        tmp_path / "unused.step",
        timeout_seconds=5,
        work_dir=tmp_path / "crash-worker",
    )

    assert result.status is Status.EVALUATOR_ERROR
    assert result.checks[0].observed["error_code"] == "l1_worker_crash"
    assert result.checks[0].observed["worker_returncode"] == 7


def test_isolated_l1_malformed_report_is_evaluator_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def malformed_command(_job_path: Path, report_path: Path) -> list[str]:
        script = (
            "from pathlib import Path; Path(__import__('sys').argv[1]).write_text('{')"
        )
        return [sys.executable, "-I", "-c", script, str(report_path)]

    monkeypatch.setattr(l1_isolated, "_worker_command", malformed_command)
    result = evaluate_l1_isolated(
        {},
        tmp_path / "unused.step",
        timeout_seconds=5,
        work_dir=tmp_path / "malformed-worker",
    )

    assert result.status is Status.EVALUATOR_ERROR
    assert result.checks[0].observed["error_code"] == "malformed_l1_worker_report"
