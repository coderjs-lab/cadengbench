from pathlib import Path

import cadquery as cq

from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.layers.l1 import evaluate_l1

GEOMETRY_PROFILE = {
    "linear_tessellation_tolerance_mm": 0.2,
    "surface_sample_count": 128,
    "surface_tolerance_mm": 0.2,
    "voxel_resolution": 8,
    "voxel_margin_fraction": 0.05,
    "seed": 13,
}
DFM_PROFILE = {
    "wall_sample_count": 32,
    "wall_neighbor_count": 8,
    "opposed_normal_cosine_min": 0.9,
    "hole_axis_tolerance_degrees": 1.0,
}


def _export(tmp_path: Path, name: str, shape: cq.Workplane) -> Path:
    path = tmp_path / f"{name}.step"
    cq.exporters.export(shape, str(path))
    return path


def _case(
    objective: str,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict:
    task_contract = {
        "task_objective": objective,
        "fidelity_profile": (
            "exact_reference"
            if objective == "exact_reconstruction"
            else "requirements_only"
        ),
    }
    if thresholds is not None:
        task_contract["reference_similarity_thresholds"] = thresholds
    return {
        "task_contract": task_contract,
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
            "geometry": GEOMETRY_PROFILE,
            "dfm": DFM_PROFILE,
        },
    }


def _exact_thresholds() -> dict[str, float]:
    return {
        "chamfer_normalized_max": 1e-9,
        "surface_f1_min": 1.0,
        "voxel_iou_min": 1.0,
        "hd95_normalized_max": 1e-9,
        "voxel_dice_min": 1.0,
    }


def test_general_task_skips_non_scoring_reference_comparison(tmp_path: Path) -> None:
    target = _export(tmp_path, "target", cq.Workplane("XY").box(10, 8, 2))
    candidate = _export(tmp_path, "candidate", cq.Workplane("XY").box(9, 8, 2))

    result = evaluate_l1(
        _case("engineering_specification"),
        candidate,
        target_step=target,
    )

    assert result.status is Status.PASSED
    diagnostics = result.metrics["reference_geometry_diagnostics"]
    assert diagnostics["status"] == "skipped"
    assert diagnostics["mode"] == "diagnostic_only"
    assert diagnostics["score_bearing"] is False
    assert diagnostics["reason"] == "non_scoring_reference_comparison_disabled"
    check = next(
        check
        for check in result.checks
        if check.check_id == "L1.reference_geometry.diagnostics"
    )
    assert check.status is Status.NOT_SCOREABLE
    assert check.required is False
    assert all(
        check.check_id != "L1.reference_geometry.exact_reconstruction"
        for check in result.checks
    )


def test_general_task_skips_optional_diagnostics_after_required_failure(
    tmp_path: Path, monkeypatch
) -> None:
    target = _export(tmp_path, "target", cq.Workplane("XY").box(10, 8, 2))
    candidate = _export(tmp_path, "candidate", cq.Workplane("XY").box(9, 8, 2))
    case = _case("engineering_specification")
    case["requirements"].append(
        {
            "requirement_id": "R-BBOX",
            "category": "global_envelope",
            "criticality": "required",
            "value_and_tolerance": {
                "axis_assignment": "free_permutation",
                "x": {"value": 20.0, "tolerance": 0.01},
                "y": {"value": 20.0, "tolerance": 0.01},
                "z": {"value": 20.0, "tolerance": 0.01},
            },
            "check_ids": ["hidden.check.bbox_dimensions.v1"],
        }
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("non-scoring comparison should have been skipped")

    monkeypatch.setattr("cadengbench_p_eval.layers.l1.compare_geometry", fail_if_called)
    result = evaluate_l1(case, candidate, target_step=target)

    assert result.status is Status.FAILED
    diagnostics = result.metrics["reference_geometry_diagnostics"]
    assert diagnostics["status"] == "skipped"
    assert diagnostics["reason"] == "required_engineering_requirement_failed"


def test_exact_reconstruction_scores_matching_geometry(tmp_path: Path) -> None:
    target = _export(tmp_path, "target", cq.Workplane("XY").box(10, 8, 2))
    candidate = _export(tmp_path, "candidate", cq.Workplane("XY").box(10, 8, 2))

    result = evaluate_l1(
        _case("exact_reconstruction", thresholds=_exact_thresholds()),
        candidate,
        target_step=target,
    )

    assert result.status is Status.PASSED
    diagnostics = result.metrics["reference_geometry_diagnostics"]
    assert diagnostics["score_bearing"] is True
    gate = next(
        check
        for check in result.checks
        if check.check_id == "L1.reference_geometry.exact_reconstruction"
    )
    assert gate.status is Status.PASSED
    assert gate.required is True


def test_exact_reconstruction_rejects_geometry_outside_thresholds(
    tmp_path: Path,
) -> None:
    target = _export(tmp_path, "target", cq.Workplane("XY").box(10, 8, 2))
    candidate = _export(tmp_path, "candidate", cq.Workplane("XY").box(9, 8, 2))

    result = evaluate_l1(
        _case("exact_reconstruction", thresholds=_exact_thresholds()),
        candidate,
        target_step=target,
    )

    assert result.status is Status.FAILED
    gate = next(
        check
        for check in result.checks
        if check.check_id == "L1.reference_geometry.exact_reconstruction"
    )
    assert gate.failure_code == "exact_reconstruction_similarity_mismatch"
    assert gate.observed["failed_thresholds"]


def test_exact_reconstruction_requires_frozen_thresholds(tmp_path: Path) -> None:
    target = _export(tmp_path, "target", cq.Workplane("XY").box(10, 8, 2))

    result = evaluate_l1(
        _case("exact_reconstruction"),
        target,
        target_step=target,
    )

    assert result.status is Status.EVALUATOR_ERROR
    errors = [
        check for check in result.checks if check.status is Status.EVALUATOR_ERROR
    ]
    assert len(errors) == 1
    assert "requires reference_similarity_thresholds" in errors[0].summary
