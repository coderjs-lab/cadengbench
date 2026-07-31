"""Strict nominal engineering correctness and DFM evaluation (L1)."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cadengbench_p_eval.case import LoadedCase
from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status
from cadengbench_p_eval.dfm.evaluator import evaluate_dfm
from cadengbench_p_eval.geometry.features import extract_cylindrical_features
from cadengbench_p_eval.geometry.fingerprint import geometry_fingerprint
from cadengbench_p_eval.geometry.measurements import parameter_measurements
from cadengbench_p_eval.geometry.metrics import compare_geometry
from cadengbench_p_eval.geometry.requirements import evaluate_requirements
from cadengbench_p_eval.geometry.step import StepGeometryError, load_step

_TASK_OBJECTIVES = {
    "engineering_specification",
    "exact_reconstruction",
    "functional_edit",
}
_EXACT_RECONSTRUCTION_GATES = {
    "chamfer_normalized_max": (
        "chamfer_normalized_by_target_diagonal",
        "max",
    ),
    "surface_f1_min": ("surface_f1", "min"),
    "voxel_iou_min": ("voxel_iou", "min"),
    "hd95_normalized_max": (
        "hd95_normalized_by_target_diagonal",
        "max",
    ),
    "voxel_dice_min": ("voxel_dice", "min"),
    "volume_relative_error_max": ("volume_relative_error", "max"),
    "surface_area_relative_error_max": (
        "surface_area_relative_error",
        "max",
    ),
    "centroid_normalized_max": (
        "centroid_distance_normalized_by_target_diagonal",
        "max",
    ),
    "bbox_dimension_normalized_max": (
        "bbox_dimension_max_error_normalized_by_target_diagonal",
        "max",
    ),
}
_REQUIRED_EXACT_RECONSTRUCTION_GATES = {
    "chamfer_normalized_max",
    "surface_f1_min",
    "voxel_iou_min",
}


def _data(case: LoadedCase | Mapping[str, Any]) -> Mapping[str, Any]:
    return case.data if isinstance(case, LoadedCase) else case


def _task_objective(data: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    task_contract = data.get("task_contract")
    if not isinstance(task_contract, Mapping):
        raise ValueError("case task_contract must be an object")
    objective = task_contract.get("task_objective", "engineering_specification")
    if objective not in _TASK_OBJECTIVES:
        raise ValueError(f"unsupported task objective: {objective!r}")
    return str(objective), task_contract


def _exact_reconstruction_thresholds(
    task_contract: Mapping[str, Any],
) -> dict[str, float]:
    raw = task_contract.get("reference_similarity_thresholds")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "exact_reconstruction requires reference_similarity_thresholds"
        )
    unknown = set(raw) - set(_EXACT_RECONSTRUCTION_GATES)
    missing = _REQUIRED_EXACT_RECONSTRUCTION_GATES - set(raw)
    if unknown or missing:
        raise ValueError(
            "invalid exact-reconstruction thresholds: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    thresholds: dict[str, float] = {}
    for key, value in raw.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"exact-reconstruction threshold {key} is invalid")
        number = float(value)
        if key.endswith("_min") and number > 1.0:
            raise ValueError(f"exact-reconstruction threshold {key} must be <= 1")
        thresholds[str(key)] = number
    return thresholds


def _exact_reconstruction_check(
    metrics: Mapping[str, Any],
    task_contract: Mapping[str, Any],
) -> CheckResult:
    thresholds = _exact_reconstruction_thresholds(task_contract)
    observed: dict[str, float] = {}
    failed: list[str] = []
    for threshold_key, threshold in thresholds.items():
        metric_key, direction = _EXACT_RECONSTRUCTION_GATES[threshold_key]
        value = metrics.get(metric_key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"reference metric {metric_key} is unavailable")
        measured = float(value)
        observed[metric_key] = measured
        if (direction == "min" and measured < threshold) or (
            direction == "max" and measured > threshold
        ):
            failed.append(threshold_key)
    return CheckResult(
        check_id="L1.reference_geometry.exact_reconstruction",
        status=Status.PASSED if not failed else Status.FAILED,
        failure_code=None if not failed else "exact_reconstruction_similarity_mismatch",
        expected={"thresholds": thresholds},
        observed={"metrics": observed, "failed_thresholds": failed},
        metrics={
            "gate_count": len(thresholds),
            "failed_gate_count": len(failed),
        },
        summary=(
            "reference-geometry thresholds passed"
            if not failed
            else "candidate did not meet the exact-reconstruction thresholds"
        ),
    )


def evaluate_l1(
    case: LoadedCase | Mapping[str, Any],
    candidate_step: Path,
    *,
    target_step: Path | None = None,
    is_parameter_variant: bool = False,
    l0_passed: bool = True,
) -> LayerResult:
    """Evaluate B-Rep facts, requirements, DFM, and reference diagnostics.

    Reference metrics are non-scoring for ordinary engineering tasks. They gate
    L1 only when ``task_objective`` is explicitly ``exact_reconstruction``.
    Parameter variants and functional edits are not compared with a nominal STEP.
    """

    started = time.perf_counter()
    if not l0_passed:
        check = CheckResult(
            check_id="L1.upstream_l0",
            status=Status.NOT_REACHED,
            summary="L1 was not reached because L0 did not pass",
        )
        return LayerResult.from_checks(
            "L1", (check,), runtime_seconds=time.perf_counter() - started
        )
    data = _data(case)
    profile = data.get("evaluator_profile")
    if not isinstance(profile, Mapping):
        check = CheckResult(
            check_id="L1.case_contract",
            status=Status.EVALUATOR_ERROR,
            summary="case has no evaluator profile",
        )
        return LayerResult.from_checks(
            "L1", (check,), runtime_seconds=time.perf_counter() - started
        )
    geometry_profile = profile.get("geometry")
    dfm_profile = profile.get("dfm")
    if not isinstance(geometry_profile, Mapping) or not isinstance(
        dfm_profile, Mapping
    ):
        check = CheckResult(
            check_id="L1.case_contract",
            status=Status.EVALUATOR_ERROR,
            summary="case lacks geometry or DFM profile sections",
        )
        return LayerResult.from_checks(
            "L1", (check,), runtime_seconds=time.perf_counter() - started
        )
    try:
        candidate = load_step(Path(candidate_step))
    except StepGeometryError as exc:
        check = CheckResult(
            check_id="L1.candidate_step",
            status=Status.EVALUATOR_ERROR,
            summary=f"L0-approved candidate STEP could not be loaded: {exc}",
        )
        return LayerResult.from_checks(
            "L1", (check,), runtime_seconds=time.perf_counter() - started
        )
    requirements = data.get("requirements")
    checks: list[CheckResult] = []
    layer_metrics: dict[str, Any] = {"candidate_facts": candidate.facts()}
    try:
        task_objective, task_contract = _task_objective(data)
    except ValueError as exc:
        task_objective = "invalid"
        task_contract = {}
        checks.append(
            CheckResult(
                check_id="L1.task_objective",
                status=Status.EVALUATOR_ERROR,
                summary=str(exc),
            )
        )
    exact_reconstruction = task_objective == "exact_reconstruction"
    diagnostic_mode = (
        "exact_reconstruction_scoring"
        if exact_reconstruction
        else "diagnostic_only"
    )
    cylindrical_inventory = None
    try:
        cylindrical_inventory = extract_cylindrical_features(candidate)
        layer_metrics["geometry_fingerprint"] = geometry_fingerprint(
            candidate,
            surface_sample_count=int(geometry_profile["surface_sample_count"]),
            seed=int(geometry_profile.get("seed", 0)),
        )
        layer_metrics["geometry_analysis_backend"] = "occt-brep-native.v2"
        layer_metrics["surface_meshing_used"] = False
        layer_metrics["parameter_measurements"] = parameter_measurements(
            data, candidate, holes=cylindrical_inventory.holes
        )
    except StepGeometryError as exc:
        checks.append(
            CheckResult(
                check_id="L1.geometry.analysis",
                status=Status.FAILED,
                required=True,
                failure_code="geometry_analysis_failed",
                summary=(
                    "candidate geometry could not support the required deterministic "
                    f"analysis: {type(exc).__name__}: {exc}"
                ),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        checks.append(
            CheckResult(
                check_id="L1.geometry.analysis",
                status=Status.EVALUATOR_ERROR,
                required=True,
                summary=(
                    "candidate fingerprint or parameter measurement extraction failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )
    else:
        checks.append(
            CheckResult(
                check_id="L1.geometry.analysis",
                status=Status.PASSED,
                required=True,
                metrics={
                    "parameter_measurement_count": len(
                        layer_metrics["parameter_measurements"]
                    )
                },
                summary=(
                    "candidate geometry fingerprint and frozen parameter "
                    "measurements were extracted"
                ),
            )
        )
    evaluated_requirement_checks: list[CheckResult] | None = None
    required_requirement_failed = False
    if isinstance(requirements, list) and requirements:
        evaluated_requirement_checks = evaluate_requirements(
            requirements,
            candidate,
            axis_tolerance_degrees=float(
                dfm_profile.get("hole_axis_tolerance_degrees", 1.0)
            ),
            cylindrical_inventory=cylindrical_inventory,
        )
        required_requirement_failed = any(
            check.required and check.status is Status.FAILED
            for check in evaluated_requirement_checks
        )
    if is_parameter_variant or task_objective == "functional_edit":
        layer_metrics["reference_geometry_diagnostics"] = {
            "status": "not_applicable",
            "mode": diagnostic_mode,
            "task_objective": task_objective,
            "reason": (
                "parameter_variant"
                if is_parameter_variant
                else "functional_edit_has_no_nominally_comparable_target"
            ),
        }
    elif target_step is None:
        layer_metrics["reference_geometry_diagnostics"] = {
            "status": "unavailable",
            "mode": diagnostic_mode,
            "task_objective": task_objective,
            "reason": "target_step_not_supplied",
        }
        checks.append(
            CheckResult(
                check_id="L1.reference_geometry.diagnostics",
                status=(
                    Status.EVALUATOR_ERROR
                    if exact_reconstruction
                    else Status.NOT_SCOREABLE
                ),
                required=exact_reconstruction,
                summary="reference STEP is unavailable for geometry comparison",
            )
        )
    elif not exact_reconstruction:
        if required_requirement_failed:
            reason = "required_engineering_requirement_failed"
            summary = (
                "optional reference diagnostics skipped after a required "
                "engineering requirement failed"
            )
        elif int(candidate.topology.get("solid_count", 0)) > 8:
            reason = "complex_multibody_non_scoring_diagnostic"
            summary = (
                "optional reference diagnostics skipped for a complex "
                "multibody engineering task"
            )
        else:
            reason = "non_scoring_reference_comparison_disabled"
            summary = (
                "optional reference comparison is disabled for ordinary "
                "engineering tasks"
            )
        layer_metrics["reference_geometry_diagnostics"] = {
            "status": "skipped",
            "mode": diagnostic_mode,
            "task_objective": task_objective,
            "score_bearing": False,
            "reason": reason,
        }
        checks.append(
            CheckResult(
                check_id="L1.reference_geometry.diagnostics",
                status=Status.NOT_SCOREABLE,
                required=False,
                summary=summary,
            )
        )
    else:
        try:
            target = load_step(Path(target_step))
            reference_metrics = compare_geometry(
                candidate,
                target,
                geometry_profile,
            )
            exact_check = (
                _exact_reconstruction_check(reference_metrics, task_contract)
                if exact_reconstruction
                else None
            )
            layer_metrics["reference_geometry_diagnostics"] = {
                "status": "computed",
                "mode": diagnostic_mode,
                "task_objective": task_objective,
                "score_bearing": exact_reconstruction,
                **reference_metrics,
            }
            checks.append(
                CheckResult(
                    check_id="L1.reference_geometry.diagnostics",
                    status=Status.PASSED,
                    required=False,
                    metrics={
                        "chamfer_mm": reference_metrics["chamfer_mm"],
                        "surface_f1": reference_metrics["surface_f1"],
                        "voxel_iou": reference_metrics["voxel_iou"],
                    },
                    summary=(
                        "reference-geometry metrics computed as scoring inputs"
                        if exact_reconstruction
                        else "reference-geometry metrics computed for diagnostics only"
                    ),
                )
            )
            if exact_check is not None:
                checks.append(exact_check)
        except (KeyError, TypeError, ValueError, StepGeometryError) as exc:
            layer_metrics["reference_geometry_diagnostics"] = {
                "status": "unavailable",
                "mode": diagnostic_mode,
                "task_objective": task_objective,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            checks.append(
                CheckResult(
                    check_id="L1.reference_geometry.diagnostics",
                    status=(
                        Status.EVALUATOR_ERROR
                        if exact_reconstruction
                        else Status.NOT_SCOREABLE
                    ),
                    required=exact_reconstruction,
                    summary=(
                        "reference-geometry comparison failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
    if not isinstance(requirements, list):
        checks.append(
            CheckResult(
                check_id="L1.requirement_contract",
                status=Status.EVALUATOR_ERROR,
                summary="case requirements must be a list",
            )
        )
    elif not requirements:
        checks.append(
            CheckResult(
                check_id="L1.requirement_contract",
                status=Status.NOT_SCOREABLE,
                summary="case declares no nominal engineering requirements",
            )
        )
        layer_metrics["requirement_coverage"] = {
            "requirement_count": 0,
            "required_requirement_count": 0,
            "categories": [],
            "reference_similarity_used": exact_reconstruction,
        }
    else:
        required_count = sum(
            isinstance(requirement, Mapping)
            and requirement.get("criticality", "required") == "required"
            and "hidden.check.solid_count.v1"
            not in (requirement.get("check_ids") or [])
            for requirement in requirements
        )
        categories = sorted(
            {
                str(requirement["category"])
                for requirement in requirements
                if isinstance(requirement, Mapping)
                and isinstance(requirement.get("category"), str)
                and requirement["category"]
            }
        )
        coverage = {
            "requirement_count": len(requirements),
            "required_requirement_count": required_count,
            "categories": categories,
            "reference_similarity_used": exact_reconstruction,
        }
        layer_metrics["requirement_coverage"] = coverage
        checks.append(
            CheckResult(
                check_id="L1.requirement_contract",
                status=Status.PASSED,
                metrics=coverage,
                summary="typed candidate-local requirement ledger is present",
            )
        )
        assert evaluated_requirement_checks is not None
        checks.extend(evaluated_requirement_checks)
    dfm_case = data.get("dfm")
    if not isinstance(dfm_case, Mapping):
        checks.append(
            CheckResult(
                check_id="L1.DFM.contract",
                status=Status.EVALUATOR_ERROR,
                required=False,
                summary="case DFM contract must be an object",
            )
        )
    else:
        checks.extend(
            evaluate_dfm(
                dfm_case,
                candidate,
                dfm_profile=dfm_profile,
                geometry_profile=geometry_profile,
                cylindrical_inventory=cylindrical_inventory,
                skip_expensive_wall_reason=(
                    "wall-thickness extraction skipped after a deterministic "
                    "required geometry requirement failed"
                    if required_requirement_failed
                    else None
                ),
            )
        )
    return LayerResult.from_checks(
        "L1",
        checks,
        metrics=layer_metrics,
        runtime_seconds=time.perf_counter() - started,
    )
