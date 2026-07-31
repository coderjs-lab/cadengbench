"""Orchestrate the two evidence-gated CADEngBench-P DFM families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cadengbench_p_eval.contracts import CheckResult, Status
from cadengbench_p_eval.geometry.features import (
    CylindricalFeatureInventory,
    extract_cylindrical_features,
)
from cadengbench_p_eval.geometry.step import StepGeometry

from .contracts import (
    MINIMUM_HOLE_GEOMETRY,
    MINIMUM_WALL_THICKNESS,
    DFMGate,
    resolve_dfm_gate,
)
from .holes import CHECK_ID as HOLE_CHECK_ID
from .holes import evaluate_minimum_hole_geometry
from .wall import CHECK_ID as WALL_CHECK_ID
from .wall import evaluate_minimum_wall_thickness


def _error(check_id: str, gate: DFMGate, exc: Exception) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        status=Status.EVALUATOR_ERROR,
        required=gate.required,
        summary=f"DFM evaluator failed: {type(exc).__name__}: {exc}",
    )


def evaluate_dfm(
    dfm_case: Mapping[str, Any],
    geometry: StepGeometry,
    *,
    dfm_profile: Mapping[str, Any],
    geometry_profile: Mapping[str, Any],
    cylindrical_inventory: CylindricalFeatureInventory | None = None,
    skip_expensive_wall_reason: str | None = None,
) -> tuple[CheckResult, ...]:
    """Measure both families and emit verdicts only through resolved gates."""

    wall_gate = resolve_dfm_gate(dfm_case, MINIMUM_WALL_THICKNESS)
    hole_gate = resolve_dfm_gate(dfm_case, MINIMUM_HOLE_GEOMETRY)
    if hole_gate.status is not Status.PASSED:
        hole_result = CheckResult(
            check_id=HOLE_CHECK_ID,
            status=hole_gate.status,
            required=hole_gate.required,
            summary=hole_gate.reason,
        )
    else:
        try:
            inventory = cylindrical_inventory or extract_cylindrical_features(geometry)
            hole_result = evaluate_minimum_hole_geometry(hole_gate, inventory)
        except Exception as exc:
            hole_result = _error(HOLE_CHECK_ID, hole_gate, exc)
    if wall_gate.status is not Status.PASSED:
        wall_result = CheckResult(
            check_id=WALL_CHECK_ID,
            status=wall_gate.status,
            required=wall_gate.required,
            summary=wall_gate.reason,
        )
    elif skip_expensive_wall_reason is not None:
        wall_result = CheckResult(
            check_id=WALL_CHECK_ID,
            status=(
                Status.NOT_REACHED
                if wall_gate.required
                else Status.NOT_SCOREABLE
            ),
            required=wall_gate.required,
            summary=skip_expensive_wall_reason,
        )
    else:
        try:
            wall_result = evaluate_minimum_wall_thickness(
                wall_gate,
                geometry,
                sample_count=int(dfm_profile.get("wall_sample_count", 1024)),
                neighbor_count=int(dfm_profile.get("wall_neighbor_count", 32)),
                opposed_normal_cosine_min=float(
                    dfm_profile.get("opposed_normal_cosine_min", 0.9)
                ),
                tessellation_tolerance_mm=float(
                    geometry_profile.get("linear_tessellation_tolerance_mm", 0.1)
                ),
                seed=int(geometry_profile.get("seed", 0)),
            )
        except Exception as exc:
            wall_result = _error(WALL_CHECK_ID, wall_gate, exc)
    return (wall_result, hole_result)
