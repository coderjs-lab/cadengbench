"""Minimum cylindrical-hole geometry measurement and compliance check."""

from __future__ import annotations

from typing import Any

from cadengbench_p_eval.contracts import CheckResult, Status
from cadengbench_p_eval.geometry.features import CylindricalFeatureInventory

from .contracts import MINIMUM_HOLE_GEOMETRY, DFMGate

CHECK_ID = "L1.DFM.minimum_hole_geometry"
EXPECTED_HOLE_COUNT_FIELDS = (
    "expected_minimum_feature_count",
    "minimum_required_hole_count",
    "required_hole_count",
    "expected_hole_count",
)


def _expected_required_hole_count(gate: DFMGate) -> tuple[int | None, str | None]:
    declarations: list[tuple[str, int]] = []
    for field in EXPECTED_HOLE_COUNT_FIELDS:
        if field not in gate.raw_contract:
            continue
        value = gate.raw_contract[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None, f"{field} must be a non-negative integer"
        declarations.append((field, value))
    if not declarations:
        return None, None
    unique_values = {value for _, value in declarations}
    if len(unique_values) != 1:
        fields = ", ".join(field for field, _ in declarations)
        return None, f"conflicting required-hole counts are declared by: {fields}"
    return declarations[0][1], None


def measure_hole_geometry(
    inventory: CylindricalFeatureInventory,
) -> dict[str, Any]:
    holes = inventory.holes
    return {
        "extractor_version": "internal-near-full-cylinder.v1",
        "hole_count": len(holes),
        "minimum_diameter_mm": min((hole.diameter_mm for hole in holes), default=None),
        "maximum_depth_to_diameter": max(
            (hole.depth_to_diameter for hole in holes), default=None
        ),
        "holes": [hole.as_dict() for hole in holes],
        "limitations": [
            "Only unambiguous internal near-complete cylindrical faces are covered.",
            "Thread form, tool selection, and hole-bottom geometry are not inferred.",
        ],
    }


def evaluate_minimum_hole_geometry(
    gate: DFMGate,
    inventory: CylindricalFeatureInventory,
) -> CheckResult:
    if gate.family != MINIMUM_HOLE_GEOMETRY:
        raise ValueError("hole evaluator received the wrong DFM gate")
    measured = measure_hole_geometry(inventory)
    if gate.status is Status.NOT_APPLICABLE:
        return CheckResult(
            check_id=CHECK_ID,
            status=Status.NOT_APPLICABLE,
            required=gate.required,
            summary=gate.reason,
            observed=measured,
        )
    expected_hole_count, expected_hole_count_error = _expected_required_hole_count(gate)
    expected = {
        "minimum_diameter_mm": gate.thresholds.get("minimum_diameter_mm"),
        "maximum_depth_to_diameter": gate.thresholds.get("maximum_depth_to_diameter"),
        "minimum_required_hole_count": expected_hole_count,
    }
    if expected_hole_count_error is not None:
        return CheckResult(
            check_id=CHECK_ID,
            status=Status.NOT_SCOREABLE,
            required=gate.required,
            summary=expected_hole_count_error,
            expected=expected,
            observed=measured,
            evidence=gate.evidence(),
        )
    if expected_hole_count is not None and expected_hole_count > measured["hole_count"]:
        if not gate.extractor_certified:
            return CheckResult(
                check_id=CHECK_ID,
                status=Status.NOT_SCOREABLE,
                required=gate.required,
                summary=(
                    "required holes are declared, but an uncertified extractor "
                    "cannot establish that they are missing"
                ),
                expected=expected,
                observed=measured,
                evidence=gate.evidence(),
                metrics={"recognized_hole_count": measured["hole_count"]},
            )
        return CheckResult(
            check_id=CHECK_ID,
            status=Status.FAILED,
            required=gate.required,
            failure_code="required_hole_features_missing",
            summary="fewer supported cylindrical holes were found than required",
            expected=expected,
            observed=measured,
            evidence=gate.evidence(),
            metrics={"recognized_hole_count": measured["hole_count"]},
        )
    if measured["hole_count"] == 0:
        return CheckResult(
            check_id=CHECK_ID,
            status=Status.NOT_SCOREABLE,
            required=gate.required,
            summary=(
                "the family is applicable, but no supported cylindrical hole "
                "was recognized and no missing required hole can be established"
            ),
            expected=expected,
            observed=measured,
            evidence=gate.evidence(),
            metrics={"recognized_hole_count": 0},
        )
    if not gate.scoreable:
        return CheckResult(
            check_id=CHECK_ID,
            status=Status.NOT_SCOREABLE,
            required=gate.required,
            summary=gate.reason,
            expected=expected,
            observed=measured,
            evidence=gate.evidence(),
            metrics={
                "minimum_diameter_mm": measured["minimum_diameter_mm"],
                "maximum_depth_to_diameter": measured["maximum_depth_to_diameter"],
            },
        )
    diameter_passed = measured["minimum_diameter_mm"] >= expected["minimum_diameter_mm"]
    depth_ratio_passed = (
        measured["maximum_depth_to_diameter"] <= expected["maximum_depth_to_diameter"]
    )
    passed = bool(diameter_passed and depth_ratio_passed)
    observed = {
        **measured,
        "minimum_diameter_passed": bool(diameter_passed),
        "maximum_depth_to_diameter_passed": bool(depth_ratio_passed),
    }
    return CheckResult(
        check_id=CHECK_ID,
        status=Status.PASSED if passed else Status.FAILED,
        required=gate.required,
        failure_code=None if passed else "minimum_hole_geometry_violation",
        summary=(
            "minimum hole geometry satisfies the frozen process profile"
            if passed
            else "minimum hole geometry violates the frozen process profile"
        ),
        expected=expected,
        observed=observed,
        evidence=gate.evidence(),
        metrics={
            "minimum_diameter_mm": measured["minimum_diameter_mm"],
            "maximum_depth_to_diameter": measured["maximum_depth_to_diameter"],
        },
    )
