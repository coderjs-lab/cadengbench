"""Deterministic registry for CADEngBench-P nominal geometry requirements."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from cadengbench_p_eval.contracts import CheckResult, Status

from .features import CylindricalFeatureInventory, extract_cylindrical_features
from .step import StepGeometry

SOLID_COUNT = "hidden.check.solid_count.v1"
BBOX_DIMENSIONS = "hidden.check.bbox_dimensions.v1"
CYLINDRICAL_OPENING_GROUP = "hidden.check.cylindrical_opening_group.v1"
CYLINDRICAL_OPENING_PARAMETERIZATION = (
    "hidden.check.cylindrical_opening_parameterization.v1"
)
MAX_CYLINDRICAL_CANDIDATES = 2_048


RequirementEvaluator = Callable[
    [Mapping[str, Any], StepGeometry, CylindricalFeatureInventory | None, float],
    tuple[bool, Any, Any, str | None],
]


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _vector(value: Any) -> np.ndarray | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return None
    result = [_numeric(component) for component in value]
    if any(component is None for component in result):
        return None
    return np.asarray(result, dtype=np.float64)


def _unit(value: np.ndarray) -> np.ndarray | None:
    magnitude = float(np.linalg.norm(value))
    return None if magnitude <= 1e-12 else value / magnitude


def _parallel(first: Sequence[float], second: np.ndarray, tolerance_deg: float) -> bool:
    left = _unit(np.asarray(first, dtype=np.float64))
    right = _unit(second)
    if left is None or right is None:
        return False
    dot = min(1.0, abs(float(np.dot(left, right))))
    return dot >= math.cos(math.radians(tolerance_deg))


def _solid_count(
    requirement: Mapping[str, Any],
    geometry: StepGeometry,
    _inventory: CylindricalFeatureInventory | None,
    _axis_tolerance: float,
) -> tuple[bool, Any, Any, str | None]:
    contract = requirement.get("value_and_tolerance")
    expected = contract.get("body_count") if isinstance(contract, Mapping) else None
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("solid-count requirement needs a positive integer body_count")
    observed = {
        "solid_count": geometry.topology["solid_count"],
        "is_valid": geometry.topology["is_valid"],
    }
    passed = observed == {"solid_count": expected, "is_valid": True}
    return (
        passed,
        {"solid_count": expected, "is_valid": True},
        observed,
        None if passed else "solid_count_mismatch",
    )


def _bbox_dimensions(
    requirement: Mapping[str, Any],
    geometry: StepGeometry,
    _inventory: CylindricalFeatureInventory | None,
    _axis_tolerance: float,
) -> tuple[bool, Any, Any, str | None]:
    contract = requirement.get("value_and_tolerance")
    if not isinstance(contract, Mapping):
        raise ValueError("bounding-box requirement needs value_and_tolerance")
    observed_lengths = dict(zip(("x", "y", "z"), geometry.bbox.lengths, strict=True))
    axis_assignment = contract.get("axis_assignment", "fixed")
    expected_payload: dict[str, Any] = {}
    observed_payload: dict[str, Any] = {}
    expected_records: list[tuple[str, float, float]] = []
    for axis in ("x", "y", "z"):
        axis_contract = contract.get(axis)
        if not isinstance(axis_contract, Mapping):
            raise ValueError(f"bounding-box requirement lacks axis {axis}")
        expected = _numeric(axis_contract.get("value"))
        tolerance = _numeric(axis_contract.get("tolerance"))
        if expected is None or tolerance is None or tolerance < 0.0:
            raise ValueError("bounding-box values and tolerances must be finite")
        expected_records.append((axis, expected, tolerance))
        expected_payload[axis] = {"value_mm": expected, "tolerance_mm": tolerance}

    if axis_assignment == "free_permutation":
        expected_sorted = sorted(
            expected_records, key=lambda value: (value[1], value[0])
        )
        observed_sorted = sorted(
            observed_lengths.items(), key=lambda value: (value[1], value[0])
        )
        matches: list[dict[str, Any]] = []
        passed = True
        for (source_axis, expected, tolerance), (observed_axis, observed) in zip(
            expected_sorted, observed_sorted, strict=True
        ):
            dimension_passed = math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=tolerance
            )
            matches.append(
                {
                    "source_axis": source_axis,
                    "observed_axis": observed_axis,
                    "expected_mm": expected,
                    "observed_mm": float(observed),
                    "tolerance_mm": tolerance,
                    "absolute_error_mm": abs(float(observed) - expected),
                    "passed": dimension_passed,
                }
            )
            passed = passed and dimension_passed
        expected_payload["axis_assignment"] = "free_permutation"
        observed_payload = {
            "axis_assignment": "free_permutation",
            "matches": matches,
        }
        return (
            passed,
            expected_payload,
            observed_payload,
            None if passed else "bbox_dimensions_mismatch",
        )
    if axis_assignment != "fixed":
        raise ValueError(f"unsupported bounding-box axis assignment: {axis_assignment}")

    passed = True
    for axis, expected, tolerance in expected_records:
        observed = float(observed_lengths[axis])
        axis_passed = math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance)
        observed_payload[axis] = {
            "value_mm": observed,
            "absolute_error_mm": abs(observed - expected),
            "passed": axis_passed,
        }
        passed = passed and axis_passed
    return (
        passed,
        expected_payload,
        observed_payload,
        None if passed else "bbox_dimensions_mismatch",
    )


def _maximum_center_matching(
    expected: list[np.ndarray],
    observed: list[np.ndarray],
    tolerance_mm: float,
) -> int:
    adjacency = [
        [
            index
            for index, candidate in enumerate(observed)
            if float(np.linalg.norm(center - candidate)) <= tolerance_mm
        ]
        for center in expected
    ]
    assignments: dict[int, int] = {}

    def assign(expected_index: int, visited: set[int]) -> bool:
        for observed_index in adjacency[expected_index]:
            if observed_index in visited:
                continue
            visited.add(observed_index)
            previous = assignments.get(observed_index)
            if previous is None or assign(previous, visited):
                assignments[observed_index] = expected_index
                return True
        return False

    return sum(assign(index, set()) for index in range(len(expected)))


def _orientation_free_opening_inventories(
    requirements: Sequence[Mapping[str, Any]],
    inventory: CylindricalFeatureInventory,
) -> dict[str, CylindricalFeatureInventory]:
    """Assign overlapping public hole groups without double-counting faces."""

    groups: list[tuple[str, int, float, float]] = []
    for requirement in requirements:
        contract = requirement.get("value_and_tolerance")
        check_ids = requirement.get("check_ids") or []
        if (
            CYLINDRICAL_OPENING_GROUP not in check_ids
            or not isinstance(contract, Mapping)
            or contract.get("axis_assignment") != "free"
            or contract.get("depth_scoring") != "not_disclosed"
            or contract.get("position_scoring") != "not_disclosed"
        ):
            continue
        quantity = contract.get("quantity")
        diameter = _numeric(contract.get("diameter"))
        tolerance = _numeric(contract.get("diameter_tolerance"))
        if (
            not isinstance(quantity, int)
            or isinstance(quantity, bool)
            or quantity < 1
            or diameter is None
            or tolerance is None
            or tolerance < 0.0
        ):
            continue
        groups.append(
            (
                str(requirement.get("requirement_id") or "unknown"),
                quantity,
                diameter,
                tolerance,
            )
        )
    if len(groups) < 2:
        return {}

    holes = list(inventory.holes)
    slots = [
        group_index
        for group_index, (_, quantity, _, _) in enumerate(groups)
        for _ in range(quantity)
    ]
    adjacency: list[list[int]] = []
    for group_index in slots:
        _, _, diameter, tolerance = groups[group_index]
        adjacency.append(
            sorted(
                (
                    index
                    for index, hole in enumerate(holes)
                    if math.isclose(
                        hole.diameter_mm,
                        diameter,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    )
                ),
                key=lambda index: (
                    abs(holes[index].diameter_mm - diameter),
                    index,
                ),
            )
        )

    hole_to_slot: dict[int, int] = {}

    def assign(slot_index: int, visited: set[int]) -> bool:
        for hole_index in adjacency[slot_index]:
            if hole_index in visited:
                continue
            visited.add(hole_index)
            previous = hole_to_slot.get(hole_index)
            if previous is None or assign(previous, visited):
                hole_to_slot[hole_index] = slot_index
                return True
        return False

    for slot_index in range(len(slots)):
        assign(slot_index, set())

    assigned: list[list[Any]] = [[] for _ in groups]
    for hole_index, slot_index in sorted(hole_to_slot.items()):
        assigned[slots[slot_index]].append(holes[hole_index])

    for hole_index, hole in enumerate(holes):
        if hole_index in hole_to_slot:
            continue
        matching = [
            (
                abs(hole.diameter_mm - diameter) / max(tolerance, 1e-12),
                group_index,
            )
            for group_index, (_, _, diameter, tolerance) in enumerate(groups)
            if math.isclose(
                hole.diameter_mm,
                diameter,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        ]
        if matching:
            _, group_index = min(matching)
            assigned[group_index].append(hole)

    return {
        requirement_id: CylindricalFeatureInventory(
            assessments=tuple(assigned[group_index])
        )
        for group_index, (requirement_id, _, _, _) in enumerate(groups)
    }


def _cylindrical_opening_group(
    requirement: Mapping[str, Any],
    _geometry: StepGeometry,
    inventory: CylindricalFeatureInventory | None,
    axis_tolerance_deg: float,
) -> tuple[bool, Any, Any, str | None]:
    contract = requirement.get("value_and_tolerance")
    if not isinstance(contract, Mapping):
        raise ValueError("cylindrical-opening requirement lacks its contract")
    quantity = contract.get("quantity")
    diameter = _numeric(contract.get("diameter"))
    depth = _numeric(contract.get("depth"))
    axis = _vector(contract.get("axis_direction"))
    axis_free = contract.get("axis_assignment") == "free"
    depth_required = contract.get("depth_scoring", "required") == "required"
    position_required = contract.get("position_scoring", "required") == "required"
    diameter_tolerance = _numeric(contract.get("diameter_tolerance"))
    position_tolerance = _numeric(contract.get("position_tolerance"))
    if diameter_tolerance is None and diameter is not None:
        diameter_tolerance = max(0.05, abs(diameter) * 0.001)
    if position_tolerance is None:
        position_tolerance = 0.05
    depth_tolerance = _numeric(contract.get("depth_tolerance"))
    if depth_tolerance is None and depth is not None:
        depth_tolerance = max(0.05, abs(depth) * 0.001)
    centers_raw = contract.get("centers") or []
    centers = (
        [_vector(value) for value in centers_raw]
        if isinstance(centers_raw, list)
        else []
    )
    valid = (
        isinstance(quantity, int)
        and not isinstance(quantity, bool)
        and 1 <= quantity <= MAX_CYLINDRICAL_CANDIDATES
        and diameter is not None
        and diameter > 0.0
        and (not depth_required or (depth is not None and depth > 0.0))
        and (axis_free or (axis is not None and _unit(axis) is not None))
        and diameter_tolerance is not None
        and diameter_tolerance >= 0.0
        and depth_tolerance is not None
        and depth_tolerance >= 0.0
        and position_tolerance >= 0.0
        and (
            not position_required
            or (
                len(centers) in {0, quantity}
                and all(center is not None for center in centers)
            )
        )
    )
    if not valid:
        raise ValueError("cylindrical-opening requirement is malformed")
    holes = list(inventory.holes if inventory is not None else ())
    if len(holes) > MAX_CYLINDRICAL_CANDIDATES:
        return (
            False,
            {"quantity": quantity},
            {"candidate_count": len(holes)},
            "cylindrical_candidate_limit_exceeded",
        )
    eligible = [
        hole
        for hole in holes
        if math.isclose(
            hole.diameter_mm, diameter, rel_tol=0.0, abs_tol=diameter_tolerance
        )
        and (
            not depth_required
            or math.isclose(
                hole.depth_mm,
                depth,
                rel_tol=0.0,
                abs_tol=depth_tolerance,
            )
        )
        and (axis_free or _parallel(hole.axis_direction, axis, axis_tolerance_deg))
    ]
    matched_centers = quantity
    if position_required and centers:
        matched_centers = _maximum_center_matching(
            [center for center in centers if center is not None],
            [np.asarray(hole.center_mm) for hole in eligible],
            position_tolerance,
        )
    passed = len(eligible) == quantity and matched_centers == quantity
    expected_payload = {
        "quantity": quantity,
        "diameter_mm": diameter,
        "diameter_tolerance_mm": diameter_tolerance,
        "depth_scoring": "required" if depth_required else "not_disclosed",
        "depth_mm": depth if depth_required else None,
        "depth_tolerance_mm": depth_tolerance if depth_required else None,
        "axis_assignment": "free" if axis_free else "fixed",
        "axis_direction": None if axis_free else axis.tolist(),
        "axis_tolerance_degrees": None if axis_free else axis_tolerance_deg,
        "position_scoring": "required" if position_required else "not_disclosed",
        "centers_mm": (
            [center.tolist() for center in centers if center is not None]
            if position_required
            else []
        ),
        "position_tolerance_mm": position_tolerance if position_required else None,
    }
    observed_payload = {
        "recognized_hole_count": len(holes),
        "dimension_axis_match_count": len(eligible),
        "position_match_count": matched_centers,
        "matched_face_ids": [hole.face_id for hole in eligible],
        "recognized_holes": [hole.as_dict() for hole in holes],
    }
    return (
        passed,
        expected_payload,
        observed_payload,
        None if passed else "cylindrical_opening_group_mismatch",
    )


CHECK_REGISTRY: dict[str, RequirementEvaluator] = {
    SOLID_COUNT: _solid_count,
    BBOX_DIMENSIONS: _bbox_dimensions,
    CYLINDRICAL_OPENING_GROUP: _cylindrical_opening_group,
}


def evaluate_requirements(
    requirements: Sequence[Mapping[str, Any]],
    geometry: StepGeometry,
    *,
    axis_tolerance_degrees: float = 1.0,
    cylindrical_inventory: CylindricalFeatureInventory | None = None,
) -> tuple[CheckResult, ...]:
    """Evaluate every declared check without silently accepting unknown IDs."""

    if not math.isfinite(axis_tolerance_degrees) or axis_tolerance_degrees < 0.0:
        raise ValueError("axis tolerance must be finite and non-negative")
    needs_cylinders = any(
        CYLINDRICAL_OPENING_GROUP in (requirement.get("check_ids") or [])
        for requirement in requirements
    )
    inventory = (
        cylindrical_inventory
        if cylindrical_inventory is not None
        else extract_cylindrical_features(geometry)
        if needs_cylinders
        else None
    )
    orientation_free_inventories = (
        _orientation_free_opening_inventories(requirements, inventory)
        if inventory is not None
        else {}
    )
    results: list[CheckResult] = []
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "unknown")
        declared_required = requirement.get("criticality", "required") == "required"
        check_ids = requirement.get("check_ids")
        if not isinstance(check_ids, list) or not check_ids:
            results.append(
                CheckResult(
                    check_id=f"L1.requirement.{requirement_id}:unbound",
                    status=Status.NOT_SCOREABLE,
                    required=declared_required,
                    summary="requirement has no deterministic check binding",
                    evidence=[{"requirement_id": requirement_id}],
                )
            )
            continue
        for source_check_id in check_ids:
            required = declared_required and source_check_id != SOLID_COUNT
            check_id = f"L1.requirement.{requirement_id}:{source_check_id}"
            if source_check_id == CYLINDRICAL_OPENING_PARAMETERIZATION:
                results.append(
                    CheckResult(
                        check_id=check_id,
                        status=Status.NOT_APPLICABLE,
                        required=False,
                        summary=(
                            "shared-parameter behavior is delegated to L2 replay; "
                            "L1 verifies the nominal repeated geometry"
                        ),
                        evidence=[
                            {
                                "requirement_id": requirement_id,
                                "source_check_id": str(source_check_id),
                                "delegated_layer": "L2",
                            }
                        ],
                    )
                )
                continue
            evaluator = CHECK_REGISTRY.get(str(source_check_id))
            evidence = [
                {
                    "requirement_id": requirement_id,
                    "source_check_id": str(source_check_id),
                }
            ]
            if evaluator is None:
                results.append(
                    CheckResult(
                        check_id=check_id,
                        status=Status.NOT_SCOREABLE,
                        required=required,
                        summary=(
                            "no certified deterministic implementation is registered"
                        ),
                        evidence=evidence,
                    )
                )
                continue
            try:
                requirement_inventory = orientation_free_inventories.get(
                    requirement_id, inventory
                )
                passed, expected, observed, failure_code = evaluator(
                    requirement,
                    geometry,
                    requirement_inventory,
                    axis_tolerance_degrees,
                )
            except (TypeError, ValueError, KeyError) as exc:
                results.append(
                    CheckResult(
                        check_id=check_id,
                        status=Status.EVALUATOR_ERROR,
                        required=required,
                        summary=f"invalid requirement contract: {exc}",
                        evidence=evidence,
                    )
                )
                continue
            results.append(
                CheckResult(
                    check_id=check_id,
                    status=Status.PASSED if passed else Status.FAILED,
                    required=required,
                    failure_code=failure_code,
                    summary=(
                        "deterministic requirement satisfied"
                        if passed
                        else "deterministic requirement violated"
                    ),
                    expected=expected,
                    observed=observed,
                    evidence=evidence,
                )
            )
    return tuple(results)
