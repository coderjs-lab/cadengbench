"""Deterministic geometry measurements used by L2 relation contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .features import CylindricalFace, extract_cylindrical_features
from .step import StepGeometry


def parameter_measurements(
    case: Mapping[str, Any],
    geometry: StepGeometry,
    *,
    holes: Sequence[CylindricalFace] | None = None,
) -> dict[str, float]:
    """Measure the public observables declared by one compiled evaluator case.

    Only frozen measurement bindings are interpreted. Unknown or incomplete
    bindings are omitted so downstream relation checks become ``not_scoreable``
    instead of silently using an invented measurement.
    """

    values = {
        "solid_count": float(geometry.topology["solid_count"]),
        "volume_mm3": float(geometry.volume_mm3),
        "surface_area_mm2": float(geometry.surface_area_mm2),
        "bbox_x_mm": float(geometry.bbox.lengths[0]),
        "bbox_y_mm": float(geometry.bbox.lengths[1]),
        "bbox_z_mm": float(geometry.bbox.lengths[2]),
        "centroid_x_mm": float(geometry.centroid_mm[0]),
        "centroid_y_mm": float(geometry.centroid_mm[1]),
        "centroid_z_mm": float(geometry.centroid_mm[2]),
    }
    recognized_holes = tuple(
        holes if holes is not None else extract_cylindrical_features(geometry).holes
    )
    values["recognized_hole_count"] = float(len(recognized_holes))
    if recognized_holes:
        values["minimum_hole_diameter_mm"] = min(
            hole.diameter_mm for hole in recognized_holes
        )
        values["maximum_hole_depth_to_diameter"] = max(
            hole.depth_to_diameter for hole in recognized_holes
        )

    requirements = {
        str(requirement.get("requirement_id")): requirement
        for requirement in case.get("requirements", [])
        if isinstance(requirement, Mapping)
    }
    parameter_system = case.get("parameter_system")
    parameter_records = (
        parameter_system.get("parameters", [])
        if isinstance(parameter_system, Mapping)
        else []
    )
    profile = case.get("evaluator_profile")
    dfm_profile = profile.get("dfm") if isinstance(profile, Mapping) else None
    axis_tolerance = float(
        dfm_profile.get("hole_axis_tolerance_degrees", 1.0)
        if isinstance(dfm_profile, Mapping)
        else 1.0
    )
    axis_cosine = math.cos(math.radians(axis_tolerance))

    for parameter in parameter_records:
        if not isinstance(parameter, Mapping):
            continue
        binding = parameter.get("measurement_binding")
        if not isinstance(binding, Mapping):
            continue
        key = binding.get("measurement_key")
        kind = binding.get("measurement_kind")
        if not isinstance(key, str) or not key:
            continue
        if kind == "bounding_box_axis":
            axis_index = {"x": 0, "y": 1, "z": 2}.get(binding.get("axis"))
            if axis_index is not None:
                values[key] = float(geometry.bbox.lengths[axis_index])
            continue

        if kind == "bounding_box_unordered_dimension":
            requirement = requirements.get(str(binding.get("requirement_id")))
            path = binding.get("requirement_value_path")
            expected: Any = requirement
            if not isinstance(path, list) or not path:
                continue
            for component in path:
                if not isinstance(component, str) or not isinstance(expected, Mapping):
                    expected = None
                    break
                expected = expected.get(component)
            if isinstance(expected, bool) or not isinstance(expected, (int, float)):
                continue
            expected_value = float(expected)
            values[key] = float(
                min(
                    geometry.bbox.lengths,
                    key=lambda value: (
                        abs(float(value) - expected_value),
                        float(value),
                    ),
                )
            )
            continue

        requirement = requirements.get(str(binding.get("requirement_id")))
        contract = (
            requirement.get("value_and_tolerance")
            if isinstance(requirement, Mapping)
            else None
        )
        if not isinstance(contract, Mapping):
            continue
        expected_axis = contract.get("axis_direction")
        axis_free = contract.get("axis_assignment") == "free"
        centers = contract.get("centers")
        quantity = contract.get("quantity")
        if (
            not isinstance(centers, list)
            or isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity < 1
        ):
            continue
        if axis_free:
            eligible = list(recognized_holes)
        else:
            if not isinstance(expected_axis, list) or len(expected_axis) != 3:
                continue
            expected_axis_values = tuple(float(value) for value in expected_axis)
            axis_norm = math.sqrt(sum(value * value for value in expected_axis_values))
            if axis_norm <= 1e-12:
                continue
            eligible = [
                hole
                for hole in recognized_holes
                if abs(
                    sum(
                        left * right
                        for left, right in zip(
                            hole.axis_direction,
                            expected_axis_values,
                            strict=True,
                        )
                    )
                    / axis_norm
                )
                >= axis_cosine
            ]
        selected: list[CylindricalFace] = []
        if axis_free:
            expected_diameter = contract.get("diameter")
            if isinstance(expected_diameter, (int, float)) and not isinstance(
                expected_diameter, bool
            ):
                selected = sorted(
                    eligible,
                    key=lambda hole: (
                        abs(hole.diameter_mm - float(expected_diameter)),
                        hole.face_id,
                    ),
                )[:quantity]
        elif centers:
            remaining = list(eligible)
            for center in centers:
                if not isinstance(center, list) or len(center) != 3 or not remaining:
                    continue
                expected_center = tuple(float(value) for value in center)
                nearest = min(
                    remaining,
                    key=lambda hole: math.dist(hole.center_mm, expected_center),
                )
                selected.append(nearest)
                remaining.remove(nearest)
        elif len(eligible) == quantity:
            # A centerless group is measurable only when axis and cardinality
            # identify the complete group without an arbitrary face choice.
            selected = eligible
        if len(selected) != quantity:
            continue
        if kind == "cylindrical_opening_diameter":
            values[key] = sum(hole.diameter_mm for hole in selected) / len(selected)
        elif kind == "cylindrical_opening_depth":
            values[key] = sum(hole.depth_mm for hole in selected) / len(selected)
    return values
