"""Deterministic minimum-wall estimate with explicit certification gating."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from cadengbench_p_eval.contracts import CheckResult, Status
from cadengbench_p_eval.geometry.metrics import (
    BREP_SURFACE_SAMPLER,
    sample_brep_surface,
)
from cadengbench_p_eval.geometry.step import StepGeometry

from .contracts import MINIMUM_WALL_THICKNESS, DFMGate

CHECK_ID = "L1.DFM.minimum_wall_thickness"
EXTRACTOR_VERSION = "opposing-surfaces-and-coaxial-cylinders.v2"


def _vector(value: Any) -> np.ndarray:
    return np.asarray((float(value.x), float(value.y), float(value.z)))


def _unit(value: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(value))
    if magnitude <= 1e-12:
        raise ValueError("zero-length vector")
    return value / magnitude


def _geometry_type(entity: Any) -> str:
    try:
        return str(entity.geomType()).upper()
    except Exception:
        return "UNKNOWN"


def _planar_candidates(geometry: StepGeometry) -> list[dict[str, Any]]:
    planes: list[dict[str, Any]] = []
    for index, face in enumerate(geometry.shape.Faces()):
        if _geometry_type(face) != "PLANE":
            continue
        try:
            planes.append(
                {
                    "index": index,
                    "normal": _unit(_vector(face.normalAt())),
                    "center": _vector(face.Center()),
                    "area": float(face.Area()),
                }
            )
        except Exception:
            continue
    if not planes:
        return []
    area_floor = max(value["area"] for value in planes) * 0.005
    candidates: list[dict[str, Any]] = []
    for left_index, first in enumerate(planes):
        for second in planes[left_index + 1 :]:
            normal_dot = float(np.dot(first["normal"], second["normal"]))
            if normal_dot > -0.985:
                continue
            delta = second["center"] - first["center"]
            spacing = abs(float(np.dot(delta, first["normal"])))
            tangential = float(
                np.linalg.norm(delta - np.dot(delta, first["normal"]) * first["normal"])
            )
            overlap_scale = 0.5 * (
                math.sqrt(max(first["area"], 0.0)) + math.sqrt(max(second["area"], 0.0))
            )
            if (
                spacing <= 1e-7
                or min(first["area"], second["area"]) < area_floor
                or tangential > overlap_scale
            ):
                continue
            candidates.append(
                {
                    "value_mm": spacing,
                    "method": "opposing_parallel_planar_face_spacing",
                    "face_indices": [first["index"], second["index"]],
                    "face_areas_mm2": [first["area"], second["area"]],
                }
            )
    return candidates


def _cylindrical_candidates(geometry: StepGeometry) -> list[dict[str, Any]]:
    cylinders: list[dict[str, Any]] = []
    for index, face in enumerate(geometry.shape.Faces()):
        if _geometry_type(face) != "CYLINDER":
            continue
        try:
            cylinder = face._geomAdaptor().Cylinder()
            axis = cylinder.Axis()
            direction = _unit(
                np.asarray(
                    (
                        axis.Direction().X(),
                        axis.Direction().Y(),
                        axis.Direction().Z(),
                    )
                )
            )
            location = np.asarray(
                (axis.Location().X(), axis.Location().Y(), axis.Location().Z())
            )
            u_min, u_max, v_min, v_max = face.uvBounds()
            normal, point = face.normalAt(
                (float(u_min) + float(u_max)) / 2.0,
                (float(v_min) + float(v_max)) / 2.0,
            )
            point_value = _vector(point)
            along = float(np.dot(point_value - location, direction))
            radial = _unit(point_value - (location + along * direction))
            internal = bool(float(np.dot(_unit(_vector(normal)), radial)) < -0.5)
            cylinders.append(
                {
                    "index": index,
                    "radius": float(cylinder.Radius()),
                    "direction": direction,
                    "location": location,
                    "internal": internal,
                }
            )
        except Exception:
            continue
    candidates: list[dict[str, Any]] = []
    for left_index, first in enumerate(cylinders):
        for second in cylinders[left_index + 1 :]:
            if first["internal"] == second["internal"]:
                continue
            if abs(float(np.dot(first["direction"], second["direction"]))) < 0.985:
                continue
            offset = second["location"] - first["location"]
            perpendicular = (
                offset - np.dot(offset, first["direction"]) * first["direction"]
            )
            if float(np.linalg.norm(perpendicular)) > 1e-4:
                continue
            thickness = abs(first["radius"] - second["radius"])
            if thickness <= 1e-7:
                continue
            candidates.append(
                {
                    "value_mm": thickness,
                    "method": "coaxial_internal_external_radius_delta",
                    "face_indices": [first["index"], second["index"]],
                }
            )
    return candidates


def _sampled_candidate(
    geometry: StepGeometry,
    *,
    sample_count: int,
    neighbor_count: int,
    opposed_normal_cosine_min: float,
    tessellation_tolerance_mm: float,
    seed: int,
) -> dict[str, Any] | None:
    if sample_count < 2 or neighbor_count < 1:
        return None
    if not math.isfinite(tessellation_tolerance_mm) or tessellation_tolerance_mm <= 0.0:
        raise ValueError("legacy tessellation tolerance must be finite and positive")
    samples = sample_brep_surface(
        geometry,
        count=sample_count,
        seed=seed,
    )
    k = min(sample_count, neighbor_count + 1)
    distances, indices = cKDTree(samples.points).query(samples.points, k=k, workers=1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    candidates: list[float] = []
    minimum_distance = max(geometry.bbox.diagonal * 1e-8, 1e-7)
    for source_index in range(sample_count):
        for distance, target_index in zip(
            distances[source_index, 1:], indices[source_index, 1:], strict=True
        ):
            if (
                not math.isfinite(float(distance))
                or float(distance) <= minimum_distance
            ):
                continue
            normal_dot = float(
                np.dot(samples.normals[source_index], samples.normals[target_index])
            )
            if normal_dot > -opposed_normal_cosine_min:
                continue
            displacement = samples.points[target_index] - samples.points[source_index]
            alignment = abs(
                float(
                    np.dot(
                        displacement / float(np.linalg.norm(displacement)),
                        samples.normals[source_index],
                    )
                )
            )
            if alignment < 0.75:
                continue
            candidates.append(float(distance))
            break
    if not candidates:
        return None
    return {
        "value_mm": float(np.quantile(candidates, 0.05)),
        "method": "sampled_brep_opposing_surface_distance_q05",
        "surface_sampling_backend": BREP_SURFACE_SAMPLER,
        "surface_meshing_used": False,
        "qualified_sample_count": len(candidates),
        "surface_sample_count": sample_count,
    }


def estimate_wall_thickness(
    geometry: StepGeometry,
    *,
    sample_count: int,
    neighbor_count: int,
    opposed_normal_cosine_min: float,
    tessellation_tolerance_mm: float,
    seed: int,
) -> dict[str, Any]:
    """Return a deterministic estimate; authority is resolved separately."""

    analytic = [*_planar_candidates(geometry), *_cylindrical_candidates(geometry)]
    sampled = _sampled_candidate(
        geometry,
        sample_count=sample_count,
        neighbor_count=neighbor_count,
        opposed_normal_cosine_min=opposed_normal_cosine_min,
        tessellation_tolerance_mm=tessellation_tolerance_mm,
        seed=seed,
    )
    fallback = {
        "value_mm": min(geometry.bbox.lengths),
        "method": "bounding_box_minimum_dimension_upper_bound",
    }
    qualified_candidates = [*analytic]
    if sampled is not None:
        qualified_candidates.append(sampled)
    selected_pool = qualified_candidates or [fallback]
    selected = min(selected_pool, key=lambda value: float(value["value_mm"]))
    ordered = sorted(
        [*qualified_candidates, fallback],
        key=lambda value: (float(value["value_mm"]), str(value["method"])),
    )
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "minimum_wall_thickness_mm": float(selected["value_mm"]),
        "selected_method": selected["method"],
        "qualified_candidate_count": len(qualified_candidates),
        "candidate_count": len(ordered),
        "candidates": ordered,
        "limitations": [
            "This is a deterministic opposing-surface estimate, not a full "
            "medial-axis field.",
            "A compliance verdict requires item-level extractor certification.",
        ],
    }


def evaluate_minimum_wall_thickness(
    gate: DFMGate,
    geometry: StepGeometry,
    *,
    sample_count: int,
    neighbor_count: int,
    opposed_normal_cosine_min: float,
    tessellation_tolerance_mm: float,
    seed: int,
) -> CheckResult:
    if gate.family != MINIMUM_WALL_THICKNESS:
        raise ValueError("wall evaluator received the wrong DFM gate")
    measured = estimate_wall_thickness(
        geometry,
        sample_count=sample_count,
        neighbor_count=neighbor_count,
        opposed_normal_cosine_min=opposed_normal_cosine_min,
        tessellation_tolerance_mm=tessellation_tolerance_mm,
        seed=seed,
    )
    expected = {"minimum_mm": gate.thresholds.get("minimum_mm")}
    if gate.status is Status.NOT_APPLICABLE:
        return CheckResult(
            check_id=CHECK_ID,
            status=Status.NOT_APPLICABLE,
            required=gate.required,
            summary=gate.reason,
            observed=measured,
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
                "minimum_wall_thickness_mm": measured["minimum_wall_thickness_mm"]
            },
        )
    passed = measured["minimum_wall_thickness_mm"] >= gate.thresholds["minimum_mm"]
    return CheckResult(
        check_id=CHECK_ID,
        status=Status.PASSED if passed else Status.FAILED,
        required=gate.required,
        failure_code=None if passed else "minimum_wall_thickness_violation",
        summary=(
            "minimum wall thickness satisfies the frozen process profile"
            if passed
            else "minimum wall thickness violates the frozen process profile"
        ),
        expected=expected,
        observed=measured,
        evidence=gate.evidence(),
        metrics={"minimum_wall_thickness_mm": measured["minimum_wall_thickness_mm"]},
    )
