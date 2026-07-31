"""Deterministic B-Rep reference-geometry diagnostics.

The metric vector is descriptive unless an evaluator case explicitly declares
the ``exact_reconstruction`` task objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec
from OCP.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_REVERSED

from .step import FloatArray, StepGeometry, StepGeometryError, contains_points

BREP_SURFACE_SAMPLER = "trimmed-brep-area-weighted.v1"
BREP_DISTANCE_BACKEND = "occt-point-to-brep-boundary.v1"


@dataclass(frozen=True)
class SurfaceSamples:
    points: FloatArray
    normals: FloatArray


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _sample_trimmed_face(
    face: cq.Face,
    *,
    count: int,
    rng: np.random.Generator,
) -> SurfaceSamples:
    """Sample one trimmed face without constructing a triangle mesh."""

    u_min, u_max, v_min, v_max = map(float, face.uvBounds())
    if not all(math.isfinite(value) for value in (u_min, u_max, v_min, v_max)):
        raise StepGeometryError("B-Rep face has non-finite parameter bounds")
    if u_max <= u_min or v_max <= v_min:
        raise StepGeometryError("B-Rep face has degenerate parameter bounds")
    target_candidates = max(count * 4, 32)
    attempt_limit = max(target_candidates * 64, 2048)
    adaptor = BRepAdaptor_Surface(face.wrapped, True)
    orientation = -1.0 if face.wrapped.Orientation() == TopAbs_REVERSED else 1.0
    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    jacobians: list[float] = []
    attempts = 0
    while len(points) < target_candidates and attempts < attempt_limit:
        batch_size = min(
            max((target_candidates - len(points)) * 3, 32),
            attempt_limit - attempts,
        )
        parameters = rng.random((batch_size, 2))
        attempts += batch_size
        for u_fraction, v_fraction in parameters:
            u_value = u_min + float(u_fraction) * (u_max - u_min)
            v_value = v_min + float(v_fraction) * (v_max - v_min)
            state = BRepClass_FaceClassifier(
                face.wrapped,
                gp_Pnt2d(u_value, v_value),
                1e-9,
            ).State()
            if state not in (TopAbs_IN, TopAbs_ON):
                continue
            point = gp_Pnt()
            derivative_u = gp_Vec()
            derivative_v = gp_Vec()
            try:
                adaptor.D1(
                    u_value,
                    v_value,
                    point,
                    derivative_u,
                    derivative_v,
                )
                normal = derivative_u.Crossed(derivative_v)
                jacobian = float(normal.Magnitude())
            except Exception:
                continue
            if not math.isfinite(jacobian) or jacobian <= 1e-14:
                continue
            points.append((float(point.X()), float(point.Y()), float(point.Z())))
            normals.append(
                (
                    orientation * float(normal.X()) / jacobian,
                    orientation * float(normal.Y()) / jacobian,
                    orientation * float(normal.Z()) / jacobian,
                )
            )
            jacobians.append(jacobian)
            if len(points) >= target_candidates:
                break
    if len(points) < count:
        raise StepGeometryError(
            "trimmed B-Rep face could not provide enough valid surface samples"
        )
    weights = np.asarray(jacobians, dtype=np.float64)
    weights /= weights.sum()
    selected = rng.choice(len(points), size=count, replace=False, p=weights)
    return SurfaceSamples(
        points=np.asarray(points, dtype=np.float64)[selected],
        normals=np.asarray(normals, dtype=np.float64)[selected],
    )


def sample_brep_surface(
    geometry: StepGeometry,
    *,
    count: int,
    seed: int,
) -> SurfaceSamples:
    """Area-weighted samples drawn directly from trimmed B-Rep faces."""

    count = _positive_int(count, "surface sample count")
    faces = list(geometry.shape.Faces())
    areas = np.asarray([float(face.Area()) for face in faces], dtype=np.float64)
    valid = np.isfinite(areas) & (areas > 1e-14)
    if not faces or not bool(np.any(valid)):
        raise StepGeometryError("B-Rep contains no positive-area faces")
    valid_indices = np.flatnonzero(valid)
    probabilities = areas[valid] / areas[valid].sum()
    rng = np.random.default_rng(int(seed))
    selected_faces = rng.choice(valid_indices, size=count, p=probabilities)
    face_counts = np.bincount(selected_faces, minlength=len(faces))
    point_blocks: list[FloatArray] = []
    normal_blocks: list[FloatArray] = []
    for face_index, face_count in enumerate(face_counts):
        if face_count == 0:
            continue
        samples = _sample_trimmed_face(
            faces[face_index],
            count=int(face_count),
            rng=rng,
        )
        point_blocks.append(samples.points)
        normal_blocks.append(samples.normals)
    if not point_blocks:
        raise StepGeometryError("B-Rep surface sampling produced no points")
    return SurfaceSamples(
        points=np.concatenate(point_blocks, axis=0),
        normals=np.concatenate(normal_blocks, axis=0),
    )


def _boundary_shape(geometry: StepGeometry) -> cq.Compound:
    faces = list(geometry.shape.Faces())
    if not faces:
        raise StepGeometryError("B-Rep boundary contains no faces")
    return cq.Compound.makeCompound(faces)


def _point_to_boundary_distances(
    samples: SurfaceSamples,
    target: StepGeometry,
    *,
    chunk_size: int = 512,
) -> FloatArray:
    boundary = _boundary_shape(target)
    distances: list[float] = []
    for start in range(0, len(samples.points), chunk_size):
        vertices = [
            cq.Vertex.makeVertex(float(point[0]), float(point[1]), float(point[2]))
            for point in samples.points[start : start + chunk_size]
        ]
        distances.extend(float(value) for value in boundary.distances(*vertices))
    result = np.asarray(distances, dtype=np.float64)
    if len(result) != len(samples.points) or not np.isfinite(result).all():
        raise StepGeometryError("point-to-B-Rep distance evaluation failed")
    noise_floor = max(target.bbox.diagonal * 1e-12, 1e-9)
    result[result <= noise_floor] = 0.0
    return result


def brep_surface_distance_metrics(
    candidate: StepGeometry,
    target: StepGeometry,
    candidate_samples: SurfaceSamples,
    target_samples: SurfaceSamples,
    *,
    tolerance_mm: float,
) -> dict[str, Any]:
    """Symmetric distances from sampled points to the opposite exact boundary."""

    tolerance = _finite_positive(tolerance_mm, "surface tolerance")
    candidate_to_target = _point_to_boundary_distances(candidate_samples, target)
    target_to_candidate = _point_to_boundary_distances(target_samples, candidate)
    precision = float(np.mean(candidate_to_target <= tolerance))
    recall = float(np.mean(target_to_candidate <= tolerance))
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    candidate_hd95 = float(np.quantile(candidate_to_target, 0.95))
    target_hd95 = float(np.quantile(target_to_candidate, 0.95))
    chamfer = 0.5 * (
        float(candidate_to_target.mean()) + float(target_to_candidate.mean())
    )
    target_diagonal = max(target.bbox.diagonal, 1e-12)
    return {
        "surface_sampling_backend": BREP_SURFACE_SAMPLER,
        "surface_distance_backend": BREP_DISTANCE_BACKEND,
        "surface_meshing_used": False,
        "chamfer_mm": chamfer,
        "chamfer_normalized_by_target_diagonal": chamfer / target_diagonal,
        "candidate_to_target_mean_mm": float(candidate_to_target.mean()),
        "target_to_candidate_mean_mm": float(target_to_candidate.mean()),
        "surface_precision": precision,
        "surface_recall": recall,
        "surface_f1": f1,
        "surface_tolerance_mm": tolerance,
        "candidate_to_target_hd95_mm": candidate_hd95,
        "target_to_candidate_hd95_mm": target_hd95,
        "hd95_mm": max(candidate_hd95, target_hd95),
        "hd95_normalized_by_target_diagonal": max(candidate_hd95, target_hd95)
        / target_diagonal,
    }


def _relative_error(observed: float, expected: float) -> float | None:
    if abs(expected) <= 1e-15:
        return None
    return abs(observed - expected) / abs(expected)


def mass_property_errors(
    candidate: StepGeometry, target: StepGeometry
) -> dict[str, Any]:
    candidate_lengths = np.asarray(candidate.bbox.lengths, dtype=np.float64)
    target_lengths = np.asarray(target.bbox.lengths, dtype=np.float64)
    bbox_error = np.abs(candidate_lengths - target_lengths)
    centroid_error = np.asarray(candidate.centroid_mm) - np.asarray(target.centroid_mm)
    target_diagonal = max(target.bbox.diagonal, 1e-12)
    centroid_distance = float(np.linalg.norm(centroid_error))
    bbox_max_error = float(bbox_error.max())
    return {
        "volume_absolute_error_mm3": abs(candidate.volume_mm3 - target.volume_mm3),
        "volume_relative_error": _relative_error(
            candidate.volume_mm3, target.volume_mm3
        ),
        "surface_area_absolute_error_mm2": abs(
            candidate.surface_area_mm2 - target.surface_area_mm2
        ),
        "surface_area_relative_error": _relative_error(
            candidate.surface_area_mm2, target.surface_area_mm2
        ),
        "centroid_distance_mm": centroid_distance,
        "centroid_distance_normalized_by_target_diagonal": (
            centroid_distance / target_diagonal
        ),
        "centroid_axis_error_mm": {
            axis: float(abs(value))
            for axis, value in zip(("x", "y", "z"), centroid_error, strict=True)
        },
        "bbox_dimension_error_mm": {
            axis: float(value)
            for axis, value in zip(("x", "y", "z"), bbox_error, strict=True)
        },
        "bbox_dimension_max_error_mm": bbox_max_error,
        "bbox_dimension_max_error_normalized_by_target_diagonal": (
            bbox_max_error / target_diagonal
        ),
    }


def target_frozen_voxel_iou(
    candidate: StepGeometry,
    target: StepGeometry,
    *,
    resolution: int,
    margin_fraction: float,
) -> dict[str, Any]:
    """Measure IoU on a grid derived only from the frozen target geometry.

    A candidate extending beyond that grid receives zero strict IoU. The in-grid
    value is still retained for diagnostics, preventing unmeasured out-of-frame
    volume from inflating the public metric.
    """

    resolution = _positive_int(resolution, "voxel resolution")
    if resolution < 4:
        raise ValueError("voxel resolution must be at least 4")
    margin = float(margin_fraction)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("voxel margin fraction must be finite and non-negative")
    target_min = np.asarray(target.bbox.minimum, dtype=np.float64)
    target_max = np.asarray(target.bbox.maximum, dtype=np.float64)
    span = target_max - target_min
    padding = np.maximum(span * margin, np.maximum(span, 1.0) * 1e-6)
    grid_min = target_min - padding
    grid_max = target_max + padding
    step = (grid_max - grid_min) / resolution
    axes = [
        grid_min[index] + (np.arange(resolution, dtype=np.float64) + 0.5) * step[index]
        for index in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    target_occupancy = contains_points(target, grid)
    candidate_occupancy = contains_points(candidate, grid)
    intersection = int(np.count_nonzero(target_occupancy & candidate_occupancy))
    union = int(np.count_nonzero(target_occupancy | candidate_occupancy))
    in_grid_iou = 1.0 if union == 0 else intersection / union
    target_count = int(np.count_nonzero(target_occupancy))
    candidate_count = int(np.count_nonzero(candidate_occupancy))
    in_grid_precision = 1.0 if candidate_count == 0 else intersection / candidate_count
    in_grid_recall = 1.0 if target_count == 0 else intersection / target_count
    occupancy_sum = target_count + candidate_count
    in_grid_dice = 1.0 if occupancy_sum == 0 else 2.0 * intersection / occupancy_sum
    candidate_min = np.asarray(candidate.bbox.minimum, dtype=np.float64)
    candidate_max = np.asarray(candidate.bbox.maximum, dtype=np.float64)
    extent_epsilon = max(float(step.max()) * 1e-6, 1e-9)
    within_grid = bool(
        np.all(candidate_min >= grid_min - extent_epsilon)
        and np.all(candidate_max <= grid_max + extent_epsilon)
    )
    return {
        "voxel_iou": float(in_grid_iou if within_grid else 0.0),
        "in_grid_voxel_iou": float(in_grid_iou),
        "voxel_dice": float(in_grid_dice if within_grid else 0.0),
        "in_grid_voxel_dice": float(in_grid_dice),
        "in_grid_voxel_precision": float(in_grid_precision),
        "in_grid_voxel_recall": float(in_grid_recall),
        "candidate_within_frozen_grid": within_grid,
        "resolution_per_axis": resolution,
        "voxel_count": int(resolution**3),
        "intersection_voxel_count": intersection,
        "union_voxel_count": union,
        "target_occupied_voxel_count": target_count,
        "candidate_occupied_voxel_count": candidate_count,
        "grid_min_mm": grid_min.tolist(),
        "grid_max_mm": grid_max.tolist(),
        "cell_size_mm": step.tolist(),
        "margin_fraction": margin,
    }


def compare_geometry(
    candidate: StepGeometry,
    target: StepGeometry,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the complete nominal-reference L1 metric vector."""

    sample_count = _positive_int(
        profile.get("surface_sample_count"), "surface sample count"
    )
    seed = int(profile.get("seed", 0))
    candidate_samples = sample_brep_surface(
        candidate,
        count=sample_count,
        seed=seed,
    )
    target_samples = sample_brep_surface(
        target,
        count=sample_count,
        seed=seed + 1,
    )
    return {
        "metric_contract": "cadengbench-p.nominal-world-frame-geometry.v2",
        "alignment_applied": False,
        "sample_count_per_shape": sample_count,
        "candidate_surface_seed": seed,
        "target_surface_seed": seed + 1,
        **brep_surface_distance_metrics(
            candidate,
            target,
            candidate_samples,
            target_samples,
            tolerance_mm=profile.get("surface_tolerance_mm"),
        ),
        **target_frozen_voxel_iou(
            candidate,
            target,
            resolution=profile.get("voxel_resolution"),
            margin_fraction=profile.get("voxel_margin_fraction", 0.0),
        ),
        **mass_property_errors(candidate, target),
    }
