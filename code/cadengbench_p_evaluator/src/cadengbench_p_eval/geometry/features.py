"""Certified geometric feature recognizers used by L1 and DFM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .step import StepGeometry

HOLE_EXTRACTOR_VERSION = "internal-near-full-cylinder.v1"


def _vector(value: Any) -> np.ndarray:
    return np.asarray((float(value.x), float(value.y), float(value.z)))


def _unit(value: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(value))
    if magnitude <= 1e-12:
        raise ValueError("zero-length direction")
    return value / magnitude


def _geometry_type(entity: Any) -> str:
    try:
        return str(entity.geomType()).upper()
    except Exception:
        return "UNKNOWN"


def _stable_face_id(payload: dict[str, Any], duplicate: int) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    suffix = hashlib.sha256(encoded).hexdigest()[:12]
    return f"face_{suffix}" if duplicate == 1 else f"face_{suffix}_{duplicate}"


@dataclass(frozen=True)
class CylindricalFace:
    face_id: str
    diameter_mm: float
    depth_mm: float
    depth_to_diameter: float
    axis_direction: tuple[float, float, float]
    axis_location_mm: tuple[float, float, float]
    center_mm: tuple[float, float, float]
    internal: bool
    near_full_cylinder: bool
    circular_edge_count: int
    is_hole: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["axis_direction"] = list(self.axis_direction)
        value["axis_location_mm"] = list(self.axis_location_mm)
        value["center_mm"] = list(self.center_mm)
        value["extractor_version"] = HOLE_EXTRACTOR_VERSION
        return value


@dataclass(frozen=True)
class CylindricalFeatureInventory:
    assessments: tuple[CylindricalFace, ...]

    @property
    def holes(self) -> tuple[CylindricalFace, ...]:
        return tuple(value for value in self.assessments if value.is_hole)

    def as_dict(self) -> dict[str, Any]:
        return {
            "extractor_version": HOLE_EXTRACTOR_VERSION,
            "hole_count": len(self.holes),
            "holes": [value.as_dict() for value in self.holes],
            "all_cylindrical_face_assessments": [
                value.as_dict() for value in self.assessments
            ],
            "limitations": [
                "Blind holes without two circular boundary edges may be missed.",
                "Intersecting, threaded, or non-cylindrical holes are not certified.",
            ],
        }


def extract_cylindrical_features(
    geometry: StepGeometry,
) -> CylindricalFeatureInventory:
    """Recognize unambiguous internal, near-complete cylindrical openings."""

    measured: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for face in geometry.shape.Faces():
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
                    ),
                    dtype=np.float64,
                )
            )
            location = np.asarray(
                (axis.Location().X(), axis.Location().Y(), axis.Location().Z()),
                dtype=np.float64,
            )
            u_min, u_max, v_min, v_max = face.uvBounds()
            normal, point = face.normalAt(
                (float(u_min) + float(u_max)) / 2.0,
                (float(v_min) + float(v_max)) / 2.0,
            )
            point_value = _vector(point)
            normal_value = _unit(_vector(normal))
            along = float(np.dot(point_value - location, direction))
            axis_point = location + along * direction
            radial = _unit(point_value - axis_point)
            internal = bool(float(np.dot(normal_value, radial)) < -0.5)
            projections = [
                float(np.dot(_vector(vertex.Center()) - location, direction))
                for vertex in face.Vertices()
            ]
            depth = (
                max(projections) - min(projections)
                if len(projections) >= 2
                else abs(float(v_max) - float(v_min))
            )
            radius = float(cylinder.Radius())
            diameter = 2.0 * radius
            angular_span = abs(float(u_max) - float(u_min))
            circular_edges = sum(
                _geometry_type(edge) == "CIRCLE" for edge in face.Edges()
            )
            near_full = angular_span >= 5.5
            is_hole = bool(
                internal and near_full and circular_edges >= 2 and depth > 1e-8
            )
            center = _vector(face.Center())
            signature = {
                "diameter_mm": round(diameter, 9),
                "depth_mm": round(abs(depth), 9),
                "axis_direction": [round(float(value), 9) for value in direction],
                "axis_location_mm": [round(float(value), 9) for value in location],
                "center_mm": [round(float(value), 9) for value in center],
                "internal": internal,
                "near_full_cylinder": near_full,
                "circular_edge_count": circular_edges,
            }
            measured.append(
                (
                    signature,
                    {
                        "diameter_mm": diameter,
                        "depth_mm": abs(depth),
                        "axis_direction": tuple(float(value) for value in direction),
                        "axis_location_mm": tuple(float(value) for value in location),
                        "center_mm": tuple(float(value) for value in center),
                        "internal": internal,
                        "near_full_cylinder": near_full,
                        "circular_edge_count": circular_edges,
                        "is_hole": is_hole,
                    },
                )
            )
        except Exception:
            continue
    measured.sort(key=lambda value: json.dumps(value[0], sort_keys=True))
    duplicate_counts: dict[str, int] = {}
    results: list[CylindricalFace] = []
    for signature, values in measured:
        digest = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()[:12]
        duplicate_counts[digest] = duplicate_counts.get(digest, 0) + 1
        diameter = float(values["diameter_mm"])
        results.append(
            CylindricalFace(
                face_id=_stable_face_id(signature, duplicate_counts[digest]),
                depth_to_diameter=float(values["depth_mm"]) / diameter,
                **values,
            )
        )
    return CylindricalFeatureInventory(assessments=tuple(results))
