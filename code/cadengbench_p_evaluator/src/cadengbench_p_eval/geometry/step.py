"""Source-neutral STEP loading and deterministic B-Rep facts."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class StepGeometryError(ValueError):
    """A STEP artifact cannot provide valid solid geometry."""


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _vector(value: Any) -> tuple[float, float, float]:
    return (float(value.x), float(value.y), float(value.z))


def _geometry_type(entity: Any) -> str:
    try:
        return str(entity.geomType()).upper()
    except Exception:
        return "UNKNOWN"


@dataclass(frozen=True)
class BoundingBox:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def lengths(self) -> tuple[float, float, float]:
        return tuple(
            upper - lower
            for lower, upper in zip(self.minimum, self.maximum, strict=True)
        )

    @property
    def diagonal(self) -> float:
        return math.sqrt(sum(length * length for length in self.lengths))

    def as_dict(self) -> dict[str, float]:
        xlen, ylen, zlen = self.lengths
        return {
            "xmin": self.minimum[0],
            "ymin": self.minimum[1],
            "zmin": self.minimum[2],
            "xmax": self.maximum[0],
            "ymax": self.maximum[1],
            "zmax": self.maximum[2],
            "xlen": xlen,
            "ylen": ylen,
            "zlen": zlen,
            "diagonal": self.diagonal,
        }


@dataclass(frozen=True)
class StepGeometry:
    path: Path
    shape: cq.Shape
    solids: tuple[cq.Shape, ...]
    bbox: BoundingBox
    volume_mm3: float
    surface_area_mm2: float
    centroid_mm: tuple[float, float, float]
    topology: dict[str, Any]

    def facts(self) -> dict[str, Any]:
        return {
            "units": {"length": "mm", "area": "mm^2", "volume": "mm^3"},
            "bbox": self.bbox.as_dict(),
            "volume_mm3": self.volume_mm3,
            "surface_area_mm2": self.surface_area_mm2,
            "centroid_mm": list(self.centroid_mm),
            "topology": dict(self.topology),
        }


@dataclass(frozen=True)
class TriangleMesh:
    triangles: FloatArray
    normals: FloatArray
    areas: FloatArray

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


def _mass_properties(
    solids: Iterable[cq.Shape],
) -> tuple[float, float, tuple[float, float, float]]:
    volume = 0.0
    area = 0.0
    weighted_center = np.zeros(3, dtype=np.float64)
    for solid in solids:
        if not bool(solid.isValid()):
            raise StepGeometryError("STEP contains an invalid solid")
        solid_volume = abs(float(solid.Volume()))
        solid_area = float(solid.Area())
        center = np.asarray(_vector(solid.Center()), dtype=np.float64)
        if (
            not _finite_positive(solid_volume)
            or not _finite_positive(solid_area)
            or not np.isfinite(center).all()
        ):
            raise StepGeometryError("STEP has invalid mass properties")
        volume += solid_volume
        area += solid_area
        weighted_center += solid_volume * center
    if not _finite_positive(volume) or not _finite_positive(area):
        raise StepGeometryError("STEP has non-positive aggregate mass properties")
    centroid = weighted_center / volume
    return volume, area, tuple(float(value) for value in centroid)


def load_step(path: Path) -> StepGeometry:
    """Import a millimetre STEP artifact without source-family assumptions."""

    path = Path(path).resolve()
    if not path.is_file():
        raise StepGeometryError(f"STEP artifact does not exist: {path}")
    try:
        shape = cq.importers.importStep(str(path)).val()
    except Exception as exc:
        raise StepGeometryError(
            f"STEP import failed: {type(exc).__name__}: {exc}"
        ) from exc
    if shape is None:
        raise StepGeometryError("STEP import produced no shape")
    solids = tuple(shape.Solids())
    if not solids:
        raise StepGeometryError("STEP contains no solids")
    volume, area, centroid = _mass_properties(solids)
    try:
        raw_bbox = shape.BoundingBox()
        bbox = BoundingBox(
            minimum=(float(raw_bbox.xmin), float(raw_bbox.ymin), float(raw_bbox.zmin)),
            maximum=(float(raw_bbox.xmax), float(raw_bbox.ymax), float(raw_bbox.zmax)),
        )
    except Exception as exc:
        raise StepGeometryError(f"STEP bounding box failed: {exc}") from exc
    if not all(math.isfinite(value) for value in (*bbox.minimum, *bbox.maximum)):
        raise StepGeometryError("STEP bounding box contains non-finite values")
    faces = list(shape.Faces())
    edges = list(shape.Edges())
    topology = {
        "solid_count": len(solids),
        "shell_count": len(shape.Shells()),
        "face_count": len(faces),
        "edge_count": len(edges),
        "vertex_count": len(shape.Vertices()),
        "is_valid": bool(shape.isValid()) and all(solid.isValid() for solid in solids),
        "face_type_counts": dict(sorted(Counter(map(_geometry_type, faces)).items())),
        "edge_type_counts": dict(sorted(Counter(map(_geometry_type, edges)).items())),
    }
    if topology["is_valid"] is not True:
        raise StepGeometryError("STEP aggregate topology is invalid")
    return StepGeometry(
        path=path,
        shape=shape,
        solids=solids,
        bbox=bbox,
        volume_mm3=volume,
        surface_area_mm2=area,
        centroid_mm=centroid,
        topology=topology,
    )


def tessellate(geometry: StepGeometry, tolerance_mm: float) -> TriangleMesh:
    """Tessellate every solid in world coordinates at a frozen deflection."""

    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0.0:
        raise ValueError("tessellation tolerance must be finite and positive")
    triangle_blocks: list[FloatArray] = []
    for solid in geometry.solids:
        try:
            vertices, indices = solid.tessellate(tolerance_mm)
        except Exception as exc:
            raise StepGeometryError(f"STEP tessellation failed: {exc}") from exc
        points = np.asarray([_vector(vertex) for vertex in vertices], dtype=np.float64)
        if not len(points) or not len(indices):
            continue
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.ndim != 2 or index_array.shape[1] != 3:
            raise StepGeometryError("tessellation returned malformed triangles")
        triangle_blocks.append(points[index_array])
    if not triangle_blocks:
        raise StepGeometryError("STEP tessellation produced no triangles")
    triangles = np.concatenate(triangle_blocks, axis=0)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    doubled_areas = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(doubled_areas) & (doubled_areas > 1e-14)
    triangles = triangles[valid]
    cross = cross[valid]
    doubled_areas = doubled_areas[valid]
    if not len(triangles):
        raise StepGeometryError("STEP tessellation contains only degenerate triangles")
    normals = cross / doubled_areas[:, None]
    return TriangleMesh(
        triangles=triangles,
        normals=normals,
        areas=0.5 * doubled_areas,
    )


def contains_points(
    geometry: StepGeometry,
    points: FloatArray,
    *,
    tolerance_mm: float = 1e-7,
) -> NDArray[np.bool_]:
    """Classify world-frame points against the imported solid set."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    result = np.zeros(len(values), dtype=np.bool_)
    for index, point in enumerate(values):
        vector = cq.Vector(float(point[0]), float(point[1]), float(point[2]))
        result[index] = any(
            bool(solid.isInside(vector, tolerance_mm)) for solid in geometry.solids
        )
    return result
