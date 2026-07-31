"""Deterministic source geometry and support-region inspection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cadquery as cq

from cadengbench_p_l3_extended.io import read_json, sha256_file

AXES = ("x", "y", "z")


class GeometryAuditError(RuntimeError):
    """A source item cannot be inspected deterministically."""


def source_step(repo_root: Path, item_id: str) -> Path:
    item_root = (
        repo_root
        / "data/production/cadengbench_p_1800_v2_two_source_r25/dataset/items"
        / item_id
    )
    consumer = read_json(item_root / "benchmark/consumer_ready_item_v2.json")
    source = consumer["source"]
    if source["source_key"] == "fusion360":
        path = item_root / "source/fusion360_occ_reconstruction.step"
    else:
        path = repo_root / source["canonical_geometry_artifact"]
    if not path.is_file():
        raise GeometryAuditError(f"missing canonical STEP for {item_id}: {path}")
    return path


def _vector(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def inspect_geometry(
    step_path: Path,
    *,
    normal_alignment_min: float,
    plane_tolerance_relative: float,
    minimum_region_area_fraction: float,
    maximum_region_surface_count: int,
) -> dict[str, Any]:
    workplane = cq.importers.importStep(str(step_path))
    solids = workplane.solids().vals()
    faces = workplane.faces().vals()
    if not solids:
        raise GeometryAuditError(f"STEP contains no solids: {step_path}")
    combined = workplane if len(solids) == 1 else workplane.combine()
    bbox = combined.val().BoundingBox()
    lower = [float(bbox.xmin), float(bbox.ymin), float(bbox.zmin)]
    upper = [float(bbox.xmax), float(bbox.ymax), float(bbox.zmax)]
    lengths = [upper[index] - lower[index] for index in range(3)]
    diagonal = math.sqrt(sum(length * length for length in lengths))
    tolerance = max(diagonal * plane_tolerance_relative, 1e-7)
    total_area = sum(float(face.Area()) for face in faces)
    total_volume = sum(float(solid.Volume()) for solid in solids)
    regions: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(AXES):
        for side, target in (("min", lower[axis_index]), ("max", upper[axis_index])):
            matches = []
            for face_index, face in enumerate(faces):
                if face.geomType() != "PLANE":
                    continue
                normal = _vector(face.normalAt())
                if abs(normal[axis_index]) < normal_alignment_min:
                    continue
                face_bbox = face.BoundingBox()
                face_lower = [face_bbox.xmin, face_bbox.ymin, face_bbox.zmin]
                face_upper = [face_bbox.xmax, face_bbox.ymax, face_bbox.zmax]
                if (
                    abs(float(face_lower[axis_index]) - target) > tolerance
                    or abs(float(face_upper[axis_index]) - target) > tolerance
                ):
                    continue
                matches.append(
                    {
                        "face_index_diagnostic_only": face_index,
                        "area_mm2": float(face.Area()),
                        "centroid_mm": _vector(face.Center()),
                        "normal": normal,
                    }
                )
            if not matches:
                continue
            area = sum(float(match["area_mm2"]) for match in matches)
            area_fraction = area / total_area if total_area else 0.0
            regions.append(
                {
                    "region_id": f"axis:{axis}:{side}",
                    "axis": axis,
                    "side": side,
                    "surface_count": len(matches),
                    "area_mm2": area,
                    "area_fraction": area_fraction,
                    "qualified_single_support": (
                        area_fraction >= minimum_region_area_fraction
                        and len(matches) <= maximum_region_surface_count
                    ),
                    "surfaces": matches,
                }
            )
    qualified = [
        region for region in regions if bool(region["qualified_single_support"])
    ]
    qualified.sort(
        key=lambda region: (
            -float(region["area_fraction"]),
            str(region["region_id"]),
        )
    )
    return {
        "step_sha256": sha256_file(step_path),
        "solid_count": len(solids),
        "face_count": len(faces),
        "surface_area_mm2": total_area,
        "volume_mm3": total_volume,
        "bbox_mm": {
            "minimum": lower,
            "maximum": upper,
            "lengths": lengths,
            "diagonal": diagonal,
        },
        "axis_extreme_planar_regions": regions,
        "qualified_single_support_regions": qualified,
    }


def extension_candidate(
    geometry: Mapping[str, Any],
    *,
    existing_solver_qualified: bool,
) -> tuple[bool, str]:
    if existing_solver_qualified:
        return False, "already_solver_qualified_v1"
    if int(geometry["solid_count"]) != 1:
        return False, "solid_count_not_one"
    regions = geometry["qualified_single_support_regions"]
    if not regions:
        return False, "no_qualified_single_support_region"
    return True, "single_support_body_acceleration_candidate"
