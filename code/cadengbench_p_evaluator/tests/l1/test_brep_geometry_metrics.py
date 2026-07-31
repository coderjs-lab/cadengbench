from pathlib import Path

import cadquery as cq
import pytest

from cadengbench_p_eval.geometry.metrics import compare_geometry
from cadengbench_p_eval.geometry.step import load_step

PROFILE = {
    "linear_tessellation_tolerance_mm": 0.2,
    "surface_sample_count": 256,
    "surface_tolerance_mm": 0.1,
    "voxel_resolution": 32,
    "voxel_margin_fraction": 0.05,
    "seed": 17,
    "hard_gates": {"voxel_iou_min": None, "surface_f1_min": None},
}


def _export(tmp_path: Path, name: str, shape: cq.Workplane) -> Path:
    path = tmp_path / f"{name}.step"
    cq.exporters.export(shape, str(path))
    return path


def test_brep_metrics_do_not_invoke_tessellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = load_step(_export(tmp_path, "box", cq.Workplane("XY").box(10, 8, 2)))

    def reject_tessellation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("L1 reference metrics must not tessellate")

    monkeypatch.setattr(cq.Shape, "tessellate", reject_tessellation)
    result = compare_geometry(geometry, geometry, PROFILE)

    assert result["surface_meshing_used"] is False
    assert result["surface_f1"] == 1.0
    assert result["chamfer_mm"] == 0.0


def test_brep_metrics_accept_independent_equivalent_exports(tmp_path: Path) -> None:
    shape = cq.Workplane("XY").box(20, 12, 4).faces(">Z").workplane().hole(3)
    first = load_step(_export(tmp_path, "first", shape))
    second = load_step(_export(tmp_path, "second", shape))

    result = compare_geometry(first, second, PROFILE)

    assert result["metric_contract"] == "cadengbench-p.nominal-world-frame-geometry.v2"
    assert result["surface_f1"] == 1.0
    assert result["chamfer_mm"] == 0.0
    assert result["voxel_iou"] == 1.0


def test_brep_metrics_support_curved_trimmed_surfaces(tmp_path: Path) -> None:
    shape = cq.Workplane("XY").cylinder(12, 5).faces(">Z").workplane().hole(2)
    first = load_step(_export(tmp_path, "curved_first", shape))
    second = load_step(_export(tmp_path, "curved_second", shape))

    result = compare_geometry(first, second, PROFILE)

    assert result["surface_meshing_used"] is False
    assert result["surface_f1"] == 1.0
    assert result["chamfer_mm"] == 0.0


def test_brep_metrics_detect_dimensional_mutation(tmp_path: Path) -> None:
    target = load_step(_export(tmp_path, "target", cq.Workplane("XY").box(10, 8, 2)))
    candidate = load_step(
        _export(tmp_path, "candidate", cq.Workplane("XY").box(11, 8, 2))
    )

    result = compare_geometry(candidate, target, PROFILE)

    assert result["surface_f1"] < 1.0
    assert result["chamfer_mm"] > 0.0
    assert result["hd95_mm"] > 0.0
    assert result["voxel_iou"] < 1.0
    assert result["voxel_dice"] < 1.0
