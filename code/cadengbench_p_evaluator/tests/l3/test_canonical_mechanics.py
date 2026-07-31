from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cadengbench_p_eval.integrity import verify_evidence_manifest
from cadengbench_p_eval.io import sha256_file, write_json
from cadengbench_p_eval.physics.calculix import run_probe_card
from cadengbench_p_eval.physics.runner import qualify_physics_card

from .test_contracts import _card

pytestmark = [pytest.mark.integration, pytest.mark.l3]


def _requirements() -> tuple[object, Path]:
    gmsh = pytest.importorskip("gmsh")
    ccx = Path(os.environ.get("CADENGBENCH_CCX", ""))
    if not ccx.is_file():
        pytest.skip("CADENGBENCH_CCX does not identify a CalculiX executable")
    return gmsh, ccx


def test_cantilever_matches_euler_bernoulli_response(tmp_path: Path) -> None:
    _requirements()
    import cadquery as cq

    length = 100.0
    width = 10.0
    height = 10.0
    step = tmp_path / "cantilever.step"
    cq.exporters.export(
        cq.Workplane("YZ").rect(width, height).extrude(length), str(step)
    )
    card = _card()
    result = run_probe_card(
        card,
        step_path=step,
        output_dir=tmp_path / "run",
        ccx_path=_requirements()[1],
        timeout_seconds=180,
    )
    assert result["final_status"] == "probe_pass", result
    force = 0.1 * width * height
    inertia = width * height**3 / 12.0
    expected = force * length**3 / (3.0 * 70000.0 * inertia)
    observed = result["metrics"]["max_displacement_mm"]
    assert observed == pytest.approx(expected, rel=0.25)
    assert result["metrics"]["force_equilibrium_relative_error"] <= 1e-5


def test_axial_bar_matches_closed_form_extension(tmp_path: Path) -> None:
    _requirements()
    import cadquery as cq

    length = 100.0
    width = 10.0
    height = 10.0
    step = tmp_path / "axial.step"
    cq.exporters.export(
        cq.Workplane("YZ").rect(width, height).extrude(length), str(step)
    )
    card = _card()
    card["scenario_family"] = "axial_tension"
    card["load"]["traction_vector_mpa"] = [0.1, 0.0, 0.0]
    result = run_probe_card(
        card,
        step_path=step,
        output_dir=tmp_path / "run",
        ccx_path=_requirements()[1],
        timeout_seconds=180,
    )
    assert result["final_status"] == "probe_pass", result
    expected = 0.1 * length / 70000.0
    observed = result["metrics"]["max_displacement_mm"]
    assert observed == pytest.approx(expected, rel=0.15)
    assert result["metrics"]["applied_force_n"][0] == pytest.approx(10.0)
    assert result["stages"]["mesh"]["loaded_area_mm2"] == pytest.approx(100.0)
    assert result["metrics"]["regularized_von_mises_mpa"]["p95"] == pytest.approx(
        0.1, rel=0.15
    )


def test_axial_cylinder_normalizes_traction_to_exact_face_area(tmp_path: Path) -> None:
    _requirements()
    import cadquery as cq

    radius = 5.0
    step = tmp_path / "axial_cylinder.step"
    cq.exporters.export(cq.Workplane("YZ").circle(radius).extrude(100.0), str(step))
    card = _card()
    card["scenario_family"] = "axial_tension"
    card["load"]["traction_vector_mpa"] = [0.1, 0.0, 0.0]
    result = run_probe_card(
        card,
        step_path=step,
        output_dir=tmp_path / "run",
        ccx_path=_requirements()[1],
        timeout_seconds=180,
    )
    assert result["final_status"] == "probe_pass", result
    exact_area = 3.141592653589793 * radius**2
    assert result["stages"]["mesh"]["loaded_area_mm2"] == pytest.approx(
        exact_area, rel=1e-8
    )
    assert result["metrics"]["applied_force_n"][0] == pytest.approx(
        0.1 * exact_area, rel=1e-8
    )


def test_axial_visualization_preserves_quadratic_tetrahedra(tmp_path: Path) -> None:
    _requirements()
    vtk = pytest.importorskip("vtk")
    import cadquery as cq

    step = tmp_path / "axial.step"
    cq.exporters.export(cq.Workplane("YZ").rect(10.0, 10.0).extrude(100.0), str(step))
    card = _card()
    card["scenario_family"] = "axial_tension"
    card["load"]["traction_vector_mpa"] = [0.1, 0.0, 0.0]
    output = tmp_path / "run"
    result = run_probe_card(
        card,
        step_path=step,
        output_dir=output,
        ccx_path=_requirements()[1],
        timeout_seconds=180,
        render=True,
    )
    assert result["final_status"] == "probe_pass", result
    assert result["stages"]["visualization"]["status"] == "passed", result
    assert (output / "von_mises.png").stat().st_size > 10_000

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(output / "von_mises.vtu"))
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfCells() > 0
    assert grid.GetCellType(0) == vtk.VTK_QUADRATIC_TETRA
    assert grid.GetCell(0).GetNumberOfPoints() == 10


def test_multiple_volumes_fail_at_geometry_stage(tmp_path: Path) -> None:
    _requirements()
    import cadquery as cq

    first = cq.Workplane("XY").box(10.0, 10.0, 10.0).val()
    second = cq.Workplane("XY").box(10.0, 10.0, 10.0).translate((30.0, 0.0, 0.0)).val()
    step = tmp_path / "two_volumes.step"
    cq.exporters.export(cq.Compound.makeCompound([first, second]), str(step))
    result = run_probe_card(
        _card(),
        step_path=step,
        output_dir=tmp_path / "run",
        ccx_path=_requirements()[1],
        timeout_seconds=180,
    )
    assert result["final_status"] == "failed"
    assert result["failure"]["stage"] == "geometry"
    assert result["failure"]["owner"] == "model"


def test_missing_selector_region_fails_at_selector_stage(tmp_path: Path) -> None:
    _requirements()
    import cadquery as cq

    step = tmp_path / "box.step"
    cq.exporters.export(cq.Workplane("YZ").rect(10.0, 10.0).extrude(100.0), str(step))
    card = _card()
    card["selectors"]["fixed"]["minimum_count"] = 2
    result = run_probe_card(
        card,
        step_path=step,
        output_dir=tmp_path / "run",
        ccx_path=_requirements()[1],
        timeout_seconds=180,
    )
    assert result["final_status"] == "failed"
    assert result["failure"]["stage"] == "selectors"
    assert result["failure"]["owner"] == "model"


def test_small_deformation_limit_produces_requirement_failure(tmp_path: Path) -> None:
    _requirements()
    import cadquery as cq

    step = tmp_path / "cantilever.step"
    cq.exporters.export(cq.Workplane("YZ").rect(10.0, 10.0).extrude(100.0), str(step))
    card = _card()
    card["analysis"]["max_normalized_displacement"] = 1e-10
    result = run_probe_card(
        card,
        step_path=step,
        output_dir=tmp_path / "run",
        ccx_path=_requirements()[1],
        timeout_seconds=180,
    )
    assert result["final_status"] == "requirement_fail"
    assert result["stages"]["requirements"]["status"] == "failed"
    assert result["checks"]["small_deformation"]["passed"] is False


def test_source_qualification_embeds_reference_response(tmp_path: Path) -> None:
    _requirements()
    import cadquery as cq

    step = tmp_path / "source.step"
    cq.exporters.export(cq.Workplane("YZ").rect(10.0, 10.0).extrude(100.0), str(step))
    card = _card()
    card["scenario_family"] = "axial_tension"
    card["load"]["traction_vector_mpa"] = [0.1, 0.0, 0.0]
    card_path = tmp_path / "card.json"
    promoted_path = tmp_path / "promoted.json"
    write_json(card_path, card)
    qualification = qualify_physics_card(
        card_path=card_path,
        reference_step=step,
        output_dir=tmp_path / "qualification",
        promoted_card_path=promoted_path,
        ccx_path=_requirements()[1],
        timeout_seconds=180,
    )
    assert qualification["promoted"] is True
    promoted = json.loads(promoted_path.read_text(encoding="utf-8"))
    assert promoted["status"] == "solver_qualified"
    assert promoted["qualification"]["source_probe_metrics"][
        "regularized_von_mises_p95_mpa"
    ] == pytest.approx(0.1, rel=0.15)
    qualification_root = tmp_path / "qualification"
    verify_evidence_manifest(qualification_root)
    assert (qualification_root / "result_target.txt").read_text(
        encoding="ascii"
    ).strip() == "result.json"
    assert (qualification_root / "result.sha256").read_text(
        encoding="ascii"
    ).strip() == sha256_file(qualification_root / "result.json")
