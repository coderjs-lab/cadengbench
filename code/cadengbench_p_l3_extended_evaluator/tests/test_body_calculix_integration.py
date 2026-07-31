from __future__ import annotations

import math
import os
from pathlib import Path

import cadquery as cq
import pytest

from cadengbench_p_l3_extended.body_solver import run_body_card


def _bar_card() -> dict:
    return {
        "schema_version": "cadengbench-p.l3-body-acceleration-card.v1",
        "card_id": "p_benchcad_999999:canonical-body-acceleration:v1",
        "item_id": "p_benchcad_999999",
        "status": "geometry_qualified",
        "source_intent_claimed": False,
        "scenario_family": "single_support_body_acceleration",
        "geometry": {"units": "mm", "expected_solid_count": 1},
        "selector": {
            "kind": "axis_extreme_planar_surfaces_v2",
            "region_id": "axis:x:min",
            "axis": "x",
            "side": "min",
            "normal_alignment_min": 0.98,
            "plane_tolerance_relative": 1e-6,
            "minimum_count": 1,
            "maximum_count": 4,
            "minimum_area_fraction": 0.001,
        },
        "material": {
            "profile_id": "canonical-70gpa-density",
            "model": "linear_elastic_isotropic",
            "youngs_modulus_mpa": 70000.0,
            "poisson_ratio": 0.33,
            "density_tonne_per_mm3": 2.7e-9,
        },
        "load": {
            "kind": "body_acceleration_probe",
            "acceleration_vector_mm_s2": [9810000.0, 0.0, 0.0],
            "distribution": "c3d10_consistent_nodal_body_force_v1",
            "control": "fixed_acceleration",
        },
        "mesh": {
            "element_type": "C3D10",
            "algorithm_3d": 10,
            "second_order_linear": True,
            "target_long_axis_divisions": 24,
            "target_short_axis_divisions": 5,
            "minimum_sicn": 0.02,
            "maximum_elements": 250000,
        },
        "analysis": {
            "type": "linear_static",
            "small_deformation": True,
            "max_equilibrium_relative_error": 1e-5,
            "max_normalized_displacement": 0.02,
            "stress_exclusion_mesh_lengths": 2.0,
            "regularized_stress_percentiles": [95.0, 99.0],
        },
        "scale_laws": {
            "resultant_force_exponent": 3.0,
            "regularized_stress_exponent": 1.0,
            "max_displacement_exponent": 2.0,
            "normalized_displacement_exponent": 1.0,
        },
        "provenance": {
            "audit_sha256": "a" * 64,
            "profile_sha256": "b" * 64,
            "annotation_method": (
                "deterministic_largest_axis_extreme_planar_support_v1"
            ),
        },
    }


@pytest.mark.integration
def test_axial_body_force_matches_closed_form(tmp_path: Path) -> None:
    ccx = Path(os.environ.get("CADENGBENCH_CCX", ""))
    if not ccx.is_file():
        pytest.skip("CalculiX is unavailable")
    length = 100.0
    step = tmp_path / "bar.step"
    cq.exporters.export(cq.Workplane("XY").box(length, 10.0, 10.0), str(step))
    result = run_body_card(
        _bar_card(),
        step_path=step,
        output_dir=tmp_path / "run",
        ccx_path=ccx,
        timeout_seconds=300,
    )
    assert result["final_status"] == "probe_pass"
    density = 2.7e-9
    acceleration = 9810000.0
    youngs_modulus = 70000.0
    expected = density * acceleration * length**2 / (2.0 * youngs_modulus)
    observed = float(result["metrics"]["max_displacement_mm"])
    assert math.isclose(observed, expected, rel_tol=0.05)
    assert result["metrics"]["force_equilibrium_relative_error"] <= 1e-5
