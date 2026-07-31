from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest


@pytest.fixture
def profile() -> dict[str, Any]:
    return {
        "schema_version": "cadengbench-p.l3-response-profile.v1",
        "profile_id": "test-response-v1",
        "status": "development_only",
        "source_intent_claimed": False,
        "material": {
            "model": "linear_elastic_isotropic",
            "youngs_modulus_mpa": 70000.0,
            "poisson_ratio": 0.33,
            "yield_strength_mpa": 250.0,
            "safety_factor": 2.0,
        },
        "calibration": {
            "load_control": "fixed_resultant_force",
            "target_reference_stress_utilization": 0.5,
            "target_reference_normalized_displacement": 0.01,
            "minimum_load_scale": 0.01,
            "maximum_load_scale": 1000.0,
            "maximum_traction_mpa": 100.0,
        },
        "fidelity": {
            "metrics": [
                "regularized_von_mises_p95_mpa",
                "normalized_max_displacement",
                "normalized_compliance",
            ],
            "maximum_symmetric_log_error": {
                "regularized_von_mises_p95_mpa": 0.22314355131420976,
                "normalized_max_displacement": 0.22314355131420976,
                "normalized_compliance": 0.22314355131420976,
            },
            "mesh_uncertainty_multiplier": 3.0,
            "component_score": "exp_negative_normalized_log_error",
            "aggregate": "geometric_mean",
        },
        "engineering_requirements": {
            "maximum_stress_utilization": 1.0,
            "maximum_normalized_displacement": 0.02,
            "maximum_candidate_reference_displacement_ratio": 1.25,
            "maximum_candidate_reference_compliance_ratio": 1.25,
        },
        "numerical": {
            "maximum_force_control_relative_error": 1e-5,
            "maximum_equilibrium_relative_error": 1e-5,
            "maximum_linearity_relative_error": 0.02,
            "ratio_epsilon": 1e-12,
        },
    }


@pytest.fixture
def card() -> dict[str, Any]:
    selector = {
        "kind": "axis_extreme_planar_surfaces_v2",
        "region_id": "axis:x:min",
        "axis": "x",
        "side": "min",
        "normal_alignment_min": 0.98,
        "plane_tolerance_relative": 1e-6,
        "minimum_count": 1,
        "maximum_count": 64,
        "minimum_area_fraction": 0.002,
    }
    qualification = {
        "schema_version": "cadengbench-p.l3-card-qualification.v1",
        "item_id": "p_benchcad_000001",
        "card_sha256": "1" * 64,
        "reference_step_sha256": "2" * 64,
        "result_sha256": "3" * 64,
        "evaluator_build_hash": "4" * 64,
        "runtime_lock_sha256": "5" * 64,
        "result_status": "probe_pass",
        "promoted": True,
        "source_probe_metrics": {
            "normalized_max_displacement": 1e-6,
            "regularized_von_mises_p95_mpa": 0.1,
            "regularized_von_mises_p99_mpa": 0.1,
            "stress_concentration_p95": 1.0,
            "normalized_compliance": 1.0,
            "force_equilibrium_relative_error": 1e-8,
        },
    }
    return {
        "schema_version": "cadengbench-p.l3-physics-card.v2",
        "card_id": "p_benchcad_000001:probe:v1",
        "item_id": "p_benchcad_000001",
        "status": "solver_qualified",
        "source_intent_claimed": False,
        "scenario_family": "axial_tension",
        "geometry": {"units": "mm", "expected_solid_count": 1},
        "selectors": {
            "fixed": selector,
            "loaded": {**selector, "region_id": "axis:x:max", "side": "max"},
        },
        "material": {
            "profile_id": "probe",
            "model": "linear_elastic_isotropic",
            "youngs_modulus_mpa": 70000.0,
            "poisson_ratio": 0.33,
        },
        "load": {
            "kind": "surface_traction_probe",
            "traction_vector_mpa": [0.1, 0.0, 0.0],
            "distribution": "boundary_triangle_consistent_nodal_v1",
        },
        "mesh": {
            "element_type": "C3D10",
            "algorithm_3d": 10,
            "high_order_optimization": "none",
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
        "checks": [
            "mesh_quality",
            "selector_stability",
            "solver_completion",
            "force_equilibrium",
            "finite_response",
            "small_deformation",
            "regularized_von_mises",
            "normalized_compliance",
        ],
        "qualification": qualification,
        "provenance": {
            "annotation_candidate_sha256": "a" * 64,
            "context_sha256": "b" * 64,
            "profile_sha256": "c" * 64,
            "annotator_model": "model",
            "annotator_revision": "revision",
        },
    }


def solver_result(
    *,
    stress: float = 62.5,
    displacement: float = 0.000625,
    compliance: float = 10.0,
    area: float = 100.0,
    force: list[float] | None = None,
    build_hash: str = "d" * 64,
    runtime_hash: str = "e" * 64,
) -> dict[str, Any]:
    return {
        "schema_version": "cadengbench-p.l3-probe-result.v1",
        "item_id": "p_benchcad_000001",
        "final_status": "probe_pass",
        "evaluator_build_hash": build_hash,
        "runtime_lock_sha256": runtime_hash,
        "stages": {
            "geometry": {"status": "passed"},
            "selectors": {"status": "passed"},
            "mesh": {"status": "passed", "loaded_area_mm2": area},
            "solver": {"status": "passed"},
            "result_sanity": {"status": "passed"},
        },
        "metrics": {
            "regularized_von_mises_mpa": {"p95": stress, "p99": stress * 1.1},
            "normalized_max_displacement": displacement,
            "normalized_compliance": compliance,
            "force_equilibrium_relative_error": 1e-8,
            "applied_force_n": force or [6250.0, 0.0, 0.0],
        },
    }


@pytest.fixture
def base_result() -> dict[str, Any]:
    return solver_result(
        stress=0.1,
        displacement=1e-6,
        compliance=10.0,
        area=100.0,
        force=[10.0, 0.0, 0.0],
    )


@pytest.fixture
def result_factory():
    return solver_result


@pytest.fixture
def deep_copy():
    return deepcopy
