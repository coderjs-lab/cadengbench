import copy

import pytest

from cadengbench_p_l3_extended.contracts import ContractError, validate_body_card


def _card() -> dict:
    return {
        "schema_version": "cadengbench-p.l3-body-acceleration-card.v1",
        "card_id": "p_benchcad_000001:body-acceleration:v1",
        "item_id": "p_benchcad_000001",
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
            "maximum_count": 64,
            "minimum_area_fraction": 0.002,
        },
        "material": {
            "profile_id": "probe",
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
            "max_equilibrium_relative_error": 0.001,
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


def test_valid_body_card() -> None:
    validate_body_card(_card())


def test_acceleration_must_point_away_from_support() -> None:
    card = copy.deepcopy(_card())
    card["load"]["acceleration_vector_mm_s2"][0] *= -1
    with pytest.raises(ContractError, match="away"):
        validate_body_card(card)
