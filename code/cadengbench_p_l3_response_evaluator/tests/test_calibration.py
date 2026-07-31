from __future__ import annotations

import math

import pytest

from cadengbench_p_l3_response.calibration import compile_response_contract
from cadengbench_p_l3_response.io import sha256_json
from cadengbench_p_l3_response.schemas import validate_document
from cadengbench_p_l3_response.solver import force_controlled_card


def _contract(profile, card, base_result):
    return compile_response_contract(
        profile=profile,
        profile_sha256="a" * 64,
        card=card,
        card_sha256="b" * 64,
        nominal_reference_step_sha256="c" * 64,
        reference_result=base_result,
        reference_result_sha256="f" * 64,
    )


def test_reference_calibration_targets_half_allowable(profile, card, base_result):
    contract = _contract(profile, card, base_result)
    assert contract["material"]["allowable_stress_mpa"] == 125.0
    assert contract["loading"]["controlling_constraint"] == "stress_utilization"
    assert contract["loading"]["load_scale"] == pytest.approx(625.0)
    assert contract["loading"]["target_resultant_force_n"] == pytest.approx(
        [6250.0, 0.0, 0.0]
    )
    assert contract["reference_base"]["predicted_scaled_metrics"][
        "regularized_von_mises_p95_mpa"
    ] == pytest.approx(62.5)
    body = dict(contract)
    digest = body.pop("contract_sha256")
    assert digest == sha256_json(body)
    validate_document("contract", contract)


def test_calibrated_profile_marks_frozen_threshold_authority(
    profile, card, base_result
):
    profile["status"] = "calibrated"
    contract = _contract(profile, card, base_result)
    assert contract["status"] == "calibrated"
    assert (
        contract["threshold_authority"]
        == "benchmark_assigned_reference_relative_calibrated_v1"
    )
    validate_document("contract", contract)


def test_mesh_uncertainty_can_expand_fidelity_limit(profile, card, base_result):
    mesh_study = {
        "comparisons_to_fine": {
            "production": {
                metric: {"observed": 1.0, "fine": 2.0}
                for metric in (
                    "regularized_von_mises_p95_mpa",
                    "normalized_max_displacement",
                    "normalized_compliance",
                )
            }
        }
    }
    contract = compile_response_contract(
        profile=profile,
        profile_sha256="a" * 64,
        card=card,
        card_sha256="b" * 64,
        nominal_reference_step_sha256="c" * 64,
        reference_result=base_result,
        reference_result_sha256="f" * 64,
        mesh_study=mesh_study,
        mesh_study_sha256="9" * 64,
    )
    expected = 3.0 * math.log(2.0)
    assert contract["fidelity"]["effective_maximum_symmetric_log_error"][
        "normalized_compliance"
    ] == pytest.approx(expected)


def test_geometry_specific_traction_preserves_total_force(profile, card, base_result):
    contract = _contract(profile, card, base_result)
    controlled = force_controlled_card(
        card,
        target_force_n=contract["loading"]["target_resultant_force_n"],
        loaded_area_mm2=50.0,
        maximum_equilibrium_relative_error=5e-5,
        role="candidate",
        variant_id="nominal",
        contract_sha256=contract["contract_sha256"],
    )
    assert controlled["status"] == "geometry_qualified"
    assert "qualification" not in controlled
    assert controlled["load"]["traction_vector_mpa"] == pytest.approx([125.0, 0.0, 0.0])
    assert controlled["analysis"]["max_equilibrium_relative_error"] == 5e-5
