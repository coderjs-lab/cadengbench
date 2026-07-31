from __future__ import annotations

from cadengbench_p_l3_response.calibration import compile_response_contract
from cadengbench_p_l3_response.provenance import evaluator_identity
from cadengbench_p_l3_response.scoring import (
    failed_solver_pair_result,
    score_response_pair,
    skipped_pair_result,
)


def _contract(profile, card, base_result):
    base_result["evaluator_build_hash"] = "d" * 64
    base_result["runtime_lock_sha256"] = "e" * 64
    contract = compile_response_contract(
        profile=profile,
        profile_sha256="a" * 64,
        card=card,
        card_sha256="b" * 64,
        nominal_reference_step_sha256="c" * 64,
        reference_result=base_result,
        reference_result_sha256="f" * 64,
    )
    contract["evaluator"] = evaluator_identity()
    from cadengbench_p_l3_response.io import sha256_json

    body = dict(contract)
    body.pop("contract_sha256")
    contract["contract_sha256"] = sha256_json(body)
    return contract


def test_exact_response_passes(profile, card, base_result, result_factory):
    contract = _contract(profile, card, base_result)
    reference = result_factory()
    candidate = result_factory()
    result = score_response_pair(
        contract,
        reference_result=reference,
        candidate_result=candidate,
        variant_id="nominal",
        runtime_seconds=1.0,
        nominal_reference=True,
    )
    assert result["final_verdict"] == "pass"
    assert result["response_fidelity"]["aggregate_score"] == 1.0
    assert result["engineering_requirements"]["status"] == "passed"


def test_weakened_candidate_fails_engineering_requirements(
    profile, card, base_result, result_factory
):
    contract = _contract(profile, card, base_result)
    reference = result_factory()
    candidate = result_factory(stress=150.0, displacement=0.012, compliance=18.0)
    result = score_response_pair(
        contract,
        reference_result=reference,
        candidate_result=candidate,
        variant_id="nominal",
        runtime_seconds=1.0,
        nominal_reference=True,
    )
    assert result["final_verdict"] == "requirement_fail"
    assert result["margins"]["stress_mpa"] == -25.0
    assert (
        result["engineering_requirements"]["checks"][
            "candidate_reference_compliance_ratio"
        ]["passed"]
        is False
    )


def test_stronger_but_different_candidate_fails_fidelity_only(
    profile, card, base_result, result_factory
):
    contract = _contract(profile, card, base_result)
    reference = result_factory()
    candidate = result_factory(stress=40.0, displacement=0.0003, compliance=7.0)
    result = score_response_pair(
        contract,
        reference_result=reference,
        candidate_result=candidate,
        variant_id="nominal",
        runtime_seconds=1.0,
        nominal_reference=True,
    )
    assert result["engineering_requirements"]["status"] == "passed"
    assert result["response_fidelity"]["status"] == "failed"
    assert result["final_verdict"] == "response_fidelity_fail"


def test_force_mismatch_is_numerical_failure(
    profile, card, base_result, result_factory
):
    contract = _contract(profile, card, base_result)
    candidate = result_factory(force=[6000.0, 0.0, 0.0])
    result = score_response_pair(
        contract,
        reference_result=result_factory(),
        candidate_result=candidate,
        variant_id="nominal",
        runtime_seconds=1.0,
        nominal_reference=True,
    )
    assert result["final_verdict"] == "numerical_validity_fail"


def test_skipped_and_mesh_failure_remain_typed(profile, card, base_result):
    contract = _contract(profile, card, base_result)
    skipped = skipped_pair_result(contract, variant_id="v1", upstream_status="failed")
    assert skipped["final_verdict"] == "skipped_upstream"
    failed = failed_solver_pair_result(
        contract,
        variant_id="v2",
        result_evidence={
            "stages": {
                "geometry": {"status": "passed"},
                "selectors": {"status": "passed"},
                "mesh": {"status": "failed"},
            },
            "failure": {"stage": "mesh", "owner": "model"},
        },
        role="candidate",
        phase="preflight",
        runtime_seconds=0.1,
    )
    assert failed["final_verdict"] == "mesh_fail"
    assert failed["scoreable"] is True
