from __future__ import annotations

import os
from pathlib import Path

import pytest

from cadengbench_p_l3_response.io import read_json, verify_evidence_commit, write_json
from cadengbench_p_l3_response.runner import (
    bind_family_manifest,
    calibrate_item,
    evaluate_pair,
    run_family,
)

pytestmark = pytest.mark.integration


def _requirements() -> Path:
    pytest.importorskip("gmsh")
    ccx = Path(os.environ.get("CADENGBENCH_CCX", ""))
    runtime_lock = Path(os.environ.get("CADENGBENCH_L3_RUNTIME_LOCK", ""))
    if not ccx.is_file() or not runtime_lock.is_file():
        pytest.skip("pinned CalculiX and runtime lock are required")
    return ccx


def _write_inputs(tmp_path: Path, profile: dict, card: dict) -> tuple[Path, Path]:
    profile_path = tmp_path / "profile.json"
    card_path = tmp_path / "card.json"
    write_json(profile_path, profile)
    write_json(card_path, card)
    return profile_path, card_path


def test_actual_axial_fixed_force_exact_and_weakened(
    tmp_path: Path, profile: dict, card: dict
) -> None:
    ccx = _requirements()
    import cadquery as cq

    reference = tmp_path / "reference.step"
    weak = tmp_path / "weak.step"
    cq.exporters.export(
        cq.Workplane("YZ").rect(10.0, 10.0).extrude(100.0), str(reference)
    )
    cq.exporters.export(cq.Workplane("YZ").rect(7.0, 7.0).extrude(100.0), str(weak))
    profile_path, card_path = _write_inputs(tmp_path, profile, card)
    calibration = tmp_path / "calibration"
    contract = calibrate_item(
        profile_path=profile_path,
        card_path=card_path,
        reference_step=reference,
        output_dir=calibration,
        ccx_path=ccx,
        timeout_seconds=180,
    )
    contract_path = calibration / "response_contract.json"
    variants_path = tmp_path / "variants.json"
    write_json(
        variants_path,
        {
            "item_id": contract["item_id"],
            "variants": [
                {
                    "variant_id": "nominal",
                    "upstream_status": "passed",
                    "reference_step_path": str(reference),
                    "candidate_step_path": str(reference),
                },
                {
                    "variant_id": "weak-7x7",
                    "upstream_status": "passed",
                    "reference_step_path": str(reference),
                    "candidate_step_path": str(weak),
                },
            ],
        },
    )
    manifest_path = tmp_path / "family_manifest.json"
    bind_family_manifest(
        contract_path=contract_path,
        card_path=card_path,
        variants_spec_path=variants_path,
        output_path=manifest_path,
    )
    family_dir = tmp_path / "family"
    family = run_family(
        manifest_path=manifest_path,
        output_dir=family_dir,
        ccx_path=ccx,
        workers=1,
        timeout_seconds=180,
    )
    exact = read_json(family_dir / "variants/nominal/pair_result.json")
    weakened = read_json(family_dir / "variants/weak-7x7/pair_result.json")
    assert family["variant_count"] == 2
    assert family["status_counts"] == {"pass": 1, "requirement_fail": 1}
    assert family["end_to_end_structural_pass_rate"] == 0.5
    assert family["engineering_requirement_pass_rate"] == 0.5
    assert family["mean_response_fidelity_score"] == pytest.approx(
        (1.0 + weakened["response_fidelity"]["aggregate_score"]) / 2.0
    )
    assert contract["loading"]["controlling_constraint"] == "stress_utilization"
    assert exact["final_verdict"] == "pass", exact
    assert weakened["final_verdict"] == "requirement_fail", weakened
    assert (
        weakened["candidate"]["metrics"]["regularized_von_mises_p95_mpa"]
        > contract["engineering_requirements"]["maximum_regularized_von_mises_p95_mpa"]
    )
    assert weakened["response_fidelity"]["metrics"]["regularized_von_mises_p95_mpa"][
        "candidate_reference_ratio"
    ] == pytest.approx(100.0 / 49.0, rel=0.12)
    verify_evidence_commit(family_dir)
    verify_evidence_commit(family_dir / "variants/nominal")
    verify_evidence_commit(family_dir / "variants/weak-7x7")


def test_actual_cantilever_weakened_section_fails(
    tmp_path: Path, profile: dict, card: dict
) -> None:
    ccx = _requirements()
    import cadquery as cq

    card["scenario_family"] = "cantilever_transverse"
    card["load"]["traction_vector_mpa"] = [0.0, 0.0, -0.1]
    reference = tmp_path / "reference.step"
    weak = tmp_path / "weak.step"
    cq.exporters.export(
        cq.Workplane("YZ").rect(10.0, 10.0).extrude(100.0), str(reference)
    )
    cq.exporters.export(cq.Workplane("YZ").rect(10.0, 7.0).extrude(100.0), str(weak))
    profile_path, card_path = _write_inputs(tmp_path, profile, card)
    calibration = tmp_path / "calibration"
    calibrate_item(
        profile_path=profile_path,
        card_path=card_path,
        reference_step=reference,
        output_dir=calibration,
        ccx_path=ccx,
        timeout_seconds=180,
    )
    result = evaluate_pair(
        contract_path=calibration / "response_contract.json",
        card_path=card_path,
        reference_step=reference,
        candidate_step=weak,
        variant_id="weak-height-7",
        output_dir=tmp_path / "weak-result",
        ccx_path=ccx,
        timeout_seconds=180,
    )
    assert result["final_verdict"] in {"requirement_fail", "response_fidelity_fail"}
    assert result["response_fidelity"]["status"] == "failed"
    assert (
        result["candidate"]["metrics"]["normalized_max_displacement"]
        > result["reference"]["metrics"]["normalized_max_displacement"]
    )
