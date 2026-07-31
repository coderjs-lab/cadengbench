from __future__ import annotations

from pathlib import Path

from cadengbench_p_l3_extended import body_response
from cadengbench_p_l3_extended.io import sha256_file, write_json
from cadengbench_p_l3_extended.provenance import evaluator_build_hash


def _solver_result() -> dict:
    return {
        "final_status": "probe_pass",
        "evaluator_build_hash": evaluator_build_hash(),
        "runtime_lock_sha256": "1" * 64,
        "stages": {
            name: {"status": "passed"}
            for name in ("geometry", "selectors", "mesh", "solver")
        },
        "metrics": {
            "regularized_von_mises_mpa": {"p95": 10.0},
            "normalized_max_displacement": 0.001,
            "body_force_stress_concentration_p95": 2.0,
            "force_equilibrium_relative_error": 1e-8,
            "applied_force_n": [100.0, 0.0, 0.0],
        },
    }


def test_identical_body_response_pair_passes(monkeypatch, tmp_path: Path) -> None:
    card = tmp_path / "card.json"
    step = tmp_path / "part.step"
    write_json(card, {"item_id": "p_benchcad_000001"})
    step.write_text("step", encoding="utf-8")
    contract = {
        "item_id": "p_benchcad_000001",
        "contract_sha256": "2" * 64,
        "bindings": {
            "base_card_sha256": sha256_file(card),
            "nominal_reference_step_sha256": sha256_file(step),
            "body_evaluator_build_hash": evaluator_build_hash(),
            "l3_runtime_lock_sha256": "1" * 64,
        },
        "fidelity": {
            "maximum_symmetric_log_error": {
                metric: 0.25 for metric in body_response.BODY_METRICS
            }
        },
        "engineering_requirements": {
            "maximum_regularized_von_mises_p95_mpa": 125.0,
            "maximum_normalized_displacement": 0.02,
            "maximum_candidate_reference_stress_ratio": 1.5,
            "maximum_candidate_reference_displacement_ratio": 1.25,
            "maximum_candidate_reference_stress_concentration_ratio": 1.5,
        },
        "numerical": {
            "maximum_equilibrium_relative_error": 0.001,
            "ratio_epsilon": 1e-12,
        },
        "evaluator": {
            "name": "cadengbench-p-l3-extended",
            "version": "0.2.0",
            "build_hash": evaluator_build_hash(),
        },
    }
    monkeypatch.setattr(
        body_response, "load_body_response_contract", lambda _: contract
    )
    monkeypatch.setattr(
        body_response, "run_body_card", lambda *args, **kwargs: _solver_result()
    )
    result = body_response.evaluate_body_response_pair(
        contract_path=tmp_path / "contract.json",
        card_path=card,
        reference_step=step,
        candidate_step=step,
        variant_id="nominal",
        output_dir=tmp_path / "result",
        ccx_path=tmp_path / "ccx",
    )
    assert result["final_verdict"] == "pass"
    assert result["margins"]["response_fidelity_score"] == 1.0
    assert (
        result["load_scaling"]["candidate_reference_resultant_force_ratio"] == 1.0
    )
