from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import write_json

from cadengbench_p_runner import l3_benchmark
from cadengbench_p_runner.io import sha256_file
from cadengbench_p_runner.l3_benchmark import (
    apply_l3_pair_selection,
    build_binding_index,
    plan_l3_run,
    run_l3_plan,
    summarize_l3_run,
    validate_release,
)


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    item_ids = ["p_benchcad_000001", "p_fusion360_000001"]
    write_json(
        root / "RELEASE.json",
        {
            "benchmark": "cadengbench-p",
            "dataset_version": "cadengbench-p-test-300-v1",
            "item_count": 2,
            "status": "frozen",
        },
    )
    (root / "item_ids.txt").write_text("\n".join(item_ids) + "\n", encoding="utf-8")
    rows = [
        {
            "item_id": item_ids[0],
            "source_key": "benchcad",
            "complexity": {"tier": "easy"},
        },
        {
            "item_id": item_ids[1],
            "source_key": "fusion360",
            "complexity": {"tier": "hard"},
        },
    ]
    (root / "selected_items.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    for item_id in item_ids:
        write_json(root / "items" / item_id / "item.json", {"item_id": item_id})
    return root


def _layer(name: str, status: str = "passed") -> dict[str, str]:
    return {"layer": name, "status": status}


def _l02_run(tmp_path: Path) -> Path:
    root = tmp_path / "l02"
    benchcad = root / "items/p_benchcad_000001"
    write_json(
        benchcad / "result.json",
        {
            "item_id": "p_benchcad_000001",
            "protocol": "zero_to_cad",
            "layers": [_layer("L0"), _layer("L1"), _layer("L2")],
        },
    )
    write_json(
        benchcad / "l2.json",
        {
            "artifacts": {
                "parameter_bindings": [{"parameter_id": "P-001", "nominal": 10.0}],
                "replay": {"schedule": {"variants": []}, "variants": []},
            }
        },
    )
    fusion = root / "items/p_fusion360_000001"
    write_json(
        fusion / "result.json",
        {
            "item_id": "p_fusion360_000001",
            "protocol": "functional_edit",
            "layers": [_layer("L0"), _layer("L1"), _layer("L2")],
        },
    )
    return root


def _plan_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    artifact = tmp_path / "artifact.step"
    artifact.write_text("step", encoding="utf-8")
    card = tmp_path / "physics_card.json"
    contract = tmp_path / "response_contract.json"
    write_json(card, {"item_id": "p_benchcad_000001"})
    write_json(contract, {"item_id": "p_benchcad_000001"})
    bindings = tmp_path / "bindings.json"
    write_json(
        bindings,
        {
            "items": [
                {
                    "item_id": "p_benchcad_000001",
                    "applicability": "solver_qualified",
                    "card_path": str(card),
                    "nominal_reference_step_path": str(artifact),
                    "response_contract_path": str(contract),
                    "mesh_study_path": None,
                },
                {
                    "item_id": "p_fusion360_000001",
                    "applicability": "solver_qualified",
                },
            ]
        },
    )
    bank = tmp_path / "bank.json"
    write_json(
        bank,
        {
            "items": [
                {
                    "item_id": "p_benchcad_000001",
                    "status": "ready",
                    "references": [
                        {
                            "reference_id": "nominal",
                            "assignments": {"P-001": 10.0},
                            "step_path": str(artifact),
                        }
                    ],
                }
            ]
        },
    )
    alignment = tmp_path / "alignment.json"
    write_json(
        alignment,
        {
            "items": [
                {
                    "item_id": "p_benchcad_000001",
                    "status": "ready",
                    "transform": {"method": "test"},
                    "variants": [
                        {
                            "reference_id": "nominal",
                            "assignments": {"P-001": 10.0},
                            "memberships": [
                                {
                                    "variant_id": "nominal",
                                    "schedule_kind": "nominal",
                                }
                            ],
                            "upstream_status": "passed",
                            "aligned_candidate_step_path": str(artifact),
                        }
                    ],
                }
            ]
        },
    )
    profile = tmp_path / "profile.json"
    write_json(profile, {"profile_id": "test"})
    return bindings, bank, alignment, profile


def test_release_validation_checks_payload_and_metadata(tmp_path: Path) -> None:
    release = _release(tmp_path)
    assert len(validate_release(release)["item_ids"]) == 2
    assert len(validate_release(release, require_items=False)["selected"]) == 2


def test_binding_index_preserves_applicability_and_card_binding(tmp_path: Path) -> None:
    release = _release(tmp_path)
    reference = release / "items/p_benchcad_000001/source/generated_from_cadquery.step"
    reference.parent.mkdir(parents=True)
    reference.write_text("reference", encoding="utf-8")
    sidecar = tmp_path / "sidecar"
    canonical_card = (
        sidecar / "items/p_benchcad_000001/physics_card.solver_qualified.json"
    )
    write_json(
        canonical_card,
        {
            "schema_version": "cadengbench-p.l3-physics-card.v2",
            "item_id": "p_benchcad_000001",
            "scenario_family": "axial_tension",
            "qualification": {
                "promoted": True,
                "reference_step_sha256": sha256_file(reference),
            },
        },
    )
    write_json(
        sidecar / "pilot_runs/promoted_cards/p_benchcad_000001.json",
        {
            "schema_version": "cadengbench-p.l3-physics-card.v2",
            "item_id": "p_benchcad_000001",
            "scenario_family": "cantilever_transverse",
            "qualification": {
                "promoted": True,
                "reference_step_sha256": sha256_file(reference),
            },
        },
    )
    qualification = tmp_path / "qualification.json"
    write_json(
        qualification,
        {
            "records": [
                {"item_id": "p_benchcad_000001", "status": "promoted"},
                {
                    "item_id": "p_fusion360_000001",
                    "status": "not_compiled",
                    "annotation_reason": "deterministically_ineligible",
                },
            ]
        },
    )
    index = build_binding_index(
        release_root=release,
        qualification_summary_path=qualification,
        sidecar_root=sidecar,
        contract_root=tmp_path / "contracts",
        output_path=tmp_path / "bindings.json",
    )
    assert index["item_count"] == 2
    assert index["solver_qualified_count"] == 1
    assert index["items"][0]["scenario"] == "axial_tension"
    assert index["items"][0]["card_path"] == str(canonical_card.resolve())
    assert index["items"][1]["applicability"] == "not_applicable"


def test_extended_binding_index_dispatches_body_response(tmp_path: Path) -> None:
    release = _release(tmp_path)
    reference = release / "items/p_benchcad_000001/source/generated_from_cadquery.step"
    reference.parent.mkdir(parents=True)
    reference.write_text("reference", encoding="utf-8")
    fusion_reference = (
        release
        / "items/p_fusion360_000001/source/fusion360_occ_reconstruction.step"
    )
    fusion_reference.parent.mkdir(parents=True)
    fusion_reference.write_text("fusion", encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / "PROJECT_BRIEF.md").parent.mkdir(parents=True, exist_ok=True)
    (repo / "PROJECT_BRIEF.md").write_text("test", encoding="utf-8")
    card = repo / "data/body.json"
    write_json(
        card,
        {
            "schema_version": "cadengbench-p.l3-body-acceleration-card.v1",
            "item_id": "p_benchcad_000001",
            "scenario_family": "single_support_body_acceleration",
            "qualification": {"reference_step_sha256": sha256_file(reference)},
        },
    )
    applicability = repo / "benchmarks/cadengbench-p/evaluators/l3/index.jsonl"
    applicability.parent.mkdir(parents=True)
    applicability.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "item_id": "p_benchcad_000001",
                        "status": "solver_qualified",
                        "physics_card_path": "data/body.json",
                    }
                ),
                json.dumps(
                    {
                        "item_id": "p_fusion360_000001",
                        "status": "not_applicable",
                        "reason": "not_qualified",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    index = build_binding_index(
        release_root=release,
        qualification_summary_path=tmp_path / "unused.json",
        sidecar_root=tmp_path / "sidecar",
        contract_root=tmp_path / "surface-contracts",
        body_contract_root=tmp_path / "body-contracts",
        applicability_index_path=applicability,
        output_path=tmp_path / "bindings.json",
    )
    assert index["solver_qualified_count"] == 1
    assert index["family_counts"] == {"single_support_body_acceleration": 1}
    assert index["items"][0]["evaluator_mode"] == "body_acceleration_response_v1"


def test_l3_plan_runs_zero_to_cad_and_excludes_functional_edit(tmp_path: Path) -> None:
    release = _release(tmp_path)
    l02 = _l02_run(tmp_path)
    bindings, bank, alignment, profile = _plan_inputs(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = plan_l3_run(
        release_root=release,
        l02_run=l02,
        bindings_path=bindings,
        reference_bank_path=bank,
        alignment_path=alignment,
        response_profile_path=profile,
        output_path=plan_path,
    )
    assert plan["ready_item_count"] == 1
    assert plan["items"][0]["variants"][0]["l3_scoreable"] is True
    assert plan["items"][0]["variants"][0]["physical_status"] == "passed"
    assert plan["items"][1]["status"] == "not_applicable"
    assert plan["items"][1]["applicability"] == "protocol_not_applicable"


def test_l3_plan_scores_physics_independently_of_l2(
    monkeypatch, tmp_path: Path
) -> None:
    release = _release(tmp_path)
    l02 = _l02_run(tmp_path)
    result_path = l02 / "items/p_benchcad_000001/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["layers"] = [
        _layer("L0"),
        _layer("L1", "failed"),
        _layer("L2", "failed"),
    ]
    write_json(result_path, result)
    bindings, bank, alignment, profile = _plan_inputs(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = plan_l3_run(
        release_root=release,
        l02_run=l02,
        bindings_path=bindings,
        reference_bank_path=bank,
        alignment_path=alignment,
        response_profile_path=profile,
        output_path=plan_path,
    )
    variant = plan["items"][0]["variants"][0]
    assert variant["physical_status"] == "passed"
    assert variant["l3_scoreable"] is True
    assert plan["items"][0]["l2_status"] == "failed"

    def fake_command(
        command: list[str], *, stdout_path: Path, stderr_path: Path
    ) -> int:
        pair_output = Path(command[command.index("--output") + 1])
        write_json(
            pair_output / "pair_result.json",
            {
                "final_verdict": "pass",
                "stages": {
                    name: "passed"
                    for name in (
                        "rebuild",
                        "selectors",
                        "mesh",
                        "solve",
                        "numerical_validity",
                        "response_fidelity",
                        "engineering_requirements",
                    )
                },
                "margins": {"response_fidelity_score": 1.0},
                "runtime_seconds": 1.0,
            },
        )
        return 0

    monkeypatch.setattr(l3_benchmark, "_run_command", fake_command)
    output = tmp_path / "l3-run"
    execution = run_l3_plan(
        plan_path=plan_path,
        output_root=output,
        l3_cli=tmp_path / "fake-l3",
        ccx_path=tmp_path / "ccx",
    )
    assert execution["completed_pair_count"] == 1
    summary = summarize_l3_run(
        plan_path=plan_path, run_root=output, output_root=tmp_path / "tables"
    )
    assert summary["physical_pair_count"] == 1
    assert summary["physical_diagnostic_status"] == "completed"
    assert summary["physical_diagnostic_reason"] is None
    assert summary["l3_scoreable_pair_count"] == 1
    assert summary["l3_status"] == "completed"
    assert summary["l3_reason"] is None
    assert summary["l3_pass_rate"] == 1.0
    assert summary["scored_protocol"] == "zero_to_cad"
    assert "functional_edit" in summary["excluded_protocols"]
    assert summary["diagnostic_physical_stage_rates"]["solve"]["pass_rate"] == 1.0

    write_json(
        output
        / "items/p_benchcad_000001/variants"
        / variant["variant_id"]
        / "pair_result.json",
        {
            "final_verdict": "reference_not_scoreable",
            "scoreable": False,
            "stages": {
                name: "not_reached"
                for name in (
                    "rebuild",
                    "selectors",
                    "mesh",
                    "solve",
                    "numerical_validity",
                    "response_fidelity",
                    "engineering_requirements",
                )
            },
            "margins": {},
            "runtime_seconds": 1.0,
        },
    )
    corrected = summarize_l3_run(
        plan_path=plan_path,
        run_root=output,
        output_root=tmp_path / "corrected-tables",
    )
    assert corrected["physical_pair_count"] == 1
    assert corrected["l3_scoreable_pair_count"] == 0
    assert corrected["l3_pass_rate"] is None


def test_l3_plan_inherits_exact_l02_manifest_roster(tmp_path: Path) -> None:
    release = _release(tmp_path)
    l02 = _l02_run(tmp_path)
    write_json(
        l02 / "manifest.json",
        {
            "item_count": 1,
            "submission_inventory": [
                {"item_id": "p_benchcad_000001", "state": "present"}
            ],
        },
    )
    bindings, bank, alignment, profile = _plan_inputs(tmp_path)
    plan = plan_l3_run(
        release_root=release,
        l02_run=l02,
        bindings_path=bindings,
        reference_bank_path=bank,
        alignment_path=alignment,
        response_profile_path=profile,
        output_path=tmp_path / "plan.json",
    )
    assert plan["item_count"] == 1
    assert [item["item_id"] for item in plan["items"]] == [
        "p_benchcad_000001"
    ]


def test_l3_pair_selection_keeps_representatives_and_full_ablation(
    tmp_path: Path,
) -> None:
    release = _release(tmp_path)

    def variant(variant_id: str, value: float) -> dict[str, object]:
        return {
            "variant_id": variant_id,
            "assignments": {"P-001": value},
            "memberships": [
                {
                    "variant_id": variant_id,
                    "schedule_kind": (
                        "nominal" if variant_id == "nominal" else "morris"
                    ),
                }
            ],
        }

    variants = [
        variant("nominal", 10.0),
        variant("near-duplicate", 10.0),
        variant("mild", 9.0),
        variant("moderate", 8.0),
        variant("strong", 7.0),
        variant("boundary", 6.0),
    ]
    plan = {
        "schema_version": "cadengbench-p.l3-benchmark-plan.v4",
        "release_root": str(release),
        "items": [
            {
                "item_id": "p_benchcad_000001",
                "variants": variants,
            },
            {
                "item_id": "p_fusion360_000001",
                "variants": variants,
            },
        ],
    }
    policy_path = tmp_path / "selection.json"
    write_json(
        policy_path,
        {
            "schema_version": "cadengbench-p.l3-pair-selection.v1",
            "policy_id": "test-selection",
            "distance_metric": "max_symmetric_relative_parameter_delta",
            "default_selection": ["nominal", "moderate", "boundary"],
            "full_family_item_ids": ["p_fusion360_000001"],
        },
    )

    selected = apply_l3_pair_selection(
        plan=plan,
        selection_path=policy_path,
        output_path=tmp_path / "selected-plan.json",
    )

    assert selected["schema_version"] == "cadengbench-p.l3-benchmark-plan.v5"
    assert selected["pair_selection"]["candidate_pair_count"] == 12
    assert selected["pair_selection"]["selected_pair_count"] == 9
    assert [
        value["variant_id"] for value in selected["items"][0]["variants"]
    ] == ["nominal", "moderate", "boundary"]
    assert len(selected["items"][1]["variants"]) == 6
    assert selected["items"][0]["unselected_variant_count"] == 3
    assert selected["items"][1]["unselected_variant_count"] == 0


def test_l3_execution_and_tables_are_resumable(monkeypatch, tmp_path: Path) -> None:
    release = _release(tmp_path)
    l02 = _l02_run(tmp_path)
    bindings, bank, alignment, profile = _plan_inputs(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_l3_run(
        release_root=release,
        l02_run=l02,
        bindings_path=bindings,
        reference_bank_path=bank,
        alignment_path=alignment,
        response_profile_path=profile,
        output_path=plan_path,
    )

    def fake_command(
        command: list[str], *, stdout_path: Path, stderr_path: Path
    ) -> int:
        pair_output = Path(command[command.index("--output") + 1])
        write_json(
            pair_output / "pair_result.json",
            {
                "final_verdict": "pass",
                "stages": {
                    name: "passed"
                    for name in (
                        "rebuild",
                        "selectors",
                        "mesh",
                        "solve",
                        "numerical_validity",
                        "response_fidelity",
                        "engineering_requirements",
                    )
                },
                "margins": {
                    "stress_mpa": 10.0,
                    "normalized_displacement": 0.01,
                    "displacement_ratio": 0.2,
                    "compliance_ratio": 0.2,
                    "response_fidelity_score": 1.0,
                },
                "runtime_seconds": 2.5,
            },
        )
        return 0

    monkeypatch.setattr(l3_benchmark, "_run_command", fake_command)
    run = tmp_path / "l3-run"
    stale_output = run / "items/p_benchcad_000001/variants/nominal"
    stale_output.mkdir(parents=True)
    (stale_output / "partial.txt").write_text("interrupted", encoding="utf-8")
    execution = run_l3_plan(
        plan_path=plan_path,
        output_root=run,
        l3_cli=tmp_path / "fake-l3",
        ccx_path=tmp_path / "ccx",
        workers=2,
    )
    assert execution["completed_pair_count"] == 1
    archived = (
        run
        / "retry_archive/items/p_benchcad_000001/variants/nominal.attempt-0001"
        / "partial.txt"
    )
    assert archived.read_text(encoding="utf-8") == "interrupted"
    resumed = run_l3_plan(
        plan_path=plan_path,
        output_root=run,
        l3_cli=tmp_path / "fake-l3",
        ccx_path=tmp_path / "ccx",
        workers=1,
    )
    assert resumed["outcomes"][0]["status"] == "resumed"

    summary = summarize_l3_run(
        plan_path=plan_path, run_root=run, output_root=tmp_path / "tables"
    )
    assert summary["l3_pass_rate"] == 1.0
    assert summary["l3_status"] == "completed"
    assert summary["l3_reason"] is None
    assert summary["l3_family_pass_rate"] == 1.0
    assert summary["physical_pair_count"] == 1
    assert summary["l3_stage_progress_score"] == 1.0
    assert summary["l3_stage_rates"]["solve"]["pass_rate"] == 1.0
    assert (
        summary["diagnostic_physical_stage_rates"]["response_fidelity"]["pass_rate"]
        == 1.0
    )
    with (tmp_path / "tables/results.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["comparison_mode"] == "matched_parameter_response"
    assert rows[0]["final_verdict"] == "pass"
    with (tmp_path / "tables/pairs.csv").open(encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    assert pairs[0]["physical_variant_id"] == "nominal"
