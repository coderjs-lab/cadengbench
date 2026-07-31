from pathlib import Path

import pytest

import cadengbench_p_eval.runner as runner
from cadengbench_p_eval import __version__
from cadengbench_p_eval.aggregation import aggregate_results, aggregate_run
from cadengbench_p_eval.contracts import (
    CheckResult,
    EvaluationResult,
    LayerResult,
    Status,
)
from cadengbench_p_eval.io import read_json, sha256_json, write_json


def _result(item_id: str, source: str, statuses: tuple[str, str, str]) -> dict:
    layers = [
        {"layer": name, "status": status, "checks": []}
        for name, status in zip(("L0", "L1", "L2"), statuses, strict=True)
    ]
    return {
        "item_id": item_id,
        "protocol": "zero_to_cad",
        "identity": {"source_family": source},
        "overall_status": statuses[-1],
        "layers": layers,
    }


def test_aggregate_preserves_full_population_denominator() -> None:
    aggregate = aggregate_results(
        [
            _result("a", "benchcad", ("passed", "passed", "passed")),
            _result("b", "fusion", ("failed", "not_reached", "not_reached")),
        ]
    )
    assert aggregate["item_count"] == 2
    assert aggregate["overall"]["strict_pass_rate"] == 0.5
    assert aggregate["layers"]["L0"]["strict_pass_rate"] == 0.5
    assert aggregate["layers"]["L2"]["scoreable_rate"] == 0.5
    assert aggregate["layers"]["L2"]["reach_rate"] == 0.5
    assert aggregate["layers"]["L2"]["conditional_pass_rate"] == 1.0
    assert aggregate["overall"]["source_macro_strict_pass_rate"] == 0.5
    assert aggregate["protocols"]["zero_to_cad"]["item_count"] == 2


def test_aggregate_reports_partial_required_check_progress() -> None:
    result = _result("a", "benchcad", ("passed", "failed", "not_reached"))
    result["layers"][1]["checks"] = [
        {"check_id": "required-pass", "required": True, "status": "passed"},
        {"check_id": "required-fail", "required": True, "status": "failed"},
        {
            "check_id": "irrelevant",
            "required": True,
            "status": "not_applicable",
        },
    ]

    aggregate = aggregate_results([result])

    assert (
        aggregate["layers"]["L1"]["mean_required_check_pass_rate_on_reached"]
        == 0.5
    )


def test_aggregate_separates_l2_authority_modes() -> None:
    direct = _result("a", "benchcad", ("passed", "passed", "passed"))
    response = _result("b", "fusion", ("passed", "passed", "failed"))
    direct["identity"]["l2_contract_mode"] = "direct_measurement_only"
    response["identity"]["l2_contract_mode"] = "geometry_response_only"

    aggregate = aggregate_results([direct, response])

    assert aggregate["l2_authority"]["direct_measurement_only"] == {
        "item_count": 1,
        "scoreable_item_count": 1,
        "strict_pass_rate": 1.0,
    }
    assert aggregate["l2_authority"]["geometry_response_only"] == {
        "item_count": 1,
        "scoreable_item_count": 1,
        "strict_pass_rate": 0.0,
    }


def test_aggregate_reports_reference_diagnostics_without_scoring_them() -> None:
    first = _result("item-a", "benchcad", ("passed", "passed", "passed"))
    second = _result("item-b", "fusion360", ("passed", "passed", "passed"))
    first["layers"][1]["metrics"] = {
        "reference_geometry_diagnostics": {
            "status": "computed",
            "score_bearing": False,
            "chamfer_mm": 1.0,
            "surface_f1": 0.8,
            "voxel_iou": 0.7,
        }
    }
    second["layers"][1]["metrics"] = {
        "reference_geometry_diagnostics": {
            "status": "computed",
            "score_bearing": True,
            "chamfer_mm": 3.0,
            "surface_f1": 1.0,
            "voxel_iou": 0.9,
        }
    }

    diagnostics = aggregate_results([first, second])["reference_geometry_diagnostics"]

    assert diagnostics["computed_item_count"] == 2
    assert diagnostics["coverage_rate"] == 1.0
    assert diagnostics["exact_reconstruction_scoring_item_count"] == 1
    assert diagnostics["metric_means"]["chamfer_mm"] == 2.0
    assert diagnostics["metric_means"]["surface_f1"] == 0.9
    assert diagnostics["metric_means"]["voxel_iou"] == pytest.approx(0.8)
    assert diagnostics["score_bearing_for_general_tasks"] is False


def test_source_macro_rate_does_not_overweight_larger_subset() -> None:
    aggregate = aggregate_results(
        [
            _result("a", "benchcad", ("passed", "passed", "passed")),
            _result("b", "benchcad", ("passed", "passed", "passed")),
            _result("c", "benchcad", ("passed", "passed", "passed")),
            _result("d", "fusion", ("failed", "not_reached", "not_reached")),
        ]
    )

    assert aggregate["overall"]["strict_pass_rate"] == 0.75
    assert aggregate["overall"]["source_macro_strict_pass_rate"] == 0.5


def test_publication_rate_is_withheld_for_required_not_scoreable_check() -> None:
    record = _result("a", "benchcad", ("passed", "passed", "not_scoreable"))
    record["layers"][2]["checks"] = [
        {
            "check_id": "L2.qualified_domains",
            "status": "not_scoreable",
            "required": True,
        }
    ]

    aggregate = aggregate_results([record])

    assert aggregate["overall"]["publication_valid"] is False
    assert aggregate["overall"]["publication_strict_pass_rate"] is None
    assert aggregate["overall"]["required_not_scoreable_item_count"] == 1


def test_optional_not_scoreable_check_does_not_invalidate_publication() -> None:
    record = _result("a", "benchcad", ("passed", "passed", "passed"))
    record["layers"][2]["checks"] = [
        {
            "check_id": "L2.optional_diagnostic",
            "status": "not_scoreable",
            "required": False,
        }
    ]

    aggregate = aggregate_results([record])

    assert aggregate["overall"]["publication_valid"] is True
    assert aggregate["overall"]["publication_strict_pass_rate"] == 1.0


def test_dataset_owned_l2_failure_is_excluded_from_model_denominator() -> None:
    passed = _result("a", "benchcad", ("passed", "passed", "passed"))
    excluded = _result(
        "b", "fusion360", ("passed", "passed", "not_scoreable")
    )
    excluded["layers"][2]["checks"] = [
        {
            "check_id": "L2.feasibility_contract",
            "status": "not_scoreable",
            "required": True,
            "summary": (
                "invalid feasibility contract: nominal assignment violates a "
                "frozen feasibility relation"
            ),
        }
    ]

    aggregate = aggregate_results([passed, excluded])

    assert aggregate["overall"]["raw_strict_pass_rate"] == 0.5
    assert aggregate["overall"]["strict_pass_rate"] == 1.0
    assert aggregate["overall"]["scoreable_item_count"] == 1
    assert aggregate["overall"]["benchmark_excluded_item_count"] == 1
    assert aggregate["layers"]["L2"]["scoreable_item_count"] == 1
    assert aggregate["overall"]["publication_valid"] is True


def test_evaluator_error_in_optional_check_invalidates_run() -> None:
    record = _result("a", "fusion360", ("passed", "passed", "passed"))
    record["layers"][1]["checks"] = [
        {
            "check_id": "L1.optional_metric",
            "status": "evaluator_error",
            "required": False,
        }
    ]

    aggregate = aggregate_results([record])

    assert aggregate["overall"]["publication_valid"] is False
    assert aggregate["overall"]["evaluator_error_item_count"] == 1


def test_non_scoreable_overall_invalidates_even_without_required_ns_check() -> None:
    record = _result("a", "benchcad", ("passed", "passed", "not_scoreable"))
    record["layers"][2]["checks"] = [
        {
            "check_id": "L2.not_applicable_protocol",
            "status": "not_applicable",
            "required": False,
        }
    ]

    aggregate = aggregate_results([record])

    assert aggregate["overall"]["required_not_scoreable_item_count"] == 0
    assert aggregate["overall"]["publication_valid"] is False


def _write_integrity_run(run_root: Path, item_ids: list[str]) -> None:
    inventory = [
        {
            "index": index,
            "item_id": item_id,
            "path": f"{item_id}/submission.py",
            "state": "present",
            "sha256": "c" * 64,
            "size_bytes": 1,
        }
        for index, item_id in enumerate(item_ids)
    ]
    identity = {
        "evalset_manifest_hash": "d" * 64,
        "evalset_semantic_hash": "e" * 64,
        "evaluator_build_hash": "a" * 64,
        "environment_hash": "b" * 64,
        "submission_inventory_hash": sha256_json(inventory),
        "model_id": "test-model",
    }
    manifest = {
        "schema_version": runner.RUN_MANIFEST_SCHEMA,
        "identity": identity,
        "submission_inventory": inventory,
        "item_count": len(item_ids),
        "model_id": "test-model",
        **identity,
        "run_hash": sha256_json(identity),
    }
    write_json(run_root / "manifest.json", manifest)
    for item_id in item_ids:
        checks = tuple(
            LayerResult.from_checks(
                layer,
                (CheckResult(f"{layer}.synthetic", Status.PASSED),),
            )
            for layer in ("L0", "L1", "L2")
        )
        result = EvaluationResult(
            schema_version=runner.RESULT_SCHEMA,
            evaluator_version=__version__,
            item_id=item_id,
            protocol="zero_to_cad",
            identity={
                "source_family": "benchcad",
                "evaluator_build_hash": identity["evaluator_build_hash"],
                "environment_hash": identity["environment_hash"],
            },
            layers=checks,
            overall_status=Status.PASSED,
            environment={"environment_hash": identity["environment_hash"]},
            runtime_seconds=0.1,
        )
        runner._write_result(run_root / "items" / item_id, result)


def test_standalone_aggregate_is_bound_to_complete_run_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_integrity_run(run_root, ["item-a", "item-b"])
    result = run_root / "items/item-b/result.json"
    result.unlink()

    with pytest.raises(ValueError, match="incomplete"):
        aggregate_run(run_root)


def test_standalone_aggregate_rejects_tampered_result_digest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_integrity_run(run_root, ["item-a"])
    result = run_root / "items/item-a/result.json"
    result.write_text(result.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="digest validation"):
        aggregate_run(run_root)


def test_standalone_aggregate_accepts_complete_hash_bound_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_integrity_run(run_root, ["item-a", "item-b"])

    aggregate = aggregate_run(run_root)

    assert aggregate["item_count"] == 2
    assert aggregate["run_hash"]
    assert aggregate["overall"]["publication_valid"] is True
    assert read_json(run_root / "aggregate.json") == aggregate
    assert read_json(run_root / "aggregate/metrics.json") == aggregate
