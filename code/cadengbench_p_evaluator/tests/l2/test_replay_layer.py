from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import pytest

from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status
from cadengbench_p_eval.layers.l2 import evaluate_l2
from cadengbench_p_eval.parametrics.domains import parse_parameter_domains
from cadengbench_p_eval.parametrics.relations import parse_relation
from cadengbench_p_eval.parametrics.replay import (
    ReplayError,
    VariantEvaluationContext,
    VariantExecution,
    bind_parameter_manifest,
    model_parameter_values,
    replay_schedule,
)
from cadengbench_p_eval.parametrics.schedules import generate_directed_schedule


def passing_layer(name: str) -> LayerResult:
    return LayerResult.from_checks(
        name, [CheckResult(f"{name}.synthetic", Status.PASSED)]
    )


def fingerprint(parameters: Mapping[str, float]) -> str:
    return "|".join(f"{key}={parameters[key]:.9f}" for key in sorted(parameters))


def test_parameter_manifest_binding_is_exact(
    parameter_system: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    bindings = bind_parameter_manifest(parameter_system, submission_contract)
    assert bindings["P-001"].name == "length"
    assert bindings["P-002"].nominal == 4.0

    duplicate = deepcopy(submission_contract)
    duplicate["parameter_manifest"]["P-002"]["name"] = "length"
    with pytest.raises(ReplayError) as captured:
        bind_parameter_manifest(parameter_system, duplicate)
    assert captured.value.failure_code == "parameter_binding_not_independent"

    wrong_default = deepcopy(submission_contract)
    wrong_default["default_params"]["length"] = 11.0
    with pytest.raises(ReplayError) as captured:
        bind_parameter_manifest(parameter_system, wrong_default)
    assert captured.value.failure_code == "parameter_default_mismatch"


def test_replay_detects_ignored_parameter_and_never_allows_nominal_target(
    parameter_system: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    bindings = bind_parameter_manifest(parameter_system, submission_contract)
    domains = parse_parameter_domains(parameter_system)
    schedule = generate_directed_schedule(
        domains,
        {parameter_id: value.nominal for parameter_id, value in bindings.items()},
        points_per_parameter=2,
        max_variants=2,
    )
    seen_contexts: list[VariantEvaluationContext] = []

    def execute_ignored(
        parameters: Mapping[str, float], variant_id: str
    ) -> VariantExecution:
        del parameters, variant_id
        return VariantExecution(Status.PASSED, fingerprint="hard-coded-shape")

    def l0(
        execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult:
        del execution
        seen_contexts.append(context)
        return passing_layer("L0")

    def l1(
        execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult:
        del execution
        assert context.nominal_reference_allowed is False
        assert context.parameter_matched_reference is None
        return passing_layer("L1")

    result = replay_schedule(
        schedule,
        bindings,
        execute=execute_ignored,
        evaluate_l0=l0,
        evaluate_l1=l1,
        baseline_execution=VariantExecution(
            Status.PASSED, fingerprint="hard-coded-shape"
        ),
    )
    assert result.status is Status.FAILED
    assert any(check.failure_code == "parameter_ignored" for check in result.checks)
    assert len(seen_contexts) == 2


def test_replay_checks_measurement_relation(
    parameter_system: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    bindings = bind_parameter_manifest(parameter_system, submission_contract)
    domains = parse_parameter_domains(parameter_system)
    schedule = generate_directed_schedule(
        domains,
        {parameter_id: value.nominal for parameter_id, value in bindings.items()},
        points_per_parameter=2,
        max_variants=2,
    )
    relation = parse_relation(
        {
            "relation_id": "length_readback",
            "predicate": {
                "type": "comparison",
                "op": "eq",
                "left": {
                    "type": "variable",
                    "source": "measurement",
                    "key": "length",
                },
                "right": {
                    "type": "variable",
                    "source": "parameter",
                    "key": "P-001",
                },
                "tolerance": 1e-9,
            },
            "authority": "synthetic",
            "evidence_ref": "test",
            "required": True,
        }
    )

    def execute(parameters: Mapping[str, float], variant_id: str) -> VariantExecution:
        del variant_id
        return VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint(parameters),
            measurements={"length": parameters["length"]},
        )

    result = replay_schedule(
        schedule,
        bindings,
        execute=execute,
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
        relations=[relation],
        baseline_execution=VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint({"length": 10.0, "width": 4.0}),
            measurements={"length": 10.0},
        ),
    )
    assert result.status is Status.PASSED


def test_replay_requires_candidate_solid_count_preservation(
    parameter_system: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    bindings = bind_parameter_manifest(parameter_system, submission_contract)
    domains = parse_parameter_domains(parameter_system)
    schedule = generate_directed_schedule(
        domains,
        {parameter_id: value.nominal for parameter_id, value in bindings.items()},
        points_per_parameter=2,
        max_variants=2,
    )

    result = replay_schedule(
        schedule,
        bindings,
        execute=lambda parameters, _variant_id: VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint(parameters),
            measurements={"solid_count": 2.0},
        ),
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
        baseline_execution=VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint({"length": 10.0, "width": 4.0}),
            measurements={"solid_count": 1.0},
        ),
    )

    preservation = [
        check
        for check in result.checks
        if check.check_id.endswith(".solid_count_preserved")
    ]
    assert preservation
    assert all(check.required for check in preservation)
    assert all(check.status is Status.FAILED for check in preservation)
    assert all(
        check.failure_code == "solid_count_changed_under_perturbation"
        for check in preservation
    )
    assert result.status is Status.FAILED


def test_replay_treats_missing_bound_measurement_as_model_failure(
    parameter_system: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    bindings = bind_parameter_manifest(parameter_system, submission_contract)
    domains = parse_parameter_domains(parameter_system)
    schedule = generate_directed_schedule(
        domains,
        {parameter_id: value.nominal for parameter_id, value in bindings.items()},
        points_per_parameter=2,
        max_variants=2,
    )
    relation = parse_relation(
        {
            "relation_id": "length_readback",
            "predicate": {
                "type": "comparison",
                "op": "eq",
                "left": {
                    "type": "variable",
                    "source": "measurement",
                    "key": "length",
                },
                "right": {
                    "type": "variable",
                    "source": "parameter",
                    "key": "P-001",
                },
                "tolerance": 1e-9,
            },
            "authority": "synthetic",
            "evidence_ref": "test",
            "required": True,
        }
    )

    result = replay_schedule(
        schedule,
        bindings,
        execute=lambda parameters, _variant_id: VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint(parameters),
            measurements={},
        ),
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
        relations=[relation],
        baseline_execution=VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint({"length": 10.0, "width": 4.0}),
            measurements={"length": 10.0},
        ),
    )

    relation_checks = [
        check
        for check in result.checks
        if ".relation.length_readback" in check.check_id
    ]
    assert relation_checks
    assert all(check.status is Status.FAILED for check in relation_checks)
    assert all(
        check.failure_code == "relation_measurement_missing"
        for check in relation_checks
    )
    assert result.status is Status.FAILED


def test_failed_variant_marks_dependent_checks_not_reached(
    parameter_system: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    bindings = bind_parameter_manifest(parameter_system, submission_contract)
    domains = parse_parameter_domains(parameter_system)
    schedule = generate_directed_schedule(
        domains,
        {parameter_id: value.nominal for parameter_id, value in bindings.items()},
        points_per_parameter=2,
        max_variants=2,
    )

    result = replay_schedule(
        schedule,
        bindings,
        execute=lambda *_: VariantExecution(
            Status.FAILED,
            failure_code="solid_count_mismatch",
        ),
        evaluate_l0=lambda *_: LayerResult.from_checks(
            "L0", [CheckResult("L0.synthetic", Status.FAILED)]
        ),
        evaluate_l1=lambda *_: LayerResult("L1", Status.NOT_REACHED),
        baseline_execution=VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint({"length": 10.0, "width": 4.0}),
        ),
    )

    assert result.status is Status.FAILED
    assert not any(check.status is Status.NOT_SCOREABLE for check in result.checks)
    assert any(
        check.check_id.endswith("parameter_response")
        and check.status is Status.NOT_REACHED
        for check in result.checks
    )


def test_l2_without_qualified_domains_is_explicitly_not_scoreable(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    for parameter in l2_case["parameter_system"]["parameters"]:
        parameter["domain"] = None
    calls = 0

    def execute(parameters: Mapping[str, float], variant_id: str) -> VariantExecution:
        nonlocal calls
        del parameters, variant_id
        calls += 1
        return VariantExecution(Status.PASSED, fingerprint="unused")

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=execute,
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )
    assert result.status is Status.NOT_SCOREABLE
    assert result.metrics["variant_count"] == 0
    assert calls == 0


def test_l2_does_not_guess_variant_requirement_dependencies(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    l2_case["parameter_system"]["parameters"][0]["variant_requirement_contract"] = None
    calls = 0

    def execute(*_args: Any) -> VariantExecution:
        nonlocal calls
        calls += 1
        return VariantExecution(Status.PASSED, fingerprint="unused")

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=execute,
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    assert result.status is Status.NOT_SCOREABLE
    assert calls == 0
    dependency = next(
        check
        for check in result.checks
        if check.check_id == "L2.variant_requirement_contracts"
    )
    assert dependency.observed["missing_parameter_ids"] == ["P-001"]


def test_complete_l2_passes_with_parameter_responsive_submission(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    def execute(parameters: Mapping[str, float], variant_id: str) -> VariantExecution:
        del variant_id
        return VariantExecution(Status.PASSED, fingerprint=fingerprint(parameters))

    def l1(
        execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult:
        del execution
        assert not context.nominal_reference_allowed
        assert context.parameter_matched_reference is None
        return passing_layer("L1")

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=execute,
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=l1,
    )
    assert result.status is Status.PASSED
    assert result.metrics["parameter_response_coverage"] == 1.0
    assert result.metrics["variant_count"] > 0


def test_l2_refuses_partial_parameter_coverage_before_replay(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    l2_case["evaluator_profile"]["l2"]["max_variants"] = 1
    calls = 0

    def execute(*_args: Any) -> VariantExecution:
        nonlocal calls
        calls += 1
        return VariantExecution(Status.PASSED, fingerprint="unused")

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=execute,
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    coverage = next(
        check
        for check in result.checks
        if check.check_id == "L2.schedule.parameter_coverage"
    )
    assert result.status is Status.NOT_SCOREABLE
    assert coverage.status is Status.NOT_SCOREABLE
    assert coverage.observed["missing_parameter_ids"]
    assert result.metrics["parameter_response_coverage"] < 1.0
    assert calls == 0


def test_l2_rejects_infeasible_nominal_contract_before_execution(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    l2_case["parameter_system"]["feasibility_relations"] = [
        {
            "relation_id": "length_must_be_below_nine",
            "predicate": _parameter_upper_bound(9.0),
            "authority": "synthetic_test_oracle",
            "evidence_ref": "tests/l2/domain.json",
            "required": True,
        }
    ]
    calls = 0

    def execute(*_args: Any) -> VariantExecution:
        nonlocal calls
        calls += 1
        return VariantExecution(Status.PASSED, fingerprint="unused")

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=execute,
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    assert result.status is Status.NOT_SCOREABLE
    assert calls == 0
    feasibility = next(
        check for check in result.checks if check.check_id == "L2.feasibility_contract"
    )
    assert "nominal assignment violates" in feasibility.summary


def test_l2_uses_directed_probes_for_discrete_parameter_system(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    l2_case["parameter_system"]["parameters"][1]["domain"] = {
        "kind": "discrete",
        "values": [2.0, 4.0, 8.0],
        "authority": "synthetic_test_oracle",
        "evidence_ref": "tests/l2/fixtures/domain.json",
    }

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=lambda parameters, _variant_id: VariantExecution(
            Status.PASSED, fingerprint=fingerprint(parameters)
        ),
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    assert result.status is Status.PASSED
    morris = next(
        check for check in result.checks if check.check_id == "L2.schedule.morris"
    )
    assert morris.status is Status.NOT_APPLICABLE
    assert result.metrics["variant_count"] > 0


def test_l2_rejects_hard_coded_submission(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=lambda *_: VariantExecution(
            Status.PASSED, fingerprint="hard-coded-shape"
        ),
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )
    assert result.status is Status.FAILED
    assert result.metrics["failed_variant_count"] > 0


def _parameter_upper_bound(value: float) -> dict[str, Any]:
    return {
        "type": "comparison",
        "op": "le",
        "left": {
            "type": "variable",
            "source": "parameter",
            "key": "P-001",
        },
        "right": {"type": "constant", "value": value},
    }


def test_l2_proves_explicit_solver_obligation(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    l2_case["parameter_system"]["solver_obligations"] = [
        {
            "obligation_id": "length_within_upper_domain",
            "assumptions": [],
            "conclusion": _parameter_upper_bound(15.0),
            "authority": "synthetic_test_oracle",
            "evidence_ref": "tests/l2/domain.json",
            "required": True,
        }
    ]

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=lambda parameters, _variant_id: VariantExecution(
            Status.PASSED, fingerprint=fingerprint(parameters)
        ),
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    assert result.status is Status.PASSED
    assert result.metrics["solver_checked_count"] == 1
    assert result.metrics["solver_proved_count"] == 1
    assert result.metrics["z3_success_rate"] == 1.0


def test_l2_treats_solver_counterexample_as_dataset_qualification_failure(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    l2_case["parameter_system"]["solver_obligations"] = [
        {
            "obligation_id": "invalid_upper_bound",
            "assumptions": [],
            "conclusion": _parameter_upper_bound(9.0),
            "authority": "synthetic_test_oracle",
            "evidence_ref": "tests/l2/domain.json",
            "required": True,
        }
    ]

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=lambda parameters, _variant_id: VariantExecution(
            Status.PASSED, fingerprint=fingerprint(parameters)
        ),
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    assert result.status is Status.NOT_SCOREABLE
    counterexample = next(
        check
        for check in result.checks
        if check.check_id.endswith("invalid_upper_bound")
    )
    assert counterexample.failure_code is None
    assert "dataset qualification failed" in counterexample.summary
    assert counterexample.observed["counterexample"]


def test_l2_derives_dependent_parameters_before_model_execution(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    derived = l2_case["parameter_system"]["parameters"][1]
    derived["parameter_kind"] = "derived"
    derived["domain"] = None
    derived["variant_requirement_contract"] = None
    derived["dependencies"] = ["P-001"]
    derived["derived_expression"] = {
        "type": "arithmetic",
        "op": "mul",
        "args": [
            {"type": "constant", "value": 0.4},
            {"type": "variable", "source": "parameter", "key": "P-001"},
        ],
    }
    l2_case["parameter_system"]["relations"] = [
        {
            "relation_id": "derived_width_readback",
            "predicate": {
                "type": "comparison",
                "op": "eq",
                "left": {
                    "type": "variable",
                    "source": "measurement",
                    "key": "width",
                },
                "right": {
                    "type": "variable",
                    "source": "parameter",
                    "key": "P-002",
                },
                "tolerance": 1e-9,
            },
            "authority": "synthetic_test_oracle",
            "evidence_ref": "tests/l2/derived.json",
            "required": True,
        }
    ]

    bindings = bind_parameter_manifest(l2_case["parameter_system"], submission_contract)
    translated = model_parameter_values(bindings, {"P-001": 12.0})
    assert translated["length"] == 12.0
    assert translated["width"] == pytest.approx(4.8)

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=lambda parameters, _variant_id: VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint(parameters),
            measurements={"width": parameters["width"]},
        ),
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    assert result.status is Status.PASSED
    assert result.metrics["l2_i_independent_parameter_count"] == 1
    assert result.metrics["l2_i_derived_parameter_count"] == 1
    assert result.metrics["l2_c_propagation_integrity_score"] == 1.0


def test_l2_executes_explicit_invalid_region_rejection(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    l2_case["parameter_system"]["invalid_probes"] = [
        {
            "probe_id": "negative_length",
            "assignments": {"P-001": -1.0, "P-002": 4.0},
            "expected_behavior": "reject",
            "safe_measurements": [],
            "allowed_failure_codes": ["invalid_parameter"],
            "authority": "synthetic_test_oracle",
            "evidence_ref": "tests/l2/invalid.json",
            "required": True,
        }
    ]

    def execute(parameters: Mapping[str, float], variant_id: str) -> VariantExecution:
        del variant_id
        if parameters["length"] < 0.0:
            return VariantExecution(
                Status.FAILED,
                failure_code="invalid_parameter",
            )
        return VariantExecution(Status.PASSED, fingerprint=fingerprint(parameters))

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=execute,
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    assert result.status is Status.PASSED
    assert result.metrics["l2_p_invalid_probe_count"] == 1
    assert result.metrics["l2_p_invalid_handling_rate"] == 1.0


def test_l2_accepts_only_measurement_bound_safe_invalid_handling(
    l2_case: dict[str, Any], submission_contract: dict[str, Any]
) -> None:
    l2_case["parameter_system"]["invalid_probes"] = [
        {
            "probe_id": "clamped_length",
            "assignments": {"P-001": 100.0, "P-002": 4.0},
            "expected_behavior": "safe_output",
            "safe_measurements": [{"key": "length", "value": 15.0, "tolerance": 1e-9}],
            "allowed_failure_codes": [],
            "authority": "synthetic_test_oracle",
            "evidence_ref": "tests/l2/invalid.json",
            "required": True,
        }
    ]

    def execute(parameters: Mapping[str, float], variant_id: str) -> VariantExecution:
        del variant_id
        effective_length = min(parameters["length"], 15.0)
        effective = {**parameters, "length": effective_length}
        return VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint(effective),
            measurements={"length": effective_length},
        )

    result = evaluate_l2(
        l2_case,
        submission_contract,
        execute=execute,
        evaluate_l0=lambda *_: passing_layer("L0"),
        evaluate_l1=lambda *_: passing_layer("L1"),
    )

    assert result.status is Status.PASSED
    assert result.metrics["l2_p_invalid_handling_rate"] == 1.0
