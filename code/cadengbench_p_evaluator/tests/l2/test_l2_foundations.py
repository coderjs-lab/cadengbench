from __future__ import annotations

from pathlib import Path

import pytest

from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status
from cadengbench_p_eval.parametrics.domains import (
    DomainError,
    domain_probe_values,
    normalize_assignment,
    parse_domain,
)
from cadengbench_p_eval.parametrics.relations import (
    RelationContext,
    parse_predicate,
    parse_relation,
    prove_implication,
)
from cadengbench_p_eval.parametrics.replay import (
    CanonicalGeometryEvidence,
    ParameterBinding,
    VariantEvaluationContext,
    VariantExecution,
    replay_schedule,
)
from cadengbench_p_eval.parametrics.schedules import (
    PerturbationSchedule,
    ScheduleError,
    ScheduleVariant,
    combine_schedules,
    deduplicate_schedule,
    generate_coupled_schedule,
    generate_directed_schedule,
    generate_morris_schedule,
)


def _domain(minimum: float, maximum: float) -> dict[str, object]:
    return {
        "kind": "continuous",
        "minimum": minimum,
        "maximum": maximum,
        "minimum_inclusive": True,
        "maximum_inclusive": True,
        "authority": "synthetic_test_oracle",
        "evidence_ref": "tests/l2/fixtures/domain.json",
    }


def _variable(key: str, *, source: str = "parameter") -> dict[str, str]:
    return {"type": "variable", "source": source, "key": key}


def _constant(value: float) -> dict[str, object]:
    return {"type": "constant", "value": value}


def _comparison(
    op: str,
    left: dict[str, object] | dict[str, str],
    right: dict[str, object] | dict[str, str],
) -> dict[str, object]:
    return {
        "type": "comparison",
        "op": op,
        "left": left,
        "right": right,
    }


def _passing_layer(name: str) -> LayerResult:
    return LayerResult.from_checks(
        name, (CheckResult(f"{name}.synthetic", Status.PASSED),)
    )


def _binding(nominal: float = 5.0) -> dict[str, ParameterBinding]:
    return {
        "P-001": ParameterBinding(
            parameter_id="P-001",
            name="length",
            unit="mm",
            role="principal_dimension",
            nominal=nominal,
        )
    }


def _single_variant_schedule(value: float = 8.0) -> PerturbationSchedule:
    return PerturbationSchedule(
        kind="directed",
        variants=(
            ScheduleVariant(
                variant_id="directed-0000-P-001",
                schedule_kind="directed",
                assignments={"P-001": value},
                changed_parameters=("P-001",),
                predecessor_id="nominal",
            ),
        ),
    )


def test_assignment_normalization_and_probe_values_are_domain_strict() -> None:
    continuous = parse_domain(_domain(0.0, 10.0), nominal=5.0)
    assert normalize_assignment({"P-001": continuous}, {"P-001": 3}) == {"P-001": 3.0}
    assert domain_probe_values(continuous, count=3) == (0.0, 10.0, 5.0)

    with pytest.raises(DomainError, match="outside"):
        normalize_assignment({"P-001": continuous}, {"P-001": 11.0})
    with pytest.raises(DomainError, match="missing"):
        normalize_assignment({"P-001": continuous}, {})
    with pytest.raises(DomainError, match="unknown"):
        normalize_assignment({"P-001": continuous}, {"P-001": 5.0, "P-002": 2.0})


def test_combined_schedule_deduplicates_assignments_and_preserves_origins() -> None:
    directed = PerturbationSchedule(
        kind="directed",
        variants=(
            ScheduleVariant(
                "directed-a",
                "directed",
                {"P-001": 1.0},
                ("P-001",),
                predecessor_id="nominal",
            ),
        ),
    )
    morris = PerturbationSchedule(
        kind="morris",
        variants=(
            ScheduleVariant(
                "morris-a",
                "morris",
                {"P-001": 1.0},
                ("P-001",),
                predecessor_id="morris-start",
                trajectory_index=0,
                step_index=1,
            ),
        ),
        seed=7,
    )

    combined = combine_schedules(directed, morris)

    assert len(combined.variants) == 1
    assert combined.metadata["duplicate_assignment_count"] == 1
    assert {
        (origin.schedule_kind, origin.variant_id)
        for origin in combined.variants[0].origins
    } == {("directed", "directed-a"), ("morris", "morris-a")}
    assert combined.covered_parameters == ("P-001",)


def test_deduplication_remaps_predecessor_to_retained_assignment() -> None:
    schedule = PerturbationSchedule(
        kind="synthetic",
        variants=(
            ScheduleVariant("first", "a", {"P-001": 1.0}, ("P-001",)),
            ScheduleVariant("duplicate", "b", {"P-001": 1.0}, ("P-001",)),
            ScheduleVariant(
                "successor",
                "b",
                {"P-001": 2.0},
                ("P-001",),
                predecessor_id="duplicate",
            ),
        ),
    )

    deduplicated = deduplicate_schedule(schedule)

    assert [variant.variant_id for variant in deduplicated.variants] == [
        "first",
        "successor",
    ]
    assert deduplicated.variants[1].predecessor_id == "first"
    assert deduplicated.variants[1].origins[0].predecessor_id == "first"


def test_directed_and_morris_accept_typed_feasibility_predicates() -> None:
    domains = {"P-001": parse_domain(_domain(0.0, 10.0), nominal=5.0)}
    directed_predicate = parse_predicate(
        _comparison("le", _variable("P-001"), _constant(7.0))
    )
    directed = generate_directed_schedule(
        domains,
        {"P-001": 5.0},
        feasibility=directed_predicate,
    )
    assert [variant.assignments["P-001"] for variant in directed.variants] == [0.0]
    assert directed.metadata["feasibility_kind"] == "typed"

    unit_domains = {"P-001": parse_domain(_domain(0.0, 1.0), nominal=0.5)}
    morris_predicate = parse_predicate(
        _comparison("le", _variable("P-001"), _constant(2.0 / 3.0))
    )
    morris = generate_morris_schedule(
        unit_domains,
        levels=4,
        trajectories=2,
        seed=11,
        feasibility=morris_predicate,
        max_attempts=100,
    )
    context = RelationContext()
    for variant in morris.variants:
        context = RelationContext(parameters=variant.assignments)
        assert morris_predicate.evaluate(context)
    assert morris.metadata["feasibility_kind"] == "typed"
    assert morris.metadata["input_variant_count"] == 4
    assert morris.metadata["deduplicated_variant_count"] == 2
    assert all(len(variant.origins) == 2 for variant in morris.variants)


def test_typed_feasibility_with_nonparameter_input_fails_as_contract_error() -> None:
    domains = {"P-001": parse_domain(_domain(0.0, 10.0), nominal=5.0)}
    invalid = parse_predicate(
        _comparison("gt", _variable("clearance", source="measurement"), _constant(0.0))
    )
    with pytest.raises(ScheduleError, match="invalid frozen feasibility"):
        generate_directed_schedule(domains, {"P-001": 5.0}, feasibility=invalid)


def test_coupled_schedule_is_deterministic_and_relation_feasible() -> None:
    domains = {
        "radius": parse_domain(_domain(1.0, 5.0), nominal=3.0),
        "diameter": parse_domain(_domain(2.0, 10.0), nominal=6.0),
    }
    twice_radius = {
        "type": "arithmetic",
        "op": "mul",
        "args": [_constant(2.0), _variable("radius")],
    }
    predicate = parse_predicate(_comparison("eq", _variable("diameter"), twice_radius))

    first = generate_coupled_schedule(
        domains,
        {"radius": 3.0, "diameter": 6.0},
        feasibility=predicate,
    )
    second = generate_coupled_schedule(
        domains,
        {"radius": 3.0, "diameter": 6.0},
        feasibility=predicate,
    )

    assert first.as_dict() == second.as_dict()
    assert len(first.variants) == 2
    assert all(
        variant.changed_parameters == ("diameter", "radius")
        for variant in first.variants
    )
    assert all(
        predicate.evaluate(RelationContext(parameters=variant.assignments))
        for variant in first.variants
    )
    assert first.metadata["rejected_infeasible_variants"] == 2


def test_smt_checks_antecedent_before_translating_conclusion() -> None:
    domains = {"radius": parse_domain(_domain(1.0, 5.0), nominal=2.0)}
    infeasible = parse_predicate(_comparison("lt", _variable("radius"), _constant(0.0)))
    unsupported_conclusion = parse_predicate(
        _comparison("gt", _variable("observed", source="measurement"), _constant(0.0))
    )

    result = prove_implication(domains, [infeasible], unsupported_conclusion)

    assert result.status == "infeasible_assumptions"
    assert not result.proved


def test_smt_invalid_empty_domain_contract_is_evaluator_owned() -> None:
    result = prove_implication(
        {}, [], parse_predicate(_comparison("eq", _constant(1.0), _constant(1.0)))
    )
    assert result.status == "invalid_contract"
    assert not result.proved


def test_smt_substitutes_derived_parameter_dag() -> None:
    domains = {"radius": parse_domain(_domain(1.0, 5.0), nominal=2.0)}
    conclusion = parse_predicate(
        _comparison(
            "eq",
            _variable("diameter"),
            {
                "type": "arithmetic",
                "op": "mul",
                "args": [_constant(2.0), _variable("radius")],
            },
        )
    )
    proof = prove_implication(
        domains,
        [],
        conclusion,
        derived_expressions={
            "diameter": {
                "type": "arithmetic",
                "op": "mul",
                "args": [_constant(2.0), _variable("radius")],
            }
        },
    )

    assert proof.proved


def test_replay_ignores_raw_step_bytes_for_geometry_response(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.step"
    variant_path = tmp_path / "variant.step"
    baseline_path.write_bytes(b"baseline raw STEP bytes")
    variant_path.write_bytes(b"different raw STEP bytes")

    result = replay_schedule(
        _single_variant_schedule(),
        _binding(),
        execute=lambda *_: VariantExecution(
            Status.PASSED,
            fingerprint="same-canonical-brep",
            artifact=str(variant_path),
        ),
        evaluate_l0=lambda *_: _passing_layer("L0"),
        evaluate_l1=lambda *_: _passing_layer("L1"),
        baseline_execution=VariantExecution(
            Status.PASSED,
            fingerprint="same-canonical-brep",
            artifact=str(baseline_path),
        ),
    )

    response = next(
        check
        for check in result.checks
        if check.check_id.endswith("parameter_response")
    )
    assert response.status is Status.FAILED
    assert response.failure_code == "parameter_ignored"
    assert response.observed["raw_artifact_bytes_used"] is False


def test_replay_uses_canonical_geometry_hook_measurements() -> None:
    relation = parse_relation(
        {
            "relation_id": "length_readback",
            "predicate": _comparison(
                "eq",
                _variable("length", source="measurement"),
                _variable("P-001"),
            ),
            "authority": "synthetic_test_oracle",
            "evidence_ref": "tests/l2/fixtures/relation.json",
            "required": True,
        }
    )

    def canonical_geometry(
        execution: VariantExecution, context: VariantEvaluationContext
    ) -> CanonicalGeometryEvidence:
        del execution
        assignments = context.public_assignments
        return CanonicalGeometryEvidence(
            measurements={"length": assignments["P-001"]},
            source="test_brep_measurement_hook",
        )

    result = replay_schedule(
        _single_variant_schedule(),
        _binding(),
        execute=lambda *_: VariantExecution(
            Status.PASSED,
            fingerprint="raw-byte-hash-that-must-be-ignored",
            measurements={"length": -100.0},
        ),
        evaluate_l0=lambda *_: _passing_layer("L0"),
        evaluate_l1=lambda *_: _passing_layer("L1"),
        relations=(relation,),
        baseline_execution=VariantExecution(
            Status.PASSED,
            fingerprint="another-raw-byte-hash",
            measurements={"length": -100.0},
        ),
        canonical_geometry=canonical_geometry,
    )

    assert result.status is Status.PASSED
    response = next(
        check
        for check in result.checks
        if check.check_id.endswith("parameter_response")
    )
    assert response.observed["fingerprint_changed"] is None
    assert response.observed["measurements_changed"] is True
    assert response.observed["candidate_evidence_source"] == (
        "test_brep_measurement_hook"
    )


def test_canonical_geometry_hook_failure_is_evaluator_owned() -> None:
    def failing_hook(
        execution: VariantExecution, context: VariantEvaluationContext
    ) -> CanonicalGeometryEvidence:
        del execution, context
        raise RuntimeError("synthetic B-Rep extractor failure")

    result = replay_schedule(
        _single_variant_schedule(),
        _binding(),
        execute=lambda *_: VariantExecution(Status.PASSED, fingerprint="variant"),
        evaluate_l0=lambda *_: _passing_layer("L0"),
        evaluate_l1=lambda *_: _passing_layer("L1"),
        baseline_execution=VariantExecution(Status.PASSED, fingerprint="baseline"),
        canonical_geometry=failing_hook,
    )

    assert result.status is Status.EVALUATOR_ERROR
    assert not any(check.status is Status.FAILED for check in result.checks)
    assert any(
        check.check_id == "L2.variant.nominal.canonical_geometry"
        and check.status is Status.EVALUATOR_ERROR
        for check in result.checks
    )


def test_malformed_runtime_relation_is_not_a_model_failure() -> None:
    divide_by_zero = {
        "type": "arithmetic",
        "op": "div",
        "args": [_constant(1.0), _constant(0.0)],
    }
    relation = parse_relation(
        {
            "relation_id": "invalid_frozen_relation",
            "predicate": _comparison(
                "eq", _variable("value", source="measurement"), divide_by_zero
            ),
            "authority": "synthetic_test_oracle",
            "evidence_ref": "tests/l2/fixtures/relation.json",
            "required": True,
        }
    )

    result = replay_schedule(
        _single_variant_schedule(),
        _binding(),
        execute=lambda *_: VariantExecution(
            Status.PASSED,
            fingerprint="changed",
            measurements={"value": 1.0},
        ),
        evaluate_l0=lambda *_: _passing_layer("L0"),
        evaluate_l1=lambda *_: _passing_layer("L1"),
        relations=(relation,),
        baseline_execution=VariantExecution(
            Status.PASSED,
            fingerprint="baseline",
            measurements={"value": 1.0},
        ),
    )

    relation_check = next(
        check
        for check in result.checks
        if check.check_id.endswith("relation.invalid_frozen_relation")
    )
    assert relation_check.status is Status.NOT_SCOREABLE
    assert relation_check.failure_code is None
    assert result.status is Status.NOT_SCOREABLE
