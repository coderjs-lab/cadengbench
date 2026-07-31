from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.geometry.step import StepGeometryError
from cadengbench_p_eval.parametrics.domains import parse_domain
from cadengbench_p_eval.parametrics.functional import (
    FunctionalEditError,
    FunctionalEvaluationContext,
    HookOutcome,
    evaluate_functional_edit,
    parse_functional_edit_contract,
)
from cadengbench_p_eval.parametrics.relations import (
    MissingRelationValue,
    RelationContext,
    RelationError,
    parse_predicate,
    parse_relation,
    prove_implication,
)
from cadengbench_p_eval.protocols.functional_edit import (
    FunctionalEditAdapter,
    FunctionalSourceResult,
    expected_solid_count,
)


def qualified_domain(minimum: float, maximum: float) -> dict[str, object]:
    return {
        "kind": "continuous",
        "minimum": minimum,
        "maximum": maximum,
        "minimum_inclusive": True,
        "maximum_inclusive": True,
        "authority": "synthetic_test_oracle",
        "evidence_ref": "tests/l2/fixtures/domain.json",
    }


def test_expected_solid_count_falls_back_to_verified_baseline_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FunctionalSourceResult(
        normalized_result={
            "checks": [
                {
                    "check_id": "core.solid_count_preserved",
                    "passed": True,
                    "evidence": {"passed": True},
                }
            ]
        },
        source_result={},
        acceptance_contract={},
        candidate_step=None,
        baseline_step=Path("baseline.step"),
    )
    monkeypatch.setattr(
        "cadengbench_p_eval.protocols.functional_edit.load_step",
        lambda path: SimpleNamespace(solids=(object(), object())),
    )

    assert expected_solid_count(source) == 2


def test_candidate_step_geometry_analysis_error_is_a_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(FunctionalEditAdapter)
    adapter.case = SimpleNamespace(
        data={
            "evaluator_profile": {
                "geometry": {"seed": 0, "surface_sample_count": 64}
            }
        }
    )
    monkeypatch.setattr(
        "cadengbench_p_eval.protocols.functional_edit.load_step",
        lambda _path: (_ for _ in ()).throw(
            StepGeometryError(
                "trimmed B-Rep face could not provide enough valid surface samples"
            )
        ),
    )

    result = adapter._analyze_variant(
        tmp_path / "candidate.step",
        value=1.0,
        details={},
    )

    assert result.status is Status.FAILED
    assert result.failure_code == "geometry_analysis_failed"


def variable(source: str, key: str) -> dict[str, str]:
    return {"type": "variable", "source": source, "key": key}


def constant(value: float) -> dict[str, object]:
    return {"type": "constant", "value": value}


def comparison(
    op: str,
    left: dict[str, object] | dict[str, str],
    right: dict[str, object] | dict[str, str],
    tolerance: float = 0.0,
) -> dict[str, object]:
    return {
        "type": "comparison",
        "op": op,
        "left": left,
        "right": right,
        "tolerance": tolerance,
    }


def relation_record(
    relation_id: str, predicate: dict[str, object]
) -> dict[str, object]:
    return {
        "relation_id": relation_id,
        "predicate": predicate,
        "authority": "synthetic_test_oracle",
        "evidence_ref": "tests/l2/fixtures/relation.json",
        "required": True,
    }


def test_relation_dsl_evaluates_parameter_measurement_binding() -> None:
    twice_radius = {
        "type": "arithmetic",
        "op": "mul",
        "args": [constant(2.0), variable("parameter", "radius")],
    }
    relation = parse_relation(
        relation_record(
            "diameter_binding",
            comparison("eq", variable("measurement", "diameter"), twice_radius, 1e-6),
        )
    )
    assert relation.evaluate(
        RelationContext(parameters={"radius": 3.0}, measurements={"diameter": 6.0})
    )
    assert not relation.evaluate(
        RelationContext(parameters={"radius": 3.0}, measurements={"diameter": 7.0})
    )


def test_relation_dsl_rejects_code_and_missing_values() -> None:
    with pytest.raises(RelationError):
        parse_predicate(
            comparison(
                "eq",
                {"type": "python", "code": "__import__('os').system('id')"},
                constant(0.0),
            )
        )
    relation = parse_relation(
        relation_record(
            "missing", comparison("eq", variable("measurement", "x"), constant(1.0))
        )
    )
    with pytest.raises(MissingRelationValue):
        relation.evaluate(RelationContext())


def test_z3_implication_proves_and_returns_counterexample() -> None:
    domains = {
        "radius": parse_domain(qualified_domain(1.0, 5.0), nominal=2.0),
        "diameter": parse_domain(qualified_domain(2.0, 10.0), nominal=4.0),
    }
    twice_radius = {
        "type": "arithmetic",
        "op": "mul",
        "args": [constant(2.0), variable("parameter", "radius")],
    }
    assumption = parse_predicate(
        comparison("eq", variable("parameter", "diameter"), twice_radius)
    )
    proved = prove_implication(
        domains,
        [assumption],
        parse_predicate(
            comparison(
                "ge",
                variable("parameter", "diameter"),
                variable("parameter", "radius"),
            )
        ),
    )
    refuted = prove_implication(
        domains,
        [assumption],
        parse_predicate(
            comparison(
                "lt",
                variable("parameter", "diameter"),
                variable("parameter", "radius"),
            )
        ),
    )
    assert proved.status == "proved"
    assert refuted.status == "counterexample"
    assert set(refuted.counterexample) == {"radius", "diameter"}


def test_z3_does_not_accept_vacuous_inconsistent_assumptions() -> None:
    domains = {"radius": parse_domain(qualified_domain(1.0, 5.0), nominal=2.0)}
    assumptions = [
        parse_predicate(
            comparison("lt", variable("parameter", "radius"), constant(0.0))
        )
    ]

    result = prove_implication(
        domains,
        assumptions,
        parse_predicate(
            comparison("ge", variable("parameter", "radius"), constant(1.0))
        ),
    )

    assert result.status == "infeasible_assumptions"
    assert not result.proved


def functional_contract() -> dict[str, object]:
    return {
        "contract_id": "functional-edit-001",
        "predicates": [
            {
                "predicate_id": "target_value",
                "category": "target",
                "authority": "verified_replay",
                "evidence_ref": "target.json",
                "required": True,
                "evaluator": {
                    "kind": "relation",
                    "predicate": comparison(
                        "eq",
                        variable("measurement", "edited_diameter"),
                        variable("parameter", "requested_diameter"),
                        1e-6,
                    ),
                },
            },
            {
                "predicate_id": "flow_area_increased",
                "category": "effect",
                "authority": "analytic_geometry",
                "evidence_ref": "effect.json",
                "required": True,
                "evaluator": {
                    "kind": "relation",
                    "predicate": comparison(
                        "gt",
                        variable("measurement", "flow_area"),
                        variable("baseline_measurement", "flow_area"),
                    ),
                },
            },
            {
                "predicate_id": "solid_count_preserved",
                "category": "preservation",
                "authority": "brep_measurement",
                "evidence_ref": "preservation.json",
                "required": True,
                "evaluator": {
                    "kind": "hook",
                    "hook_id": "preserve_solid_count",
                    "options": {},
                },
            },
        ],
    }


def test_functional_edit_checks_target_effect_and_preservation() -> None:
    contract = parse_functional_edit_contract(functional_contract())
    context = FunctionalEvaluationContext(
        parameters={"requested_diameter": 8.0},
        measurements={"edited_diameter": 8.0, "flow_area": 50.0},
        baseline_measurements={"flow_area": 30.0, "solid_count": 1.0},
        metadata={"solid_count": 1.0},
    )

    def preserve_solid_count(
        value: FunctionalEvaluationContext, options: object
    ) -> HookOutcome:
        del options
        observed = value.metadata["solid_count"]
        expected = value.baseline_measurements["solid_count"]
        return HookOutcome(observed == expected, expected=expected, observed=observed)

    result = evaluate_functional_edit(
        contract, context, hooks={"preserve_solid_count": preserve_solid_count}
    )
    assert result.status is Status.PASSED
    assert {check.status for check in result.checks} == {Status.PASSED}


def test_functional_edit_missing_hook_is_not_scoreable() -> None:
    contract = parse_functional_edit_contract(functional_contract())
    context = FunctionalEvaluationContext(
        parameters={"requested_diameter": 8.0},
        measurements={"edited_diameter": 8.0, "flow_area": 50.0},
        baseline_measurements={"flow_area": 30.0},
    )
    result = evaluate_functional_edit(contract, context)
    assert result.status is Status.NOT_SCOREABLE
    assert any(check.status is Status.NOT_SCOREABLE for check in result.checks)


def test_functional_contract_requires_all_three_obligations() -> None:
    raw = functional_contract()
    raw["predicates"] = raw["predicates"][:2]
    with pytest.raises(FunctionalEditError, match="preservation"):
        parse_functional_edit_contract(raw)
