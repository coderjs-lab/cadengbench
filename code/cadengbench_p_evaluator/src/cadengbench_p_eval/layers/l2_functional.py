"""Protocol-E L2: deterministic functional edit and post-edit robustness."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status
from cadengbench_p_eval.layers.l2 import replay_diagnostics
from cadengbench_p_eval.parametrics.domains import (
    DomainError,
    parse_parameter_domains,
)
from cadengbench_p_eval.parametrics.functional import (
    FunctionalEvaluationContext,
    HookOutcome,
    evaluate_functional_edit,
    parse_functional_edit_contract,
)
from cadengbench_p_eval.parametrics.relations import parse_relation
from cadengbench_p_eval.parametrics.replay import (
    ParameterBinding,
    replay_schedule,
)
from cadengbench_p_eval.parametrics.schedules import (
    ScheduleError,
    combine_schedules,
    generate_directed_schedule,
    generate_morris_schedule,
)
from cadengbench_p_eval.protocols.functional_edit import (
    FunctionalEditAdapter,
    FunctionalSourceResult,
)


def _source_check_hook(
    source: FunctionalSourceResult,
):
    checks = source.checks

    def hook(
        context: FunctionalEvaluationContext, options: Mapping[str, Any]
    ) -> HookOutcome:
        del context
        check_id = options.get("check_id")
        if not isinstance(check_id, str) or not check_id:
            raise ValueError("source-check hook requires check_id")
        record = checks.get(check_id)
        if record is None:
            return HookOutcome(
                False,
                summary=f"source adapter omitted {check_id}",
                expected={"source_check": check_id},
                observed=None,
            )
        return HookOutcome(
            record.get("passed") is True,
            summary=f"normalized deterministic source check: {check_id}",
            expected=record.get("expected"),
            observed=record.get("evidence"),
            evidence=[{"source_check_id": check_id}],
        )

    return hook


def _functional_contract(source: FunctionalSourceResult) -> dict[str, Any]:
    contract = source.acceptance_contract
    contract_id = str(contract.get("contract_id"))
    return {
        "contract_id": contract_id,
        "predicates": [
            {
                "predicate_id": "target_parameter_assignment",
                "category": "target",
                "authority": "verified_source_parameter_binding",
                "evidence_ref": (
                    "functional_edit_acceptance_contract#/parameter_contract"
                ),
                "required": True,
                "evaluator": {
                    "kind": "hook",
                    "hook_id": "source_check",
                    "options": {"check_id": "core.target_parameter_assignment"},
                },
            },
            {
                "predicate_id": "geometry_mediated_engineering_effect",
                "category": "effect",
                "authority": "verified_unique_target_replay",
                "evidence_ref": (
                    "functional_edit_acceptance_contract#/hidden_reference_artifacts/target_step"
                ),
                "required": True,
                "evaluator": {
                    "kind": "hook",
                    "hook_id": "source_check",
                    "options": {"check_id": "core.target_brep_similarity"},
                },
            },
            {
                "predicate_id": "parameter_direction",
                "category": "effect",
                "authority": "verified_source_parameter_binding",
                "evidence_ref": (
                    "functional_edit_acceptance_contract#/publication_projection/target_operation"
                ),
                "required": True,
                "evaluator": {
                    "kind": "relation",
                    "predicate": {
                        "type": "comparison",
                        "op": (
                            "gt"
                            if float(
                                source.acceptance_contract["parameter_contract"][
                                    "target_value"
                                ]
                            )
                            > float(
                                source.acceptance_contract["parameter_contract"][
                                    "base_value"
                                ]
                            )
                            else "lt"
                            if float(
                                source.acceptance_contract["parameter_contract"][
                                    "target_value"
                                ]
                            )
                            < float(
                                source.acceptance_contract["parameter_contract"][
                                    "base_value"
                                ]
                            )
                            else "eq"
                        ),
                        "left": {
                            "type": "variable",
                            "source": "parameter",
                            "key": "E-P-001",
                        },
                        "right": {
                            "type": "variable",
                            "source": "baseline_parameter",
                            "key": "E-P-001",
                        },
                        "tolerance": 1e-9,
                    },
                },
            },
            {
                "predicate_id": "non_target_source_preserved",
                "category": "preservation",
                "authority": "source_native_masked_digest",
                "evidence_ref": ("functional_edit_acceptance_contract#/core_checks"),
                "required": True,
                "evaluator": {
                    "kind": "hook",
                    "hook_id": "source_check",
                    "options": {"check_id": "core.non_target_source_preserved"},
                },
            },
            {
                "predicate_id": "solid_count_preserved",
                "category": "preservation",
                "authority": "deterministic_brep_measurement",
                "evidence_ref": ("functional_edit_acceptance_contract#/core_checks"),
                "required": True,
                "evaluator": {
                    "kind": "hook",
                    "hook_id": "source_check",
                    "options": {"check_id": "core.solid_count_preserved"},
                },
            },
        ],
    }


def _post_edit_schedule(
    case: Mapping[str, Any], domains: Mapping[str, Any], nominal: float
):
    profile = case["evaluator_profile"]["l2"]
    maximum = int(profile["max_variants"])
    parameter_count = len(domains)
    morris_budget = min(maximum, parameter_count + 1)
    schedules = []
    if morris_budget >= parameter_count + 1:
        schedules.append(
            generate_morris_schedule(
                domains,
                levels=int(profile["morris_levels"]),
                trajectories=max(1, int(profile["morris_trajectories"])),
                seed=int(profile["seed"]),
                max_variants=morris_budget,
            )
        )
    remaining = maximum - sum(len(schedule.variants) for schedule in schedules)
    if remaining:
        schedules.append(
            generate_directed_schedule(
                domains,
                {"E-P-001": nominal},
                points_per_parameter=max(
                    2, int(profile["directed_points_per_parameter"])
                ),
                max_variants=remaining,
            )
        )
    if not schedules:
        raise ScheduleError("post-edit profile produced no replay schedule")
    return schedules[0] if len(schedules) == 1 else combine_schedules(*schedules)


def evaluate_l2_functional(
    case: Mapping[str, Any],
    *,
    adapter: FunctionalEditAdapter,
    source: FunctionalSourceResult,
) -> LayerResult:
    """Evaluate target, effect, preservation, and edited-domain robustness."""

    checks: list[CheckResult] = []
    artifacts: dict[str, Any] = {
        "source_evaluation": source.normalized_result,
        "candidate_effect_contracts": source.acceptance_contract.get(
            "candidate_effect_contracts", []
        ),
        "candidate_preservation_contracts": source.acceptance_contract.get(
            "candidate_preservation_contracts", []
        ),
    }
    parameter = source.acceptance_contract.get("parameter_contract") or {}
    target = float(parameter["target_value"])
    base = float(parameter["base_value"])
    unit = str(parameter["canonical_unit"])

    checks.append(
        CheckResult(
            "L2.I.source_native_binding",
            Status.PASSED,
            expected={"parameter_id": parameter.get("parameter_id")},
            observed={
                "source_pointer": parameter.get("source_pointer"),
                "source_key": source.acceptance_contract.get("source_key"),
            },
            evidence=[
                {
                    "authority": "verified_edit_binding",
                    "evidence_ref": (
                        "functional_edit_acceptance_contract#/parameter_contract"
                    ),
                }
            ],
        )
    )
    core = parse_functional_edit_contract(_functional_contract(source))
    functional = evaluate_functional_edit(
        core,
        FunctionalEvaluationContext(
            parameters={"E-P-001": target},
            baseline_parameters={"E-P-001": base},
        ),
        hooks={"source_check": _source_check_hook(source)},
    )
    checks.extend(functional.checks)
    artifacts["functional_predicates"] = functional.as_dict()

    parameter_system = case.get("parameter_system")
    try:
        if not isinstance(parameter_system, Mapping):
            raise DomainError("functional case has no parameter system")
        domains = parse_parameter_domains(parameter_system, require_complete=True)
        schedule = _post_edit_schedule(case, domains, target)
    except (DomainError, KeyError, ScheduleError, TypeError, ValueError) as exc:
        checks.append(
            CheckResult(
                "L2.P.post_edit_schedule",
                Status.NOT_SCOREABLE,
                summary=str(exc),
            )
        )
        return LayerResult.from_checks("L2", checks, artifacts=artifacts)

    bindings = {
        "E-P-001": ParameterBinding(
            parameter_id="E-P-001",
            name="edit_value",
            unit=unit,
            role="functional_edit_target",
            nominal=target,
        )
    }
    relation = parse_relation(
        {
            "relation_id": "REL-E-P-001-MEASUREMENT",
            "authority": "source_native_replay",
            "evidence_ref": ("functional_edit_acceptance_contract#/parameter_contract"),
            "required": True,
            "predicate": {
                "type": "comparison",
                "op": "eq",
                "left": {
                    "type": "variable",
                    "source": "measurement",
                    "key": "E-P-001",
                },
                "right": {
                    "type": "variable",
                    "source": "parameter",
                    "key": "E-P-001",
                },
                "tolerance": 1e-6,
            },
        }
    )
    baseline = adapter.baseline_execution()
    replay = replay_schedule(
        schedule,
        bindings,
        execute=adapter.execute_variant,
        evaluate_l0=adapter.l0,
        evaluate_l1=adapter.l1,
        relations=(relation,),
        baseline_execution=baseline,
    )
    checks.append(
        CheckResult(
            "L2.P.post_edit_schedule",
            Status.PASSED,
            metrics={"variant_count": len(replay.variants)},
        )
    )
    checks.extend(replay.checks)
    artifacts["post_edit_schedule"] = schedule.as_dict()
    artifacts["post_edit_replay"] = replay.as_dict()
    replay_metrics, replay_artifacts = replay_diagnostics(replay, domains)
    artifacts.update(replay_artifacts)

    passed = sum(variant.status is Status.PASSED for variant in replay.variants)
    relation_checks = [
        check
        for variant in replay.variants
        for check in variant.checks
        if ".relation." in check.check_id
    ]
    relation_passed = sum(check.status is Status.PASSED for check in relation_checks)
    return LayerResult.from_checks(
        "L2",
        checks,
        metrics={
            "l2_i_binding_count": 1,
            "l2_p_variant_count": len(replay.variants),
            "replay_robustness": passed / len(replay.variants)
            if replay.variants
            else None,
            "l2_c_relation_count": len(relation_checks),
            "parameter_integrity_score": (
                relation_passed / len(relation_checks) if relation_checks else None
            ),
            "l2_e_predicate_count": len(functional.checks),
            "llm_judge_used": False,
            **replay_metrics,
        },
        artifacts=artifacts,
    )
