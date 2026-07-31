"""Public L2 evaluator: qualified parametric replay and functional edits."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status
from cadengbench_p_eval.parametrics.domains import (
    ContinuousDomain,
    DiscreteDomain,
    DomainError,
    UnqualifiedDomainError,
    parse_parameter_domains,
)
from cadengbench_p_eval.parametrics.functional import (
    FunctionalEditContract,
    FunctionalEditError,
    FunctionalEvaluationContext,
    FunctionalHook,
    evaluate_functional_edit,
    parse_functional_edit_contract,
)
from cadengbench_p_eval.parametrics.invalid import (
    InvalidProbeError,
    evaluate_invalid_probes,
    parse_invalid_probes,
)
from cadengbench_p_eval.parametrics.relations import (
    MissingRelationValue,
    RelationContext,
    RelationError,
    TypedRelation,
    parse_predicate,
    parse_relation,
    prove_implication,
)
from cadengbench_p_eval.parametrics.replay import (
    EvaluateL0,
    EvaluateL1,
    ExecuteVariant,
    ReplayError,
    ReplayResult,
    VariantExecution,
    bind_parameter_manifest,
    independent_parameter_values,
    model_parameter_values,
    morris_elementary_effects,
    replay_schedule,
    resolve_public_parameter_values,
)
from cadengbench_p_eval.parametrics.schedules import (
    PerturbationSchedule,
    ScheduleError,
    combine_schedules,
    generate_coupled_schedule,
    generate_directed_schedule,
    generate_morris_schedule,
)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _valid_variant_requirement_contract(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("kind")
        in {"direct_measurement_only", "geometry_response_only"}
        and isinstance(value.get("authority"), str)
        and bool(value.get("authority"))
        and isinstance(value.get("evidence_ref"), str)
        and bool(value.get("evidence_ref"))
        and set(value) == {"kind", "authority", "evidence_ref"}
    )


def _functional_checks(
    case: Mapping[str, Any],
    contract: FunctionalEditContract | Mapping[str, Any] | None,
    context: FunctionalEvaluationContext | None,
    hooks: Mapping[str, FunctionalHook] | None,
) -> tuple[list[CheckResult], dict[str, Any] | None]:
    available = bool((case.get("functional_edit") or {}).get("available", False))
    if contract is None:
        status = Status.NOT_SCOREABLE if available else Status.NOT_APPLICABLE
        return [
            CheckResult(
                "L2.functional_edit.contract",
                status,
                required=available,
                summary=(
                    "functional edit is declared but no normalized "
                    "deterministic contract was supplied"
                    if available
                    else "item has no functional edit contract"
                ),
            )
        ], None
    if context is None:
        return [
            CheckResult(
                "L2.functional_edit.evidence",
                Status.NOT_SCOREABLE,
                summary="functional edit measurements are absent",
            )
        ], None
    try:
        parsed = (
            contract
            if isinstance(contract, FunctionalEditContract)
            else parse_functional_edit_contract(contract)
        )
        result = evaluate_functional_edit(parsed, context, hooks=hooks)
    except (FunctionalEditError, RelationError) as exc:
        return [
            CheckResult(
                "L2.functional_edit.contract",
                Status.NOT_SCOREABLE,
                summary=str(exc),
            )
        ], None
    return list(result.checks), result.as_dict()


def _parse_relations(records: Sequence[Mapping[str, Any]]) -> tuple[TypedRelation, ...]:
    parsed = tuple(parse_relation(record) for record in records)
    identifiers = [relation.relation_id for relation in parsed]
    if len(identifiers) != len(set(identifiers)):
        raise RelationError("typed relation IDs must be unique")
    return parsed


def _solver_checks(
    records: Any,
    domains: Mapping[str, Any],
    *,
    timeout_ms: int,
    derived_expressions: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[CheckResult], list[dict[str, Any]], dict[str, Any]]:
    """Check explicit symbolic obligations; absence is not rewarded."""

    if records in (None, []):
        return (
            [
                CheckResult(
                    "L2.solver.applicability",
                    Status.NOT_APPLICABLE,
                    required=False,
                    summary="item declares no solver-eligible relation obligations",
                )
            ],
            [],
            {
                "solver_obligation_count": 0,
                "solver_checked_count": 0,
                "solver_proved_count": 0,
                "z3_success_rate": None,
            },
        )
    if not isinstance(records, list):
        return (
            [
                CheckResult(
                    "L2.solver.contract",
                    Status.NOT_SCOREABLE,
                    summary="solver_obligations must be a list",
                )
            ],
            [],
            {
                "solver_obligation_count": 0,
                "solver_checked_count": 0,
                "solver_proved_count": 0,
                "z3_success_rate": None,
            },
        )

    checks: list[CheckResult] = []
    proofs: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    checked = 0
    proved = 0
    for index, raw in enumerate(records):
        fallback_id = f"invalid-{index:04d}"
        obligation_id = (
            str(raw.get("obligation_id"))
            if isinstance(raw, Mapping) and raw.get("obligation_id")
            else fallback_id
        )
        check_id = f"L2.solver.{obligation_id}"
        required = bool(raw.get("required", True)) if isinstance(raw, Mapping) else True
        try:
            if not isinstance(raw, Mapping):
                raise RelationError("solver obligation must be an object")
            if obligation_id in identifiers or obligation_id == fallback_id:
                raise RelationError("solver obligation IDs must be unique and nonempty")
            if not isinstance(raw.get("required", True), bool):
                raise RelationError("solver obligation required must be a boolean")
            authority = raw.get("authority")
            evidence_ref = raw.get("evidence_ref")
            if not isinstance(authority, str) or not authority:
                raise RelationError("solver obligation authority is required")
            if not isinstance(evidence_ref, str) or not evidence_ref:
                raise RelationError("solver obligation evidence_ref is required")
            assumptions_raw = raw.get("assumptions", [])
            conclusion_raw = raw.get("conclusion")
            if not isinstance(assumptions_raw, list) or not isinstance(
                conclusion_raw, Mapping
            ):
                raise RelationError(
                    "solver obligation requires assumptions and one conclusion"
                )
            assumptions = tuple(parse_predicate(value) for value in assumptions_raw)
            conclusion = parse_predicate(conclusion_raw)
            identifiers.add(obligation_id)
            proof = prove_implication(
                domains,
                assumptions,
                conclusion,
                timeout_ms=timeout_ms,
                derived_expressions=derived_expressions,
            )
        except RelationError as exc:
            checks.append(
                CheckResult(
                    check_id,
                    Status.NOT_SCOREABLE,
                    required=required,
                    summary=f"invalid solver obligation: {exc}",
                )
            )
            proofs.append(
                {
                    "obligation_id": obligation_id,
                    "status": "invalid_contract",
                    "reason": str(exc),
                }
            )
            continue
        except Exception as exc:  # solver backend boundary
            checks.append(
                CheckResult(
                    check_id,
                    Status.EVALUATOR_ERROR,
                    required=required,
                    summary=f"solver backend failed: {type(exc).__name__}: {exc}",
                )
            )
            proofs.append(
                {
                    "obligation_id": obligation_id,
                    "status": "solver_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        proof_record = {
            "obligation_id": obligation_id,
            "status": proof.status,
            "counterexample": proof.counterexample,
            "reason": proof.reason,
            "authority": authority,
            "evidence_ref": evidence_ref,
            "timeout_ms": timeout_ms,
        }
        proofs.append(proof_record)
        if proof.status in {"proved", "counterexample"}:
            checked += 1
        if proof.proved:
            proved += 1
            status = Status.PASSED
            failure_code = None
            summary = "domain and assumptions imply the declared relation"
        elif proof.status == "counterexample":
            status = Status.NOT_SCOREABLE
            failure_code = None
            summary = (
                "dataset qualification failed: the declared relation is not "
                "implied by its frozen domain"
            )
        else:
            status = Status.NOT_SCOREABLE
            failure_code = None
            summary = f"solver proof is {proof.status}: {proof.reason or 'no reason'}"
        checks.append(
            CheckResult(
                check_id,
                status,
                required=required,
                failure_code=failure_code,
                summary=summary,
                observed=proof_record,
                evidence=[{"authority": authority, "evidence_ref": evidence_ref}],
            )
        )
    metrics = {
        "solver_obligation_count": len(records),
        "solver_checked_count": checked,
        "solver_proved_count": proved,
        "z3_success_rate": proved / checked if checked else None,
    }
    return checks, proofs, metrics


def _baseline_execution(
    execute: ExecuteVariant,
    bindings: Mapping[str, Any],
) -> VariantExecution:
    public_nominal = independent_parameter_values(bindings)
    try:
        model_nominal = model_parameter_values(bindings, public_nominal)
        result = execute(model_nominal, "nominal")
        if not isinstance(result, VariantExecution):
            raise TypeError("execution callback returned an invalid result")
        return result
    except Exception as exc:  # evaluator callback boundary
        return VariantExecution(
            Status.EVALUATOR_ERROR,
            details={"exception_type": type(exc).__name__, "message": str(exc)},
        )


def _baseline_check(execution: VariantExecution) -> CheckResult:
    return CheckResult(
        "L2.nominal_execution",
        execution.status,
        failure_code=execution.failure_code,
        observed=execution.as_dict(),
    )


def _rate(checks: Sequence[CheckResult]) -> float | None:
    executed = [
        check for check in checks if check.status in {Status.PASSED, Status.FAILED}
    ]
    if not executed:
        return None
    return sum(check.status is Status.PASSED for check in executed) / len(executed)


def _is_boundary_origin(
    assignments: Mapping[str, float],
    changed_parameters: Sequence[str],
    domains: Mapping[str, Any],
) -> bool:
    if len(changed_parameters) != 1:
        return False
    parameter_id = changed_parameters[0]
    domain = domains.get(parameter_id)
    value = assignments.get(parameter_id)
    if value is None:
        return False
    if isinstance(domain, ContinuousDomain):
        lower = domain.from_unit_interval(0.0)
        upper = domain.from_unit_interval(1.0)
        return math.isclose(value, lower, rel_tol=0.0, abs_tol=1e-12) or math.isclose(
            value, upper, rel_tol=0.0, abs_tol=1e-12
        )
    if isinstance(domain, DiscreteDomain):
        return value in {domain.values[0], domain.values[-1]}
    return False


def replay_diagnostics(
    replay: ReplayResult,
    domains: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response_checks = [
        check
        for variant in replay.variants
        for check in variant.checks
        if ".parameter_response" in check.check_id
        and check.status is not Status.NOT_APPLICABLE
    ]
    relation_checks = [
        check
        for variant in replay.variants
        for check in variant.checks
        if ".relation." in check.check_id
    ]
    origin_statuses: dict[str, list[Status]] = {
        "directed": [],
        "morris": [],
        "coupled": [],
    }
    boundary_statuses: list[Status] = []
    dfm_evaluated = 0
    dfm_preserved = 0
    failure_taxonomy: dict[str, int] = {}
    for variant in replay.variants:
        for origin in variant.variant.origins:
            if origin.schedule_kind in origin_statuses:
                origin_statuses[origin.schedule_kind].append(variant.status)
            if origin.schedule_kind == "directed" and _is_boundary_origin(
                variant.variant.assignments,
                origin.changed_parameters,
                domains,
            ):
                boundary_statuses.append(variant.status)
        dfm_checks = [
            check
            for check in variant.l1.checks
            if check.check_id.startswith("L1.DFM.")
            and check.status in {Status.PASSED, Status.FAILED}
        ]
        if dfm_checks:
            dfm_evaluated += 1
            if all(check.status is Status.PASSED for check in dfm_checks):
                dfm_preserved += 1
        if variant.execution.failure_code:
            failure_taxonomy[variant.execution.failure_code] = (
                failure_taxonomy.get(variant.execution.failure_code, 0) + 1
            )
        for check in variant.checks:
            if check.status is Status.FAILED:
                key = check.failure_code or check.check_id
                failure_taxonomy[key] = failure_taxonomy.get(key, 0) + 1

    def status_rate(values: Sequence[Status]) -> float | None:
        executed = [
            value for value in values if value in {Status.PASSED, Status.FAILED}
        ]
        return (
            sum(value is Status.PASSED for value in executed) / len(executed)
            if executed
            else None
        )

    effects = morris_elementary_effects(
        replay,
        domain_spans={
            parameter_id: domain.maximum - domain.minimum
            for parameter_id, domain in domains.items()
            if isinstance(domain, ContinuousDomain)
        },
    )
    metrics = {
        "l2_p_replay_robustness": status_rate(
            [variant.status for variant in replay.variants]
        ),
        "l2_p_parameter_response_rate": _rate(response_checks),
        "l2_p_directed_robustness": status_rate(origin_statuses["directed"]),
        "l2_p_morris_robustness": status_rate(origin_statuses["morris"]),
        "l2_p_coupled_robustness": status_rate(origin_statuses["coupled"]),
        "l2_p_boundary_robustness": status_rate(boundary_statuses),
        "l2_c_propagation_integrity_score": _rate(relation_checks),
        "l2_dfm_robustness": (dfm_preserved / dfm_evaluated if dfm_evaluated else None),
        "l2_dfm_evaluated_variant_count": dfm_evaluated,
        "morris_elementary_effect_count": len(effects),
    }
    artifacts = {
        "morris_elementary_effects": effects,
        "failure_taxonomy": dict(sorted(failure_taxonomy.items())),
        "component_check_counts": {
            "parameter_response": len(response_checks),
            "behavioral_relation": len(relation_checks),
            "boundary_origin": len(boundary_statuses),
            "dfm_evaluated_variant": dfm_evaluated,
        },
    }
    return metrics, artifacts


def _schedule_plan(
    domains: Mapping[str, Any],
    nominals: Mapping[str, float],
    config: Mapping[str, Any],
    *,
    feasibility: Callable[[Mapping[str, float]], bool] | None = None,
) -> tuple[list[PerturbationSchedule], list[CheckResult]]:
    maximum = _positive_int(config.get("max_variants"), "l2.max_variants")
    points = _positive_int(
        config.get("directed_points_per_parameter"),
        "l2.directed_points_per_parameter",
    )
    levels = _positive_int(config.get("morris_levels"), "l2.morris_levels")
    trajectories = _positive_int(
        config.get("morris_trajectories"), "l2.morris_trajectories"
    )
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("l2.seed must be an integer")

    parameter_count = len(domains)
    morris_points = parameter_count + 1
    schedules: list[PerturbationSchedule] = []
    checks: list[CheckResult] = []
    morris_applicable = all(
        isinstance(domain, ContinuousDomain) for domain in domains.values()
    )
    coupled_applicable = parameter_count >= 2
    minimum_directed = parameter_count
    minimum_morris = morris_points if morris_applicable else 0
    minimum_coupled = 1 if coupled_applicable else 0
    minimum_complete_budget = minimum_directed + minimum_morris + minimum_coupled

    if maximum < minimum_complete_budget:
        if morris_applicable:
            checks.append(
                CheckResult(
                    "L2.schedule.morris",
                    Status.NOT_SCOREABLE,
                    summary=(
                        "variant budget cannot fit one complete Morris trajectory "
                        "alongside required directed/coupled coverage"
                    ),
                    observed={
                        "available_budget": maximum,
                        "minimum_complete_budget": minimum_complete_budget,
                    },
                )
            )
        else:
            checks.append(
                CheckResult(
                    "L2.schedule.morris",
                    Status.NOT_APPLICABLE,
                    required=False,
                    summary=(
                        "Morris trajectories require an all-continuous parameter system"
                    ),
                )
            )
        checks.append(
            CheckResult(
                "L2.schedule.directed",
                Status.NOT_SCOREABLE,
                summary=(
                    "variant budget cannot provide one directed probe per "
                    "public parameter"
                ),
                observed={
                    "available_budget": maximum,
                    "minimum_complete_budget": minimum_complete_budget,
                },
            )
        )
        checks.append(
            CheckResult(
                "L2.schedule.coupled",
                Status.NOT_SCOREABLE if coupled_applicable else Status.NOT_APPLICABLE,
                required=coupled_applicable,
                summary=(
                    "variant budget cannot provide a coupled probe"
                    if coupled_applicable
                    else "coupled probes require at least two public parameters"
                ),
            )
        )
        return schedules, checks

    remaining = maximum
    if not morris_applicable:
        checks.append(
            CheckResult(
                "L2.schedule.morris",
                Status.NOT_APPLICABLE,
                required=False,
                summary=(
                    "Morris trajectories require an all-continuous parameter system; "
                    "explicit directed probes cover this item"
                ),
            )
        )
    else:
        available = remaining - minimum_directed - minimum_coupled
        trajectory_count = min(trajectories, available // morris_points)
        morris_budget = trajectory_count * morris_points
        try:
            morris = generate_morris_schedule(
                domains,
                levels=levels,
                trajectories=trajectory_count,
                seed=seed,
                max_variants=morris_budget,
                feasibility=feasibility,
            )
        except ScheduleError as exc:
            checks.append(
                CheckResult(
                    "L2.schedule.morris",
                    Status.NOT_SCOREABLE,
                    summary=str(exc),
                )
            )
        else:
            schedules.append(morris)
            remaining -= len(morris.variants)
            checks.append(
                CheckResult(
                    "L2.schedule.morris",
                    Status.PASSED,
                    metrics={"variant_count": len(morris.variants)},
                )
            )
    directed_budget = min(
        parameter_count * max(points, 2),
        remaining - minimum_coupled,
    )
    if directed_budget >= minimum_directed:
        try:
            directed = generate_directed_schedule(
                domains,
                nominals,
                points_per_parameter=max(points, 2),
                max_variants=directed_budget,
                feasibility=feasibility,
            )
        except ScheduleError as exc:
            checks.append(
                CheckResult(
                    "L2.schedule.directed",
                    Status.NOT_SCOREABLE,
                    summary=str(exc),
                )
            )
        else:
            schedules.append(directed)
            remaining -= len(directed.variants)
            checks.append(
                CheckResult(
                    "L2.schedule.directed",
                    Status.PASSED,
                    metrics={"variant_count": len(directed.variants)},
                )
            )
    else:
        checks.append(
            CheckResult(
                "L2.schedule.directed",
                Status.NOT_SCOREABLE,
                summary="variant budget cannot cover every parameter directly",
            )
        )

    if not coupled_applicable:
        checks.append(
            CheckResult(
                "L2.schedule.coupled",
                Status.NOT_APPLICABLE,
                required=False,
                summary="coupled probes require at least two public parameters",
            )
        )
    elif remaining >= minimum_coupled:
        try:
            coupled = generate_coupled_schedule(
                domains,
                nominals,
                points_per_parameter=max(points, 2),
                max_variants=remaining,
                feasibility=feasibility,
            )
        except ScheduleError as exc:
            checks.append(
                CheckResult(
                    "L2.schedule.coupled",
                    Status.NOT_SCOREABLE,
                    summary=str(exc),
                )
            )
        else:
            schedules.append(coupled)
            checks.append(
                CheckResult(
                    "L2.schedule.coupled",
                    Status.PASSED,
                    metrics={"variant_count": len(coupled.variants)},
                )
            )
    else:
        checks.append(
            CheckResult(
                "L2.schedule.coupled",
                Status.NOT_SCOREABLE,
                summary="variant budget was exhausted before coupled probes",
            )
        )
    return schedules, checks


def evaluate_l2(
    case: Mapping[str, Any],
    submission_contract: Mapping[str, Any],
    *,
    execute: ExecuteVariant,
    evaluate_l0: EvaluateL0,
    evaluate_l1: EvaluateL1,
    baseline_execution: VariantExecution | None = None,
    relation_contracts: Sequence[Mapping[str, Any]] | None = None,
    functional_contract: FunctionalEditContract | Mapping[str, Any] | None = None,
    functional_context: FunctionalEvaluationContext | None = None,
    functional_hooks: Mapping[str, FunctionalHook] | None = None,
) -> LayerResult:
    """Evaluate L2 without inventing ranges or using an LLM as a judge."""

    checks: list[CheckResult] = []
    artifacts: dict[str, Any] = {}
    parameter_system = case.get("parameter_system")
    if not isinstance(parameter_system, Mapping):
        return LayerResult.from_checks(
            "L2",
            [
                CheckResult(
                    "L2.parameter_contract",
                    Status.EVALUATOR_ERROR,
                    summary="case has no parameter_system",
                )
            ],
        )

    try:
        bindings = bind_parameter_manifest(parameter_system, submission_contract)
    except ReplayError as exc:
        return LayerResult.from_checks(
            "L2",
            [
                CheckResult(
                    "L2.parameter_manifest",
                    Status.FAILED,
                    failure_code=exc.failure_code,
                    summary=str(exc),
                )
            ],
        )
    checks.append(
        CheckResult(
            "L2.parameter_manifest",
            Status.PASSED,
            metrics={"binding_count": len(bindings)},
        )
    )
    artifacts["parameter_bindings"] = [
        binding.as_dict() for binding in bindings.values()
    ]

    functional_checks, functional_artifact = _functional_checks(
        case,
        functional_contract,
        functional_context,
        functional_hooks,
    )

    parameter_records = parameter_system.get("parameters")
    qualified_records = (
        [
            record
            for record in parameter_records
            if isinstance(record, Mapping) and record.get("domain") is not None
        ]
        if isinstance(parameter_records, list)
        else []
    )
    direct_measurement_count = sum(
        isinstance(record.get("variant_requirement_contract"), Mapping)
        and record["variant_requirement_contract"].get("kind")
        == "direct_measurement_only"
        for record in qualified_records
    )
    geometry_response_count = sum(
        isinstance(record.get("variant_requirement_contract"), Mapping)
        and record["variant_requirement_contract"].get("kind")
        == "geometry_response_only"
        for record in qualified_records
    )
    artifacts["variant_contract_coverage"] = {
        "direct_measurement_parameter_count": direct_measurement_count,
        "geometry_response_parameter_count": geometry_response_count,
    }
    missing_variant_contracts = sorted(
        str(record.get("parameter_id"))
        for record in qualified_records
        if not _valid_variant_requirement_contract(
            record.get("variant_requirement_contract")
        )
    )
    if missing_variant_contracts:
        checks.append(
            CheckResult(
                "L2.variant_requirement_contracts",
                Status.NOT_SCOREABLE,
                summary=(
                    "qualified parameters lack explicit dependency/update semantics"
                ),
                observed={"missing_parameter_ids": missing_variant_contracts},
            )
        )
        checks.extend(functional_checks)
        return LayerResult.from_checks(
            "L2",
            checks,
            metrics={
                "parameter_count": len(bindings),
                "qualified_domain_count": 0,
                "variant_count": 0,
            },
            artifacts=artifacts,
        )
    checks.append(
        CheckResult(
            "L2.variant_requirement_contracts",
            Status.PASSED if qualified_records else Status.NOT_APPLICABLE,
            required=bool(qualified_records),
            metrics={"qualified_contract_count": len(qualified_records)},
        )
    )

    try:
        domains = parse_parameter_domains(parameter_system, require_complete=True)
    except UnqualifiedDomainError as exc:
        checks.append(
            CheckResult(
                "L2.qualified_domains",
                Status.NOT_SCOREABLE,
                summary=str(exc),
                metrics={"qualified_count": 0, "parameter_count": len(bindings)},
            )
        )
        checks.extend(functional_checks)
        if functional_artifact is not None:
            artifacts["functional_edit"] = functional_artifact
        return LayerResult.from_checks(
            "L2",
            checks,
            metrics={
                "parameter_count": len(bindings),
                "qualified_domain_count": 0,
                "variant_count": 0,
            },
            artifacts=artifacts,
        )
    except DomainError as exc:
        checks.append(
            CheckResult(
                "L2.qualified_domains",
                Status.NOT_SCOREABLE,
                summary=f"invalid frozen domain contract: {exc}",
            )
        )
        checks.extend(functional_checks)
        return LayerResult.from_checks("L2", checks, artifacts=artifacts)
    checks.append(
        CheckResult(
            "L2.qualified_domains",
            Status.PASSED,
            metrics={"qualified_count": len(domains)},
        )
    )

    raw_relations = (
        relation_contracts
        if relation_contracts is not None
        else parameter_system.get("relations", [])
    )
    if not isinstance(raw_relations, Sequence) or isinstance(
        raw_relations, (str, bytes)
    ):
        checks.append(
            CheckResult(
                "L2.relation_contract",
                Status.NOT_SCOREABLE,
                summary="typed relations must be a list",
            )
        )
        relations: tuple[TypedRelation, ...] = ()
    else:
        try:
            relations = _parse_relations(raw_relations)
        except RelationError as exc:
            checks.append(
                CheckResult(
                    "L2.relation_contract",
                    Status.NOT_SCOREABLE,
                    summary=str(exc),
                )
            )
            relations = ()
        else:
            checks.append(
                CheckResult(
                    "L2.relation_contract",
                    Status.PASSED if relations else Status.NOT_APPLICABLE,
                    required=bool(relations),
                    metrics={"relation_count": len(relations)},
                )
            )

    profile = case.get("evaluator_profile")
    config = profile.get("l2") if isinstance(profile, Mapping) else None
    if not isinstance(config, Mapping):
        checks.append(
            CheckResult(
                "L2.schedule.profile",
                Status.EVALUATOR_ERROR,
                summary="case evaluator profile has no L2 section",
            )
        )
        checks.extend(functional_checks)
        return LayerResult.from_checks("L2", checks, artifacts=artifacts)

    try:
        z3_timeout_ms = _positive_int(
            config.get("z3_timeout_ms", 1000), "l2.z3_timeout_ms"
        )
    except ValueError as exc:
        checks.append(
            CheckResult(
                "L2.solver.profile",
                Status.EVALUATOR_ERROR,
                summary=str(exc),
            )
        )
        checks.extend(functional_checks)
        return LayerResult.from_checks("L2", checks, artifacts=artifacts)
    derived_expressions = {
        str(record["parameter_id"]): dict(record["derived_expression"])
        for record in parameter_records or []
        if isinstance(record, Mapping)
        and record.get("parameter_kind") == "derived"
        and isinstance(record.get("derived_expression"), Mapping)
    }
    solver_checks, solver_proofs, solver_metrics = _solver_checks(
        parameter_system.get("solver_obligations", []),
        domains,
        timeout_ms=z3_timeout_ms,
        derived_expressions=derived_expressions,
    )
    checks.extend(solver_checks)
    artifacts["solver_proofs"] = solver_proofs

    nominals = independent_parameter_values(bindings)
    raw_feasibility = parameter_system.get("feasibility_relations", [])
    feasibility_relations: tuple[TypedRelation, ...] = ()
    if raw_feasibility not in (None, []):
        try:
            if not isinstance(raw_feasibility, list):
                raise RelationError("feasibility_relations must be a list")
            feasibility_relations = _parse_relations(raw_feasibility)
            nominal_context = RelationContext(
                parameters=resolve_public_parameter_values(bindings, nominals)
            )
            if not all(
                relation.evaluate(nominal_context) for relation in feasibility_relations
            ):
                raise RelationError(
                    "nominal assignment violates a frozen feasibility relation"
                )
        except (MissingRelationValue, RelationError, ReplayError) as exc:
            checks.append(
                CheckResult(
                    "L2.feasibility_contract",
                    Status.NOT_SCOREABLE,
                    summary=f"invalid feasibility contract: {exc}",
                )
            )
            checks.extend(functional_checks)
            return LayerResult.from_checks(
                "L2", checks, metrics=solver_metrics, artifacts=artifacts
            )
        checks.append(
            CheckResult(
                "L2.feasibility_contract",
                Status.PASSED,
                metrics={"relation_count": len(feasibility_relations)},
            )
        )
    else:
        checks.append(
            CheckResult(
                "L2.feasibility_contract",
                Status.NOT_APPLICABLE,
                required=False,
                summary="item declares no coupled feasibility relations",
            )
        )

    def feasibility(assignments: Mapping[str, float]) -> bool:
        try:
            complete_assignments = resolve_public_parameter_values(
                bindings, assignments
            )
        except ReplayError as exc:
            raise ScheduleError(
                f"derived feasibility assignment failed: {exc}"
            ) from exc
        context = RelationContext(parameters=complete_assignments)
        try:
            return all(relation.evaluate(context) for relation in feasibility_relations)
        except RelationError as exc:
            raise ScheduleError(f"feasibility evaluation failed: {exc}") from exc

    try:
        invalid_probes = parse_invalid_probes(
            parameter_system.get("invalid_probes", []),
            domains,
            bindings,
            feasibility=feasibility if feasibility_relations else None,
        )
    except (InvalidProbeError, ScheduleError) as exc:
        checks.append(
            CheckResult(
                "L2.invalid_probe_contract",
                Status.NOT_SCOREABLE,
                summary=f"invalid frozen invalid-region contract: {exc}",
            )
        )
        checks.extend(functional_checks)
        return LayerResult.from_checks(
            "L2", checks, metrics=solver_metrics, artifacts=artifacts
        )
    checks.append(
        CheckResult(
            "L2.invalid_probe_contract",
            Status.PASSED if invalid_probes else Status.NOT_APPLICABLE,
            required=bool(invalid_probes),
            metrics={"invalid_probe_count": len(invalid_probes)},
            summary=(
                "item declares no evidence-backed invalid-region probes"
                if not invalid_probes
                else ""
            ),
        )
    )

    try:
        schedules, schedule_checks = _schedule_plan(
            domains,
            nominals,
            config,
            feasibility=feasibility if feasibility_relations else None,
        )
    except ValueError as exc:
        checks.append(
            CheckResult(
                "L2.schedule.profile",
                Status.EVALUATOR_ERROR,
                summary=str(exc),
            )
        )
        checks.extend(functional_checks)
        return LayerResult.from_checks("L2", checks, artifacts=artifacts)
    checks.extend(schedule_checks)
    if not schedules:
        checks.append(
            CheckResult(
                "L2.schedule.parameter_coverage",
                Status.NOT_SCOREABLE,
                summary="no complete replay schedule could be generated",
                expected={"parameter_ids": sorted(domains)},
                observed={
                    "covered_parameter_ids": [],
                    "missing_parameter_ids": sorted(domains),
                },
            )
        )
        checks.extend(functional_checks)
        return LayerResult.from_checks(
            "L2",
            checks,
            metrics={
                "parameter_count": len(bindings),
                "qualified_domain_count": len(domains),
                "variant_count": 0,
                "parameter_response_coverage": 0.0,
                **solver_metrics,
            },
            artifacts=artifacts,
        )
    schedule = schedules[0] if len(schedules) == 1 else combine_schedules(*schedules)
    artifacts["schedule"] = schedule.as_dict()
    expected_parameters = set(domains)
    covered_parameters = set(schedule.covered_parameters)
    missing_parameters = sorted(expected_parameters - covered_parameters)
    extra_parameters = sorted(covered_parameters - expected_parameters)
    if extra_parameters:
        checks.append(
            CheckResult(
                "L2.schedule.parameter_coverage",
                Status.EVALUATOR_ERROR,
                summary="generated schedule references parameters outside the contract",
                observed={"extra_parameter_ids": extra_parameters},
            )
        )
    elif missing_parameters:
        checks.append(
            CheckResult(
                "L2.schedule.parameter_coverage",
                Status.NOT_SCOREABLE,
                summary=(
                    "the frozen variant budget or feasibility contract does not "
                    "exercise every public parameter"
                ),
                expected={"parameter_ids": sorted(expected_parameters)},
                observed={
                    "covered_parameter_ids": sorted(covered_parameters),
                    "missing_parameter_ids": missing_parameters,
                },
            )
        )
    else:
        checks.append(
            CheckResult(
                "L2.schedule.parameter_coverage",
                Status.PASSED,
                metrics={"covered_parameter_count": len(covered_parameters)},
            )
        )
    if missing_parameters or extra_parameters:
        checks.extend(functional_checks)
        if functional_artifact is not None:
            artifacts["functional_edit"] = functional_artifact
        return LayerResult.from_checks(
            "L2",
            checks,
            metrics={
                "parameter_count": len(bindings),
                "qualified_domain_count": len(domains),
                "variant_count": 0,
                "parameter_response_coverage": (
                    len(covered_parameters) / len(domains) if domains else 0.0
                ),
                **solver_metrics,
            },
            artifacts=artifacts,
        )

    nominal_execution = baseline_execution or _baseline_execution(execute, bindings)
    checks.append(_baseline_check(nominal_execution))
    replay = replay_schedule(
        schedule,
        bindings,
        execute=execute,
        evaluate_l0=evaluate_l0,
        evaluate_l1=evaluate_l1,
        relations=relations,
        baseline_execution=nominal_execution,
    )
    checks.extend(replay.checks)
    invalid_evaluation = evaluate_invalid_probes(
        invalid_probes,
        bindings,
        execute=execute,
        evaluate_l0=evaluate_l0,
        evaluate_l1=evaluate_l1,
    )
    checks.extend(invalid_evaluation.checks)
    checks.extend(functional_checks)
    artifacts["replay"] = replay.as_dict()
    artifacts["invalid_probe_replay"] = list(invalid_evaluation.records)
    if functional_artifact is not None:
        artifacts["functional_edit"] = functional_artifact

    replay_metrics, replay_artifacts = replay_diagnostics(replay, domains)
    artifacts.update(replay_artifacts)

    passed_variants = sum(
        variant.status is Status.PASSED for variant in replay.variants
    )
    failed_variants = sum(
        variant.status is Status.FAILED for variant in replay.variants
    )
    invalid_behavior_checks = [
        check
        for check in invalid_evaluation.checks
        if check.check_id.endswith(".behavior")
    ]
    return LayerResult.from_checks(
        "L2",
        checks,
        metrics={
            "parameter_count": len(bindings),
            "qualified_domain_count": len(domains),
            "variant_count": len(replay.variants),
            "passed_variant_count": passed_variants,
            "failed_variant_count": failed_variants,
            "parameter_response_coverage": len(schedule.covered_parameters)
            / len(domains),
            "l2_i_binding_count": len(bindings),
            "l2_i_independent_parameter_count": len(domains),
            "l2_i_derived_parameter_count": len(bindings) - len(domains),
            "direct_measurement_parameter_count": direct_measurement_count,
            "geometry_response_parameter_count": geometry_response_count,
            "l2_p_variant_count": len(replay.variants),
            "l2_c_runtime_relation_count": len(relations),
            "l2_e_applicable": functional_artifact is not None,
            "l2_p_invalid_probe_count": len(invalid_probes),
            "l2_p_invalid_handling_rate": _rate(invalid_behavior_checks),
            "llm_judge_used": False,
            **replay_metrics,
            **solver_metrics,
        },
        artifacts=artifacts,
    )
