"""Deterministic functional-edit target, effect, and preservation checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from cadengbench_p_eval.contracts import CheckResult, Status, reduce_required_status

from .relations import (
    MissingRelationValue,
    RelationContext,
    RelationError,
    TypedRelation,
    parse_predicate,
)


class FunctionalEditError(ValueError):
    """A functional-edit acceptance contract is malformed."""


_CATEGORIES = {"target", "effect", "preservation"}


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FunctionalEditError(f"{field} must be a nonempty string")
    return value


def _check_keys(raw: Mapping[str, Any], allowed: set[str], label: str) -> None:
    extras = set(raw) - allowed
    if extras:
        raise FunctionalEditError(f"unsupported {label} fields: {sorted(extras)}")


@dataclass(frozen=True)
class FunctionalEvaluationContext:
    parameters: Mapping[str, float] = field(default_factory=dict)
    measurements: Mapping[str, float] = field(default_factory=dict)
    baseline_parameters: Mapping[str, float] = field(default_factory=dict)
    baseline_measurements: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, float] = field(default_factory=dict)

    def relation_context(self) -> RelationContext:
        return RelationContext(
            parameters=self.parameters,
            measurements=self.measurements,
            baseline_parameters=self.baseline_parameters,
            baseline_measurements=self.baseline_measurements,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class HookOutcome:
    passed: bool
    summary: str = ""
    expected: Any = None
    observed: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)


class FunctionalHook(Protocol):
    def __call__(
        self, context: FunctionalEvaluationContext, options: Mapping[str, Any]
    ) -> HookOutcome | bool: ...


@dataclass(frozen=True)
class FunctionalPredicate:
    predicate_id: str
    category: str
    authority: str
    evidence_ref: str
    required: bool
    relation: TypedRelation | None = None
    hook_id: str | None = None
    hook_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FunctionalEditContract:
    contract_id: str
    predicates: tuple[FunctionalPredicate, ...]


@dataclass(frozen=True)
class FunctionalEditResult:
    contract_id: str
    checks: tuple[CheckResult, ...]
    status: Status

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "status": self.status.value,
            "checks": [check.as_dict() for check in self.checks],
        }


def parse_functional_edit_contract(raw: Mapping[str, Any]) -> FunctionalEditContract:
    """Parse a declarative contract.

    Executable source and free-form judges are deliberately invalid.
    """

    if not isinstance(raw, Mapping):
        raise FunctionalEditError("functional edit contract must be an object")
    _check_keys(raw, {"contract_id", "predicates"}, "functional edit contract")
    contract_id = _string(raw.get("contract_id"), "contract_id")
    records = raw.get("predicates")
    if not isinstance(records, list) or not records:
        raise FunctionalEditError("functional edit contract requires predicates")

    predicates: list[FunctionalPredicate] = []
    identifiers: set[str] = set()
    categories: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise FunctionalEditError(f"predicate {index} must be an object")
        _check_keys(
            record,
            {
                "predicate_id",
                "category",
                "authority",
                "evidence_ref",
                "required",
                "evaluator",
            },
            "functional predicate",
        )
        predicate_id = _string(record.get("predicate_id"), "predicate_id")
        if predicate_id in identifiers:
            raise FunctionalEditError(f"duplicate functional predicate: {predicate_id}")
        identifiers.add(predicate_id)
        category = _string(record.get("category"), "category")
        if category not in _CATEGORIES:
            raise FunctionalEditError(f"unsupported functional category: {category}")
        categories.add(category)
        required = record.get("required", True)
        if not isinstance(required, bool):
            raise FunctionalEditError("functional predicate required must be boolean")
        evaluator = record.get("evaluator")
        if not isinstance(evaluator, Mapping):
            raise FunctionalEditError(
                "functional predicate evaluator must be an object"
            )
        kind = evaluator.get("kind")
        relation: TypedRelation | None = None
        hook_id: str | None = None
        options: Mapping[str, Any] = {}
        if kind == "relation":
            _check_keys(evaluator, {"kind", "predicate"}, "relation evaluator")
            relation = TypedRelation(
                relation_id=predicate_id,
                predicate=parse_predicate(evaluator.get("predicate")),
                authority=_string(record.get("authority"), "authority"),
                evidence_ref=_string(record.get("evidence_ref"), "evidence_ref"),
                required=required,
            )
        elif kind == "hook":
            _check_keys(evaluator, {"kind", "hook_id", "options"}, "hook evaluator")
            hook_id = _string(evaluator.get("hook_id"), "hook_id")
            raw_options = evaluator.get("options", {})
            if not isinstance(raw_options, Mapping):
                raise FunctionalEditError("hook options must be an object")
            options = dict(raw_options)
        else:
            raise FunctionalEditError("evaluator.kind must be relation or hook")
        predicates.append(
            FunctionalPredicate(
                predicate_id=predicate_id,
                category=category,
                authority=_string(record.get("authority"), "authority"),
                evidence_ref=_string(record.get("evidence_ref"), "evidence_ref"),
                required=required,
                relation=relation,
                hook_id=hook_id,
                hook_options=options,
            )
        )
    missing = sorted(_CATEGORIES - categories)
    if missing:
        raise FunctionalEditError(
            "evaluation-ready functional contracts require target, effect, "
            "and preservation; "
            f"missing={missing}"
        )
    return FunctionalEditContract(contract_id, tuple(predicates))


def _relation_check(
    predicate: FunctionalPredicate, context: FunctionalEvaluationContext
) -> CheckResult:
    assert predicate.relation is not None
    check_id = f"L2.functional.{predicate.category}.{predicate.predicate_id}"
    try:
        passed = predicate.relation.evaluate(context.relation_context())
    except MissingRelationValue as exc:
        return CheckResult(
            check_id,
            Status.NOT_SCOREABLE,
            required=predicate.required,
            summary=str(exc),
        )
    except RelationError as exc:
        return CheckResult(
            check_id,
            Status.EVALUATOR_ERROR,
            required=predicate.required,
            summary=str(exc),
        )
    return CheckResult(
        check_id,
        Status.PASSED if passed else Status.FAILED,
        required=predicate.required,
        failure_code=None if passed else f"functional_{predicate.category}_violated",
        evidence=[
            {"authority": predicate.authority, "evidence_ref": predicate.evidence_ref}
        ],
    )


def _hook_check(
    predicate: FunctionalPredicate,
    context: FunctionalEvaluationContext,
    hooks: Mapping[str, FunctionalHook],
) -> CheckResult:
    assert predicate.hook_id is not None
    check_id = f"L2.functional.{predicate.category}.{predicate.predicate_id}"
    hook = hooks.get(predicate.hook_id)
    if hook is None:
        return CheckResult(
            check_id,
            Status.NOT_SCOREABLE,
            required=predicate.required,
            summary=f"deterministic hook is unavailable: {predicate.hook_id}",
        )
    try:
        raw = hook(context, predicate.hook_options)
        outcome = raw if isinstance(raw, HookOutcome) else HookOutcome(bool(raw))
    except Exception as exc:  # deterministic extension boundary
        return CheckResult(
            check_id,
            Status.EVALUATOR_ERROR,
            required=predicate.required,
            summary=f"{type(exc).__name__}: {exc}",
        )
    return CheckResult(
        check_id,
        Status.PASSED if outcome.passed else Status.FAILED,
        required=predicate.required,
        failure_code=None
        if outcome.passed
        else f"functional_{predicate.category}_violated",
        summary=outcome.summary,
        expected=outcome.expected,
        observed=outcome.observed,
        metrics=outcome.metrics,
        evidence=[
            {"authority": predicate.authority, "evidence_ref": predicate.evidence_ref},
            *outcome.evidence,
        ],
    )


def evaluate_functional_edit(
    contract: FunctionalEditContract,
    context: FunctionalEvaluationContext,
    *,
    hooks: Mapping[str, FunctionalHook] | None = None,
) -> FunctionalEditResult:
    """Evaluate only deterministic predicates or explicitly registered hooks."""

    registry = hooks or {}
    checks = tuple(
        _relation_check(predicate, context)
        if predicate.relation is not None
        else _hook_check(predicate, context, registry)
        for predicate in contract.predicates
    )
    return FunctionalEditResult(
        contract_id=contract.contract_id,
        checks=checks,
        status=reduce_required_status(checks),
    )
