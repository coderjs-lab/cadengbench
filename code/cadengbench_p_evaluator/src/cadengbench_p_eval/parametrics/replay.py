"""Parameter binding and source-neutral L2 replay orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from cadengbench_p_eval.contracts import (
    CheckResult,
    LayerResult,
    Status,
    reduce_required_status,
)

from .relations import (
    ArithmeticExpression,
    ConstantExpression,
    MissingRelationValue,
    RelationContext,
    RelationError,
    TypedRelation,
    VariableExpression,
    parse_expression,
)
from .schedules import PerturbationSchedule, ScheduleVariant


class ReplayError(ValueError):
    """A submission cannot be bound to the public parametric contract."""

    def __init__(self, failure_code: str, message: str):
        super().__init__(message)
        self.failure_code = failure_code


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError("invalid_parameter_contract", f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReplayError("invalid_parameter_contract", f"{field} must be finite")
    return number


def _expression_parameter_keys(expression: Any) -> set[str]:
    if isinstance(expression, ConstantExpression):
        return set()
    if isinstance(expression, VariableExpression):
        if expression.source != "parameter":
            raise RelationError(
                "derived expressions may reference parameter values only"
            )
        return {expression.key}
    if isinstance(expression, ArithmeticExpression):
        return set().union(
            *(_expression_parameter_keys(argument) for argument in expression.args)
        )
    raise RelationError("derived expression contains an unsupported node")


@dataclass(frozen=True)
class ParameterBinding:
    parameter_id: str
    name: str
    unit: str
    role: str
    nominal: float
    parameter_kind: str = "independent"
    dependencies: tuple[str, ...] = ()
    derived_expression: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_id": self.parameter_id,
            "name": self.name,
            "unit": self.unit,
            "role": self.role,
            "nominal": self.nominal,
            "parameter_kind": self.parameter_kind,
            "dependencies": list(self.dependencies),
            "derived_expression": (
                dict(self.derived_expression)
                if self.derived_expression is not None
                else None
            ),
        }


def bind_parameter_manifest(
    parameter_system: Mapping[str, Any],
    submission_contract: Mapping[str, Any],
) -> dict[str, ParameterBinding]:
    """Bind every public `P-*` ID to exactly one model parameter name."""

    records = parameter_system.get("parameters")
    defaults = submission_contract.get("default_params")
    manifest = submission_contract.get("parameter_manifest")
    if not isinstance(records, list) or not records:
        raise ReplayError("invalid_parameter_contract", "public parameters are absent")
    if not isinstance(defaults, Mapping) or not isinstance(manifest, Mapping):
        raise ReplayError(
            "invalid_submission_interface",
            "submission inspection must expose default_params and parameter_manifest",
        )

    expected: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ReplayError(
                "invalid_parameter_contract",
                f"public parameter {index} is not an object",
            )
        parameter_id = record.get("parameter_id")
        if not isinstance(parameter_id, str) or not parameter_id:
            raise ReplayError(
                "invalid_parameter_contract", "public parameter ID is absent"
            )
        if parameter_id in expected:
            raise ReplayError(
                "invalid_parameter_contract",
                f"duplicate public parameter: {parameter_id}",
            )
        expected[parameter_id] = record

    if set(manifest) != set(expected):
        missing = sorted(set(expected) - set(manifest))
        extra = sorted(set(manifest) - set(expected))
        raise ReplayError(
            "parameter_manifest_mismatch",
            "manifest IDs differ from the public contract; "
            f"missing={missing}, extra={extra}",
        )

    bound_names: set[str] = set()
    bindings: dict[str, ParameterBinding] = {}
    for parameter_id in sorted(expected):
        public = expected[parameter_id]
        model = manifest[parameter_id]
        if not isinstance(model, Mapping):
            raise ReplayError(
                "invalid_submission_interface",
                f"manifest entry for {parameter_id} must be an object",
            )
        name = model.get("name")
        unit = model.get("unit")
        role = model.get("role")
        if not all(isinstance(value, str) and value for value in (name, unit, role)):
            raise ReplayError(
                "invalid_submission_interface",
                f"manifest entry for {parameter_id} requires name, unit, and role",
            )
        assert isinstance(name, str) and isinstance(unit, str) and isinstance(role, str)
        if name in bound_names:
            raise ReplayError(
                "parameter_binding_not_independent",
                f"multiple public IDs bind to DEFAULT_PARAMS[{name!r}]",
            )
        if name not in defaults:
            raise ReplayError(
                "parameter_default_missing",
                f"DEFAULT_PARAMS[{name!r}] is absent",
            )
        expected_unit = public.get("unit")
        expected_role = public.get("role", public.get("parameter_role"))
        if unit != expected_unit or role != expected_role:
            raise ReplayError(
                "parameter_manifest_mismatch",
                f"unit or role mismatch for {parameter_id}",
            )
        nominal = _finite_number(public.get("nominal"), f"{parameter_id}.nominal")
        parameter_kind = public.get("parameter_kind", "independent")
        if parameter_kind not in {"independent", "derived"}:
            raise ReplayError(
                "invalid_parameter_contract",
                f"{parameter_id}.parameter_kind must be independent or derived",
            )
        raw_dependencies = public.get("dependencies", [])
        raw_expression = public.get("derived_expression")
        if parameter_kind == "derived":
            if (
                not isinstance(raw_dependencies, list)
                or not raw_dependencies
                or not all(
                    isinstance(dependency, str) and dependency
                    for dependency in raw_dependencies
                )
                or len(raw_dependencies) != len(set(raw_dependencies))
            ):
                raise ReplayError(
                    "invalid_parameter_contract",
                    f"{parameter_id} requires unique derived dependencies",
                )
            if not isinstance(raw_expression, Mapping):
                raise ReplayError(
                    "invalid_parameter_contract",
                    f"{parameter_id} requires derived_expression",
                )
            try:
                expression = parse_expression(raw_expression)
                referenced_parameters = _expression_parameter_keys(expression)
            except RelationError as exc:
                raise ReplayError(
                    "invalid_parameter_contract",
                    f"{parameter_id} has invalid derived_expression: {exc}",
                ) from exc
            if referenced_parameters != set(raw_dependencies):
                raise ReplayError(
                    "invalid_parameter_contract",
                    f"{parameter_id} dependencies do not match its expression",
                )
        elif raw_dependencies not in (None, []) or raw_expression is not None:
            raise ReplayError(
                "invalid_parameter_contract",
                f"independent parameter {parameter_id} cannot declare derivation",
            )
        default = _finite_number(defaults[name], f"DEFAULT_PARAMS[{name!r}]")
        tolerance = max(1e-9, abs(nominal) * 1e-9)
        if not math.isclose(default, nominal, rel_tol=0.0, abs_tol=tolerance):
            raise ReplayError(
                "parameter_default_mismatch",
                f"DEFAULT_PARAMS[{name!r}]={default} differs from nominal {nominal}",
            )
        bound_names.add(name)
        bindings[parameter_id] = ParameterBinding(
            parameter_id=parameter_id,
            name=name,
            unit=unit,
            role=role,
            nominal=nominal,
            parameter_kind=str(parameter_kind),
            dependencies=tuple(str(value) for value in raw_dependencies or []),
            derived_expression=(
                dict(raw_expression) if isinstance(raw_expression, Mapping) else None
            ),
        )
    known_ids = set(bindings)
    for binding in bindings.values():
        if binding.parameter_kind != "derived":
            continue
        missing_dependencies = sorted(set(binding.dependencies) - known_ids)
        if missing_dependencies or binding.parameter_id in binding.dependencies:
            raise ReplayError(
                "invalid_parameter_contract",
                f"{binding.parameter_id} has invalid dependencies: "
                f"{missing_dependencies or [binding.parameter_id]}",
            )
    nominal_independent = {
        parameter_id: binding.nominal
        for parameter_id, binding in bindings.items()
        if binding.parameter_kind == "independent"
    }
    resolved_nominal = resolve_public_parameter_values(bindings, nominal_independent)
    for parameter_id, binding in bindings.items():
        tolerance = max(1e-9, abs(binding.nominal) * 1e-9)
        if not math.isclose(
            resolved_nominal[parameter_id],
            binding.nominal,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ReplayError(
                "derived_parameter_nominal_mismatch",
                f"derived nominal for {parameter_id} does not match its expression",
            )
    return bindings


def independent_parameter_values(
    bindings: Mapping[str, ParameterBinding],
) -> dict[str, float]:
    return {
        parameter_id: binding.nominal
        for parameter_id, binding in bindings.items()
        if binding.parameter_kind == "independent"
    }


def resolve_public_parameter_values(
    bindings: Mapping[str, ParameterBinding],
    independent_assignments: Mapping[str, float],
) -> dict[str, float]:
    """Evaluate a finite, acyclic derived-parameter DAG."""

    independent_ids = {
        parameter_id
        for parameter_id, binding in bindings.items()
        if binding.parameter_kind == "independent"
    }
    if set(independent_assignments) != independent_ids:
        missing = sorted(independent_ids - set(independent_assignments))
        extra = sorted(set(independent_assignments) - independent_ids)
        raise ReplayError(
            "variant_assignment_mismatch",
            "independent assignment IDs differ from bindings; "
            f"missing={missing}, extra={extra}",
        )
    values = {
        parameter_id: _finite_number(
            independent_assignments[parameter_id], f"assignment.{parameter_id}"
        )
        for parameter_id in sorted(independent_ids)
    }
    pending = {
        parameter_id: binding
        for parameter_id, binding in bindings.items()
        if binding.parameter_kind == "derived"
    }
    while pending:
        progressed = False
        for parameter_id in sorted(tuple(pending)):
            binding = pending[parameter_id]
            if not set(binding.dependencies).issubset(values):
                continue
            assert binding.derived_expression is not None
            try:
                value = parse_expression(binding.derived_expression).evaluate(
                    RelationContext(parameters=values)
                )
            except RelationError as exc:
                raise ReplayError(
                    "derived_parameter_evaluation_failed",
                    f"{parameter_id} could not be evaluated: {exc}",
                ) from exc
            values[parameter_id] = _finite_number(
                value, f"derived_assignment.{parameter_id}"
            )
            pending.pop(parameter_id)
            progressed = True
        if not progressed:
            raise ReplayError(
                "derived_parameter_cycle",
                "derived parameter dependencies are cyclic or incomplete: "
                f"{sorted(pending)}",
            )
    return values


def model_parameter_values(
    bindings: Mapping[str, ParameterBinding],
    public_assignments: Mapping[str, float],
) -> dict[str, float]:
    """Translate a complete public assignment into `build(params)` keys."""

    resolved = resolve_public_parameter_values(bindings, public_assignments)
    translated: dict[str, float] = {}
    for parameter_id, binding in bindings.items():
        translated[binding.name] = _finite_number(
            resolved[parameter_id], f"assignment.{parameter_id}"
        )
    return translated


@dataclass(frozen=True)
class VariantExecution:
    status: Status
    fingerprint: str | None = None
    artifact: str | None = None
    measurements: dict[str, float] = field(default_factory=dict)
    failure_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is Status.FAILED and not self.failure_code:
            raise ValueError("failed variant executions require a failure_code")
        if self.status is not Status.FAILED and self.failure_code:
            raise ValueError("failure_code is reserved for failed variant executions")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.status is Status.PASSED
            if self.status in {Status.PASSED, Status.FAILED}
            else None,
            "fingerprint": self.fingerprint,
            "artifact": self.artifact,
            "measurements": dict(sorted(self.measurements.items())),
            "failure_code": self.failure_code,
            "details": self.details,
        }


@dataclass(frozen=True)
class CanonicalGeometryEvidence:
    """Canonical B-Rep response evidence returned by an evaluator hook.

    ``artifact`` paths and raw STEP bytes are intentionally absent. A response
    is established only from a canonical topology/geometry fingerprint and/or
    deterministic B-Rep measurements.
    """

    fingerprint: str | None = None
    measurements: dict[str, float] = field(default_factory=dict)
    source: str = "canonical_brep_hook"

    def __post_init__(self) -> None:
        if self.fingerprint is not None and (
            not isinstance(self.fingerprint, str) or not self.fingerprint
        ):
            raise ValueError("canonical geometry fingerprint must be nonempty")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("canonical geometry evidence source must be nonempty")
        normalized: dict[str, float] = {}
        for key, value in self.measurements.items():
            if not isinstance(key, str) or not key:
                raise ValueError("canonical measurement keys must be nonempty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"canonical measurement {key!r} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"canonical measurement {key!r} must be finite")
            normalized[key] = number
        object.__setattr__(self, "measurements", normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "measurements": dict(sorted(self.measurements.items())),
            "source": self.source,
            "raw_artifact_bytes_used": False,
        }


@dataclass(frozen=True)
class VariantEvaluationContext:
    variant: ScheduleVariant
    public_assignments: Mapping[str, float]
    parameter_matched_reference: str | None = None

    @property
    def nominal_reference_allowed(self) -> bool:
        """Perturbed variants must never be scored against the nominal target shape."""

        return False


class ExecuteVariant(Protocol):
    def __call__(
        self, model_parameters: Mapping[str, float], variant_id: str
    ) -> VariantExecution: ...


class EvaluateL0(Protocol):
    def __call__(
        self, execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult: ...


class EvaluateL1(Protocol):
    def __call__(
        self, execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult: ...


class ExtractCanonicalGeometry(Protocol):
    def __call__(
        self, execution: VariantExecution, context: VariantEvaluationContext
    ) -> CanonicalGeometryEvidence: ...


@dataclass(frozen=True)
class ReplayVariantResult:
    variant: ScheduleVariant
    execution: VariantExecution
    l0: LayerResult
    l1: LayerResult
    canonical_geometry: CanonicalGeometryEvidence | None
    checks: tuple[CheckResult, ...]
    status: Status

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.as_dict(),
            "execution": self.execution.as_dict(),
            "l0": self.l0.as_dict(),
            "l1": self.l1.as_dict(),
            "canonical_geometry": (
                self.canonical_geometry.as_dict()
                if self.canonical_geometry is not None
                else None
            ),
            "checks": [check.as_dict() for check in self.checks],
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ReplayResult:
    schedule: PerturbationSchedule
    variants: tuple[ReplayVariantResult, ...]

    @property
    def checks(self) -> tuple[CheckResult, ...]:
        return tuple(check for variant in self.variants for check in variant.checks)

    @property
    def status(self) -> Status:
        return reduce_required_status(self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.as_dict(),
            "status": self.status.value,
            "variants": [variant.as_dict() for variant in self.variants],
        }


def morris_elementary_effects(
    replay: ReplayResult,
    *,
    domain_spans: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Extract auditable Morris elementary effects from canonical evidence.

    Numeric effects use deterministic B-Rep measurement packets. A fingerprint
    transition is retained as categorical response evidence when no scalar
    measurement describes the changed feature.
    """

    by_id: dict[str, ReplayVariantResult] = {}
    for result in replay.variants:
        by_id[result.variant.variant_id] = result
        for origin in result.variant.origins:
            by_id[origin.variant_id] = result

    records: list[dict[str, Any]] = []
    for result in replay.variants:
        current = result.canonical_geometry
        for origin in result.variant.origins:
            if (
                origin.schedule_kind != "morris"
                or len(origin.changed_parameters) != 1
                or origin.predecessor_id is None
            ):
                continue
            parameter_id = origin.changed_parameters[0]
            predecessor = by_id.get(origin.predecessor_id)
            if predecessor is None:
                records.append(
                    {
                        "variant_id": result.variant.variant_id,
                        "origin_variant_id": origin.variant_id,
                        "parameter_id": parameter_id,
                        "status": "missing_predecessor",
                    }
                )
                continue
            input_delta = (
                result.variant.assignments[parameter_id]
                - predecessor.variant.assignments[parameter_id]
            )
            if math.isclose(input_delta, 0.0, rel_tol=0.0, abs_tol=1e-15):
                records.append(
                    {
                        "variant_id": result.variant.variant_id,
                        "origin_variant_id": origin.variant_id,
                        "parameter_id": parameter_id,
                        "status": "zero_input_delta",
                    }
                )
                continue
            previous = predecessor.canonical_geometry
            common_measurements = (
                sorted(set(current.measurements) & set(previous.measurements))
                if current is not None and previous is not None
                else []
            )
            span = (domain_spans or {}).get(parameter_id)
            normalized_delta = (
                input_delta / span
                if isinstance(span, (int, float)) and float(span) > 0.0
                else None
            )
            measurement_effects = {}
            for key in common_measurements:
                output_delta = current.measurements[key] - previous.measurements[key]
                measurement_effects[key] = {
                    "output_delta": output_delta,
                    "effect_per_parameter_unit": output_delta / input_delta,
                    "effect_per_normalized_step": (
                        output_delta / normalized_delta
                        if normalized_delta not in (None, 0.0)
                        else None
                    ),
                }
            records.append(
                {
                    "variant_id": result.variant.variant_id,
                    "origin_variant_id": origin.variant_id,
                    "predecessor_id": origin.predecessor_id,
                    "trajectory_index": origin.trajectory_index,
                    "step_index": origin.step_index,
                    "parameter_id": parameter_id,
                    "input_delta": input_delta,
                    "normalized_input_delta": normalized_delta,
                    "fingerprint_changed": (
                        current.fingerprint != previous.fingerprint
                        if current is not None
                        and previous is not None
                        and current.fingerprint is not None
                        and previous.fingerprint is not None
                        else None
                    ),
                    "measurement_effects": measurement_effects,
                    "status": (
                        "measured"
                        if measurement_effects
                        else "categorical_response_only"
                        if current is not None and previous is not None
                        else "missing_geometry_evidence"
                    ),
                }
            )
    return records


def _execution_check(variant_id: str, execution: VariantExecution) -> CheckResult:
    return CheckResult(
        check_id=f"L2.variant.{variant_id}.execution",
        status=execution.status,
        failure_code=execution.failure_code,
        observed=execution.as_dict(),
    )


def _layer_check(variant_id: str, layer: LayerResult) -> CheckResult:
    failure_code = None
    if layer.status is Status.FAILED:
        failure_code = f"{layer.layer.lower()}_variant_failed"
    return CheckResult(
        check_id=f"L2.variant.{variant_id}.{layer.layer.lower()}",
        status=layer.status,
        failure_code=failure_code,
        observed={"layer": layer.layer, "status": layer.status.value},
    )


def _response_check(
    variant: ScheduleVariant,
    geometry: CanonicalGeometryEvidence | None,
    predecessor: CanonicalGeometryEvidence | None,
    *,
    check_id: str | None = None,
) -> CheckResult:
    check_id = check_id or f"L2.variant.{variant.variant_id}.parameter_response"
    if not variant.changed_parameters:
        return CheckResult(check_id, Status.NOT_APPLICABLE, required=False)
    if predecessor is None or geometry is None:
        return CheckResult(
            check_id,
            Status.NOT_SCOREABLE,
            summary=(
                "canonical B-Rep evidence is required to detect ignored parameters"
            ),
        )
    fingerprint_changed: bool | None = None
    if predecessor.fingerprint is not None and geometry.fingerprint is not None:
        fingerprint_changed = predecessor.fingerprint != geometry.fingerprint

    measurements_changed: bool | None = None
    measurement_packets_incompatible = False
    if (
        predecessor.measurements
        and geometry.measurements
        and set(predecessor.measurements) == set(geometry.measurements)
    ):
        measurements_changed = any(
            not math.isclose(
                predecessor.measurements[key],
                geometry.measurements[key],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key in predecessor.measurements
        )
    elif predecessor.measurements and geometry.measurements:
        measurement_packets_incompatible = True

    comparisons = [
        value
        for value in (fingerprint_changed, measurements_changed)
        if value is not None
    ]
    observed = {
        "fingerprint_changed": fingerprint_changed,
        "measurements_changed": measurements_changed,
        "measurement_packets_incompatible": measurement_packets_incompatible,
        "predecessor_evidence_source": predecessor.source,
        "candidate_evidence_source": geometry.source,
        "raw_artifact_bytes_used": False,
    }
    if not comparisons:
        return CheckResult(
            check_id,
            Status.NOT_SCOREABLE,
            summary=(
                "canonical geometry evidence has no comparable fingerprint or "
                "measurement packet"
            ),
            observed=observed,
        )
    if measurement_packets_incompatible and not any(comparisons):
        return CheckResult(
            check_id,
            Status.NOT_SCOREABLE,
            summary="canonical B-Rep measurement packets are not comparable",
            observed=observed,
        )
    if not any(comparisons):
        return CheckResult(
            check_id,
            Status.FAILED,
            failure_code="parameter_ignored",
            expected={"changed_parameters": list(variant.changed_parameters)},
            observed=observed,
        )
    return CheckResult(
        check_id,
        Status.PASSED,
        expected={"changed_parameters": list(variant.changed_parameters)},
        observed=observed,
    )


def _canonical_geometry_evidence(
    execution: VariantExecution,
    context: VariantEvaluationContext,
    hook: ExtractCanonicalGeometry | None,
) -> tuple[CanonicalGeometryEvidence | None, CheckResult]:
    check_id = f"L2.variant.{context.variant.variant_id}.canonical_geometry"
    if execution.status is not Status.PASSED:
        return None, CheckResult(
            check_id,
            Status.NOT_REACHED,
            required=False,
            summary="variant execution did not produce geometry",
        )
    try:
        evidence = (
            hook(execution, context)
            if hook is not None
            else CanonicalGeometryEvidence(
                fingerprint=execution.fingerprint,
                measurements=execution.measurements,
                source="execute_variant_hook",
            )
        )
        if not isinstance(evidence, CanonicalGeometryEvidence):
            raise TypeError("canonical geometry hook returned an invalid result")
    except Exception as exc:  # evaluator-owned geometry-analysis boundary
        return None, CheckResult(
            check_id,
            Status.EVALUATOR_ERROR,
            summary=f"canonical geometry hook failed: {type(exc).__name__}: {exc}",
        )
    if evidence.fingerprint is None and not evidence.measurements:
        return None, CheckResult(
            check_id,
            Status.NOT_SCOREABLE,
            summary="canonical geometry hook returned no response evidence",
        )
    return evidence, CheckResult(
        check_id,
        Status.PASSED,
        observed=evidence.as_dict(),
    )


def _solid_count_preservation_check(
    variant_id: str,
    geometry: CanonicalGeometryEvidence | None,
    baseline: CanonicalGeometryEvidence | None,
) -> CheckResult:
    check_id = f"L2.variant.{variant_id}.solid_count_preserved"
    if geometry is None:
        return CheckResult(
            check_id,
            Status.NOT_REACHED,
            required=False,
            summary="variant execution did not produce geometry",
        )
    baseline_count = (
        baseline.measurements.get("solid_count") if baseline is not None else None
    )
    variant_count = geometry.measurements.get("solid_count")
    if baseline_count is None or variant_count is None:
        return CheckResult(
            check_id,
            Status.NOT_SCOREABLE,
            required=False,
            summary=(
                "solid-count preservation is unavailable because canonical "
                "nominal or variant geometry evidence lacks solid_count"
            ),
            expected={"nominal_candidate_solid_count": baseline_count},
            observed={"variant_candidate_solid_count": variant_count},
        )
    preserved = math.isclose(
        float(variant_count),
        float(baseline_count),
        rel_tol=0.0,
        abs_tol=0.0,
    )
    return CheckResult(
        check_id,
        Status.PASSED if preserved else Status.FAILED,
        failure_code=None if preserved else "solid_count_changed_under_perturbation",
        summary=(
            "parameter perturbation preserved the candidate nominal solid count"
            if preserved
            else "parameter perturbation changed the candidate nominal solid count"
        ),
        expected={"nominal_candidate_solid_count": int(baseline_count)},
        observed={"variant_candidate_solid_count": int(variant_count)},
    )


def _relation_checks(
    variant: ScheduleVariant,
    execution: VariantExecution,
    relations: Sequence[TypedRelation],
    *,
    parameter_assignments: Mapping[str, float] | None = None,
    measurements: Mapping[str, float] | None = None,
    baseline_parameters: Mapping[str, float],
    baseline_measurements: Mapping[str, float],
) -> list[CheckResult]:
    if execution.status is not Status.PASSED:
        return [
            CheckResult(
                f"L2.variant.{variant.variant_id}.relation.{relation.relation_id}",
                Status.NOT_REACHED,
                required=relation.required,
                summary="relation was not evaluated because variant execution failed",
            )
            for relation in relations
        ]
    context = RelationContext(
        parameters=(
            parameter_assignments
            if parameter_assignments is not None
            else variant.assignments
        ),
        measurements=measurements
        if measurements is not None
        else execution.measurements,
        baseline_parameters=baseline_parameters,
        baseline_measurements=baseline_measurements,
    )
    checks: list[CheckResult] = []
    for relation in relations:
        check_id = f"L2.variant.{variant.variant_id}.relation.{relation.relation_id}"
        try:
            holds = relation.evaluate(context)
        except MissingRelationValue as exc:
            missing_model_measurement = exc.source in {
                "measurement",
                "baseline_measurement",
            }
            checks.append(
                CheckResult(
                    check_id,
                    (
                        Status.FAILED
                        if missing_model_measurement
                        else Status.NOT_SCOREABLE
                    ),
                    required=relation.required,
                    failure_code=(
                        "relation_measurement_missing"
                        if missing_model_measurement
                        else None
                    ),
                    summary=(
                        "generated geometry does not expose a required relation "
                        f"measurement: {exc.source}.{exc.key}"
                        if missing_model_measurement
                        else str(exc)
                    ),
                )
            )
        except RelationError as exc:
            checks.append(
                CheckResult(
                    check_id,
                    Status.NOT_SCOREABLE,
                    required=relation.required,
                    summary=f"invalid frozen relation contract: {exc}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    check_id,
                    Status.PASSED if holds else Status.FAILED,
                    required=relation.required,
                    failure_code=None if holds else "parameter_relation_violated",
                    evidence=[
                        {
                            "authority": relation.authority,
                            "evidence_ref": relation.evidence_ref,
                        }
                    ],
                )
            )
    return checks


def replay_schedule(
    schedule: PerturbationSchedule,
    bindings: Mapping[str, ParameterBinding],
    *,
    execute: ExecuteVariant,
    evaluate_l0: EvaluateL0,
    evaluate_l1: EvaluateL1,
    relations: Sequence[TypedRelation] = (),
    baseline_execution: VariantExecution | None = None,
    canonical_geometry: ExtractCanonicalGeometry | None = None,
) -> ReplayResult:
    """Execute variants and replay L0/L1 without a nominal target-shape oracle.

    The optional ``canonical_geometry`` hook is the authoritative geometry
    response boundary. For backward compatibility, precomputed ``fingerprint``
    and ``measurements`` fields returned by ``execute`` are treated as the same
    hook output. Raw artifact bytes are never read or compared here.
    """

    baseline_independent = independent_parameter_values(bindings)
    baseline_parameters = resolve_public_parameter_values(
        bindings, baseline_independent
    )
    known_geometry: dict[str, CanonicalGeometryEvidence] = {}
    baseline_geometry: CanonicalGeometryEvidence | None = None
    baseline_geometry_check: CheckResult | None = None
    if baseline_execution is not None:
        nominal_variant = ScheduleVariant(
            variant_id="nominal",
            schedule_kind="nominal",
            assignments=dict(baseline_independent),
            changed_parameters=(),
        )
        baseline_context = VariantEvaluationContext(
            variant=nominal_variant,
            public_assignments=baseline_parameters,
        )
        baseline_geometry, baseline_geometry_check = _canonical_geometry_evidence(
            baseline_execution, baseline_context, canonical_geometry
        )
        if baseline_geometry is not None:
            known_geometry["nominal"] = baseline_geometry
    results: list[ReplayVariantResult] = []

    for variant_index, variant in enumerate(schedule.variants):
        try:
            translated = model_parameter_values(bindings, variant.assignments)
            public_assignments = resolve_public_parameter_values(
                bindings, variant.assignments
            )
            execution = execute(translated, variant.variant_id)
            if not isinstance(execution, VariantExecution):
                raise TypeError("execution callback returned an invalid result")
        except ReplayError as exc:
            public_assignments = dict(variant.assignments)
            execution = VariantExecution(
                Status.NOT_SCOREABLE,
                details={
                    "evaluator_owned_contract": True,
                    "failure_code": exc.failure_code,
                    "message": str(exc),
                },
            )
        except Exception as exc:  # callback is an evaluator boundary
            public_assignments = dict(variant.assignments)
            execution = VariantExecution(
                Status.EVALUATOR_ERROR,
                details={"exception_type": type(exc).__name__, "message": str(exc)},
            )

        context = VariantEvaluationContext(
            variant=variant,
            public_assignments=public_assignments,
        )
        if execution.status is Status.PASSED:
            try:
                l0 = evaluate_l0(execution, context)
                if not isinstance(l0, LayerResult):
                    raise TypeError("L0 callback returned an invalid result")
            except Exception as exc:  # callback is an evaluator boundary
                l0 = LayerResult(
                    "L0",
                    Status.EVALUATOR_ERROR,
                    artifacts={
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
        else:
            l0 = LayerResult("L0", Status.NOT_REACHED)

        if l0.status is Status.PASSED:
            try:
                l1 = evaluate_l1(execution, context)
                if not isinstance(l1, LayerResult):
                    raise TypeError("L1 callback returned an invalid result")
            except Exception as exc:  # callback is an evaluator boundary
                l1 = LayerResult(
                    "L1",
                    Status.EVALUATOR_ERROR,
                    artifacts={
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
        else:
            l1 = LayerResult("L1", Status.NOT_REACHED)

        geometry, geometry_check = _canonical_geometry_evidence(
            execution, context, canonical_geometry
        )
        checks = [
            _execution_check(variant.variant_id, execution),
            _layer_check(variant.variant_id, l0),
            _layer_check(variant.variant_id, l1),
            geometry_check,
            _solid_count_preservation_check(
                variant.variant_id,
                geometry,
                baseline_geometry,
            ),
        ]
        if variant_index == 0 and baseline_geometry_check is not None:
            checks.insert(0, baseline_geometry_check)
        checks.extend(
            _relation_checks(
                variant,
                execution,
                relations,
                parameter_assignments=public_assignments,
                measurements=geometry.measurements if geometry is not None else {},
                baseline_parameters=baseline_parameters,
                baseline_measurements=(
                    baseline_geometry.measurements
                    if baseline_geometry is not None
                    else {}
                ),
            )
        )
        result = ReplayVariantResult(
            variant=variant,
            execution=execution,
            l0=l0,
            l1=l1,
            canonical_geometry=geometry,
            checks=tuple(checks),
            status=reduce_required_status(checks),
        )
        results.append(result)
        if geometry is not None:
            known_geometry[variant.variant_id] = geometry
            for origin in variant.origins:
                known_geometry[origin.variant_id] = geometry

    finalized: list[ReplayVariantResult] = []
    for result in results:
        checks = list(result.checks)
        for origin_index, origin in enumerate(result.variant.origins):
            origin_variant = ScheduleVariant(
                variant_id=result.variant.variant_id,
                schedule_kind=origin.schedule_kind,
                assignments=result.variant.assignments,
                changed_parameters=origin.changed_parameters,
                predecessor_id=origin.predecessor_id,
                trajectory_index=origin.trajectory_index,
                step_index=origin.step_index,
            )
            origin_predecessor = (
                known_geometry.get(origin.predecessor_id)
                if origin.predecessor_id is not None
                else None
            )
            response_id = (
                f"L2.variant.{result.variant.variant_id}.parameter_response"
                if origin_index == 0
                else (
                    f"L2.variant.{result.variant.variant_id}.parameter_response."
                    f"origin-{origin_index:02d}-{origin.schedule_kind}"
                )
            )
            if result.execution.status is not Status.PASSED or (
                origin.predecessor_id is not None and origin_predecessor is None
            ):
                checks.append(
                    CheckResult(
                        response_id,
                        Status.NOT_REACHED,
                        summary=(
                            "parameter response was not evaluated because the "
                            "variant or its predecessor failed execution"
                        ),
                    )
                )
            else:
                checks.append(
                    _response_check(
                        origin_variant,
                        result.canonical_geometry,
                        origin_predecessor,
                        check_id=response_id,
                    )
                )
        finalized.append(
            ReplayVariantResult(
                variant=result.variant,
                execution=result.execution,
                l0=result.l0,
                l1=result.l1,
                canonical_geometry=result.canonical_geometry,
                checks=tuple(checks),
                status=reduce_required_status(checks),
            )
        )
    return ReplayResult(schedule=schedule, variants=tuple(finalized))
