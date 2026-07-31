"""Explicit invalid-region probes for L2 parameter-interface behavior."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cadengbench_p_eval.contracts import CheckResult, Status

from .domains import ParameterDomain
from .replay import (
    EvaluateL0,
    EvaluateL1,
    ExecuteVariant,
    ParameterBinding,
    VariantEvaluationContext,
    VariantExecution,
    independent_parameter_values,
    model_parameter_values,
)
from .schedules import ScheduleVariant


class InvalidProbeError(ValueError):
    """An invalid-region probe contract is malformed or not actually invalid."""


@dataclass(frozen=True)
class SafeMeasurement:
    key: str
    value: float
    tolerance: float


@dataclass(frozen=True)
class InvalidProbe:
    probe_id: str
    assignments: dict[str, float]
    expected_behavior: str
    safe_measurements: tuple[SafeMeasurement, ...]
    allowed_failure_codes: tuple[str, ...]
    authority: str
    evidence_ref: str
    required: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "assignments": dict(sorted(self.assignments.items())),
            "expected_behavior": self.expected_behavior,
            "safe_measurements": [
                {
                    "key": measurement.key,
                    "value": measurement.value,
                    "tolerance": measurement.tolerance,
                }
                for measurement in self.safe_measurements
            ],
            "allowed_failure_codes": list(self.allowed_failure_codes),
            "authority": self.authority,
            "evidence_ref": self.evidence_ref,
            "required": self.required,
        }


@dataclass(frozen=True)
class InvalidProbeEvaluation:
    checks: tuple[CheckResult, ...]
    records: tuple[dict[str, Any], ...]


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidProbeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidProbeError(f"{field} must be finite")
    return number


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidProbeError(f"{field} must be a nonempty string")
    return value


def parse_invalid_probes(
    records: Any,
    domains: Mapping[str, ParameterDomain],
    bindings: Mapping[str, ParameterBinding],
    *,
    feasibility: Callable[[Mapping[str, float]], bool] | None = None,
) -> tuple[InvalidProbe, ...]:
    """Parse only evidence-backed probes outside a domain/feasible region."""

    if records in (None, []):
        return ()
    if not isinstance(records, list):
        raise InvalidProbeError("invalid_probes must be a list")
    independent_ids = set(independent_parameter_values(bindings))
    parsed: list[InvalidProbe] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise InvalidProbeError(f"invalid probe {index} must be an object")
        allowed = {
            "probe_id",
            "assignments",
            "expected_behavior",
            "safe_measurements",
            "allowed_failure_codes",
            "authority",
            "evidence_ref",
            "required",
        }
        extras = set(raw) - allowed
        if extras:
            raise InvalidProbeError(
                f"unsupported invalid probe fields: {sorted(extras)}"
            )
        probe_id = _string(raw.get("probe_id"), "probe_id")
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", probe_id) is None
            or probe_id in identifiers
        ):
            raise InvalidProbeError("invalid probe IDs must be safe and unique")
        identifiers.add(probe_id)
        assignments_raw = raw.get("assignments")
        if (
            not isinstance(assignments_raw, Mapping)
            or set(assignments_raw) != independent_ids
        ):
            raise InvalidProbeError(
                f"{probe_id} must assign every independent parameter exactly once"
            )
        assignments = {
            str(parameter_id): _finite(value, f"{probe_id}.{parameter_id}")
            for parameter_id, value in assignments_raw.items()
        }
        outside_domain = any(
            not domains[parameter_id].contains(assignments[parameter_id])
            for parameter_id in independent_ids
        )
        feasibility_invalid = feasibility is not None and not feasibility(assignments)
        if not outside_domain and not feasibility_invalid:
            raise InvalidProbeError(
                f"{probe_id} lies inside the complete feasible parameter region"
            )
        behavior = raw.get("expected_behavior")
        if behavior not in {"reject", "safe_output", "reject_or_safe_output"}:
            raise InvalidProbeError(f"{probe_id}.expected_behavior is unsupported")
        safe_raw = raw.get("safe_measurements", [])
        if not isinstance(safe_raw, list):
            raise InvalidProbeError(f"{probe_id}.safe_measurements must be a list")
        safe_measurements: list[SafeMeasurement] = []
        safe_keys: set[str] = set()
        for measurement_index, measurement_raw in enumerate(safe_raw):
            if not isinstance(measurement_raw, Mapping) or set(measurement_raw) != {
                "key",
                "value",
                "tolerance",
            }:
                raise InvalidProbeError(
                    f"{probe_id}.safe_measurements[{measurement_index}] is malformed"
                )
            key = _string(measurement_raw.get("key"), "safe measurement key")
            if key in safe_keys:
                raise InvalidProbeError(f"{probe_id} repeats safe measurement {key}")
            safe_keys.add(key)
            tolerance = _finite(
                measurement_raw.get("tolerance"), "safe measurement tolerance"
            )
            if tolerance < 0.0:
                raise InvalidProbeError("safe measurement tolerance cannot be negative")
            safe_measurements.append(
                SafeMeasurement(
                    key=key,
                    value=_finite(
                        measurement_raw.get("value"), "safe measurement value"
                    ),
                    tolerance=tolerance,
                )
            )
        if (
            behavior in {"safe_output", "reject_or_safe_output"}
            and not safe_measurements
        ):
            raise InvalidProbeError(
                f"{probe_id} safe handling requires expected measurements"
            )
        failure_codes_raw = raw.get("allowed_failure_codes", [])
        if not isinstance(failure_codes_raw, list) or not all(
            isinstance(value, str) and value for value in failure_codes_raw
        ):
            raise InvalidProbeError(
                f"{probe_id}.allowed_failure_codes must contain strings"
            )
        required = raw.get("required", True)
        if not isinstance(required, bool):
            raise InvalidProbeError(f"{probe_id}.required must be boolean")
        parsed.append(
            InvalidProbe(
                probe_id=probe_id,
                assignments=assignments,
                expected_behavior=str(behavior),
                safe_measurements=tuple(safe_measurements),
                allowed_failure_codes=tuple(failure_codes_raw),
                authority=_string(raw.get("authority"), "invalid probe authority"),
                evidence_ref=_string(
                    raw.get("evidence_ref"), "invalid probe evidence_ref"
                ),
                required=required,
            )
        )
    return tuple(parsed)


def _safe_output_checks(
    probe: InvalidProbe,
    execution: VariantExecution,
    context: VariantEvaluationContext,
    *,
    evaluate_l0: EvaluateL0,
    evaluate_l1: EvaluateL1,
) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = []
    if execution.status is not Status.PASSED:
        return tuple(checks)
    l0 = evaluate_l0(execution, context)
    checks.append(
        CheckResult(
            f"L2.invalid.{probe.probe_id}.safe_l0",
            l0.status,
            failure_code=(
                "invalid_probe_safe_output_l0_failed"
                if l0.status is Status.FAILED
                else None
            ),
            required=probe.required,
            observed={"layer_status": l0.status.value},
        )
    )
    if l0.status is not Status.PASSED:
        return tuple(checks)
    l1 = evaluate_l1(execution, context)
    checks.append(
        CheckResult(
            f"L2.invalid.{probe.probe_id}.safe_l1",
            l1.status,
            failure_code=(
                "invalid_probe_safe_output_l1_failed"
                if l1.status is Status.FAILED
                else None
            ),
            required=probe.required,
            observed={"layer_status": l1.status.value},
        )
    )
    for measurement in probe.safe_measurements:
        observed = execution.measurements.get(measurement.key)
        if observed is None:
            status = Status.NOT_SCOREABLE
            passed = False
        else:
            passed = math.isclose(
                float(observed),
                measurement.value,
                rel_tol=0.0,
                abs_tol=measurement.tolerance,
            )
            status = Status.PASSED if passed else Status.FAILED
        checks.append(
            CheckResult(
                f"L2.invalid.{probe.probe_id}.measurement.{measurement.key}",
                status,
                failure_code=(
                    None
                    if status is not Status.FAILED
                    else "invalid_probe_unsafe_output"
                ),
                required=probe.required,
                expected={
                    "value": measurement.value,
                    "tolerance": measurement.tolerance,
                },
                observed={"value": observed},
                summary=(
                    "safe-output measurement is absent" if observed is None else ""
                ),
            )
        )
    return tuple(checks)


def evaluate_invalid_probes(
    probes: Sequence[InvalidProbe],
    bindings: Mapping[str, ParameterBinding],
    *,
    execute: ExecuteVariant,
    evaluate_l0: EvaluateL0,
    evaluate_l1: EvaluateL1,
) -> InvalidProbeEvaluation:
    """Execute explicit invalid inputs without relabeling model failures."""

    checks: list[CheckResult] = []
    records: list[dict[str, Any]] = []
    nominals = independent_parameter_values(bindings)
    for probe in probes:
        changed = tuple(
            parameter_id
            for parameter_id in sorted(probe.assignments)
            if not math.isclose(
                probe.assignments[parameter_id],
                nominals[parameter_id],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        variant = ScheduleVariant(
            variant_id=f"invalid-{probe.probe_id}",
            schedule_kind="invalid",
            assignments=dict(probe.assignments),
            changed_parameters=changed,
            predecessor_id="nominal",
        )
        context = VariantEvaluationContext(
            variant=variant,
            public_assignments=probe.assignments,
        )
        try:
            execution = execute(
                model_parameter_values(bindings, probe.assignments),
                variant.variant_id,
            )
            if not isinstance(execution, VariantExecution):
                raise TypeError("execution callback returned an invalid result")
        except Exception as exc:  # evaluator callback boundary
            execution = VariantExecution(
                Status.EVALUATOR_ERROR,
                details={"exception_type": type(exc).__name__, "message": str(exc)},
            )

        clean_rejection = execution.status is Status.FAILED and (
            not probe.allowed_failure_codes
            or execution.failure_code in probe.allowed_failure_codes
        )
        safe_checks = _safe_output_checks(
            probe,
            execution,
            context,
            evaluate_l0=evaluate_l0,
            evaluate_l1=evaluate_l1,
        )
        safe_output = bool(safe_checks) and all(
            check.status is Status.PASSED for check in safe_checks
        )
        if execution.status is Status.EVALUATOR_ERROR:
            status = Status.EVALUATOR_ERROR
            failure_code = None
        elif probe.expected_behavior == "reject":
            status = Status.PASSED if clean_rejection else Status.FAILED
            failure_code = None if clean_rejection else "invalid_region_not_rejected"
        elif probe.expected_behavior == "safe_output":
            status = Status.PASSED if safe_output else Status.FAILED
            failure_code = None if safe_output else "invalid_region_not_handled_safely"
        else:
            accepted = clean_rejection or safe_output
            status = Status.PASSED if accepted else Status.FAILED
            failure_code = None if accepted else "invalid_region_not_handled_safely"
        check = CheckResult(
            f"L2.invalid.{probe.probe_id}.behavior",
            status,
            failure_code=failure_code,
            required=probe.required,
            expected={"behavior": probe.expected_behavior},
            observed={
                "execution": execution.as_dict(),
                "clean_rejection": clean_rejection,
                "safe_output": safe_output,
            },
            evidence=[
                {"authority": probe.authority, "evidence_ref": probe.evidence_ref}
            ],
        )
        checks.extend((check, *safe_checks))
        records.append(
            {
                "contract": probe.as_dict(),
                "execution": execution.as_dict(),
                "status": status.value,
            }
        )
    return InvalidProbeEvaluation(tuple(checks), tuple(records))
