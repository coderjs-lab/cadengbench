"""Stable result contracts shared by every evaluator layer."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    NOT_SCOREABLE = "not_scoreable"
    EVALUATOR_ERROR = "evaluator_error"
    NOT_REACHED = "not_reached"


def passed_value(status: Status) -> bool | None:
    if status is Status.PASSED:
        return True
    if status is Status.FAILED:
        return False
    return None


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: Status
    required: bool = True
    failure_code: str | None = None
    summary: str = ""
    expected: Any = None
    observed: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    runtime_seconds: float | None = None
    passed: bool | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", passed_value(self.status))
        if self.status is Status.FAILED and not self.failure_code:
            raise ValueError("failed checks require a failure_code")
        if self.status is not Status.FAILED and self.failure_code:
            raise ValueError("failure_code is reserved for failed checks")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


def reduce_required_status(checks: Iterable[CheckResult]) -> Status:
    required = [check for check in checks if check.required]
    active = [check for check in required if check.status is not Status.NOT_APPLICABLE]
    if not active:
        return Status.NOT_APPLICABLE
    statuses = {check.status for check in active}
    if Status.EVALUATOR_ERROR in statuses:
        return Status.EVALUATOR_ERROR
    if Status.FAILED in statuses:
        return Status.FAILED
    if Status.NOT_SCOREABLE in statuses:
        return Status.NOT_SCOREABLE
    if Status.NOT_REACHED in statuses:
        return Status.NOT_REACHED
    return Status.PASSED


@dataclass(frozen=True)
class LayerResult:
    layer: str
    status: Status
    checks: tuple[CheckResult, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float | None = None
    passed: bool | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", passed_value(self.status))

    @classmethod
    def from_checks(
        cls,
        layer: str,
        checks: Iterable[CheckResult],
        **kwargs: Any,
    ) -> LayerResult:
        frozen = tuple(checks)
        return cls(
            layer=layer,
            status=reduce_required_status(frozen),
            checks=frozen,
            **kwargs,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "status": self.status.value,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(frozen=True)
class EvaluationResult:
    schema_version: str
    evaluator_version: str
    item_id: str
    protocol: str
    identity: dict[str, str]
    layers: tuple[LayerResult, ...]
    overall_status: Status
    environment: dict[str, Any]
    runtime_seconds: float
    llm_judge_used: bool = False
    passed: bool | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", passed_value(self.overall_status))
        if self.llm_judge_used:
            raise ValueError("CADEngBench-P scoring cannot use an LLM judge")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator_version": self.evaluator_version,
            "item_id": self.item_id,
            "protocol": self.protocol,
            "identity": self.identity,
            "layers": [layer.as_dict() for layer in self.layers],
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "environment": self.environment,
            "runtime_seconds": self.runtime_seconds,
            "llm_judge_used": self.llm_judge_used,
        }


def _runtime_from_value(
    value: Any, field: str, *, required: bool = False
) -> float | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    runtime = float(value)
    if not math.isfinite(runtime) or runtime < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return runtime


def check_result_from_dict(value: dict[str, Any]) -> CheckResult:
    if not isinstance(value, Mapping):
        raise ValueError("check result must be an object")
    check_id = value.get("check_id")
    required = value.get("required", True)
    metrics = value.get("metrics", {})
    evidence = value.get("evidence", [])
    if not isinstance(check_id, str) or not check_id:
        raise ValueError("check_id must be a nonempty string")
    if not isinstance(required, bool):
        raise ValueError("check required must be a boolean")
    if not isinstance(metrics, Mapping):
        raise ValueError("check metrics must be an object")
    if not isinstance(evidence, list) or not all(
        isinstance(record, Mapping) for record in evidence
    ):
        raise ValueError("check evidence must be a list of objects")
    return CheckResult(
        check_id=check_id,
        status=Status(str(value["status"])),
        required=required,
        failure_code=value.get("failure_code"),
        summary=str(value.get("summary") or ""),
        expected=value.get("expected"),
        observed=value.get("observed"),
        metrics=dict(metrics),
        evidence=[dict(record) for record in evidence],
        runtime_seconds=_runtime_from_value(
            value.get("runtime_seconds"), "check runtime_seconds"
        ),
    )


def layer_result_from_dict(value: dict[str, Any]) -> LayerResult:
    if not isinstance(value, Mapping):
        raise ValueError("layer result must be an object")
    checks = value.get("checks", [])
    metrics = value.get("metrics", {})
    artifacts = value.get("artifacts", {})
    if not isinstance(checks, list):
        raise ValueError("layer checks must be a list")
    if not isinstance(metrics, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("layer metrics and artifacts must be objects")
    return LayerResult(
        layer=str(value["layer"]),
        status=Status(str(value["status"])),
        checks=tuple(check_result_from_dict(check) for check in checks),
        metrics=dict(metrics),
        artifacts=dict(artifacts),
        runtime_seconds=_runtime_from_value(
            value.get("runtime_seconds"), "layer runtime_seconds"
        ),
    )


def evaluation_result_from_dict(value: dict[str, Any]) -> EvaluationResult:
    if not isinstance(value, Mapping):
        raise ValueError("evaluation result must be an object")
    identity = value.get("identity")
    environment = value.get("environment")
    layers = value.get("layers")
    llm_judge_used = value.get("llm_judge_used", False)
    if not isinstance(identity, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("evaluation identity and environment must be objects")
    if not isinstance(layers, list):
        raise ValueError("evaluation layers must be a list")
    if not isinstance(llm_judge_used, bool):
        raise ValueError("llm_judge_used must be a boolean")
    return EvaluationResult(
        schema_version=str(value["schema_version"]),
        evaluator_version=str(value["evaluator_version"]),
        item_id=str(value["item_id"]),
        protocol=str(value["protocol"]),
        identity=dict(identity),
        layers=tuple(layer_result_from_dict(layer) for layer in layers),
        overall_status=Status(str(value["overall_status"])),
        environment=dict(environment),
        runtime_seconds=_runtime_from_value(
            value.get("runtime_seconds"),
            "evaluation runtime_seconds",
            required=True,
        ),
        llm_judge_used=llm_judge_used,
    )
