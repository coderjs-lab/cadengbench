"""Strict, evidence-qualified parameter-domain contracts for L2 replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias


class DomainError(ValueError):
    """A parameter domain is missing, ambiguous, or malformed."""


class UnqualifiedDomainError(DomainError):
    """A domain lacks the authority needed to define benchmark perturbations."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise DomainError(f"{field} must be a finite number")
    return number


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnqualifiedDomainError(f"{field} must be a nonempty string")
    return value.strip()


@dataclass(frozen=True)
class ContinuousDomain:
    """A bounded scalar interval with explicit authority and evidence."""

    minimum: float
    maximum: float
    minimum_inclusive: bool
    maximum_inclusive: bool
    authority: str
    evidence_ref: str
    kind: str = "continuous"

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise DomainError("continuous bounds must be finite")
        if self.minimum >= self.maximum:
            raise DomainError("continuous minimum must be below maximum")

    def contains(self, value: float) -> bool:
        value = float(value)
        lower = (
            value >= self.minimum if self.minimum_inclusive else value > self.minimum
        )
        upper = (
            value <= self.maximum if self.maximum_inclusive else value < self.maximum
        )
        return math.isfinite(value) and lower and upper

    def from_unit_interval(self, value: float) -> float:
        """Map a normalized coordinate into the interval without violating open ends."""

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise DomainError("normalized domain coordinate must lie in [0, 1]")
        mapped = self.minimum + value * (self.maximum - self.minimum)
        if value == 0.0 and not self.minimum_inclusive:
            mapped = math.nextafter(self.minimum, self.maximum)
        elif value == 1.0 and not self.maximum_inclusive:
            mapped = math.nextafter(self.maximum, self.minimum)
        if not self.contains(mapped):
            raise DomainError("normalized value could not be represented in the domain")
        return mapped


@dataclass(frozen=True)
class DiscreteDomain:
    """An ordered, explicitly enumerated scalar domain."""

    values: tuple[float, ...]
    authority: str
    evidence_ref: str
    kind: str = "discrete"

    def __post_init__(self) -> None:
        if not self.values:
            raise DomainError("discrete domains require at least one value")
        if any(not math.isfinite(value) for value in self.values):
            raise DomainError("discrete values must be finite")
        if len(set(self.values)) != len(self.values):
            raise DomainError("discrete values must be unique")

    def contains(self, value: float) -> bool:
        return float(value) in self.values

    def from_unit_interval(self, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise DomainError("normalized domain coordinate must lie in [0, 1]")
        if len(self.values) == 1:
            return self.values[0]
        index = min(round(value * (len(self.values) - 1)), len(self.values) - 1)
        return self.values[index]


ParameterDomain: TypeAlias = ContinuousDomain | DiscreteDomain


def normalize_assignment(
    domains: Mapping[str, ParameterDomain],
    assignment: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, float]:
    """Validate and normalize one assignment against frozen domains.

    Schedule generation and SMT qualification share this boundary so malformed
    evaluator-owned assignments cannot reach a model execution callback.
    """

    if not isinstance(assignment, Mapping):
        raise DomainError("parameter assignment must be an object")
    if not all(isinstance(key, str) and key for key in domains):
        raise DomainError("domain parameter IDs must be nonempty strings")
    if not all(isinstance(key, str) and key for key in assignment):
        raise DomainError("assignment parameter IDs must be nonempty strings")
    expected = set(domains)
    observed = set(assignment)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if require_complete and missing:
        raise DomainError(f"parameter assignment is missing values: {missing}")
    if extra:
        raise DomainError(f"parameter assignment has unknown values: {extra}")

    normalized: dict[str, float] = {}
    for parameter_id in sorted(observed):
        value = _finite_number(assignment[parameter_id], f"assignment.{parameter_id}")
        if not domains[parameter_id].contains(value):
            raise DomainError(
                f"assignment.{parameter_id} lies outside its qualified domain"
            )
        normalized[parameter_id] = value
    return normalized


def domain_probe_values(
    domain: ParameterDomain, *, count: int = 2
) -> tuple[float, ...]:
    """Return deterministic, domain-valid probe values with boundary priority."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise DomainError("domain probe count must be a positive integer")
    if isinstance(domain, ContinuousDomain):
        if count == 1:
            unit_points = (0.5,)
        elif count == 2:
            unit_points = (0.0, 1.0)
        else:
            interiors = tuple(index / (count - 1) for index in range(1, count - 1))
            unit_points = (0.0, 1.0, *interiors)
        return tuple(domain.from_unit_interval(value) for value in unit_points)
    if not isinstance(domain, DiscreteDomain):  # pragma: no cover - closed union
        raise DomainError(f"unsupported parameter domain: {type(domain).__name__}")
    if len(domain.values) <= count:
        return domain.values
    if count == 1:
        return (domain.values[(len(domain.values) - 1) // 2],)

    indices = [0, len(domain.values) - 1]
    for position in range(1, count - 1):
        index = round(position * (len(domain.values) - 1) / (count - 1))
        if index not in indices:
            indices.append(index)
    if len(indices) < count:
        indices.extend(
            index for index in range(1, len(domain.values) - 1) if index not in indices
        )
    return tuple(domain.values[index] for index in indices[:count])


_COMMON_KEYS = {"kind", "authority", "evidence_ref"}
_CONTINUOUS_KEYS = _COMMON_KEYS | {
    "minimum",
    "maximum",
    "minimum_inclusive",
    "maximum_inclusive",
}
_DISCRETE_KEYS = _COMMON_KEYS | {"values"}


def parse_domain(
    raw: Mapping[str, Any], *, nominal: float | None = None
) -> ParameterDomain:
    """Parse one frozen domain; no inferred or default ranges are permitted."""

    if not isinstance(raw, Mapping):
        raise DomainError("domain must be an object")
    kind = raw.get("kind")
    authority = _nonempty_string(raw.get("authority"), "domain.authority")
    evidence_ref = _nonempty_string(raw.get("evidence_ref"), "domain.evidence_ref")

    if kind == "continuous":
        extras = set(raw) - _CONTINUOUS_KEYS
        if extras:
            raise DomainError(f"unsupported continuous domain fields: {sorted(extras)}")
        for name in ("minimum_inclusive", "maximum_inclusive"):
            if name not in raw or not isinstance(raw[name], bool):
                raise DomainError(f"domain.{name} must be an explicit boolean")
        domain: ParameterDomain = ContinuousDomain(
            minimum=_finite_number(raw.get("minimum"), "domain.minimum"),
            maximum=_finite_number(raw.get("maximum"), "domain.maximum"),
            minimum_inclusive=raw["minimum_inclusive"],
            maximum_inclusive=raw["maximum_inclusive"],
            authority=authority,
            evidence_ref=evidence_ref,
        )
    elif kind == "discrete":
        extras = set(raw) - _DISCRETE_KEYS
        if extras:
            raise DomainError(f"unsupported discrete domain fields: {sorted(extras)}")
        values = raw.get("values")
        if not isinstance(values, list):
            raise DomainError("domain.values must be an explicit list")
        domain = DiscreteDomain(
            values=tuple(
                _finite_number(value, f"domain.values[{index}]")
                for index, value in enumerate(values)
            ),
            authority=authority,
            evidence_ref=evidence_ref,
        )
    else:
        raise DomainError("domain.kind must be 'continuous' or 'discrete'")

    if nominal is not None and not domain.contains(_finite_number(nominal, "nominal")):
        raise DomainError("nominal parameter value lies outside its qualified domain")
    return domain


def parse_parameter_domains(
    parameter_system: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, ParameterDomain]:
    """Load public domains by stable parameter ID.

    Missing domains remain explicit: callers either receive an
    :class:`UnqualifiedDomainError` or, with ``require_complete=False``, only the
    qualified subset. The function never creates fallback percentage ranges.
    """

    records = parameter_system.get("parameters")
    if not isinstance(records, list) or not records:
        raise DomainError("parameter_system.parameters must be a nonempty list")
    parsed: dict[str, ParameterDomain] = {}
    missing: list[str] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DomainError(f"parameter_system.parameters[{index}] must be an object")
        parameter_id = record.get("parameter_id")
        if not isinstance(parameter_id, str) or not parameter_id:
            raise DomainError(
                f"parameter_system.parameters[{index}] has no parameter_id"
            )
        if parameter_id in seen:
            raise DomainError(f"duplicate public parameter ID: {parameter_id}")
        seen.add(parameter_id)
        parameter_kind = record.get("parameter_kind", "independent")
        if parameter_kind not in {"independent", "derived"}:
            raise DomainError(
                f"parameter {parameter_id} has unsupported parameter_kind"
            )
        raw_domain = record.get("domain")
        if parameter_kind == "derived":
            if raw_domain is not None:
                raise DomainError(
                    f"derived parameter {parameter_id} cannot declare a sampled domain"
                )
            continue
        if raw_domain is None:
            missing.append(parameter_id)
            continue
        if not isinstance(raw_domain, Mapping):
            raise DomainError(f"domain for {parameter_id} must be an object or null")
        parsed[parameter_id] = parse_domain(raw_domain, nominal=record.get("nominal"))
    if require_complete and missing:
        raise UnqualifiedDomainError(
            "qualified domains are absent for public parameters: " + ", ".join(missing)
        )
    if require_complete and not parsed:
        raise DomainError("parameter system has no independently perturbable domain")
    return parsed
