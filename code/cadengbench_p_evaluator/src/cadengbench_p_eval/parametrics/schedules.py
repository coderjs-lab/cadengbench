"""Deterministic, feasibility-aware L2 perturbation schedules."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from typing import Any, TypeAlias

from .domains import (
    ContinuousDomain,
    DomainError,
    ParameterDomain,
    domain_probe_values,
    normalize_assignment,
)
from .relations import (
    ComparisonPredicate,
    LogicalPredicate,
    RelationContext,
    RelationError,
    TypedRelation,
)


class ScheduleError(ValueError):
    """A replay schedule cannot be generated from the frozen contract."""


FeasibilityPredicate = Callable[[Mapping[str, float]], bool]
TypedFeasibilityConstraint: TypeAlias = (
    ComparisonPredicate | LogicalPredicate | TypedRelation
)
FeasibilitySpec: TypeAlias = (
    FeasibilityPredicate
    | TypedFeasibilityConstraint
    | Sequence[FeasibilityPredicate | TypedFeasibilityConstraint]
)


@dataclass(frozen=True)
class ScheduleVariantProvenance:
    """One source-schedule origin for a canonical assignment."""

    variant_id: str
    schedule_kind: str
    changed_parameters: tuple[str, ...]
    predecessor_id: str | None = None
    trajectory_index: int | None = None
    step_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "schedule_kind": self.schedule_kind,
            "changed_parameters": list(self.changed_parameters),
            "predecessor_id": self.predecessor_id,
            "trajectory_index": self.trajectory_index,
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class ScheduleVariant:
    variant_id: str
    schedule_kind: str
    assignments: dict[str, float]
    changed_parameters: tuple[str, ...]
    predecessor_id: str | None = None
    trajectory_index: int | None = None
    step_index: int | None = None
    provenance: tuple[ScheduleVariantProvenance, ...] = ()

    @property
    def origins(self) -> tuple[ScheduleVariantProvenance, ...]:
        if self.provenance:
            return self.provenance
        return (
            ScheduleVariantProvenance(
                variant_id=self.variant_id,
                schedule_kind=self.schedule_kind,
                changed_parameters=self.changed_parameters,
                predecessor_id=self.predecessor_id,
                trajectory_index=self.trajectory_index,
                step_index=self.step_index,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "schedule_kind": self.schedule_kind,
            "assignments": dict(sorted(self.assignments.items())),
            "changed_parameters": list(self.changed_parameters),
            "predecessor_id": self.predecessor_id,
            "trajectory_index": self.trajectory_index,
            "step_index": self.step_index,
            "provenance": [origin.as_dict() for origin in self.origins],
        }


@dataclass(frozen=True)
class PerturbationSchedule:
    kind: str
    variants: tuple[ScheduleVariant, ...]
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def covered_parameters(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    parameter_id
                    for variant in self.variants
                    for origin in variant.origins
                    for parameter_id in origin.changed_parameters
                }
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "variant_count": len(self.variants),
            "covered_parameters": list(self.covered_parameters),
            "metadata": self.metadata,
            "variants": [variant.as_dict() for variant in self.variants],
        }


def _feasibility_checker(
    feasibility: FeasibilitySpec | None,
) -> tuple[FeasibilityPredicate | None, str]:
    if feasibility is None:
        return None, "none"
    if isinstance(
        feasibility, (ComparisonPredicate, LogicalPredicate, TypedRelation)
    ) or callable(feasibility):
        constraints: tuple[FeasibilityPredicate | TypedFeasibilityConstraint, ...] = (
            feasibility,
        )
    elif isinstance(feasibility, Sequence) and not isinstance(
        feasibility, (str, bytes)
    ):
        constraints = tuple(feasibility)
    else:
        raise ScheduleError(
            "feasibility must be a callable or typed predicate/relation sequence"
        )
    if not all(
        callable(constraint)
        or isinstance(
            constraint, (ComparisonPredicate, LogicalPredicate, TypedRelation)
        )
        for constraint in constraints
    ):
        raise ScheduleError("feasibility contains an unsupported constraint")

    typed = any(
        isinstance(constraint, (ComparisonPredicate, LogicalPredicate, TypedRelation))
        for constraint in constraints
    )

    def check(assignments: Mapping[str, float]) -> bool:
        context = RelationContext(parameters=assignments)
        for constraint in constraints:
            try:
                if isinstance(
                    constraint,
                    (ComparisonPredicate, LogicalPredicate, TypedRelation),
                ):
                    holds = constraint.evaluate(context)
                else:
                    holds = bool(constraint(assignments))
            except RelationError as exc:
                raise ScheduleError(
                    f"invalid frozen feasibility predicate: {exc}"
                ) from exc
            except ScheduleError:
                raise
            except Exception as exc:
                raise ScheduleError(
                    f"feasibility callback failed: {type(exc).__name__}: {exc}"
                ) from exc
            if not holds:
                return False
        return True

    return check, "typed" if typed else "callable"


def _assignment_key(assignments: Mapping[str, float]) -> tuple[tuple[str, str], ...]:
    key: list[tuple[str, str]] = []
    for parameter_id in sorted(assignments):
        value = float(assignments[parameter_id])
        if not math.isfinite(value):
            raise ScheduleError("schedule assignments must be finite")
        if value == 0.0:
            value = 0.0
        key.append((parameter_id, value.hex()))
    return tuple(key)


def _merged_origins(
    left: Sequence[ScheduleVariantProvenance],
    right: Sequence[ScheduleVariantProvenance],
) -> tuple[ScheduleVariantProvenance, ...]:
    merged: list[ScheduleVariantProvenance] = []
    seen: set[tuple[Any, ...]] = set()
    for origin in (*left, *right):
        identity = (
            origin.variant_id,
            origin.schedule_kind,
            origin.changed_parameters,
            origin.predecessor_id,
            origin.trajectory_index,
            origin.step_index,
        )
        if identity not in seen:
            seen.add(identity)
            merged.append(origin)
    return tuple(merged)


def deduplicate_schedule(schedule: PerturbationSchedule) -> PerturbationSchedule:
    """Collapse identical assignments while preserving every schedule origin."""

    if not schedule.variants:
        return schedule
    expected_parameters = set(schedule.variants[0].assignments)
    retained: list[ScheduleVariant] = []
    retained_index: dict[tuple[tuple[str, str], ...], int] = {}
    id_keys: dict[str, tuple[tuple[str, str], ...]] = {}
    aliases: dict[str, str] = {}

    for variant in schedule.variants:
        if set(variant.assignments) != expected_parameters:
            raise ScheduleError(
                "all combined variants must assign the same parameter IDs"
            )
        key = _assignment_key(variant.assignments)
        previous_key = id_keys.get(variant.variant_id)
        if previous_key is not None and previous_key != key:
            raise ScheduleError(
                f"variant ID {variant.variant_id!r} refers to multiple assignments"
            )
        id_keys[variant.variant_id] = key
        if key not in retained_index:
            aliases[variant.variant_id] = variant.variant_id
            retained_index[key] = len(retained)
            retained.append(replace(variant, provenance=variant.origins))
            continue

        index = retained_index[key]
        canonical = retained[index]
        aliases[variant.variant_id] = canonical.variant_id
        retained[index] = replace(
            canonical,
            provenance=_merged_origins(canonical.origins, variant.origins),
        )

    remapped: list[ScheduleVariant] = []
    for variant in retained:
        predecessor = variant.predecessor_id
        if predecessor is not None:
            predecessor = aliases.get(predecessor, predecessor)
            if predecessor == variant.variant_id:
                predecessor = None
        origins = tuple(
            replace(
                origin,
                predecessor_id=(
                    None
                    if origin.predecessor_id is not None
                    and aliases.get(origin.predecessor_id, origin.predecessor_id)
                    == variant.variant_id
                    else aliases.get(origin.predecessor_id, origin.predecessor_id)
                    if origin.predecessor_id is not None
                    else None
                ),
            )
            for origin in variant.origins
        )
        remapped.append(
            replace(
                variant,
                predecessor_id=predecessor,
                provenance=origins,
            )
        )

    input_count = len(schedule.variants)
    output_count = len(remapped)
    metadata = dict(schedule.metadata)
    metadata.update(
        {
            "input_variant_count": input_count,
            "deduplicated_variant_count": output_count,
            "duplicate_assignment_count": input_count - output_count,
        }
    )
    return PerturbationSchedule(
        kind=schedule.kind,
        variants=tuple(remapped),
        seed=schedule.seed,
        metadata=metadata,
    )


def _validate_nominals(
    domains: Mapping[str, ParameterDomain], nominals: Mapping[str, float]
) -> dict[str, float]:
    if not domains:
        raise ScheduleError("at least one qualified domain is required")
    missing = sorted(set(domains) - set(nominals))
    if missing:
        raise ScheduleError(f"nominal values are missing for: {missing}")
    try:
        return normalize_assignment(
            domains,
            {parameter_id: nominals[parameter_id] for parameter_id in domains},
        )
    except DomainError as exc:
        raise ScheduleError(str(exc)) from exc


def _directed_values(domain: ParameterDomain, count: int) -> list[float]:
    if count < 2:
        raise ScheduleError(
            "directed schedules require at least two points per parameter"
        )
    try:
        return list(domain_probe_values(domain, count=count))
    except DomainError as exc:
        raise ScheduleError(str(exc)) from exc


def _assignment_value_changed(left: float, right: float) -> bool:
    return not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def generate_directed_schedule(
    domains: Mapping[str, ParameterDomain],
    nominals: Mapping[str, float],
    *,
    points_per_parameter: int = 2,
    max_variants: int | None = None,
    feasibility: FeasibilitySpec | None = None,
) -> PerturbationSchedule:
    """Vary one public parameter at a time over explicit domain points."""

    baseline = _validate_nominals(domains, nominals)
    feasibility_check, feasibility_kind = _feasibility_checker(feasibility)
    if feasibility_check is not None and not feasibility_check(baseline):
        raise ScheduleError("nominal assignment violates the feasibility contract")
    if max_variants is not None and max_variants < 1:
        raise ScheduleError("max_variants must be positive")
    candidates: dict[str, list[float]] = {}
    for parameter_id in sorted(domains):
        candidates[parameter_id] = [
            value
            for value in _directed_values(domains[parameter_id], points_per_parameter)
            if _assignment_value_changed(value, baseline[parameter_id])
        ]

    variants: list[ScheduleVariant] = []
    rejected_infeasible = 0
    round_index = 0
    while any(round_index < len(values) for values in candidates.values()):
        for parameter_id in sorted(candidates):
            if round_index >= len(candidates[parameter_id]):
                continue
            assignments = dict(baseline)
            assignments[parameter_id] = candidates[parameter_id][round_index]
            if feasibility_check is not None and not feasibility_check(assignments):
                rejected_infeasible += 1
                continue
            variants.append(
                ScheduleVariant(
                    variant_id=f"directed-{len(variants):04d}-{parameter_id}",
                    schedule_kind="directed",
                    assignments=assignments,
                    changed_parameters=(parameter_id,),
                    predecessor_id="nominal",
                    step_index=round_index,
                )
            )
            if max_variants is not None and len(variants) >= max_variants:
                break
        if max_variants is not None and len(variants) >= max_variants:
            break
        round_index += 1
    if not variants:
        raise ScheduleError("directed schedule contains no non-nominal variants")
    requested = sum(len(values) for values in candidates.values())
    return deduplicate_schedule(
        PerturbationSchedule(
            kind="directed",
            variants=tuple(variants),
            metadata={
                "points_per_parameter": points_per_parameter,
                "requested_variants": requested,
                "truncated_by_budget": len(variants) + rejected_infeasible < requested,
                "rejected_infeasible_variants": rejected_infeasible,
                "feasibility_kind": feasibility_kind,
            },
        )
    )


def _morris_trajectory(
    domains: Mapping[str, ContinuousDomain],
    *,
    levels: int,
    rng: random.Random,
    trajectory_index: int,
) -> list[ScheduleVariant]:
    parameter_ids = sorted(domains)
    delta_index = levels // 2
    delta = delta_index / (levels - 1)
    directions = {parameter_id: rng.choice((-1, 1)) for parameter_id in parameter_ids}
    grid_indices: dict[str, int] = {}
    for parameter_id in parameter_ids:
        if directions[parameter_id] > 0:
            grid_indices[parameter_id] = rng.randint(0, levels - 1 - delta_index)
        else:
            grid_indices[parameter_id] = rng.randint(delta_index, levels - 1)
    order = rng.sample(parameter_ids, len(parameter_ids))

    def assignments() -> dict[str, float]:
        return {
            parameter_id: domains[parameter_id].from_unit_interval(
                grid_indices[parameter_id] / (levels - 1)
            )
            for parameter_id in parameter_ids
        }

    variants = [
        ScheduleVariant(
            variant_id=f"morris-t{trajectory_index:03d}-s000",
            schedule_kind="morris",
            assignments=assignments(),
            changed_parameters=(),
            trajectory_index=trajectory_index,
            step_index=0,
        )
    ]
    for step_index, parameter_id in enumerate(order, start=1):
        grid_indices[parameter_id] += directions[parameter_id] * delta_index
        variants.append(
            ScheduleVariant(
                variant_id=f"morris-t{trajectory_index:03d}-s{step_index:03d}",
                schedule_kind="morris",
                assignments=assignments(),
                changed_parameters=(parameter_id,),
                predecessor_id=variants[-1].variant_id,
                trajectory_index=trajectory_index,
                step_index=step_index,
            )
        )
    assert all(
        math.isclose(
            abs(
                variants[index].assignments[variants[index].changed_parameters[0]]
                - variants[index - 1].assignments[variants[index].changed_parameters[0]]
            )
            / (
                domains[variants[index].changed_parameters[0]].maximum
                - domains[variants[index].changed_parameters[0]].minimum
            ),
            delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for index in range(1, len(variants))
    )
    return variants


def generate_morris_schedule(
    domains: Mapping[str, ParameterDomain],
    *,
    levels: int,
    trajectories: int,
    seed: int,
    max_variants: int | None = None,
    feasibility: FeasibilitySpec | None = None,
    max_attempts: int | None = None,
) -> PerturbationSchedule:
    """Generate seeded, complete Morris elementary-effects trajectories.

    Each accepted source trajectory contains ``k + 1`` points and changes every
    one of the ``k`` parameters exactly once by ``Delta = p / (2 * (p - 1))`` in
    normalized coordinates. Feasibility rejection is bounded and deterministic;
    assignments shared by trajectories execute once with all origins retained.
    """

    if not domains:
        raise ScheduleError("Morris schedules require at least one domain")
    if levels < 4 or levels % 2:
        raise ScheduleError("Morris levels must be an even integer of at least four")
    if trajectories < 1:
        raise ScheduleError("Morris trajectories must be positive")
    feasibility_check, feasibility_kind = _feasibility_checker(feasibility)
    continuous: dict[str, ContinuousDomain] = {}
    for parameter_id, domain in domains.items():
        if not isinstance(domain, ContinuousDomain):
            raise ScheduleError("Morris schedules support continuous domains only")
        continuous[parameter_id] = domain

    points_per_trajectory = len(continuous) + 1
    if max_variants is None:
        accepted_target = trajectories
    else:
        if max_variants < points_per_trajectory:
            raise ScheduleError(
                "variant budget cannot fit one complete Morris trajectory"
            )
        accepted_target = min(trajectories, max_variants // points_per_trajectory)
    attempt_budget = max_attempts if max_attempts is not None else accepted_target * 20
    if attempt_budget < accepted_target:
        raise ScheduleError(
            "max_attempts cannot be below the requested trajectory count"
        )

    rng = random.Random(seed)
    accepted: list[ScheduleVariant] = []
    attempts = 0
    while len(accepted) // points_per_trajectory < accepted_target:
        if attempts >= attempt_budget:
            raise ScheduleError(
                "bounded Morris feasibility search exhausted before enough "
                "trajectories were accepted"
            )
        trajectory_index = len(accepted) // points_per_trajectory
        candidate = _morris_trajectory(
            continuous,
            levels=levels,
            rng=rng,
            trajectory_index=trajectory_index,
        )
        attempts += 1
        if feasibility_check is not None and not all(
            feasibility_check(variant.assignments) for variant in candidate
        ):
            continue
        accepted.extend(candidate)

    delta = levels / (2 * (levels - 1))
    return deduplicate_schedule(
        PerturbationSchedule(
            kind="morris",
            variants=tuple(accepted),
            seed=seed,
            metadata={
                "levels": levels,
                "delta": delta,
                "requested_trajectories": trajectories,
                "generated_trajectories": accepted_target,
                "attempts": attempts,
                "attempt_budget": attempt_budget,
                "truncated_by_budget": accepted_target < trajectories,
                "rejected_infeasible_trajectories": attempts - accepted_target,
                "feasibility_kind": feasibility_kind,
            },
        )
    )


def _changed_parameters(
    baseline: Mapping[str, float], assignments: Mapping[str, float]
) -> tuple[str, ...]:
    return tuple(
        parameter_id
        for parameter_id in sorted(baseline)
        if _assignment_value_changed(
            baseline[parameter_id], assignments[parameter_id]
        )
    )


def generate_coupled_schedule(
    domains: Mapping[str, ParameterDomain],
    nominals: Mapping[str, float],
    *,
    points_per_parameter: int = 2,
    max_variants: int | None = None,
    feasibility: FeasibilitySpec | None = None,
) -> PerturbationSchedule:
    """Generate deterministic multi-parameter probes.

    Boundary-priority all-parameter patterns are followed by pairwise boundary
    combinations. Only feasible assignments that change at least two public
    parameters are retained.
    """

    baseline = _validate_nominals(domains, nominals)
    if len(domains) < 2:
        raise ScheduleError("coupled schedules require at least two parameters")
    if points_per_parameter < 2:
        raise ScheduleError(
            "coupled schedules require at least two points per parameter"
        )
    if max_variants is not None and max_variants < 1:
        raise ScheduleError("max_variants must be positive")
    feasibility_check, feasibility_kind = _feasibility_checker(feasibility)
    if feasibility_check is not None and not feasibility_check(baseline):
        raise ScheduleError("nominal assignment violates the feasibility contract")

    probe_values: dict[str, tuple[float, ...]] = {}
    for parameter_id in sorted(domains):
        values = _directed_values(domains[parameter_id], points_per_parameter)
        probe_values[parameter_id] = tuple(
            value
            for value in values
            if _assignment_value_changed(value, baseline[parameter_id])
        )

    candidates: list[dict[str, float]] = []
    active_ids = [
        parameter_id for parameter_id in sorted(domains) if probe_values[parameter_id]
    ]
    if len(active_ids) < 2:
        raise ScheduleError(
            "fewer than two parameters have non-nominal domain probe values"
        )

    maximum_rank = max(len(probe_values[parameter_id]) for parameter_id in active_ids)
    for rank in range(maximum_rank):
        assignments = dict(baseline)
        for parameter_id in active_ids:
            values = probe_values[parameter_id]
            assignments[parameter_id] = values[min(rank, len(values) - 1)]
        candidates.append(assignments)
    for offset in (0, 1):
        assignments = dict(baseline)
        for index, parameter_id in enumerate(active_ids):
            values = probe_values[parameter_id]
            assignments[parameter_id] = values[(index + offset) % len(values)]
        candidates.append(assignments)

    for left, right in combinations(active_ids, 2):
        for left_value, right_value in product(probe_values[left], probe_values[right]):
            assignments = dict(baseline)
            assignments[left] = left_value
            assignments[right] = right_value
            candidates.append(assignments)

    unique_candidates: list[tuple[dict[str, float], tuple[str, ...]]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for candidate in candidates:
        try:
            normalized = normalize_assignment(domains, candidate)
        except DomainError as exc:  # generated assignments are evaluator-owned
            raise ScheduleError(str(exc)) from exc
        changed = _changed_parameters(baseline, normalized)
        if len(changed) < 2:
            continue
        key = _assignment_key(normalized)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((normalized, changed))

    feasible_candidates: list[tuple[dict[str, float], tuple[str, ...]]] = []
    rejected_infeasible = 0
    for assignments, changed in unique_candidates:
        if feasibility_check is not None and not feasibility_check(assignments):
            rejected_infeasible += 1
            continue
        feasible_candidates.append((assignments, changed))
    if not feasible_candidates:
        raise ScheduleError("coupled schedule contains no feasible variants")

    selected = (
        feasible_candidates
        if max_variants is None
        else feasible_candidates[:max_variants]
    )
    variants = tuple(
        ScheduleVariant(
            variant_id=f"coupled-{index:04d}",
            schedule_kind="coupled",
            assignments=assignments,
            changed_parameters=changed,
            predecessor_id="nominal",
            step_index=index,
        )
        for index, (assignments, changed) in enumerate(selected)
    )
    return deduplicate_schedule(
        PerturbationSchedule(
            kind="coupled",
            variants=variants,
            metadata={
                "points_per_parameter": points_per_parameter,
                "candidate_assignment_count": len(unique_candidates),
                "feasible_assignment_count": len(feasible_candidates),
                "rejected_infeasible_variants": rejected_infeasible,
                "truncated_by_budget": len(selected) < len(feasible_candidates),
                "feasibility_kind": feasibility_kind,
            },
        )
    )


def combine_schedules(*schedules: PerturbationSchedule) -> PerturbationSchedule:
    """Combine and assignment-deduplicate schedules without losing provenance."""

    variants = tuple(variant for schedule in schedules for variant in schedule.variants)
    combined = PerturbationSchedule(
        kind="combined",
        variants=variants,
        metadata={
            "components": [schedule.as_dict() for schedule in schedules],
            "component_count": len(schedules),
        },
    )
    return deduplicate_schedule(combined)
