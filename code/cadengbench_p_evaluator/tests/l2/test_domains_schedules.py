from __future__ import annotations

import math
from itertools import pairwise

import pytest

from cadengbench_p_eval.parametrics.domains import (
    ContinuousDomain,
    DomainError,
    UnqualifiedDomainError,
    parse_domain,
    parse_parameter_domains,
)
from cadengbench_p_eval.parametrics.schedules import (
    ScheduleError,
    generate_directed_schedule,
    generate_morris_schedule,
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


def test_domain_requires_explicit_authority_and_evidence() -> None:
    raw = qualified_domain(1.0, 2.0)
    raw["authority"] = None
    with pytest.raises(UnqualifiedDomainError):
        parse_domain(raw, nominal=1.5)


def test_domain_rejects_unknown_fields_and_outside_nominal() -> None:
    raw = qualified_domain(1.0, 2.0)
    raw["invented_range"] = True
    with pytest.raises(DomainError, match="unsupported"):
        parse_domain(raw, nominal=1.5)
    with pytest.raises(DomainError, match="outside"):
        parse_domain(qualified_domain(1.0, 2.0), nominal=3.0)


def test_missing_domains_are_not_replaced_by_default_ranges() -> None:
    system = {
        "parameters": [{"parameter_id": "P-001", "nominal": 10.0, "domain": None}]
    }
    with pytest.raises(UnqualifiedDomainError, match="P-001"):
        parse_parameter_domains(system)
    assert parse_parameter_domains(system, require_complete=False) == {}


def test_directed_schedule_is_deterministic_and_one_factor_at_a_time() -> None:
    domains = {
        "P-002": parse_domain(qualified_domain(2.0, 8.0), nominal=4.0),
        "P-001": parse_domain(qualified_domain(5.0, 15.0), nominal=10.0),
    }
    nominals = {"P-001": 10.0, "P-002": 4.0}
    first = generate_directed_schedule(domains, nominals, points_per_parameter=2)
    second = generate_directed_schedule(domains, nominals, points_per_parameter=2)
    assert first.as_dict() == second.as_dict()
    assert first.covered_parameters == ("P-001", "P-002")
    assert len(first.variants) == 4
    for variant in first.variants:
        assert len(variant.changed_parameters) == 1
        changed = variant.changed_parameters[0]
        unchanged = ({"P-001", "P-002"} - {changed}).pop()
        assert variant.assignments[unchanged] == nominals[unchanged]


def test_directed_schedule_excludes_float_noise_at_nominal() -> None:
    domains = {
        "P-001": parse_domain(
            qualified_domain(10.0, 15.870000000000001),
            nominal=15.87,
        )
    }
    schedule = generate_directed_schedule(
        domains,
        {"P-001": 15.87},
        points_per_parameter=2,
    )

    assert len(schedule.variants) == 1
    assert schedule.variants[0].assignments["P-001"] == 10.0


def test_directed_schedule_filters_explicitly_infeasible_variants() -> None:
    domains = {"P-001": parse_domain(qualified_domain(0.0, 10.0), nominal=5.0)}
    schedule = generate_directed_schedule(
        domains,
        {"P-001": 5.0},
        points_per_parameter=2,
        feasibility=lambda values: values["P-001"] <= 7.0,
    )

    assert [variant.assignments["P-001"] for variant in schedule.variants] == [0.0]
    assert schedule.metadata["rejected_infeasible_variants"] == 1


def test_morris_schedule_is_seeded_complete_and_uses_standard_delta() -> None:
    domains = {
        "P-001": parse_domain(qualified_domain(0.0, 10.0), nominal=5.0),
        "P-002": parse_domain(qualified_domain(10.0, 20.0), nominal=15.0),
        "P-003": parse_domain(qualified_domain(20.0, 30.0), nominal=25.0),
    }
    first = generate_morris_schedule(
        domains, levels=4, trajectories=2, seed=17, max_variants=8
    )
    second = generate_morris_schedule(
        domains, levels=4, trajectories=2, seed=17, max_variants=8
    )
    different = generate_morris_schedule(
        domains, levels=4, trajectories=2, seed=18, max_variants=8
    )
    assert first.as_dict() == second.as_dict()
    assert first.as_dict() != different.as_dict()
    assert len(first.variants) == 8
    expected_delta = 4 / (2 * (4 - 1))
    for trajectory_index in range(2):
        trajectory = first.variants[trajectory_index * 4 : (trajectory_index + 1) * 4]
        assert trajectory[0].changed_parameters == ()
        assert {variant.changed_parameters[0] for variant in trajectory[1:]} == set(
            domains
        )
        for previous, current in pairwise(trajectory):
            parameter_id = current.changed_parameters[0]
            domain = domains[parameter_id]
            assert isinstance(domain, ContinuousDomain)
            observed = abs(
                current.assignments[parameter_id] - previous.assignments[parameter_id]
            ) / (domain.maximum - domain.minimum)
            assert math.isclose(observed, expected_delta)


def test_morris_rejects_odd_levels_and_bounds_feasibility_search() -> None:
    domains = {"P-001": parse_domain(qualified_domain(0.0, 1.0), nominal=0.5)}
    with pytest.raises(ScheduleError, match="even"):
        generate_morris_schedule(domains, levels=5, trajectories=1, seed=1)
    with pytest.raises(ScheduleError, match="exhausted"):
        generate_morris_schedule(
            domains,
            levels=4,
            trajectories=1,
            seed=1,
            feasibility=lambda _: False,
            max_attempts=3,
        )


def test_morris_budget_never_emits_partial_trajectory() -> None:
    domains = {
        "P-001": parse_domain(qualified_domain(0.0, 1.0), nominal=0.5),
        "P-002": parse_domain(qualified_domain(0.0, 1.0), nominal=0.5),
    }
    with pytest.raises(ScheduleError, match="complete"):
        generate_morris_schedule(
            domains, levels=4, trajectories=1, seed=1, max_variants=2
        )
    schedule = generate_morris_schedule(
        domains, levels=4, trajectories=3, seed=1, max_variants=7
    )
    assert len(schedule.variants) == 6
    assert schedule.metadata["generated_trajectories"] == 2
    assert schedule.metadata["truncated_by_budget"] is True
