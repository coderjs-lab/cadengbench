"""Reference-relative response scoring and engineering requirement reduction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from cadengbench_p_l3_response.calibration import (
    RESPONSE_METRICS,
    CalibrationError,
    extract_response_metrics,
    vector_norm,
)
from cadengbench_p_l3_response.provenance import evaluator_identity
from cadengbench_p_l3_response.schemas import validate_document


class ResponseScoringError(ValueError):
    """Matched solver evidence is incomplete or incompatible."""


def _relative_error(observed: float, expected: float, epsilon: float) -> float:
    return abs(observed - expected) / max(abs(expected), epsilon)


def _vector_relative_error(
    observed: list[float], expected: list[float], epsilon: float
) -> float:
    difference = [observed[index] - expected[index] for index in range(3)]
    return vector_norm(difference) / max(vector_norm(expected), epsilon)


def _symmetric_log_error(candidate: float, reference: float, epsilon: float) -> float:
    return abs(math.log(max(candidate, epsilon) / max(reference, epsilon)))


def _require_solver_identity(
    contract: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    bindings = contract["bindings"]
    if result.get("evaluator_build_hash") != bindings["l3_evaluator_build_hash"]:
        raise ResponseScoringError("L3 evaluator build does not match the contract")
    if result.get("runtime_lock_sha256") != bindings["l3_runtime_lock_sha256"]:
        raise ResponseScoringError("L3 runtime lock does not match the contract")


def _is_complete_solve(result: Mapping[str, Any]) -> bool:
    stages = result.get("stages")
    if not isinstance(stages, Mapping):
        return False
    return all(
        isinstance(stages.get(stage), Mapping)
        and stages[stage].get("status") == "passed"
        for stage in ("geometry", "selectors", "mesh", "solver", "result_sanity")
    ) and isinstance(result.get("metrics"), Mapping)


def _stage_status(result: Mapping[str, Any], stage: str) -> str:
    stages = result.get("stages")
    value = stages.get(stage) if isinstance(stages, Mapping) else None
    return (
        str(value.get("status", "not_reached"))
        if isinstance(value, Mapping)
        else "not_reached"
    )


def _metric_payload(metrics: Mapping[str, float]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in (*RESPONSE_METRICS, "force_equilibrium_relative_error")
    }


def _base_result(
    contract: Mapping[str, Any], variant_id: str, runtime_seconds: float
) -> dict[str, Any]:
    return {
        "schema_version": "cadengbench-p.l3-response-pair-result.v1",
        "item_id": contract["item_id"],
        "variant_id": variant_id,
        "contract_sha256": contract["contract_sha256"],
        "source_intent_claimed": False,
        "scoreable": False,
        "stages": {
            "rebuild": "not_reached",
            "selectors": "not_reached",
            "mesh": "not_reached",
            "solve": "not_reached",
            "numerical_validity": "not_reached",
            "response_fidelity": "not_reached",
            "engineering_requirements": "not_reached",
        },
        "final_verdict": "evaluator_error",
        "response_fidelity": {"status": "not_reached"},
        "engineering_requirements": {"status": "not_reached"},
        "margins": {},
        "runtime_seconds": max(float(runtime_seconds), 0.0),
        "evaluator": evaluator_identity(),
    }


def skipped_pair_result(
    contract: Mapping[str, Any], *, variant_id: str, upstream_status: str
) -> dict[str, Any]:
    result = _base_result(contract, variant_id, 0.0)
    result["final_verdict"] = "skipped_upstream"
    result["failure"] = {
        "stage": "upstream",
        "owner": "model",
        "upstream_status": upstream_status,
    }
    validate_document("pair_result", result)
    return result


def failed_solver_pair_result(
    contract: Mapping[str, Any],
    *,
    variant_id: str,
    result_evidence: Mapping[str, Any],
    role: str,
    phase: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    result = _base_result(contract, variant_id, runtime_seconds)
    failure = result_evidence.get("failure")
    failure = dict(failure) if isinstance(failure, Mapping) else {}
    stage = str(failure.get("stage") or "internal")
    owner = str(failure.get("owner") or "evaluator")
    if role == "reference":
        verdict = (
            "reference_not_scoreable" if owner != "evaluator" else "evaluator_error"
        )
    elif owner == "evaluator":
        verdict = "evaluator_error"
    else:
        verdict = {
            "geometry": "rebuild_fail",
            "selectors": "selector_fail",
            "mesh": "mesh_fail",
            "deck": "solver_fail",
            "solver": "solver_fail",
            "result_sanity": "numerical_validity_fail",
            "requirements": "numerical_validity_fail",
        }.get(stage, "evaluator_error")
    result["final_verdict"] = verdict
    result["scoreable"] = role == "candidate" and owner != "evaluator"
    result["stages"] = {
        "rebuild": _stage_status(result_evidence, "geometry"),
        "selectors": _stage_status(result_evidence, "selectors"),
        "mesh": _stage_status(result_evidence, "mesh"),
        "solve": _stage_status(result_evidence, "solver"),
        "numerical_validity": (
            "failed" if verdict == "numerical_validity_fail" else "not_reached"
        ),
        "response_fidelity": "not_reached",
        "engineering_requirements": "not_reached",
    }
    result["failure"] = {
        **failure,
        "role": role,
        "phase": phase,
    }
    validate_document("pair_result", result)
    return result


def score_response_pair(
    contract: Mapping[str, Any],
    *,
    reference_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    variant_id: str,
    runtime_seconds: float,
    nominal_reference: bool,
) -> dict[str, Any]:
    """Score two complete, fixed-force CalculiX results."""

    if contract["evaluator"]["build_hash"] != evaluator_identity()["build_hash"]:
        raise ResponseScoringError(
            "response evaluator build does not match the contract"
        )
    for value in (reference_result, candidate_result):
        _require_solver_identity(contract, value)
        if not _is_complete_solve(value):
            raise ResponseScoringError(
                "matched result did not complete the solver cascade"
            )
    try:
        reference = extract_response_metrics(reference_result)
        candidate = extract_response_metrics(candidate_result)
    except CalibrationError as exc:
        raise ResponseScoringError(str(exc)) from exc

    epsilon = float(contract["numerical"]["ratio_epsilon"])
    target_force = [
        float(value) for value in contract["loading"]["target_resultant_force_n"]
    ]
    reference_force = [
        float(value) for value in reference_result["metrics"]["applied_force_n"]
    ]
    candidate_force = [
        float(value) for value in candidate_result["metrics"]["applied_force_n"]
    ]
    reference_force_error = _vector_relative_error(
        reference_force, target_force, epsilon
    )
    candidate_force_error = _vector_relative_error(
        candidate_force, target_force, epsilon
    )
    numerical_limit = float(
        contract["numerical"]["maximum_force_control_relative_error"]
    )
    equilibrium_limit = float(
        contract["numerical"]["maximum_equilibrium_relative_error"]
    )
    numerical_checks: dict[str, Any] = {
        "reference_force_control": {
            "observed": reference_force_error,
            "limit": numerical_limit,
            "passed": reference_force_error <= numerical_limit,
        },
        "candidate_force_control": {
            "observed": candidate_force_error,
            "limit": numerical_limit,
            "passed": candidate_force_error <= numerical_limit,
        },
        "reference_equilibrium": {
            "observed": reference["force_equilibrium_relative_error"],
            "limit": equilibrium_limit,
            "passed": reference["force_equilibrium_relative_error"]
            <= equilibrium_limit,
        },
        "candidate_equilibrium": {
            "observed": candidate["force_equilibrium_relative_error"],
            "limit": equilibrium_limit,
            "passed": candidate["force_equilibrium_relative_error"]
            <= equilibrium_limit,
        },
    }
    if nominal_reference:
        predicted = contract["reference_base"]["predicted_scaled_metrics"]
        linearity_limit = float(
            contract["numerical"]["maximum_linearity_relative_error"]
        )
        for metric in RESPONSE_METRICS:
            observed_error = _relative_error(
                reference[metric], float(predicted[metric]), epsilon
            )
            numerical_checks[f"reference_linearity:{metric}"] = {
                "observed": observed_error,
                "limit": linearity_limit,
                "passed": observed_error <= linearity_limit,
            }
    numerical_passed = all(bool(check["passed"]) for check in numerical_checks.values())

    fidelity_metrics: dict[str, Any] = {}
    component_scores = []
    for metric in contract["fidelity"]["metrics"]:
        reference_value = reference[metric]
        candidate_value = candidate[metric]
        ratio = candidate_value / max(reference_value, epsilon)
        error = _symmetric_log_error(candidate_value, reference_value, epsilon)
        limit = float(
            contract["fidelity"]["effective_maximum_symmetric_log_error"][metric]
        )
        component_score = math.exp(-error / limit)
        component_scores.append(component_score)
        fidelity_metrics[metric] = {
            "reference": reference_value,
            "candidate": candidate_value,
            "candidate_reference_ratio": ratio,
            "symmetric_log_error": error,
            "limit": limit,
            "margin": limit - error,
            "component_score": component_score,
            "passed": error <= limit,
        }
    fidelity_score = math.exp(
        sum(math.log(max(value, 1e-300)) for value in component_scores)
        / len(component_scores)
    )
    fidelity_passed = all(value["passed"] for value in fidelity_metrics.values())

    limits = contract["engineering_requirements"]
    stress_limit = float(limits["maximum_regularized_von_mises_p95_mpa"])
    displacement_limit = float(limits["maximum_normalized_displacement"])
    displacement_ratio = candidate["normalized_max_displacement"] / max(
        reference["normalized_max_displacement"], epsilon
    )
    compliance_ratio = candidate["normalized_compliance"] / max(
        reference["normalized_compliance"], epsilon
    )
    requirement_checks = {
        "regularized_von_mises_allowable": {
            "observed": candidate["regularized_von_mises_p95_mpa"],
            "limit": stress_limit,
            "margin": stress_limit - candidate["regularized_von_mises_p95_mpa"],
            "passed": candidate["regularized_von_mises_p95_mpa"] <= stress_limit,
        },
        "normalized_displacement_limit": {
            "observed": candidate["normalized_max_displacement"],
            "limit": displacement_limit,
            "margin": displacement_limit - candidate["normalized_max_displacement"],
            "passed": candidate["normalized_max_displacement"] <= displacement_limit,
        },
        "candidate_reference_displacement_ratio": {
            "observed": displacement_ratio,
            "limit": float(limits["maximum_candidate_reference_displacement_ratio"]),
            "margin": float(limits["maximum_candidate_reference_displacement_ratio"])
            - displacement_ratio,
            "passed": displacement_ratio
            <= float(limits["maximum_candidate_reference_displacement_ratio"]),
        },
        "candidate_reference_compliance_ratio": {
            "observed": compliance_ratio,
            "limit": float(limits["maximum_candidate_reference_compliance_ratio"]),
            "margin": float(limits["maximum_candidate_reference_compliance_ratio"])
            - compliance_ratio,
            "passed": compliance_ratio
            <= float(limits["maximum_candidate_reference_compliance_ratio"]),
        },
    }
    requirements_passed = all(check["passed"] for check in requirement_checks.values())

    reference_requirements_passed = (
        reference["regularized_von_mises_p95_mpa"] <= stress_limit
        and reference["normalized_max_displacement"] <= displacement_limit
    )
    if not numerical_passed:
        verdict = "numerical_validity_fail"
    elif not reference_requirements_passed:
        verdict = "reference_not_scoreable"
    elif not requirements_passed:
        verdict = "requirement_fail"
    elif not fidelity_passed:
        verdict = "response_fidelity_fail"
    else:
        verdict = "pass"

    result = _base_result(contract, variant_id, runtime_seconds)
    result.update(
        {
            "scoreable": verdict != "reference_not_scoreable",
            "stages": {
                "rebuild": "passed",
                "selectors": "passed",
                "mesh": "passed",
                "solve": "passed",
                "numerical_validity": "passed" if numerical_passed else "failed",
                "response_fidelity": "passed" if fidelity_passed else "failed",
                "engineering_requirements": (
                    "passed" if requirements_passed else "failed"
                ),
            },
            "final_verdict": verdict,
            "reference": {
                "metrics": _metric_payload(reference),
                "applied_force_n": reference_force,
                "requirements_passed": reference_requirements_passed,
            },
            "candidate": {
                "metrics": _metric_payload(candidate),
                "applied_force_n": candidate_force,
            },
            "numerical_validity": {
                "status": "passed" if numerical_passed else "failed",
                "checks": numerical_checks,
            },
            "response_fidelity": {
                "status": "passed" if fidelity_passed else "failed",
                "aggregate_score": fidelity_score,
                "metrics": fidelity_metrics,
            },
            "engineering_requirements": {
                "status": "passed" if requirements_passed else "failed",
                "checks": requirement_checks,
            },
            "margins": {
                "stress_mpa": requirement_checks["regularized_von_mises_allowable"][
                    "margin"
                ],
                "normalized_displacement": requirement_checks[
                    "normalized_displacement_limit"
                ]["margin"],
                "displacement_ratio": requirement_checks[
                    "candidate_reference_displacement_ratio"
                ]["margin"],
                "compliance_ratio": requirement_checks[
                    "candidate_reference_compliance_ratio"
                ]["margin"],
                "response_fidelity_score": fidelity_score,
            },
        }
    )
    validate_document("pair_result", result)
    return result
