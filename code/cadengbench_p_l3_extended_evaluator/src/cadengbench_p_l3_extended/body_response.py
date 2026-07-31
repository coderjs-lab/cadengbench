"""Matched reference/candidate scoring for body-acceleration probes."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from cadengbench_p_l3_extended.body_solver import run_body_card
from cadengbench_p_l3_extended.contracts import validate_body_card
from cadengbench_p_l3_extended.io import (
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from cadengbench_p_l3_extended.provenance import evaluator_build_hash

BODY_METRICS = (
    "regularized_von_mises_p95_mpa",
    "normalized_max_displacement",
    "body_force_stress_concentration_p95",
)


class BodyResponseError(ValueError):
    """A body-response contract or solver result is invalid."""


def _schema() -> dict[str, Any]:
    source = (
        Path(__file__).resolve().parents[2]
        / "schemas/body_response_contract.v1.schema.json"
    )
    if source.is_file():
        return read_json(source)
    packaged = resources.files("cadengbench_p_l3_extended").joinpath(
        "_schemas/body_response_contract.v1.schema.json"
    )
    return json.loads(packaged.read_text(encoding="utf-8"))


def load_body_response_profile(path: Path) -> dict[str, Any]:
    profile = read_json(path)
    if profile.get("schema_version") != "cadengbench-p.l3-body-response-profile.v1":
        raise BodyResponseError("invalid body-response profile schema")
    if profile.get("status") != "calibrated":
        raise BodyResponseError("body-response profile is not calibrated")
    if profile.get("source_intent_claimed") is not False:
        raise BodyResponseError("source-intent claims are prohibited")
    if tuple(profile["fidelity"]["metrics"]) != BODY_METRICS:
        raise BodyResponseError("body-response metric roster is not frozen")
    return profile


def load_body_response_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    errors = sorted(
        Draft202012Validator(_schema()).iter_errors(contract),
        key=lambda error: list(error.path),
    )
    if errors:
        raise BodyResponseError(errors[0].message)
    body = dict(contract)
    expected = body.pop("contract_sha256")
    if sha256_json(body) != expected:
        raise BodyResponseError("body-response contract self-hash changed")
    return contract


def _metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise BodyResponseError("body solver result has no metrics")
    stress = metrics.get("regularized_von_mises_mpa")
    if not isinstance(stress, Mapping):
        raise BodyResponseError("body solver result has no regularized stress")
    values = {
        "regularized_von_mises_p95_mpa": float(stress["p95"]),
        "normalized_max_displacement": float(metrics["normalized_max_displacement"]),
        "body_force_stress_concentration_p95": float(
            metrics["body_force_stress_concentration_p95"]
        ),
        "force_equilibrium_relative_error": float(
            metrics["force_equilibrium_relative_error"]
        ),
        "applied_force_n": math.sqrt(
            sum(float(value) ** 2 for value in metrics["applied_force_n"])
        ),
    }
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise BodyResponseError("body-response metrics must be finite and nonnegative")
    return values


def compile_body_response_contract(
    *,
    profile_path: Path,
    card_path: Path,
    reference_step: Path,
    reference_result_path: Path,
) -> dict[str, Any]:
    profile = load_body_response_profile(profile_path)
    card = read_json(card_path)
    validate_body_card(card)
    if card.get("status") != "solver_qualified":
        raise BodyResponseError("body-response calibration requires a promoted card")
    if card.get("source_intent_claimed") is not False:
        raise BodyResponseError("source-intent claims are prohibited")
    if sha256_file(reference_step) != card["qualification"]["reference_step_sha256"]:
        raise BodyResponseError("nominal reference does not match the promoted card")
    result = read_json(reference_result_path)
    if result.get("final_status") != "probe_pass":
        raise BodyResponseError("nominal body-response preflight did not pass")
    material = profile["material"]
    for key in (
        "youngs_modulus_mpa",
        "poisson_ratio",
        "density_tonne_per_mm3",
    ):
        if not math.isclose(
            float(material[key]), float(card["material"][key]), rel_tol=0, abs_tol=1e-15
        ):
            raise BodyResponseError(f"profile material mismatch: {key}")
    reference_metrics = _metrics(result)
    allowable = float(material["yield_strength_mpa"]) / float(
        material["safety_factor"]
    )
    requirements = profile["engineering_requirements"]
    body: dict[str, Any] = {
        "schema_version": "cadengbench-p.l3-body-response-contract.v1",
        "contract_id": f"{card['item_id']}:l3-body-response:v1",
        "item_id": card["item_id"],
        "status": "calibrated",
        "source_intent_claimed": False,
        "threshold_authority": (
            "benchmark_assigned_reference_relative_body_acceleration_v1"
        ),
        "profile": {
            "profile_id": profile["profile_id"],
            "profile_sha256": sha256_file(profile_path),
            "status": profile["status"],
        },
        "bindings": {
            "base_card_sha256": sha256_file(card_path),
            "nominal_reference_step_sha256": sha256_file(reference_step),
            "reference_base_result_sha256": sha256_file(reference_result_path),
            "body_evaluator_build_hash": result["evaluator_build_hash"],
            "l3_runtime_lock_sha256": result["runtime_lock_sha256"],
        },
        "scenario": card["scenario_family"],
        "material": {
            **dict(material),
            "allowable_stress_mpa": allowable,
        },
        "loading": {
            "control": "fixed_acceleration",
            "acceleration_vector_mm_s2": card["load"][
                "acceleration_vector_mm_s2"
            ],
            "density_tonne_per_mm3": card["material"][
                "density_tonne_per_mm3"
            ],
            "force_scales_with_candidate_mass": True,
        },
        "reference_base": {"metrics": reference_metrics},
        "fidelity": dict(profile["fidelity"]),
        "engineering_requirements": {
            **dict(requirements),
            "maximum_regularized_von_mises_p95_mpa": allowable
            * float(requirements["maximum_stress_utilization"]),
        },
        "numerical": dict(profile["numerical"]),
        "evaluator": {
            "name": "cadengbench-p-l3-extended",
            "version": "0.2.0",
            "build_hash": evaluator_build_hash(),
        },
    }
    contract = {**body, "contract_sha256": sha256_json(body)}
    output_errors = list(Draft202012Validator(_schema()).iter_errors(contract))
    if output_errors:
        raise BodyResponseError(output_errors[0].message)
    return contract


def calibrate_body_response(
    *,
    profile_path: Path,
    card_path: Path,
    reference_step: Path,
    output_dir: Path,
    ccx_path: Path,
    timeout_seconds: int = 300,
    force: bool = False,
) -> dict[str, Any]:
    contract_path = output_dir / "response_contract.json"
    if contract_path.is_file() and not force:
        return load_body_response_contract(contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    card = read_json(card_path)
    preflight_dir = output_dir / "nominal_reference_preflight"
    result = run_body_card(
        card,
        step_path=reference_step,
        output_dir=preflight_dir,
        ccx_path=ccx_path,
        timeout_seconds=timeout_seconds,
        render=False,
    )
    if result.get("final_status") != "probe_pass":
        raise BodyResponseError("nominal body-response preflight did not pass")
    contract = compile_body_response_contract(
        profile_path=profile_path,
        card_path=card_path,
        reference_step=reference_step,
        reference_result_path=preflight_dir / "result.json",
    )
    write_json(contract_path, contract)
    return contract


def _stage(result: Mapping[str, Any], name: str) -> str:
    stages = result.get("stages")
    stage = stages.get(name) if isinstance(stages, Mapping) else None
    return str(stage.get("status", "not_reached")) if isinstance(stage, Mapping) else (
        "not_reached"
    )


def _failed_result(
    contract: Mapping[str, Any],
    *,
    variant_id: str,
    evidence: Mapping[str, Any],
    role: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    failure = dict(evidence.get("failure") or {})
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
            "requirements": "requirement_fail",
        }.get(stage, "evaluator_error")
    return {
        "schema_version": "cadengbench-p.l3-response-pair-result.v1",
        "item_id": contract["item_id"],
        "variant_id": variant_id,
        "contract_sha256": contract["contract_sha256"],
        "source_intent_claimed": False,
        "scoreable": role == "candidate" and owner != "evaluator",
        "stages": {
            "rebuild": _stage(evidence, "geometry"),
            "selectors": _stage(evidence, "selectors"),
            "mesh": _stage(evidence, "mesh"),
            "solve": _stage(evidence, "solver"),
            "numerical_validity": "not_reached",
            "response_fidelity": "not_reached",
            "engineering_requirements": "not_reached",
        },
        "final_verdict": verdict,
        "response_fidelity": {"status": "not_reached"},
        "engineering_requirements": {"status": "not_reached"},
        "margins": {},
        "failure": {**failure, "role": role},
        "runtime_seconds": runtime_seconds,
        "evaluator": contract["evaluator"],
    }


def _symmetric_log_error(candidate: float, reference: float, epsilon: float) -> float:
    return abs(math.log(max(candidate, epsilon) / max(reference, epsilon)))


def _write_result(output_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    write_json(output_dir / "pair_result.json", result)
    with (output_dir / "row.csv").open("w", encoding="utf-8", newline="") as stream:
        row = {
            "item_id": result["item_id"],
            "variant_id": result["variant_id"],
            "final_verdict": result["final_verdict"],
            "response_fidelity_score": result["margins"].get(
                "response_fidelity_score"
            ),
            "runtime_seconds": result["runtime_seconds"],
        }
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return result


def evaluate_body_response_pair(
    *,
    contract_path: Path,
    card_path: Path,
    reference_step: Path,
    candidate_step: Path,
    variant_id: str,
    output_dir: Path,
    ccx_path: Path,
    upstream_status: str = "passed",
    timeout_seconds: int = 300,
    render: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    result_path = output_dir / "pair_result.json"
    if result_path.is_file() and not force:
        return read_json(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    contract = load_body_response_contract(contract_path)
    if sha256_file(card_path) != contract["bindings"]["base_card_sha256"]:
        raise BodyResponseError("body card does not match the response contract")
    if (
        variant_id == "nominal"
        and sha256_file(reference_step)
        != contract["bindings"]["nominal_reference_step_sha256"]
    ):
        raise BodyResponseError(
            "nominal reference does not match the response contract"
        )
    if upstream_status != "passed":
        return _write_result(
            output_dir,
            {
                "schema_version": "cadengbench-p.l3-response-pair-result.v1",
                "item_id": contract["item_id"],
                "variant_id": variant_id,
                "contract_sha256": contract["contract_sha256"],
                "source_intent_claimed": False,
                "scoreable": False,
                "stages": dict.fromkeys(
                    (
                        "rebuild",
                        "selectors",
                        "mesh",
                        "solve",
                        "numerical_validity",
                        "response_fidelity",
                        "engineering_requirements",
                    ),
                    "not_reached",
                ),
                "final_verdict": "skipped_upstream",
                "response_fidelity": {"status": "not_reached"},
                "engineering_requirements": {"status": "not_reached"},
                "margins": {},
                "runtime_seconds": 0.0,
                "evaluator": contract["evaluator"],
            },
        )
    card = read_json(card_path)
    solver_results: dict[str, dict[str, Any]] = {}
    for role, step in (("reference", reference_step), ("candidate", candidate_step)):
        solver_results[role] = run_body_card(
            card,
            step_path=step,
            output_dir=output_dir / role,
            ccx_path=ccx_path,
            timeout_seconds=timeout_seconds,
            render=render and role == "candidate",
        )
        if solver_results[role].get("final_status") not in {
            "probe_pass",
            "requirement_fail",
        }:
            return _write_result(
                output_dir,
                _failed_result(
                    contract,
                    variant_id=variant_id,
                    evidence=solver_results[role],
                    role=role,
                    runtime_seconds=time.monotonic() - started,
                ),
            )
        if (
            solver_results[role].get("evaluator_build_hash")
            != contract["bindings"]["body_evaluator_build_hash"]
            or solver_results[role].get("runtime_lock_sha256")
            != contract["bindings"]["l3_runtime_lock_sha256"]
        ):
            raise BodyResponseError("body solver identity differs from the contract")

    reference = _metrics(solver_results["reference"])
    candidate = _metrics(solver_results["candidate"])
    epsilon = float(contract["numerical"]["ratio_epsilon"])
    equilibrium_limit = float(
        contract["numerical"]["maximum_equilibrium_relative_error"]
    )
    numerical_checks = {
        role: {
            "observed": values["force_equilibrium_relative_error"],
            "limit": equilibrium_limit,
            "passed": values["force_equilibrium_relative_error"] <= equilibrium_limit,
        }
        for role, values in (("reference", reference), ("candidate", candidate))
    }
    numerical_passed = all(check["passed"] for check in numerical_checks.values())

    fidelity_metrics: dict[str, Any] = {}
    component_scores = []
    for metric in BODY_METRICS:
        error = _symmetric_log_error(candidate[metric], reference[metric], epsilon)
        limit = float(
            contract["fidelity"]["maximum_symmetric_log_error"][metric]
        )
        component = math.exp(-error / limit)
        component_scores.append(component)
        fidelity_metrics[metric] = {
            "reference": reference[metric],
            "candidate": candidate[metric],
            "candidate_reference_ratio": candidate[metric]
            / max(reference[metric], epsilon),
            "symmetric_log_error": error,
            "limit": limit,
            "margin": limit - error,
            "component_score": component,
            "passed": error <= limit,
        }
    fidelity_score = math.exp(
        sum(math.log(max(value, 1e-300)) for value in component_scores)
        / len(component_scores)
    )
    fidelity_passed = all(value["passed"] for value in fidelity_metrics.values())

    requirements = contract["engineering_requirements"]
    stress_ratio = candidate["regularized_von_mises_p95_mpa"] / max(
        reference["regularized_von_mises_p95_mpa"], epsilon
    )
    displacement_ratio = candidate["normalized_max_displacement"] / max(
        reference["normalized_max_displacement"], epsilon
    )
    concentration_ratio = candidate["body_force_stress_concentration_p95"] / max(
        reference["body_force_stress_concentration_p95"], epsilon
    )
    requirement_checks = {
        "regularized_von_mises_allowable": {
            "observed": candidate["regularized_von_mises_p95_mpa"],
            "limit": requirements["maximum_regularized_von_mises_p95_mpa"],
        },
        "normalized_displacement_limit": {
            "observed": candidate["normalized_max_displacement"],
            "limit": requirements["maximum_normalized_displacement"],
        },
        "candidate_reference_stress_ratio": {
            "observed": stress_ratio,
            "limit": requirements["maximum_candidate_reference_stress_ratio"],
        },
        "candidate_reference_displacement_ratio": {
            "observed": displacement_ratio,
            "limit": requirements[
                "maximum_candidate_reference_displacement_ratio"
            ],
        },
        "candidate_reference_stress_concentration_ratio": {
            "observed": concentration_ratio,
            "limit": requirements[
                "maximum_candidate_reference_stress_concentration_ratio"
            ],
        },
    }
    for check in requirement_checks.values():
        check["margin"] = float(check["limit"]) - float(check["observed"])
        check["passed"] = check["observed"] <= check["limit"]
    requirements_passed = all(check["passed"] for check in requirement_checks.values())
    verdict = (
        "numerical_validity_fail"
        if not numerical_passed
        else "requirement_fail"
        if not requirements_passed
        else "response_fidelity_fail"
        if not fidelity_passed
        else "pass"
    )
    result = {
        "schema_version": "cadengbench-p.l3-response-pair-result.v1",
        "item_id": contract["item_id"],
        "variant_id": variant_id,
        "contract_sha256": contract["contract_sha256"],
        "source_intent_claimed": False,
        "scoreable": True,
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
        "numerical_validity": {
            "status": "passed" if numerical_passed else "failed",
            "checks": numerical_checks,
        },
        "response_fidelity": {
            "status": "passed" if fidelity_passed else "failed",
            "metrics": fidelity_metrics,
            "score": fidelity_score,
        },
        "engineering_requirements": {
            "status": "passed" if requirements_passed else "failed",
            "checks": requirement_checks,
        },
        "reference": {
            "step_sha256": sha256_file(reference_step),
            "metrics": reference,
        },
        "candidate": {
            "step_sha256": sha256_file(candidate_step),
            "metrics": candidate,
        },
        "load_scaling": {
            "control": "fixed_acceleration",
            "candidate_reference_resultant_force_ratio": candidate[
                "applied_force_n"
            ]
            / max(reference["applied_force_n"], epsilon),
        },
        "margins": {
            "stress_mpa": requirement_checks[
                "regularized_von_mises_allowable"
            ]["margin"],
            "normalized_displacement": requirement_checks[
                "normalized_displacement_limit"
            ]["margin"],
            "stress_ratio": requirement_checks[
                "candidate_reference_stress_ratio"
            ]["margin"],
            "displacement_ratio": requirement_checks[
                "candidate_reference_displacement_ratio"
            ]["margin"],
            "body_stress_concentration_ratio": requirement_checks[
                "candidate_reference_stress_concentration_ratio"
            ]["margin"],
            "response_fidelity_score": fidelity_score,
        },
        "runtime_seconds": time.monotonic() - started,
        "evaluator": contract["evaluator"],
    }
    return _write_result(output_dir, result)
