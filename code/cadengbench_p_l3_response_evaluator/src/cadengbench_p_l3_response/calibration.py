"""Deterministic nominal-reference load and threshold calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from cadengbench_p_l3_response.io import sha256_json
from cadengbench_p_l3_response.provenance import evaluator_identity
from cadengbench_p_l3_response.schemas import validate_document

RESPONSE_METRICS = (
    "regularized_von_mises_p95_mpa",
    "normalized_max_displacement",
    "normalized_compliance",
)


class CalibrationError(ValueError):
    """The source evidence cannot support a response contract."""


def vector_norm(vector: list[float] | tuple[float, ...]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def extract_response_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise CalibrationError("solver result has no metrics")
    stress = metrics.get("regularized_von_mises_mpa")
    if not isinstance(stress, Mapping):
        raise CalibrationError("solver result has no regularized stress")
    values = {
        "regularized_von_mises_p95_mpa": float(stress["p95"]),
        "normalized_max_displacement": float(metrics["normalized_max_displacement"]),
        "normalized_compliance": float(metrics["normalized_compliance"]),
        "force_equilibrium_relative_error": float(
            metrics["force_equilibrium_relative_error"]
        ),
    }
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise CalibrationError("solver metrics must be finite and nonnegative")
    return values


def extract_loaded_area(result: Mapping[str, Any]) -> float:
    stages = result.get("stages")
    mesh = stages.get("mesh") if isinstance(stages, Mapping) else None
    if not isinstance(mesh, Mapping) or mesh.get("status") != "passed":
        raise CalibrationError("solver result has no passing mesh evidence")
    area = float(mesh["loaded_area_mm2"])
    if not math.isfinite(area) or area <= 0:
        raise CalibrationError("loaded area must be finite and positive")
    return area


def _mesh_uncertainty(
    mesh_study: Mapping[str, Any] | None,
) -> dict[str, float]:
    output = dict.fromkeys(RESPONSE_METRICS, 0.0)
    if mesh_study is None:
        return output
    comparisons = mesh_study.get("comparisons_to_fine")
    production = (
        comparisons.get("production") if isinstance(comparisons, Mapping) else None
    )
    if not isinstance(production, Mapping):
        raise CalibrationError("mesh study lacks production-to-fine comparisons")
    for metric in RESPONSE_METRICS:
        comparison = production.get(metric)
        if not isinstance(comparison, Mapping):
            raise CalibrationError(f"mesh study lacks metric: {metric}")
        observed = max(abs(float(comparison["observed"])), 1e-30)
        fine = max(abs(float(comparison["fine"])), 1e-30)
        output[metric] = abs(math.log(observed / fine))
    return output


def compile_response_contract(
    *,
    profile: Mapping[str, Any],
    profile_sha256: str,
    card: Mapping[str, Any],
    card_sha256: str,
    nominal_reference_step_sha256: str,
    reference_result: Mapping[str, Any],
    reference_result_sha256: str,
    mesh_study: Mapping[str, Any] | None = None,
    mesh_study_sha256: str | None = None,
) -> dict[str, Any]:
    """Compile an immutable item-level force and response contract."""

    validate_document("profile", dict(profile))
    if card.get("status") != "solver_qualified":
        raise CalibrationError(
            "response calibration requires a solver-qualified L3 card"
        )
    if card.get("source_intent_claimed") is not False:
        raise CalibrationError("source-intent claims are prohibited")
    if reference_result.get("final_status") != "probe_pass":
        raise CalibrationError("nominal reference preflight did not pass")
    material = profile["material"]
    card_material = card["material"]
    if not math.isclose(
        float(material["youngs_modulus_mpa"]),
        float(card_material["youngs_modulus_mpa"]),
        rel_tol=0,
        abs_tol=1e-9,
    ) or not math.isclose(
        float(material["poisson_ratio"]),
        float(card_material["poisson_ratio"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise CalibrationError("profile material does not match the frozen L3 card")

    metrics = extract_response_metrics(reference_result)
    area = extract_loaded_area(reference_result)
    traction = [float(value) for value in card["load"]["traction_vector_mpa"]]
    traction_norm = vector_norm(traction)
    if traction_norm <= 0:
        raise CalibrationError("base traction must be nonzero")
    base_force = [value * area for value in traction]
    allowable = float(material["yield_strength_mpa"]) / float(material["safety_factor"])
    calibration = profile["calibration"]
    candidates = {
        "stress_utilization": float(calibration["target_reference_stress_utilization"])
        * allowable
        / max(metrics["regularized_von_mises_p95_mpa"], 1e-30),
        "normalized_displacement": float(
            calibration["target_reference_normalized_displacement"]
        )
        / max(metrics["normalized_max_displacement"], 1e-30),
        "maximum_traction": float(calibration["maximum_traction_mpa"]) / traction_norm,
        "maximum_load_scale": float(calibration["maximum_load_scale"]),
    }
    controlling_constraint, load_scale = min(
        candidates.items(), key=lambda item: item[1]
    )
    if not math.isfinite(load_scale) or load_scale < float(
        calibration["minimum_load_scale"]
    ):
        raise CalibrationError(
            "reference requires a load scale below the profile minimum"
        )
    target_force = [value * load_scale for value in base_force]
    scaled_traction = [value * load_scale for value in traction]
    predicted = {
        "regularized_von_mises_p95_mpa": metrics["regularized_von_mises_p95_mpa"]
        * load_scale,
        "normalized_max_displacement": metrics["normalized_max_displacement"]
        * load_scale,
        "normalized_compliance": metrics["normalized_compliance"],
        "force_equilibrium_relative_error": metrics["force_equilibrium_relative_error"],
    }
    uncertainty = _mesh_uncertainty(mesh_study)
    fidelity = profile["fidelity"]
    base_limits = {
        metric: float(fidelity["maximum_symmetric_log_error"][metric])
        for metric in RESPONSE_METRICS
    }
    multiplier = float(fidelity["mesh_uncertainty_multiplier"])
    effective_limits = {
        metric: max(base_limits[metric], uncertainty[metric] * multiplier)
        for metric in RESPONSE_METRICS
    }
    requirements = profile["engineering_requirements"]
    body: dict[str, Any] = {
        "schema_version": "cadengbench-p.l3-response-contract.v1",
        "contract_id": f"{card['item_id']}:l3-response:v1",
        "item_id": card["item_id"],
        "status": (
            "calibrated"
            if profile["status"] == "calibrated"
            else "development_calibrated"
        ),
        "source_intent_claimed": False,
        "threshold_authority": (
            "benchmark_assigned_reference_relative_calibrated_v1"
            if profile["status"] == "calibrated"
            else "benchmark_assigned_reference_relative_development"
        ),
        "profile": {
            "profile_id": profile["profile_id"],
            "profile_sha256": profile_sha256,
            "status": profile["status"],
        },
        "bindings": {
            "base_card_sha256": card_sha256,
            "nominal_reference_step_sha256": nominal_reference_step_sha256,
            "reference_base_result_sha256": reference_result_sha256,
            "mesh_study_sha256": mesh_study_sha256,
            "l3_evaluator_build_hash": reference_result["evaluator_build_hash"],
            "l3_runtime_lock_sha256": reference_result["runtime_lock_sha256"],
        },
        "scenario": card["scenario_family"],
        "material": {
            **dict(material),
            "allowable_stress_mpa": allowable,
        },
        "loading": {
            "control": "fixed_resultant_force",
            "base_traction_vector_mpa": traction,
            "base_loaded_area_mm2": area,
            "base_resultant_force_n": base_force,
            "load_scale": load_scale,
            "controlling_constraint": controlling_constraint,
            "target_resultant_force_n": target_force,
            "predicted_reference_traction_vector_mpa": scaled_traction,
        },
        "reference_base": {
            "metrics": metrics,
            "predicted_scaled_metrics": predicted,
        },
        "fidelity": {
            "metrics": list(fidelity["metrics"]),
            "base_maximum_symmetric_log_error": base_limits,
            "mesh_symmetric_log_uncertainty": uncertainty,
            "effective_maximum_symmetric_log_error": effective_limits,
            "component_score": fidelity["component_score"],
            "aggregate": fidelity["aggregate"],
        },
        "engineering_requirements": {
            "maximum_stress_utilization": float(
                requirements["maximum_stress_utilization"]
            ),
            "maximum_regularized_von_mises_p95_mpa": allowable
            * float(requirements["maximum_stress_utilization"]),
            "maximum_normalized_displacement": float(
                requirements["maximum_normalized_displacement"]
            ),
            "maximum_candidate_reference_displacement_ratio": float(
                requirements["maximum_candidate_reference_displacement_ratio"]
            ),
            "maximum_candidate_reference_compliance_ratio": float(
                requirements["maximum_candidate_reference_compliance_ratio"]
            ),
        },
        "numerical": dict(profile["numerical"]),
        "evaluator": evaluator_identity(),
    }
    contract = {**body, "contract_sha256": sha256_json(body)}
    validate_document("contract", contract)
    return contract
