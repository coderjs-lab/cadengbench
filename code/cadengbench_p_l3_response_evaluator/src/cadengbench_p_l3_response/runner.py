"""Transactional calibration, matched-pair, and perturbation-family runners."""

from __future__ import annotations

import csv
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cadengbench_p_l3_response.calibration import (
    CalibrationError,
    compile_response_contract,
    extract_loaded_area,
)
from cadengbench_p_l3_response.io import (
    read_json,
    sha256_file,
    sha256_json,
    write_evidence_commit,
    write_json,
)
from cadengbench_p_l3_response.schemas import (
    load_contract,
    load_family_manifest,
    load_profile,
    validate_document,
)
from cadengbench_p_l3_response.scoring import (
    ResponseScoringError,
    failed_solver_pair_result,
    score_response_pair,
    skipped_pair_result,
)
from cadengbench_p_l3_response.solver import (
    SolverAdapterError,
    force_controlled_card,
    prepare_output,
    run_solver_job,
)

ROW_FIELDS = (
    "item_id",
    "variant_id",
    "rebuild",
    "selectors",
    "mesh",
    "solve",
    "numerical_validity",
    "response_fidelity",
    "engineering_requirements",
    "final_verdict",
    "stress_margin_mpa",
    "normalized_displacement_margin",
    "displacement_ratio_margin",
    "compliance_ratio_margin",
    "response_fidelity_score",
    "runtime_seconds",
)


class ResponseRunnerError(RuntimeError):
    """A response-evaluator run cannot be completed reproducibly."""


def _complete_solver_result(result: Mapping[str, Any]) -> bool:
    stages = result.get("stages")
    return (
        isinstance(stages, Mapping)
        and all(
            isinstance(stages.get(stage), Mapping)
            and stages[stage].get("status") == "passed"
            for stage in ("geometry", "selectors", "mesh", "solver", "result_sanity")
        )
        and isinstance(result.get("metrics"), Mapping)
    )


def _result_row(result: Mapping[str, Any]) -> dict[str, Any]:
    stages = result["stages"]
    margins = result.get("margins") or {}
    return {
        "item_id": result["item_id"],
        "variant_id": result["variant_id"],
        "rebuild": stages["rebuild"],
        "selectors": stages["selectors"],
        "mesh": stages["mesh"],
        "solve": stages["solve"],
        "numerical_validity": stages["numerical_validity"],
        "response_fidelity": stages["response_fidelity"],
        "engineering_requirements": stages["engineering_requirements"],
        "final_verdict": result["final_verdict"],
        "stress_margin_mpa": margins.get("stress_mpa"),
        "normalized_displacement_margin": margins.get("normalized_displacement"),
        "displacement_ratio_margin": margins.get("displacement_ratio"),
        "compliance_ratio_margin": margins.get("compliance_ratio"),
        "response_fidelity_score": margins.get("response_fidelity_score"),
        "runtime_seconds": result["runtime_seconds"],
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calibrate_item(
    *,
    profile_path: Path,
    card_path: Path,
    reference_step: Path,
    output_dir: Path,
    ccx_path: Path,
    mesh_study_path: Path | None = None,
    timeout_seconds: int = 300,
    force: bool = False,
) -> dict[str, Any]:
    prepare_output(output_dir, force=force)
    profile = load_profile(profile_path)
    card = read_json(card_path)
    preflight_dir = output_dir / "nominal_reference_preflight"
    try:
        preflight = run_solver_job(
            card_path=card_path,
            step_path=reference_step,
            output_dir=preflight_dir,
            ccx_path=ccx_path,
            timeout_seconds=timeout_seconds,
            render=False,
        )
    except SolverAdapterError as exc:
        raise ResponseRunnerError(str(exc)) from exc
    if (
        not _complete_solver_result(preflight)
        or preflight.get("final_status") != "probe_pass"
    ):
        raise ResponseRunnerError("nominal reference preflight did not pass")
    mesh_study = read_json(mesh_study_path) if mesh_study_path else None
    try:
        contract = compile_response_contract(
            profile=profile,
            profile_sha256=sha256_file(profile_path),
            card=card,
            card_sha256=sha256_file(card_path),
            nominal_reference_step_sha256=sha256_file(reference_step),
            reference_result=preflight,
            reference_result_sha256=sha256_file(preflight_dir / "result.json"),
            mesh_study=mesh_study,
            mesh_study_sha256=(
                sha256_file(mesh_study_path) if mesh_study_path else None
            ),
        )
    except CalibrationError as exc:
        raise ResponseRunnerError(str(exc)) from exc
    write_json(output_dir / "response_contract.json", contract)
    write_evidence_commit(output_dir, output_dir / "response_contract.json")
    return contract


def _write_controlled_card(
    *,
    base_card: dict[str, Any],
    contract: Mapping[str, Any],
    area: float,
    role: str,
    variant_id: str,
    path: Path,
) -> None:
    card = force_controlled_card(
        base_card,
        target_force_n=[
            float(value) for value in contract["loading"]["target_resultant_force_n"]
        ],
        loaded_area_mm2=area,
        maximum_equilibrium_relative_error=float(
            contract["numerical"]["maximum_equilibrium_relative_error"]
        ),
        role=role,
        variant_id=variant_id,
        contract_sha256=contract["contract_sha256"],
    )
    write_json(path, card)


def evaluate_pair(
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
    prepare_output(output_dir, force=force)
    started = time.monotonic()
    contract = load_contract(contract_path)
    if sha256_file(card_path) != contract["bindings"]["base_card_sha256"]:
        raise ResponseRunnerError("base L3 card does not match the response contract")
    if (
        variant_id == "nominal"
        and sha256_file(reference_step)
        != contract["bindings"]["nominal_reference_step_sha256"]
    ):
        raise ResponseRunnerError("nominal reference STEP does not match the contract")
    if upstream_status != "passed":
        result = skipped_pair_result(
            contract, variant_id=variant_id, upstream_status=upstream_status
        )
        write_json(output_dir / "pair_result.json", result)
        _write_rows(output_dir / "row.csv", [_result_row(result)])
        write_evidence_commit(output_dir, output_dir / "pair_result.json")
        return result
    if not reference_step.is_file() or not candidate_step.is_file():
        raise ResponseRunnerError("matched reference or candidate STEP is missing")
    base_card = read_json(card_path)

    preflight_results: dict[str, dict[str, Any]] = {}
    for role, step in (("reference", reference_step), ("candidate", candidate_step)):
        try:
            preflight_results[role] = run_solver_job(
                card_path=card_path,
                step_path=step,
                output_dir=output_dir / role / "preflight",
                ccx_path=ccx_path,
                timeout_seconds=timeout_seconds,
                render=False,
            )
        except SolverAdapterError as exc:
            raise ResponseRunnerError(str(exc)) from exc
        if not _complete_solver_result(preflight_results[role]):
            result = failed_solver_pair_result(
                contract,
                variant_id=variant_id,
                result_evidence=preflight_results[role],
                role=role,
                phase="preflight",
                runtime_seconds=time.monotonic() - started,
            )
            write_json(output_dir / "pair_result.json", result)
            _write_rows(output_dir / "row.csv", [_result_row(result)])
            write_evidence_commit(output_dir, output_dir / "pair_result.json")
            return result

    card_dir = output_dir / "force_controlled_cards"
    card_dir.mkdir(parents=True)
    controlled_paths: dict[str, Path] = {}
    for role in ("reference", "candidate"):
        path = card_dir / f"{role}.json"
        _write_controlled_card(
            base_card=base_card,
            contract=contract,
            area=extract_loaded_area(preflight_results[role]),
            role=role,
            variant_id=variant_id,
            path=path,
        )
        controlled_paths[role] = path

    scaled_results: dict[str, dict[str, Any]] = {}
    for role, step in (("reference", reference_step), ("candidate", candidate_step)):
        try:
            scaled_results[role] = run_solver_job(
                card_path=controlled_paths[role],
                step_path=step,
                output_dir=output_dir / role / "force_controlled",
                ccx_path=ccx_path,
                timeout_seconds=timeout_seconds,
                render=render and role == "candidate",
            )
        except SolverAdapterError as exc:
            raise ResponseRunnerError(str(exc)) from exc
        if not _complete_solver_result(scaled_results[role]):
            result = failed_solver_pair_result(
                contract,
                variant_id=variant_id,
                result_evidence=scaled_results[role],
                role=role,
                phase="force_controlled",
                runtime_seconds=time.monotonic() - started,
            )
            write_json(output_dir / "pair_result.json", result)
            _write_rows(output_dir / "row.csv", [_result_row(result)])
            write_evidence_commit(output_dir, output_dir / "pair_result.json")
            return result
    try:
        result = score_response_pair(
            contract,
            reference_result=scaled_results["reference"],
            candidate_result=scaled_results["candidate"],
            variant_id=variant_id,
            runtime_seconds=time.monotonic() - started,
            nominal_reference=variant_id == "nominal",
        )
    except ResponseScoringError as exc:
        raise ResponseRunnerError(str(exc)) from exc
    result["reference"]["step_sha256"] = sha256_file(reference_step)
    result["candidate"]["step_sha256"] = sha256_file(candidate_step)
    write_json(output_dir / "pair_result.json", result)
    _write_rows(output_dir / "row.csv", [_result_row(result)])
    write_evidence_commit(output_dir, output_dir / "pair_result.json")
    return result


def bind_family_manifest(
    *,
    contract_path: Path,
    card_path: Path,
    variants_spec_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    specification = read_json(variants_spec_path)
    if specification.get("item_id") != contract["item_id"]:
        raise ResponseRunnerError("family specification item does not match contract")
    variants = specification.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ResponseRunnerError("family specification has no variants")
    bound = []
    seen = set()
    for value in variants:
        if not isinstance(value, Mapping):
            raise ResponseRunnerError("family variant must be an object")
        variant_id = str(value["variant_id"])
        if variant_id in seen:
            raise ResponseRunnerError("family variant IDs must be unique")
        seen.add(variant_id)
        upstream = str(value.get("upstream_status") or "not_reached")
        reference = Path(str(value.get("reference_step_path") or ""))
        candidate = Path(str(value.get("candidate_step_path") or ""))
        if upstream == "passed" and (
            not reference.is_file() or not candidate.is_file()
        ):
            raise ResponseRunnerError(
                f"passing variant lacks matched STEP files: {variant_id}"
            )
        bound.append(
            {
                "variant_id": variant_id,
                "upstream_status": upstream,
                "reference_step_path": str(reference.resolve()),
                "reference_step_sha256": sha256_file(reference)
                if reference.is_file()
                else None,
                "candidate_step_path": str(candidate.resolve()),
                "candidate_step_sha256": sha256_file(candidate)
                if candidate.is_file()
                else None,
            }
        )
    body = {
        "schema_version": "cadengbench-p.l3-response-family-manifest.v1",
        "item_id": contract["item_id"],
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "card_path": str(card_path.resolve()),
        "card_sha256": sha256_file(card_path),
        "variants": bound,
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    validate_document("family_manifest", manifest)
    write_json(output_path, manifest)
    return manifest


def run_family(
    *,
    manifest_path: Path,
    output_dir: Path,
    ccx_path: Path,
    workers: int = 1,
    timeout_seconds: int = 300,
    render: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise ResponseRunnerError("workers must be positive")
    prepare_output(output_dir, force=force)
    manifest = load_family_manifest(manifest_path)
    contract_path = Path(manifest["contract_path"])
    card_path = Path(manifest["card_path"])
    if sha256_file(contract_path) != manifest["contract_sha256"]:
        raise ResponseRunnerError("family contract binding changed")
    if sha256_file(card_path) != manifest["card_sha256"]:
        raise ResponseRunnerError("family card binding changed")
    for record in manifest["variants"]:
        if record["upstream_status"] != "passed":
            continue
        for field in ("reference", "candidate"):
            path = Path(record[f"{field}_step_path"])
            if sha256_file(path) != record[f"{field}_step_sha256"]:
                raise ResponseRunnerError(
                    f"family {field} STEP binding changed: {record['variant_id']}"
                )
    started = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_pair,
                contract_path=contract_path,
                card_path=card_path,
                reference_step=Path(record["reference_step_path"]),
                candidate_step=Path(record["candidate_step_path"]),
                variant_id=record["variant_id"],
                output_dir=output_dir / "variants" / record["variant_id"],
                ccx_path=ccx_path,
                upstream_status=record["upstream_status"],
                timeout_seconds=timeout_seconds,
                render=render,
            ): record["variant_id"]
            for record in manifest["variants"]
        }
        for future in as_completed(futures):
            variant_id = futures[future]
            try:
                results[variant_id] = future.result()
            except Exception as exc:
                contract = load_contract(contract_path)
                fallback = failed_solver_pair_result(
                    contract,
                    variant_id=variant_id,
                    result_evidence={
                        "failure": {
                            "stage": "internal",
                            "owner": "evaluator",
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    },
                    role="candidate",
                    phase="orchestration",
                    runtime_seconds=0.0,
                )
                write_json(
                    output_dir / "variants" / variant_id / "pair_result.json", fallback
                )
                results[variant_id] = fallback
    ordered = [results[key] for key in sorted(results)]
    counts = Counter(result["final_verdict"] for result in ordered)
    scoreable = [
        result
        for result in ordered
        if result["scoreable"] and result["final_verdict"] != "skipped_upstream"
    ]
    fidelity_scores = [
        float(result["response_fidelity"]["aggregate_score"])
        for result in scoreable
        if "aggregate_score" in result["response_fidelity"]
    ]
    passed = sum(result["final_verdict"] == "pass" for result in ordered)
    requirement_passed = sum(
        result["stages"]["engineering_requirements"] == "passed" for result in scoreable
    )
    summary = {
        "schema_version": "cadengbench-p.l3-response-family-result.v1",
        "item_id": manifest["item_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "variant_count": len(ordered),
        "scoreable_variant_count": len(scoreable),
        "reference_not_scoreable_count": counts.get("reference_not_scoreable", 0),
        "pass_count": passed,
        "status_counts": dict(sorted(counts.items())),
        "conditional_structural_pass_rate": passed / len(scoreable)
        if scoreable
        else None,
        "end_to_end_structural_pass_rate": passed / len(ordered),
        "engineering_requirement_pass_rate": (
            requirement_passed / len(scoreable) if scoreable else None
        ),
        "mean_response_fidelity_score": (
            sum(fidelity_scores) / len(fidelity_scores) if fidelity_scores else None
        ),
        "runtime_seconds": time.monotonic() - started,
        "variants": [_result_row(result) for result in ordered],
    }
    write_json(output_dir / "family_result.json", summary)
    _write_rows(output_dir / "results.csv", summary["variants"])
    write_evidence_commit(output_dir, output_dir / "family_result.json")
    return summary
