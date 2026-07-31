"""Transactional item and perturbation-family runners for L3."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from cadengbench_p_eval.integrity import (
    EVIDENCE_MANIFEST_NAME,
    RESULT_DIGEST_NAME,
    verify_evidence_manifest,
    write_evidence_manifest,
)
from cadengbench_p_eval.io import (
    read_json,
    sha256_file,
    sha256_json,
    write_json,
    write_text,
)
from cadengbench_p_eval.physics.calculix import (
    load_physics_card,
    run_probe_card,
    validate_physics_card,
)
from cadengbench_p_eval.provenance import physics_evaluator_build_hash

FAMILY_MANIFEST_SCHEMA = "cadengbench-p.l3-family-manifest.v1"
FAMILY_RESULT_SCHEMA = "cadengbench-p.l3-family-result.v1"
QUALIFICATION_CORPUS_SCHEMA = "cadengbench-p.l3-qualification-corpus.v1"
MESH_STUDY_SCHEMA = "cadengbench-p.l3-mesh-sensitivity-study.v1"
RESULT_TARGET_NAME = "result_target.txt"


class L3RunnerError(RuntimeError):
    """An L3 run manifest or transaction is invalid."""


def _commit_l3_output(root: Path, result_path: Path) -> None:
    resolved_root = root.resolve()
    resolved_result = result_path.resolve()
    if resolved_root not in resolved_result.parents or not resolved_result.is_file():
        raise L3RunnerError("L3 commit target is outside its evidence root")
    write_text(
        root / RESULT_TARGET_NAME,
        f"{resolved_result.relative_to(resolved_root).as_posix()}\n",
    )
    write_evidence_manifest(root)
    write_text(root / RESULT_DIGEST_NAME, f"{sha256_file(resolved_result)}\n")


def _verify_l3_output_commit(root: Path, result_path: Path) -> None:
    target_path = root / RESULT_TARGET_NAME
    digest_path = root / RESULT_DIGEST_NAME
    if not target_path.is_file() or not digest_path.is_file():
        raise L3RunnerError("L3 output commit metadata is incomplete")
    expected_target = result_path.resolve().relative_to(root.resolve()).as_posix()
    if target_path.read_text(encoding="ascii").strip() != expected_target:
        raise L3RunnerError("L3 output commit target changed")
    if digest_path.read_text(encoding="ascii").strip() != sha256_file(result_path):
        raise L3RunnerError("L3 output result digest changed")
    verify_evidence_manifest(root)


def _source_probe_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics = result["metrics"]
    stress = metrics["regularized_von_mises_mpa"]
    return {
        "normalized_max_displacement": float(metrics["normalized_max_displacement"]),
        "regularized_von_mises_p95_mpa": float(stress["p95"]),
        "regularized_von_mises_p99_mpa": float(stress["p99"]),
        "stress_concentration_p95": float(metrics["stress_concentration_p95"]),
        "normalized_compliance": float(metrics["normalized_compliance"]),
        "force_equilibrium_relative_error": float(
            metrics["force_equilibrium_relative_error"]
        ),
    }


def _safe_replaceable_output(output: Path, protected: tuple[Path, ...]) -> None:
    resolved = output.expanduser().resolve()
    if resolved == Path("/"):
        raise L3RunnerError("refusing to use the filesystem root as output")
    for raw in protected:
        path = raw.expanduser().resolve()
        if resolved == path or resolved in path.parents or path in resolved.parents:
            raise L3RunnerError(f"output overlaps protected input: {path}")


def _prepare_output(output: Path, *, force: bool) -> None:
    if not output.exists():
        output.mkdir(parents=True)
        return
    if output.is_symlink() or not output.is_dir():
        raise L3RunnerError("L3 output must be a normal directory")
    if any(output.iterdir()):
        if not force:
            raise L3RunnerError("L3 output is nonempty; use --force or a fresh path")
        shutil.rmtree(output)
        output.mkdir(parents=True)


def qualify_physics_card(
    *,
    card_path: Path,
    reference_step: Path,
    output_dir: Path,
    promoted_card_path: Path,
    ccx_path: Path,
    timeout_seconds: int = 300,
    render: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run the source geometry and promote only a fully passing probe card."""

    _safe_replaceable_output(
        output_dir, (card_path, reference_step, promoted_card_path, ccx_path)
    )
    _prepare_output(output_dir, force=force)
    if promoted_card_path.exists():
        if not force:
            raise L3RunnerError(
                "promoted card already exists; use --force or a fresh path"
            )
        if promoted_card_path.is_symlink() or not promoted_card_path.is_file():
            raise L3RunnerError("promoted card path is not a normal file")
        promoted_card_path.unlink()
    card = load_physics_card(card_path)
    result = run_probe_card(
        card,
        step_path=reference_step,
        output_dir=output_dir,
        ccx_path=ccx_path,
        timeout_seconds=timeout_seconds,
        render=render,
    )
    qualification = {
        "schema_version": "cadengbench-p.l3-card-qualification.v1",
        "item_id": card["item_id"],
        "card_sha256": sha256_file(card_path),
        "reference_step_sha256": sha256_file(reference_step),
        "result_sha256": sha256_file(output_dir / "result.json"),
        "evaluator_build_hash": result["evaluator_build_hash"],
        "runtime_lock_sha256": result.get("runtime_lock_sha256"),
        "result_status": result["final_status"],
        "promoted": result["final_status"] == "probe_pass"
        and isinstance(result.get("runtime_lock_sha256"), str),
    }
    if result["final_status"] == "probe_pass" and not qualification["promoted"]:
        qualification["promotion_blocker"] = "runtime_lock_missing"
    if qualification["promoted"]:
        qualification["source_probe_metrics"] = _source_probe_metrics(result)
        promoted = deepcopy(card)
        promoted["status"] = "solver_qualified"
        promoted["qualification"] = qualification
        validate_physics_card(promoted)
        write_json(promoted_card_path, promoted)
        qualification["promoted_card_path"] = str(promoted_card_path)
        qualification["promoted_card_sha256"] = sha256_file(promoted_card_path)
    write_json(output_dir / "qualification.json", qualification)
    _commit_l3_output(output_dir, output_dir / "result.json")
    return qualification


def _reference_step(items_root: Path, item_id: str) -> Path:
    item_root = items_root / item_id / "source"
    candidates = (
        item_root / "generated_from_cadquery.step",
        item_root / "fusion360_occ_reconstruction.step",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise L3RunnerError(f"source STEP is unavailable for {item_id}")


def _qualify_corpus_item(
    *,
    item_id: str,
    card_path: Path,
    reference_step: Path,
    qualification_dir: Path,
    promoted_card_path: Path,
    ccx_path: Path,
    timeout_seconds: int,
    render: bool,
    force: bool,
) -> dict[str, Any]:
    evidence_path = qualification_dir / "qualification.json"
    try:
        result_path = qualification_dir / "result.json"
        commit_complete = all(
            path.is_file()
            for path in (
                evidence_path,
                result_path,
                qualification_dir / EVIDENCE_MANIFEST_NAME,
                qualification_dir / RESULT_DIGEST_NAME,
                qualification_dir / RESULT_TARGET_NAME,
            )
        )
        if not force and commit_complete:
            evidence = read_json(evidence_path)
            result = read_json(result_path)
            runtime_lock = Path(os.environ.get("CADENGBENCH_L3_RUNTIME_LOCK", ""))
            runtime_lock_sha256 = (
                sha256_file(runtime_lock) if runtime_lock.is_file() else None
            )
            current = (
                evidence.get("item_id") == item_id
                and evidence.get("card_sha256") == sha256_file(card_path)
                and evidence.get("reference_step_sha256") == sha256_file(reference_step)
                and evidence.get("evaluator_build_hash")
                == physics_evaluator_build_hash()
                and result.get("evaluator_build_hash")
                == physics_evaluator_build_hash()
                and evidence.get("runtime_lock_sha256") == runtime_lock_sha256
                and result.get("runtime_lock_sha256") == runtime_lock_sha256
            )
            promoted = bool(evidence.get("promoted"))
            if current and (
                not promoted
                or (
                    promoted_card_path.is_file()
                    and evidence.get("promoted_card_sha256")
                    == sha256_file(promoted_card_path)
                )
            ):
                _verify_l3_output_commit(qualification_dir, result_path)
                return {
                    "item_id": item_id,
                    "status": "promoted" if promoted else "not_promoted",
                    "result_status": evidence.get("result_status"),
                    "failure_stage": (result.get("failure") or {}).get("stage"),
                    "resumed": True,
                    "recovered_incomplete": False,
                }
            raise L3RunnerError(f"stale qualification evidence for {item_id}")
        recovered_incomplete = qualification_dir.exists() and not commit_complete
        evidence = qualify_physics_card(
            card_path=card_path,
            reference_step=reference_step,
            output_dir=qualification_dir,
            promoted_card_path=promoted_card_path,
            ccx_path=ccx_path,
            timeout_seconds=timeout_seconds,
            render=render,
            force=force or recovered_incomplete,
        )
        result = read_json(qualification_dir / "result.json")
        return {
            "item_id": item_id,
            "status": "promoted" if evidence["promoted"] else "not_promoted",
            "result_status": evidence["result_status"],
            "failure_stage": (result.get("failure") or {}).get("stage"),
            "resumed": False,
            "recovered_incomplete": recovered_incomplete,
        }
    except Exception as exc:  # retain a complete corpus roster
        return {
            "item_id": item_id,
            "status": "evaluator_error",
            "result_status": None,
            "failure_stage": "qualification_runner",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "resumed": False,
            "recovered_incomplete": False,
        }


def _run_isolated_worker(
    *,
    specification: Mapping[str, Any],
    runtime_root: Path,
    key: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if Path(key).name != key:
        raise L3RunnerError("L3 worker key is not a safe path component")
    attempt_root = runtime_root / key / str(time.time_ns())
    attempt_root.mkdir(parents=True, exist_ok=False)
    job_path = attempt_root / "job.json"
    result_path = attempt_root / "result.json"
    write_json(
        job_path,
        {
            "schema_version": "cadengbench-p.l3-subprocess-worker-job.v1",
            **dict(specification),
        },
    )
    command = [
        sys.executable,
        "-m",
        "cadengbench_p_eval.physics.subprocess_worker",
        "--job",
        str(job_path),
        "--result",
        str(result_path),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code: int | None = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return_code = None
        timed_out = True
    write_text(attempt_root / "worker.stdout.txt", stdout)
    write_text(attempt_root / "worker.stderr.txt", stderr)
    report = {
        "schema_version": "cadengbench-p.l3-subprocess-worker-report.v1",
        "key": key,
        "operation": specification.get("operation"),
        "return_code": return_code,
        "timed_out": timed_out,
        "runtime_seconds": time.monotonic() - started,
        "job_sha256": sha256_file(job_path),
        "result_exists": result_path.is_file(),
    }
    if return_code == 0 and result_path.is_file():
        value = read_json(result_path)
        if isinstance(value, dict):
            report["status"] = "passed"
            report["result_sha256"] = sha256_file(result_path)
            write_json(attempt_root / "worker_report.json", report)
            return value, report
    report["status"] = "failed"
    write_json(attempt_root / "worker_report.json", report)
    return None, report


def _qualify_corpus_item_isolated(
    *, runtime_root: Path, process_timeout_seconds: int, **job: Any
) -> dict[str, Any]:
    item_id = str(job["item_id"])
    specification = {
        "operation": "source_qualification",
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in job.items()
        },
    }
    result, report = _run_isolated_worker(
        specification=specification,
        runtime_root=runtime_root,
        key=item_id,
        timeout_seconds=process_timeout_seconds,
    )
    if result is not None:
        return result
    return {
        "item_id": item_id,
        "status": "evaluator_error",
        "result_status": None,
        "failure_stage": "qualification_worker_process",
        "message": "isolated qualification worker failed",
        "worker_report": report,
        "resumed": False,
        "recovered_incomplete": False,
    }


def qualify_annotation_corpus(
    *,
    repo_root: Path,
    profile_path: Path,
    output_root: Path,
    ccx_path: Path,
    workers: int = 1,
    timeout_seconds: int = 300,
    render: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Source-qualify every compiled card while preserving the 300-item roster."""

    if workers < 1:
        raise L3RunnerError("qualification workers must be positive")
    profile = read_json(profile_path)
    if profile.get("schema_version") != "cadengbench-p.l3-probe-profile.v1":
        raise L3RunnerError("unsupported L3 probe profile")
    root = repo_root.resolve()
    items_root = (root / str(profile["dataset"]["items_root"])).resolve()
    requests_path = output_root / "requests.jsonl"
    if not requests_path.is_file():
        raise L3RunnerError("L3 annotation requests are unavailable")
    requests = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(requests) != 300:
        raise L3RunnerError("qualification corpus must contain exactly 300 items")
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    jobs = []
    for request in requests:
        item_id = str(request["item_id"])
        item_root = output_root / "items" / item_id
        card_path = item_root / "physics_card.json"
        validation_path = item_root / "annotation_validation.json"
        validation = read_json(validation_path) if validation_path.is_file() else {}
        if validation.get("disposition") != "geometry_qualified":
            records.append(
                {
                    "item_id": item_id,
                    "status": "not_compiled",
                    "result_status": None,
                    "failure_stage": None,
                    "annotation_disposition": validation.get("disposition"),
                    "annotation_reason": validation.get("reason"),
                    "stale_card_ignored": card_path.is_file(),
                    "resumed": False,
                    "recovered_incomplete": False,
                }
            )
            continue
        if not card_path.is_file():
            records.append(
                {
                    "item_id": item_id,
                    "status": "evaluator_error",
                    "result_status": None,
                    "failure_stage": "qualification_input_binding",
                    "message": "geometry-qualified annotation has no physics card",
                    "resumed": False,
                    "recovered_incomplete": False,
                }
            )
            continue
        actual_card_sha256 = sha256_file(card_path)
        if validation.get("card_sha256") != actual_card_sha256:
            records.append(
                {
                    "item_id": item_id,
                    "status": "evaluator_error",
                    "result_status": None,
                    "failure_stage": "qualification_input_binding",
                    "message": "physics card does not match annotation validation",
                    "resumed": False,
                    "recovered_incomplete": False,
                }
            )
            continue
        jobs.append(
            {
                "item_id": item_id,
                "card_path": card_path,
                "reference_step": _reference_step(items_root, item_id),
                "qualification_dir": item_root / "source_qualification",
                "promoted_card_path": item_root / "physics_card.solver_qualified.json",
                "ccx_path": ccx_path,
                "timeout_seconds": timeout_seconds,
                "render": render,
                "force": force,
            }
        )
    runtime_root = output_root / "qualification_runtime"
    process_timeout_seconds = timeout_seconds + 600
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _qualify_corpus_item_isolated,
                runtime_root=runtime_root,
                process_timeout_seconds=process_timeout_seconds,
                **job,
            ): str(job["item_id"])
            for job in jobs
        }
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # parent orchestration boundary
                records.append(
                    {
                        "item_id": item_id,
                        "status": "evaluator_error",
                        "result_status": None,
                        "failure_stage": "qualification_orchestrator",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "resumed": False,
                        "recovered_incomplete": False,
                    }
                )
    records.sort(key=lambda value: value["item_id"])
    status_counts = Counter(record["status"] for record in records)
    failure_stages = Counter(
        str(record["failure_stage"])
        for record in records
        if record.get("failure_stage")
    )
    summary = {
        "schema_version": QUALIFICATION_CORPUS_SCHEMA,
        "profile_sha256": sha256_file(profile_path),
        "requests_sha256": sha256_file(requests_path),
        "item_count": len(records),
        "compiled_card_count": len(jobs),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_stage_counts": dict(sorted(failure_stages.items())),
        "resumed_count": sum(bool(record["resumed"]) for record in records),
        "recovered_incomplete_count": sum(
            bool(record.get("recovered_incomplete")) for record in records
        ),
        "workers": workers,
        "timeout_seconds": timeout_seconds,
        "process_timeout_seconds": process_timeout_seconds,
        "worker_isolation": "one_short_lived_python_process_per_card_v1",
        "render": render,
        "core_dataset_mutated": False,
        "runtime_seconds": time.monotonic() - started,
        "records": records,
    }
    write_json(output_root / "qualification_corpus_summary.json", summary)
    return summary


def run_mesh_sensitivity_study(
    *,
    card_path: Path,
    reference_step: Path,
    output_root: Path,
    ccx_path: Path,
    timeout_seconds: int = 600,
    force: bool = False,
) -> dict[str, Any]:
    """Run an offline coarse/production/fine calibration of one source card."""

    _safe_replaceable_output(output_root, (card_path, reference_step, ccx_path))
    _prepare_output(output_root, force=force)
    base = load_physics_card(card_path)
    production = base["mesh"]
    levels = {
        "coarse": {
            "target_long_axis_divisions": max(
                4, round(int(production["target_long_axis_divisions"]) * 0.75)
            ),
            "target_short_axis_divisions": max(
                2, round(int(production["target_short_axis_divisions"]) * 0.8)
            ),
        },
        "production": {
            "target_long_axis_divisions": int(production["target_long_axis_divisions"]),
            "target_short_axis_divisions": int(
                production["target_short_axis_divisions"]
            ),
        },
        "fine": {
            "target_long_axis_divisions": round(
                int(production["target_long_axis_divisions"]) * 1.25
            ),
            "target_short_axis_divisions": round(
                int(production["target_short_axis_divisions"]) * 1.2
            ),
        },
    }
    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    for level, policy in levels.items():
        card = deepcopy(base)
        card["status"] = "geometry_qualified"
        card.pop("qualification", None)
        card["card_id"] = f"{base['card_id']}:mesh-study:{level}"
        card["mesh"].update(policy)
        results[level] = run_probe_card(
            card,
            step_path=reference_step,
            output_dir=output_root / level,
            ccx_path=ccx_path,
            timeout_seconds=timeout_seconds,
            render=False,
        )
    fine = results["fine"].get("metrics") or {}
    quantities = {
        "normalized_max_displacement": lambda value: value[
            "normalized_max_displacement"
        ],
        "regularized_von_mises_p95_mpa": lambda value: value[
            "regularized_von_mises_mpa"
        ]["p95"],
        "regularized_von_mises_p99_mpa": lambda value: value[
            "regularized_von_mises_mpa"
        ]["p99"],
        "normalized_compliance": lambda value: value["normalized_compliance"],
    }
    comparisons: dict[str, Any] = {}
    if fine:
        for level in ("coarse", "production"):
            metrics = results[level].get("metrics") or {}
            if not metrics:
                continue
            comparisons[level] = {
                key: {
                    "observed": float(extract(metrics)),
                    "fine": float(extract(fine)),
                    "absolute_relative_difference": abs(
                        float(extract(metrics)) - float(extract(fine))
                    )
                    / max(abs(float(extract(fine))), 1e-12),
                }
                for key, extract in quantities.items()
            }
    summary = {
        "schema_version": MESH_STUDY_SCHEMA,
        "item_id": base["item_id"],
        "card_sha256": sha256_file(card_path),
        "reference_step_sha256": sha256_file(reference_step),
        "levels": {
            level: {
                **policy,
                "final_status": results[level]["final_status"],
                "element_count": (
                    results[level].get("stages", {}).get("mesh") or {}
                ).get("element_count"),
                "runtime_seconds": results[level].get("runtime_seconds"),
            }
            for level, policy in levels.items()
        },
        "comparisons_to_fine": comparisons,
        "all_levels_passed": all(
            value["final_status"] == "probe_pass" for value in results.values()
        ),
        "diagnostic_only": True,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(output_root / "mesh_sensitivity_summary.json", summary)
    _commit_l3_output(output_root, output_root / "mesh_sensitivity_summary.json")
    return summary


def _variant_record(
    *, variant_id: str, step_path: Path, upstream_status: str
) -> dict[str, Any]:
    record = {
        "variant_id": variant_id,
        "upstream_status": upstream_status,
        "step_path": str(step_path.resolve()),
        "step_sha256": None,
    }
    if step_path.is_file():
        record["step_sha256"] = sha256_file(step_path)
    return record


def build_family_manifest_from_evaluator_run(
    *, card_path: Path, evaluator_item_root: Path, output_path: Path
) -> dict[str, Any]:
    """Bind nominal and L2 variant STEP artifacts from an existing L0-L2 run."""

    card = load_physics_card(card_path)
    if card["status"] != "solver_qualified":
        raise L3RunnerError("L3 family manifests require a solver-qualified card")
    variants: list[dict[str, Any]] = []
    l0_path = evaluator_item_root / "l0.json"
    l1_path = evaluator_item_root / "l1.json"
    nominal = evaluator_item_root / "l0/candidate.step"
    nominal_status = "not_reached"
    if l0_path.is_file() and l1_path.is_file():
        nominal_status = (
            "passed"
            if read_json(l0_path).get("status") == "passed"
            and read_json(l1_path).get("status") == "passed"
            and nominal.is_file()
            else "failed"
        )
    variants.append(
        _variant_record(
            variant_id="nominal",
            step_path=nominal,
            upstream_status=nominal_status,
        )
    )
    l2_path = evaluator_item_root / "l2.json"
    if l2_path.is_file():
        l2 = read_json(l2_path)
        replay = (l2.get("artifacts") or {}).get("replay") or {}
        for value in replay.get("variants") or []:
            if not isinstance(value, Mapping):
                continue
            variant = value.get("variant")
            variant_id = (
                variant.get("variant_id")
                if isinstance(variant, Mapping)
                else value.get("variant_id")
            )
            if not isinstance(variant_id, str) or not variant_id:
                raise L3RunnerError("L2 replay variant is missing a valid variant ID")
            candidate = (
                evaluator_item_root / "l2/variants" / variant_id / "candidate.step"
            )
            variants.append(
                _variant_record(
                    variant_id=variant_id,
                    step_path=candidate,
                    upstream_status=str(value.get("status") or "not_reached"),
                )
            )
    body = {
        "schema_version": FAMILY_MANIFEST_SCHEMA,
        "item_id": card["item_id"],
        "card_path": str(card_path.resolve()),
        "card_sha256": sha256_file(card_path),
        "source_evaluator_item_root": str(evaluator_item_root.resolve()),
        "variants": variants,
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    write_json(output_path, manifest)
    return manifest


def load_family_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != FAMILY_MANIFEST_SCHEMA
    ):
        raise L3RunnerError("unsupported L3 family manifest")
    expected = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if not isinstance(expected, str) or sha256_json(body) != expected:
        raise L3RunnerError("L3 family manifest hash is invalid")
    card_path = Path(str(manifest["card_path"]))
    if not card_path.is_file() or sha256_file(card_path) != manifest["card_sha256"]:
        raise L3RunnerError("L3 family card binding is invalid")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise L3RunnerError("L3 family manifest has no variants")
    identifiers = [str(value.get("variant_id")) for value in variants]
    if len(identifiers) != len(set(identifiers)):
        raise L3RunnerError("L3 family variant IDs are not unique")
    for record in variants:
        if record.get("upstream_status") != "passed":
            continue
        step_path = Path(str(record["step_path"]))
        if not step_path.is_file() or sha256_file(step_path) != record["step_sha256"]:
            raise L3RunnerError(
                f"L3 variant STEP binding is invalid: {record['variant_id']}"
            )
    return manifest


def _run_variant(
    *,
    card: Mapping[str, Any],
    record: Mapping[str, Any],
    output_dir: Path,
    ccx_path: Path,
    timeout_seconds: int,
    render: bool,
) -> dict[str, Any]:
    if record["upstream_status"] != "passed":
        result = {
            "schema_version": "cadengbench-p.l3-skipped-variant.v1",
            "item_id": card["item_id"],
            "variant_id": record["variant_id"],
            "upstream_status": record["upstream_status"],
            "final_status": "skipped_upstream",
            "scoreable": False,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "result.json", result)
        return result
    result = run_probe_card(
        card,
        step_path=Path(str(record["step_path"])),
        output_dir=output_dir,
        ccx_path=ccx_path,
        timeout_seconds=timeout_seconds,
        render=render,
    )
    result["variant_id"] = record["variant_id"]
    write_json(output_dir / "result.json", result)
    return result


def _run_variant_isolated(
    *,
    card_path: Path,
    record: Mapping[str, Any],
    output_dir: Path,
    ccx_path: Path,
    timeout_seconds: int,
    render: bool,
    runtime_root: Path,
) -> dict[str, Any]:
    variant_id = str(record["variant_id"])
    specification = {
        "operation": "variant",
        "card_path": str(card_path),
        "record": dict(record),
        "output_dir": str(output_dir),
        "ccx_path": str(ccx_path),
        "timeout_seconds": timeout_seconds,
        "render": render,
    }
    result, report = _run_isolated_worker(
        specification=specification,
        runtime_root=runtime_root,
        key=variant_id,
        timeout_seconds=timeout_seconds + 600,
    )
    if result is not None:
        return result
    result = {
        "schema_version": "cadengbench-p.l3-worker-failure.v1",
        "item_id": record.get("item_id"),
        "variant_id": variant_id,
        "final_status": "failed",
        "scoreable": False,
        "failure": {
            "stage": "variant_worker_process",
            "owner": "evaluator",
            "message": "isolated variant worker failed",
            "worker_report": report,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    return result


def run_family(
    *,
    manifest_path: Path,
    output_root: Path,
    ccx_path: Path,
    workers: int = 1,
    timeout_seconds: int = 300,
    render: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run every L2-passing family member and retain typed failures."""

    if workers < 1:
        raise L3RunnerError("workers must be positive")
    manifest = load_family_manifest(manifest_path)
    card_path = Path(str(manifest["card_path"]))
    _safe_replaceable_output(output_root, (manifest_path, card_path, ccx_path))
    _prepare_output(output_root, force=force)
    card = load_physics_card(card_path)
    if card["item_id"] != manifest["item_id"]:
        raise L3RunnerError("family card item ID does not match manifest")
    if card["status"] != "solver_qualified":
        raise L3RunnerError("L3 family execution requires a solver-qualified card")
    started = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    records = list(manifest["variants"])
    runtime_root = output_root / "worker_runtime"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_variant_isolated,
                card_path=card_path,
                record={**record, "item_id": manifest["item_id"]},
                output_dir=output_root / "variants" / str(record["variant_id"]),
                ccx_path=ccx_path,
                timeout_seconds=timeout_seconds,
                render=render,
                runtime_root=runtime_root,
            ): str(record["variant_id"])
            for record in records
        }
        for future in as_completed(futures):
            variant_id = futures[future]
            try:
                results[variant_id] = future.result()
            except Exception as exc:  # parent orchestration boundary
                result = {
                    "schema_version": "cadengbench-p.l3-worker-failure.v1",
                    "item_id": manifest["item_id"],
                    "variant_id": variant_id,
                    "final_status": "failed",
                    "scoreable": False,
                    "failure": {
                        "stage": "variant_orchestrator",
                        "owner": "evaluator",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
                variant_output = output_root / "variants" / variant_id
                variant_output.mkdir(parents=True, exist_ok=True)
                write_json(variant_output / "result.json", result)
                results[variant_id] = result
    counts = Counter(value["final_status"] for value in results.values())
    reached = [
        value
        for value in results.values()
        if value["final_status"] != "skipped_upstream"
    ]
    passed = sum(value["final_status"] == "probe_pass" for value in reached)
    evaluator_errors = sum(
        (value.get("failure") or {}).get("owner") == "evaluator" for value in reached
    )
    source_records = {
        str(record["variant_id"]): record for record in manifest["variants"]
    }

    def variant_summary(variant_id: str) -> dict[str, Any]:
        value = results[variant_id]
        stages = value.get("stages") or {}
        metrics = value.get("metrics") or {}
        summary = {
            "variant_id": variant_id,
            "upstream_status": source_records[variant_id]["upstream_status"],
            "geometry": (stages.get("geometry") or {}).get("status"),
            "selectors": (stages.get("selectors") or {}).get("status"),
            "mesh": (stages.get("mesh") or {}).get("status"),
            "solve": (stages.get("solver") or {}).get("status"),
            "requirements": (stages.get("requirements") or {}).get("status"),
            "final_status": value["final_status"],
            "failure_stage": (value.get("failure") or {}).get("stage"),
            "failure_owner": (value.get("failure") or {}).get("owner"),
            "margins": {
                "force_equilibrium_relative_error": metrics.get(
                    "force_equilibrium_relative_error"
                ),
                "normalized_max_displacement": metrics.get(
                    "normalized_max_displacement"
                ),
                "regularized_von_mises_mpa": metrics.get("regularized_von_mises_mpa"),
                "normalized_compliance": metrics.get("normalized_compliance"),
            },
            "runtime_seconds": value.get("runtime_seconds"),
            "result_path": str(output_root / "variants" / variant_id / "result.json"),
        }
        if variant_id == "nominal" and value["final_status"] == "probe_pass":
            reference = card["qualification"]["source_probe_metrics"]
            candidate = {
                "normalized_max_displacement": metrics["normalized_max_displacement"],
                "regularized_von_mises_p95_mpa": metrics["regularized_von_mises_mpa"][
                    "p95"
                ],
                "regularized_von_mises_p99_mpa": metrics["regularized_von_mises_mpa"][
                    "p99"
                ],
                "stress_concentration_p95": metrics["stress_concentration_p95"],
                "normalized_compliance": metrics["normalized_compliance"],
            }
            summary["nominal_reference_response"] = {
                key: {
                    "candidate": float(candidate[key]),
                    "reference": float(reference[key]),
                    "absolute_relative_error": abs(
                        float(candidate[key]) - float(reference[key])
                    )
                    / max(abs(float(reference[key])), 1e-12),
                    "diagnostic_only": True,
                }
                for key in candidate
            }
        return summary

    summary = {
        "schema_version": FAMILY_RESULT_SCHEMA,
        "item_id": manifest["item_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "card_sha256": manifest["card_sha256"],
        "variant_count": len(records),
        "l3_reached_count": len(reached),
        "probe_pass_count": passed,
        "evaluator_error_count": evaluator_errors,
        "status_counts": dict(sorted(counts.items())),
        "conditional_l3_robustness": passed / len(reached) if reached else None,
        "end_to_end_family_robustness": passed / len(records),
        "runtime_seconds": time.monotonic() - started,
        "variants": [variant_summary(variant_id) for variant_id in sorted(results)],
    }
    write_json(output_root / "family_result.json", summary)
    from cadengbench_p_eval.layers.l3 import l3_layer_from_family_result

    write_json(output_root / "l3.json", l3_layer_from_family_result(summary).as_dict())
    _commit_l3_output(output_root, output_root / "family_result.json")
    return summary
