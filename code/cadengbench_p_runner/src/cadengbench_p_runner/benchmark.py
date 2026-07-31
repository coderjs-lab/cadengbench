"""One resumable generation-to-scoring launcher for the frozen L0-L2 benchmark."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .io import (
    read_json,
    safe_relative_path,
    semantic_hash,
    sha256_file,
    write_json,
    write_text,
)
from .model_registry import load_model_spec
from .packaging import package_run
from .planner import load_run_manifest, plan_run
from .scheduler import run_generation
from .schema import validate_canonical_request

PROTOCOLS = ("zero_to_cad", "functional_edit")
SUPPORTED_RELEASE_SCHEMAS = {
    "cadengbench-p.l0-l2-release.v1",
    "cadengbench-p.l0-l2-release.v2",
    "cadengbench-p.l0-l2-release.v3",
}


def _is_frozen_release(release: Any) -> bool:
    return (
        isinstance(release, dict)
        and release.get("schema_version") in SUPPORTED_RELEASE_SCHEMAS
        and release.get("item_count") == 300
    )


def _run_command(
    command: list[str],
    log_path: Path,
    *,
    completed_summary_path: Path | None = None,
) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    write_text(
        log_path,
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    if completed.returncode != 0:
        if completed_summary_path is not None and completed_summary_path.is_file():
            summary = read_json(completed_summary_path)
            if (
                summary.get("valid_complete_run") is True
                and summary.get("execution_complete") is True
                and summary.get("result_roster_exact") is True
                and summary.get("item_evaluator_error_count") == 0
            ):
                return
        raise RuntimeError(
            "benchmark subprocess failed with code "
            f"{completed.returncode}; see {log_path}"
        )


def _plan_or_load(
    *,
    protocol: str,
    evalset_root: Path,
    model_spec_path: Path,
    run_root: Path,
    run_id: str,
    input_profile: str,
    item_ids: list[str] | None = None,
) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.is_file():
        manifest = load_run_manifest(run_root)
        observed = manifest.get("task_kind", "zero_to_cad")
        if observed != protocol:
            raise ValueError(f"existing run is for {observed}, not {protocol}")
        expected_run_id = f"{run_id}.{protocol}"
        if manifest.get("run_id") != expected_run_id:
            raise ValueError(
                f"existing {protocol} run ID differs: {manifest.get('run_id')}"
            )
        if manifest.get("input_profile") != input_profile:
            raise ValueError(f"existing {protocol} input profile differs")
        expected_model = load_model_spec(model_spec_path)
        if manifest.get("model", {}).get("spec_hash") != expected_model.digest:
            raise ValueError(f"existing {protocol} model specification differs")
        return manifest
    return plan_run(
        items_root=evalset_root / "items",
        model_spec_path=model_spec_path,
        output_root=run_root,
        run_id=f"{run_id}.{protocol}",
        input_profile=input_profile,
        task_kind=protocol,
        sample_count=1,
        item_ids=item_ids,
    )


def _request_bundle_report(run_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    request_bytes = 0
    maximum_request = ("", 0)
    visual_assets: set[tuple[str, str]] = set()
    for record in manifest["requests"]:
        request_path = safe_relative_path(run_root, record["request_path"])
        request = read_json(request_path)
        validate_canonical_request(request)
        if (
            request["item_id"] != record["item_id"]
            or request["request_hash"] != record["request_hash"]
        ):
            raise ValueError(f"request manifest binding failed: {request_path}")
        request_bytes += request_path.stat().st_size
        maximum_request = max(
            maximum_request,
            (record["item_id"], request_path.stat().st_size),
            key=lambda value: value[1],
        )
        for message in request["messages"]:
            for part in message["parts"]:
                if part["type"] != "image":
                    continue
                asset = safe_relative_path(run_root, part["asset_path"])
                if not asset.is_file() or sha256_file(asset) != part["sha256"]:
                    raise ValueError(f"request visual binding failed: {asset}")
                visual_assets.add((part["asset_path"], part["sha256"]))
    return {
        "request_count": len(manifest["requests"]),
        "request_bytes": request_bytes,
        "maximum_request_bytes": maximum_request[1],
        "maximum_request_item_id": maximum_request[0],
        "visual_asset_count": len(visual_assets),
    }


def prepare_benchmark(
    *,
    evalset_root: Path,
    model_spec_path: Path,
    output_root: Path,
    run_id: str,
    evaluator_cli: Path,
    input_profile: str,
    item_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Freeze and validate both protocol request sets without calling a model."""

    evalset_root = evalset_root.resolve()
    model_spec_path = model_spec_path.resolve()
    output_root = output_root.resolve()
    evaluator_cli = evaluator_cli.resolve()
    release_path = evalset_root / "release_manifest.json"
    release = read_json(release_path)
    if not _is_frozen_release(release):
        raise ValueError(
            f"evalset is not a frozen 300-item L0-L2 release: {release_path}"
        )
    if not evaluator_cli.is_file():
        raise ValueError(f"evaluator CLI is missing: {evaluator_cli}")
    _run_command(
        [
            str(evaluator_cli),
            "validate-l0-l2-release",
            "--evalset",
            str(evalset_root),
        ],
        output_root / "preflight.log.json",
    )

    expected_count = len(item_ids) if item_ids is not None else 300
    if expected_count < 1 or (
        item_ids is not None and len(set(item_ids)) != expected_count
    ):
        raise ValueError("item_ids must be a nonempty unique list")
    protocols: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        generation_root = output_root / protocol / "generation"
        manifest = _plan_or_load(
            protocol=protocol,
            evalset_root=evalset_root,
            model_spec_path=model_spec_path,
            run_root=generation_root,
            run_id=run_id,
            input_profile=input_profile,
            item_ids=item_ids,
        )
        report = _request_bundle_report(generation_root, manifest)
        if report["request_count"] != expected_count:
            raise ValueError(f"{protocol} did not materialize the selected roster")
        protocols[protocol] = {
            "run_manifest_hash": manifest["manifest_hash"],
            **report,
        }
    record = {
        "schema_version": "cadengbench-p.benchmark-plan.v1",
        "run_id": run_id,
        "evalset_release_hash": release["release_hash"],
        "model_spec_sha256": sha256_file(model_spec_path),
        "model_spec_hash": load_model_spec(model_spec_path).digest,
        "input_profile": input_profile,
        "item_count": expected_count,
        "item_ids": item_ids,
        "protocols": protocols,
    }
    record["benchmark_plan_hash"] = semantic_hash(record, "benchmark_plan_hash")
    path = output_root / "benchmark_plan.json"
    if path.is_file() and read_json(path) != record:
        raise ValueError("existing benchmark plan differs from requested run")
    write_json(path, record)
    return record


def run_benchmark(
    *,
    evalset_root: Path,
    model_spec_path: Path,
    output_root: Path,
    run_id: str,
    evaluator_cli: Path,
    repo_root: Path,
    input_profile: str,
    workers: int,
    item_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resume both protocols and deterministically score every successful artifact."""

    evalset_root = evalset_root.resolve()
    output_root = output_root.resolve()
    evaluator_cli = evaluator_cli.resolve()
    repo_root = repo_root.resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    benchmark_plan = prepare_benchmark(
        evalset_root=evalset_root,
        model_spec_path=model_spec_path,
        output_root=output_root,
        run_id=run_id,
        evaluator_cli=evaluator_cli,
        input_profile=input_profile,
        item_ids=item_ids,
    )

    protocol_records: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        protocol_root = output_root / protocol
        generation_root = protocol_root / "generation"
        scoring_root = protocol_root / "scoring"
        manifest = _plan_or_load(
            protocol=protocol,
            evalset_root=evalset_root,
            model_spec_path=model_spec_path,
            run_root=generation_root,
            run_id=run_id,
            input_profile=input_profile,
            item_ids=item_ids,
        )
        generation = run_generation(generation_root)
        if not generation["run_settled"]:
            raise RuntimeError(
                f"{protocol} generation is incomplete; resume the same benchmark output"
            )
        package = package_run(generation_root)
        command = [
            str(evaluator_cli),
            "evaluate",
            "--evalset",
            str(evalset_root / protocol),
            "--submissions",
            str(generation_root / "artifacts"),
            "--output",
            str(scoring_root),
            "--model-id",
            str(manifest["model"]["model_id"]),
            "--workers",
            str(workers),
            "--resume",
        ]
        if protocol == "functional_edit":
            command.extend(["--repo-root", str(repo_root)])
        for item_id in item_ids or ():
            command.extend(["--item-id", item_id])
        _run_command(
            command,
            protocol_root / "evaluate.log.json",
            completed_summary_path=scoring_root / "summary.json",
        )
        _run_command(
            [str(evaluator_cli), "aggregate", "--run", str(scoring_root)],
            protocol_root / "aggregate.log.json",
        )
        aggregate_path = scoring_root / "aggregate.json"
        if not aggregate_path.is_file():
            raise RuntimeError(f"evaluator did not emit aggregate: {aggregate_path}")
        protocol_records[protocol] = {
            "run_manifest_hash": manifest["manifest_hash"],
            "artifact_manifest_sha256": package["artifact_manifest"]["sha256"],
            "generation": {
                "expected_jobs": generation["expected_jobs"],
                "succeeded_jobs": generation["succeeded_jobs"],
                "terminal_failed_jobs": generation["terminal_failed_jobs"],
                "truncated_jobs": generation.get("truncated_jobs", 0),
                "run_settled": generation["run_settled"],
                "attempt_count": generation["attempt_count"],
                "attempt_status_counts": generation["attempt_status_counts"],
                "recorded_latency_seconds": generation[
                    "recorded_latency_seconds"
                ],
                "usage_totals": generation["usage_totals"],
            },
            "aggregate_path": aggregate_path.relative_to(output_root).as_posix(),
            "aggregate_sha256": sha256_file(aggregate_path),
        }
    summary = {
        "schema_version": "cadengbench-p.complete-benchmark-run.v1",
        "run_id": run_id,
        "benchmark_plan_hash": benchmark_plan["benchmark_plan_hash"],
        "evalset_release_hash": benchmark_plan["evalset_release_hash"],
        "model_spec_sha256": sha256_file(model_spec_path),
        "input_profile": input_profile,
        "item_count": benchmark_plan.get("item_count", 300),
        "item_ids": benchmark_plan.get("item_ids"),
        "protocols": protocol_records,
        "l3_handoff": {
            "protocol": "zero_to_cad",
            "l02_run": "zero_to_cad/scoring",
            "status": "ready_for_conditional_l3",
        },
        "valid_complete_run": True,
    }
    write_json(output_root / "benchmark_summary.json", summary)
    return summary
