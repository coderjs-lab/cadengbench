"""Build and verify evaluator-facing candidate submission bundles."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .extraction import ExtractionResult
from .io import (
    canonical_json_bytes,
    read_json,
    safe_relative_path,
    semantic_hash,
    sha256_file,
    write_json,
    write_text,
)
from .planner import load_run_manifest
from .schema import validate_candidate_submission

CANDIDATE_SCHEMA = "cadengbench-p.candidate-submission.v1"


def candidate_record_path(
    run_root: Path, item_id: str, sample_index: int, sample_count: int
) -> Path:
    if sample_count == 1:
        return run_root / "submissions" / item_id / "candidate_submission.json"
    return (
        run_root
        / "submissions"
        / item_id
        / f"sample_{sample_index:03d}"
        / "candidate_submission.json"
    )


def artifact_path(
    run_root: Path,
    item_id: str,
    sample_index: int,
    sample_count: int,
    filename: str = "submission.py",
) -> Path:
    if sample_count == 1:
        return run_root / "artifacts" / item_id / filename
    return run_root / "artifacts" / item_id / f"sample_{sample_index:03d}" / filename


def _file_record(run_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(run_root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_candidate_submission(
    *,
    run_root: Path,
    manifest: dict[str, Any],
    request: dict[str, Any],
    sample_index: int,
    attempt_record: dict[str, Any],
    extraction: ExtractionResult,
    raw_response_path: Path,
) -> dict[str, Any]:
    item_id = str(request["item_id"])
    sample_count = int(manifest["sample_count"])
    response_contract = request["response_contract"]
    output = artifact_path(
        run_root,
        item_id,
        sample_index,
        sample_count,
        str(response_contract["filename"]),
    )
    write_text(output, extraction.source)
    artifact = {
        **_file_record(run_root, output),
        "media_type": response_contract["media_type"],
        "interface_version": response_contract["interface_version"],
    }
    candidate_schema = (
        CANDIDATE_SCHEMA
        if request["schema_version"] == "cadengbench-p.canonical-model-request.v1"
        else "cadengbench-p.candidate-submission.v2"
    )
    candidate: dict[str, Any] = {
        "schema_version": candidate_schema,
        "submission_id": f"{manifest['run_id']}:{item_id}:{sample_index}",
        "run_id": manifest["run_id"],
        "item_id": item_id,
        "sample_index": sample_index,
        "request_hash": request["request_hash"],
        "run_manifest_hash": manifest["manifest_hash"],
        "model": {
            "provider": "openai_compatible",
            "requested_model_id": manifest["model"]["model_id"],
            "returned_model_id": attempt_record.get("returned_model_id"),
            "provider_request_id": attempt_record.get("provider_request_id"),
        },
        "successful_attempt": {
            "attempt_index": attempt_record["attempt_index"],
            "ledger_record_hash": attempt_record["record_hash"],
        },
        "raw_response": _file_record(run_root, raw_response_path),
        "extraction": {
            "policy": extraction.policy,
            "method": extraction.method,
            "status": "succeeded",
        },
        "artifact": artifact,
    }
    if candidate_schema == "cadengbench-p.candidate-submission.v2":
        candidate["task_kind"] = request["task_kind"]
    candidate["candidate_hash"] = semantic_hash(candidate, "candidate_hash")
    validate_candidate_submission(candidate)
    path = candidate_record_path(run_root, item_id, sample_index, sample_count)
    write_json(path, candidate)
    return candidate


def load_candidate_submission(path: Path, *, run_root: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"candidate record is not an object: {path}")
    validate_candidate_submission(value)
    for field in ("raw_response", "artifact"):
        record = value[field]
        artifact = safe_relative_path(run_root, record["path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != record["size_bytes"]
            or sha256_file(artifact) != record["sha256"]
        ):
            raise ValueError(f"candidate {field} failed hash verification: {artifact}")
    return value


def _job_key(item_id: str, sample_index: int) -> tuple[str, int]:
    return item_id, sample_index


def package_run(run_root: Path) -> dict[str, Any]:
    manifest = load_run_manifest(run_root)
    sample_count = int(manifest["sample_count"])
    expected = [
        _job_key(record["item_id"], sample_index)
        for record in manifest["requests"]
        for sample_index in range(sample_count)
    ]
    from .ledger import AttemptLedger

    ledger_records = AttemptLedger(run_root / "attempts.jsonl").records()
    by_hash = {record["record_hash"]: record for record in ledger_records}
    candidates: list[dict[str, Any]] = []
    succeeded: set[tuple[str, int]] = set()
    for item_id, sample_index in expected:
        path = candidate_record_path(run_root, item_id, sample_index, sample_count)
        if not path.is_file():
            continue
        candidate = load_candidate_submission(path, run_root=run_root)
        if candidate["run_manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError(f"candidate belongs to another run: {path}")
        key = _job_key(candidate["item_id"], int(candidate["sample_index"]))
        if key in succeeded:
            raise ValueError(f"duplicate candidate job: {key}")
        ledger_hash = candidate["successful_attempt"]["ledger_record_hash"]
        attempt = by_hash.get(ledger_hash)
        if (
            attempt is None
            or attempt["status"] != "succeeded"
            or _job_key(attempt["item_id"], int(attempt["sample_index"])) != key
            or attempt["request_hash"] != candidate["request_hash"]
            or attempt["raw_response_path"] != candidate["raw_response"]["path"]
            or attempt["raw_response_sha256"] != candidate["raw_response"]["sha256"]
        ):
            raise ValueError(
                f"candidate is not bound to one successful attempt: {path}"
            )
        succeeded.add(key)
        candidates.append(candidate)
    candidates.sort(key=lambda value: (value["item_id"], value["sample_index"]))
    manifest_path = run_root / "artifact_manifest.jsonl"
    lines = b"".join(canonical_json_bytes(value) + b"\n" for value in candidates)
    write_text(manifest_path, lines.decode("utf-8"))

    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for record in ledger_records:
        latest[_job_key(record["item_id"], int(record["sample_index"]))] = record
    terminal_failed = {
        key
        for key, record in latest.items()
        if key not in succeeded
        and record["status"]
        in {"retryable_failure", "terminal_failure", "extraction_failure"}
        and record["retry_decision"] == "stop"
    }
    truncated = {
        key
        for key, record in latest.items()
        if record.get("error_class") == "output_truncated"
    }
    pending = set(expected) - succeeded - terminal_failed
    settled = succeeded | terminal_failed
    status_counts = Counter(str(record["status"]) for record in ledger_records)
    usage_totals: dict[str, float] = {}
    for record in ledger_records:
        for key, value in (record.get("usage") or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            usage_totals[str(key)] = usage_totals.get(str(key), 0.0) + float(value)
    summary = {
        "schema_version": "cadengbench-p.candidate-run-summary.v1",
        "run_id": manifest["run_id"],
        "run_manifest_hash": manifest["manifest_hash"],
        "expected_jobs": len(expected),
        "succeeded_jobs": len(succeeded),
        "terminal_failed_jobs": len(terminal_failed),
        "truncated_jobs": len(truncated),
        "settled_jobs": len(settled),
        "pending_jobs": len(pending),
        "run_settled": len(pending) == 0,
        "valid_complete_run": len(succeeded) == len(expected),
        "attempt_count": len(ledger_records),
        "attempt_status_counts": dict(sorted(status_counts.items())),
        "recorded_latency_seconds": sum(
            float(record["latency_seconds"]) for record in ledger_records
        ),
        "usage_totals": dict(sorted(usage_totals.items())),
        "artifact_manifest": {
            "path": manifest_path.relative_to(run_root).as_posix(),
            "sha256": sha256_file(manifest_path),
            "record_count": len(candidates),
        },
    }
    write_json(run_root / "completion_summary.json", summary)
    return summary
