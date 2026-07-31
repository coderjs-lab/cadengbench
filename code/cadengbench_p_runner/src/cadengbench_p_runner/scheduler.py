"""Bounded generation scheduler with durable retry and resume semantics."""

from __future__ import annotations

import hashlib
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .extraction import (
    ExtractionError,
    ExtractionResult,
    extract_artifact,
    response_content,
    response_finish_reason,
)
from .io import read_json, safe_relative_path, sha256_file, write_json, write_text
from .ledger import AttemptLedger
from .packaging import (
    candidate_record_path,
    load_candidate_submission,
    package_run,
    write_candidate_submission,
)
from .planner import (
    dependency_lock_hash,
    load_run_manifest,
    model_from_manifest,
    runner_build_hash,
)
from .providers.base import ProviderResult
from .providers.openai_compatible import OpenAICompatibleAdapter
from .schema import validate_canonical_request

RAW_ENVELOPE_SCHEMA = "cadengbench-p.raw-provider-envelope.v1"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("attempt timestamp has no timezone")
    return parsed


@dataclass(frozen=True, order=True)
class Job:
    item_id: str
    sample_index: int
    request_path: str
    request_hash: str


@dataclass(frozen=True)
class JobOutcome:
    job: Job
    status: str
    attempt_count: int
    message: str = ""


class RequestPacer:
    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            if delay:
                time.sleep(delay)
            self._next_start = time.monotonic() + self.interval


def _raw_path(run_root: Path, job: Job, attempt_index: int) -> Path:
    return (
        run_root
        / "raw_responses"
        / job.item_id
        / f"sample_{job.sample_index:03d}"
        / f"attempt_{attempt_index:04d}.json"
    )


def _records_for(records: list[dict[str, Any]], job: Job) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["item_id"] == job.item_id
        and int(record["sample_index"]) == job.sample_index
    ]


def _stable_jitter(job: Job, attempt_index: int) -> float:
    payload = f"{job.item_id}:{job.sample_index}:{attempt_index}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return (value / (2**32 - 1)) * 0.5


def _retry_delay(job: Job, attempt_index: int, result: ProviderResult) -> float:
    exponential = min(60.0, 2.0 ** max(0, attempt_index - 1))
    return max(
        result.retry_after_seconds or 0.0,
        exponential + _stable_jitter(job, attempt_index),
    )


def _remaining_delay(record: dict[str, Any]) -> float:
    raw = record.get("next_eligible_at")
    if not isinstance(raw, str):
        return 0.0
    return max(0.0, (_parse_time(raw) - _utc_now()).total_seconds())


def _request_for(run_root: Path, job: Job) -> dict[str, Any]:
    path = safe_relative_path(run_root, job.request_path)
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"canonical request is not an object: {path}")
    validate_canonical_request(value)
    if value["item_id"] != job.item_id or value["request_hash"] != job.request_hash:
        raise ValueError(f"canonical request differs from run manifest: {path}")
    return value


def _provider_result(envelope: dict[str, Any]) -> ProviderResult:
    raw = envelope.get("provider_result")
    if not isinstance(raw, dict):
        raise ValueError("raw provider envelope has no provider_result")
    return ProviderResult(
        outcome=str(raw.get("outcome")),
        retryable=bool(raw.get("retryable")),
        http_status=raw.get("http_status"),
        error_class=raw.get("error_class"),
        error_message=raw.get("error_message"),
        provider_request_id=raw.get("provider_request_id"),
        returned_model_id=raw.get("returned_model_id"),
        usage=dict(raw.get("usage") or {}),
        body=raw.get("body"),
        response_headers=dict(raw.get("response_headers") or {}),
        provider_request_hash=raw.get("provider_request_hash"),
        retry_after_seconds=raw.get("retry_after_seconds"),
    )


def _attempt_elapsed(records: list[dict[str, Any]], envelope: dict[str, Any]) -> float:
    starts = [_parse_time(record["started_at"]) for record in records]
    starts.append(_parse_time(envelope["started_at"]))
    return max(0.0, (_parse_time(envelope["ended_at"]) - min(starts)).total_seconds())


def _finalize_envelope(
    *,
    run_root: Path,
    manifest: dict[str, Any],
    request: dict[str, Any],
    job: Job,
    envelope_path: Path,
    envelope: dict[str, Any],
    ledger: AttemptLedger,
) -> tuple[dict[str, Any], float | None]:
    if envelope.get("schema_version") != RAW_ENVELOPE_SCHEMA:
        raise ValueError(f"unsupported raw envelope: {envelope_path}")
    if (
        envelope.get("run_id") != manifest["run_id"]
        or envelope.get("item_id") != job.item_id
        or envelope.get("sample_index") != job.sample_index
        or envelope.get("request_hash") != job.request_hash
    ):
        raise ValueError(f"raw envelope identity mismatch: {envelope_path}")
    attempt_index = int(envelope["attempt_index"])
    result = _provider_result(envelope)
    extraction: ExtractionResult | None = None
    extraction_error: str | None = None
    finish_reason = response_finish_reason(result.body)
    output_truncated = (
        finish_reason is not None
        and finish_reason.strip().lower()
        in {"length", "max_tokens", "max_output_tokens"}
    )
    if result.succeeded and not output_truncated:
        try:
            extraction = extract_artifact(result.body, request["response_contract"])
        except ExtractionError as exc:
            extraction_error = str(exc)

    existing = _records_for(ledger.records(), job)
    if any(int(record["attempt_index"]) == attempt_index for record in existing):
        raise ValueError(f"attempt is already in the ledger: {envelope_path}")
    execution = manifest["execution"]
    attempts_remain = attempt_index < int(execution["max_attempts"])
    within_elapsed = _attempt_elapsed(existing, envelope) < int(
        execution["max_retry_elapsed_seconds"]
    )
    if extraction is not None:
        status = "succeeded"
        retryable = False
        retry_decision = "none"
        delay = None
    elif output_truncated:
        try:
            raw_text = response_content(result.body)
        except ExtractionError:
            raw_text = ""
        if raw_text:
            write_text(
                run_root / "artifacts" / job.item_id / "response.txt",
                raw_text,
            )
        status = "terminal_failure"
        retryable = False
        retry_decision = "stop"
        delay = None
    elif extraction_error is not None:
        try:
            raw_text = response_content(result.body)
        except ExtractionError:
            raw_text = ""
        if raw_text:
            write_text(
                run_root / "artifacts" / job.item_id / "response.txt",
                raw_text,
            )
        status = "extraction_failure"
        retryable = False
        retry_decision = "stop"
        delay = None
    elif result.retryable:
        status = "retryable_failure"
        retryable = True
        proposed_delay = _retry_delay(job, attempt_index, result)
        elapsed = _attempt_elapsed(existing, envelope)
        retry_decision = (
            "retry"
            if attempts_remain
            and within_elapsed
            and elapsed + proposed_delay <= int(execution["max_retry_elapsed_seconds"])
            else "stop"
        )
        delay = proposed_delay if retry_decision == "retry" else None
    else:
        status = "terminal_failure"
        retryable = False
        retry_decision = "stop"
        delay = None
    next_eligible = (
        (_utc_now() + timedelta(seconds=delay)).isoformat()
        if delay is not None
        else None
    )
    record = ledger.append(
        {
            "run_id": manifest["run_id"],
            "item_id": job.item_id,
            "sample_index": job.sample_index,
            "attempt_index": attempt_index,
            "request_hash": job.request_hash,
            "started_at": envelope["started_at"],
            "ended_at": envelope["ended_at"],
            "latency_seconds": float(envelope["latency_seconds"]),
            "status": status,
            "retryable": retryable,
            "retry_decision": retry_decision,
            "next_eligible_at": next_eligible,
            "http_status": result.http_status,
            "error_class": "extraction_error"
            if extraction_error is not None
            else "output_truncated"
            if output_truncated
            else result.error_class,
            "provider_request_id": result.provider_request_id,
            "returned_model_id": result.returned_model_id,
            "usage": result.usage,
            "raw_response_path": envelope_path.relative_to(run_root).as_posix(),
            "raw_response_sha256": sha256_file(envelope_path),
        }
    )
    if extraction is not None:
        write_candidate_submission(
            run_root=run_root,
            manifest=manifest,
            request=request,
            sample_index=job.sample_index,
            attempt_record=record,
            extraction=extraction,
            raw_response_path=envelope_path,
        )
    return record, delay


def _recover_success(
    *,
    run_root: Path,
    manifest: dict[str, Any],
    request: dict[str, Any],
    job: Job,
    record: dict[str, Any],
) -> None:
    envelope_path = safe_relative_path(run_root, record["raw_response_path"])
    if (
        not envelope_path.is_file()
        or sha256_file(envelope_path) != record["raw_response_sha256"]
    ):
        raise ValueError("successful ledger record lost its raw response")
    envelope = read_json(envelope_path)
    result = _provider_result(envelope)
    extraction = extract_artifact(result.body, request["response_contract"])
    write_candidate_submission(
        run_root=run_root,
        manifest=manifest,
        request=request,
        sample_index=job.sample_index,
        attempt_record=record,
        extraction=extraction,
        raw_response_path=envelope_path,
    )


def _run_job(
    *,
    run_root: Path,
    manifest: dict[str, Any],
    job: Job,
    adapter: OpenAICompatibleAdapter,
    ledger: AttemptLedger,
    pacer: RequestPacer,
) -> JobOutcome:
    request = _request_for(run_root, job)
    sample_count = int(manifest["sample_count"])
    candidate_path = candidate_record_path(
        run_root, job.item_id, job.sample_index, sample_count
    )
    records = _records_for(ledger.records(), job)
    successful = next(
        (record for record in reversed(records) if record["status"] == "succeeded"),
        None,
    )
    if candidate_path.is_file():
        candidate = load_candidate_submission(candidate_path, run_root=run_root)
        if candidate["request_hash"] != job.request_hash:
            raise ValueError("existing candidate request hash differs")
        return JobOutcome(job, "succeeded", len(records), "reused candidate")
    if successful is not None:
        _recover_success(
            run_root=run_root,
            manifest=manifest,
            request=request,
            job=job,
            record=successful,
        )
        return JobOutcome(job, "succeeded", len(records), "recovered from raw response")

    recorded_attempts = {int(record["attempt_index"]) for record in records}
    raw_directory = _raw_path(run_root, job, 1).parent
    for path in (
        sorted(raw_directory.glob("attempt_*.json")) if raw_directory.is_dir() else []
    ):
        attempt_index = int(path.stem.split("_")[-1])
        if attempt_index in recorded_attempts:
            continue
        envelope = read_json(path)
        record, delay = _finalize_envelope(
            run_root=run_root,
            manifest=manifest,
            request=request,
            job=job,
            envelope_path=path,
            envelope=envelope,
            ledger=ledger,
        )
        records.append(record)
        recorded_attempts.add(attempt_index)
        if record["status"] == "succeeded":
            return JobOutcome(
                job, "succeeded", len(records), "recovered orphan envelope"
            )
        if record["retry_decision"] == "stop":
            return JobOutcome(job, "failed", len(records), str(record["error_class"]))
        if delay:
            time.sleep(delay)

    if records and records[-1]["retry_decision"] == "stop":
        return JobOutcome(job, "failed", len(records), str(records[-1]["error_class"]))
    if records and records[-1]["retry_decision"] == "retry":
        remaining = _remaining_delay(records[-1])
        if remaining:
            time.sleep(remaining)

    maximum = int(manifest["execution"]["max_attempts"])
    timeout = int(manifest["execution"]["request_timeout_seconds"])
    attempt_index = max(recorded_attempts, default=0) + 1
    while attempt_index <= maximum:
        pacer.wait()
        started = _utc_now()
        tick = time.monotonic()
        result = adapter.generate(request, run_root=run_root, timeout_seconds=timeout)
        ended = _utc_now()
        envelope = {
            "schema_version": RAW_ENVELOPE_SCHEMA,
            "run_id": manifest["run_id"],
            "item_id": job.item_id,
            "sample_index": job.sample_index,
            "attempt_index": attempt_index,
            "request_hash": job.request_hash,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "latency_seconds": time.monotonic() - tick,
            "provider_result": result.as_dict(),
        }
        path = _raw_path(run_root, job, attempt_index)
        if path.exists():
            raise ValueError(f"refusing to overwrite raw provider response: {path}")
        write_json(path, envelope)
        record, delay = _finalize_envelope(
            run_root=run_root,
            manifest=manifest,
            request=request,
            job=job,
            envelope_path=path,
            envelope=envelope,
            ledger=ledger,
        )
        records.append(record)
        if record["status"] == "succeeded":
            return JobOutcome(job, "succeeded", len(records))
        if record["retry_decision"] == "stop":
            return JobOutcome(job, "failed", len(records), str(record["error_class"]))
        if delay:
            time.sleep(delay)
        attempt_index += 1
    return JobOutcome(job, "failed", len(records), "attempt limit exhausted")


def _jobs(manifest: dict[str, Any]) -> list[Job]:
    return [
        Job(
            item_id=record["item_id"],
            sample_index=sample_index,
            request_path=record["request_path"],
            request_hash=record["request_hash"],
        )
        for record in manifest["requests"]
        for sample_index in range(int(manifest["sample_count"]))
    ]


def _verify_and_record_environment(run_root: Path, manifest: dict[str, Any]) -> None:
    expected = manifest["runner"]
    current_build = runner_build_hash()
    current_lock = dependency_lock_hash()
    if current_build != expected["build_hash"]:
        raise ValueError("runner source differs from the frozen run manifest")
    if current_lock != expected["dependency_lock_sha256"]:
        raise ValueError("runner dependency lock differs from the frozen run manifest")
    record = {
        "schema_version": "cadengbench-p.runner-environment.v1",
        "run_manifest_hash": manifest["manifest_hash"],
        "runner_build_hash": current_build,
        "dependency_lock_sha256": current_lock,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    path = run_root / "execution_environment.json"
    if path.is_file():
        if read_json(path) != record:
            raise ValueError("resume execution environment differs from the first run")
    else:
        write_json(path, record)


def run_generation(run_root: Path) -> dict[str, Any]:
    manifest = load_run_manifest(run_root)
    _verify_and_record_environment(run_root, manifest)
    model = model_from_manifest(manifest)
    adapter = OpenAICompatibleAdapter(model)
    ledger = AttemptLedger(run_root / "attempts.jsonl")
    pacer = RequestPacer(int(manifest["execution"]["requests_per_minute"]))
    outcomes: list[JobOutcome] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=int(manifest["execution"]["max_concurrency"]),
        thread_name_prefix="cadengbench-candidate",
    ) as executor:
        futures = {
            executor.submit(
                _run_job,
                run_root=run_root,
                manifest=manifest,
                job=job,
                adapter=adapter,
                ledger=ledger,
                pacer=pacer,
            ): job
            for job in _jobs(manifest)
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as exc:
                failure = {
                    "item_id": job.item_id,
                    "sample_index": job.sample_index,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                failures.append(failure)
                write_json(
                    run_root
                    / "runner_failures"
                    / job.item_id
                    / f"sample_{job.sample_index:03d}.json",
                    failure,
                )
    summary = package_run(run_root)
    summary["runner_failures"] = len(failures)
    summary["job_outcomes"] = [
        {
            "item_id": value.job.item_id,
            "sample_index": value.job.sample_index,
            "status": value.status,
            "attempt_count": value.attempt_count,
            "message": value.message,
        }
        for value in sorted(outcomes, key=lambda value: value.job)
    ]
    summary["valid_complete_run"] = bool(summary["valid_complete_run"] and not failures)
    summary["run_settled"] = bool(summary["run_settled"] and not failures)
    write_json(run_root / "generation_summary.json", summary)
    return summary
