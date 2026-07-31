"""Append-only, hash-chained attempt ledger."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from pathlib import Path
from typing import Any

from .io import canonical_json_bytes, semantic_hash

ATTEMPT_SCHEMA = "cadengbench-p.candidate-attempt.v1"
ZERO_HASH = "0" * 64


class LedgerError(ValueError):
    """Attempt ledger is malformed or its hash chain is broken."""


def _validate_record(record: dict[str, Any], previous_hash: str) -> None:
    required = {
        "schema_version",
        "run_id",
        "item_id",
        "sample_index",
        "attempt_index",
        "request_hash",
        "started_at",
        "ended_at",
        "latency_seconds",
        "status",
        "retryable",
        "retry_decision",
        "next_eligible_at",
        "http_status",
        "error_class",
        "provider_request_id",
        "returned_model_id",
        "usage",
        "raw_response_path",
        "raw_response_sha256",
        "previous_record_hash",
        "record_hash",
    }
    if set(record) != required:
        raise LedgerError(
            "attempt ledger record fields differ from the frozen contract"
        )
    if record.get("schema_version") != ATTEMPT_SCHEMA:
        raise LedgerError(f"attempt record must use {ATTEMPT_SCHEMA}")
    if record.get("previous_record_hash") != previous_hash:
        raise LedgerError("attempt ledger previous-record hash is invalid")
    if semantic_hash(record, "record_hash") != record.get("record_hash"):
        raise LedgerError("attempt ledger record hash is invalid")
    if record.get("status") not in {
        "succeeded",
        "retryable_failure",
        "terminal_failure",
        "extraction_failure",
    }:
        raise LedgerError("attempt ledger status is invalid")


class AttemptLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.Lock()
        self._records = self._read_verified()
        self._size_bytes = self.path.stat().st_size if self.path.is_file() else 0

    def _read_verified(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        previous = ZERO_HASH
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise LedgerError(f"blank attempt ledger line: {line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerError(
                        f"invalid attempt ledger JSON at line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise LedgerError(f"attempt line {line_number} is not an object")
                _validate_record(record, previous)
                records.append(record)
                previous = record["record_hash"]
        return records

    def records(self) -> list[dict[str, Any]]:
        with self._thread_lock:
            return [dict(record) for record in self._records]

    def append(self, value: dict[str, Any]) -> dict[str, Any]:
        with self._thread_lock:
            self.path.touch(exist_ok=True)
            with self.path.open("a+", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    # Detect unsupported concurrent writers before extending the chain.
                    current_size = self.path.stat().st_size
                    if current_size != self._size_bytes:
                        on_disk = self._read_verified()
                        self._records = on_disk
                    previous = (
                        self._records[-1]["record_hash"] if self._records else ZERO_HASH
                    )
                    record = dict(value)
                    record["schema_version"] = ATTEMPT_SCHEMA
                    record["previous_record_hash"] = previous
                    record["record_hash"] = semantic_hash(record, "record_hash")
                    _validate_record(record, previous)
                    stream.write(canonical_json_bytes(record).decode("utf-8") + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    self._records.append(record)
                    self._size_bytes = stream.tell()
                    return dict(record)
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
