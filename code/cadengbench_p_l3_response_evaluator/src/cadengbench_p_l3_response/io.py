"""Deterministic JSON, hashing, and evidence helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

EVIDENCE_MANIFEST = "evidence_manifest.json"
RESULT_DIGEST = "result.sha256"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_json_bytes(value))


def write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def write_evidence_commit(root: Path, result_path: Path) -> None:
    resolved_root = root.resolve()
    resolved_result = result_path.resolve()
    if not resolved_result.is_file() or resolved_root not in resolved_result.parents:
        raise ValueError("result must be a file inside the evidence root")
    excluded = {EVIDENCE_MANIFEST, RESULT_DIGEST}
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in excluded:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "cadengbench-p.l3-response-evidence-manifest.v1",
        "result_path": resolved_result.relative_to(resolved_root).as_posix(),
        "files": records,
    }
    write_json(root / EVIDENCE_MANIFEST, manifest)
    write_text(root / RESULT_DIGEST, f"{sha256_file(result_path)}\n")


def verify_evidence_commit(root: Path) -> None:
    manifest_path = root / EVIDENCE_MANIFEST
    digest_path = root / RESULT_DIGEST
    manifest = read_json(manifest_path)
    result = root / str(manifest["result_path"])
    if not result.is_file():
        raise ValueError("committed result is missing")
    if digest_path.read_text(encoding="ascii").strip() != sha256_file(result):
        raise ValueError("committed result digest changed")
    expected = {str(record["path"]): record for record in manifest["files"]}
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {EVIDENCE_MANIFEST, RESULT_DIGEST}
    }
    if set(expected) != set(observed):
        raise ValueError("evidence file roster changed")
    for relative, path in observed.items():
        record = expected[relative]
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"evidence changed: {relative}")
