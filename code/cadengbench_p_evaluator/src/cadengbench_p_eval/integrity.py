"""Hash-bound evidence bundle construction and verification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .io import read_json, sha256_file, sha256_json, write_json

EVIDENCE_MANIFEST_NAME = "evidence_manifest.json"
EVIDENCE_MANIFEST_SCHEMA = "cadengbench-p.evidence-manifest.v1"
RESULT_NAME = "result.json"
RESULT_DIGEST_NAME = "result.sha256"
_EXCLUDED_NAMES = {EVIDENCE_MANIFEST_NAME, RESULT_NAME, RESULT_DIGEST_NAME}


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("evidence paths must be nonempty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe evidence path: {value!r}")
    return value


def _evidence_records(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink():
        raise ValueError("evidence root cannot be a symlink")
    if not root.is_dir():
        raise ValueError(f"evidence root does not exist: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"evidence bundles cannot contain symlinks: {path}")
        if not path.is_file() or (path.parent == root and path.name in _EXCLUDED_NAMES):
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def build_evidence_manifest(root: Path) -> dict[str, Any]:
    """Describe every committed evidence file except result commit metadata."""

    records = _evidence_records(root)
    manifest: dict[str, Any] = {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA,
        "file_count": len(records),
        "files": records,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def write_evidence_manifest(root: Path) -> dict[str, Any]:
    manifest = build_evidence_manifest(root)
    write_json(root / EVIDENCE_MANIFEST_NAME, manifest)
    return manifest


def verify_evidence_manifest(root: Path) -> dict[str, Any]:
    """Verify manifest semantics, exact file roster, size, and SHA-256 values."""

    path = root / EVIDENCE_MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"evidence bundle is missing {EVIDENCE_MANIFEST_NAME}")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("evidence manifest must be an object")
    if value.get("schema_version") != EVIDENCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported evidence manifest schema")
    files = value.get("files")
    if not isinstance(files, list) or value.get("file_count") != len(files):
        raise ValueError("evidence manifest file_count is inconsistent")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("evidence manifest contains a malformed file record")
        relative = _safe_relative_path(record.get("path"))
        if relative in seen:
            raise ValueError(f"evidence manifest repeats path {relative!r}")
        seen.add(relative)
        digest = record.get("sha256")
        size = record.get("size_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"evidence record {relative!r} has an invalid digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"evidence record {relative!r} has an invalid size")
    payload = dict(value)
    recorded_hash = payload.pop("manifest_hash", None)
    if not isinstance(recorded_hash, str) or sha256_json(payload) != recorded_hash:
        raise ValueError("evidence manifest failed its semantic hash validation")
    expected = build_evidence_manifest(root)
    if dict(value) != expected:
        raise ValueError("evidence bundle file roster or content hash has changed")
    return dict(value)
