"""Build identity for the standalone response evaluator."""

from __future__ import annotations

import hashlib
from pathlib import Path

NAME = "cadengbench-p-l3-response"
VERSION = "0.1.0"


def _identity_files() -> list[tuple[str, Path]]:
    package_root = Path(__file__).resolve().parent
    entries = [
        (path.relative_to(package_root).as_posix(), path)
        for path in package_root.rglob("*.py")
    ]
    source_schema_root = package_root.parents[1] / "schemas"
    schema_root = (
        source_schema_root if source_schema_root.is_dir() else package_root / "_schemas"
    )
    entries.extend(
        (f"_schemas/{path.name}", path) for path in schema_root.glob("*.json")
    )
    return sorted(entries, key=lambda entry: entry[0])


def evaluator_build_hash() -> str:
    digest = hashlib.sha256()
    for logical_path, path in _identity_files():
        relative = logical_path.encode("ascii")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def evaluator_identity() -> dict[str, str]:
    return {"name": NAME, "version": VERSION, "build_hash": evaluator_build_hash()}
