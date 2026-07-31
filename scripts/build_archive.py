#!/usr/bin/env python3
"""Refresh manifests and build a deterministic sub-50 MB review archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 50 * 1024 * 1024
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv-a",
    ".venv-p",
    ".solver",
    "__pycache__",
    "outputs",
}
EXCLUDED_FILES = {".DS_Store", ".env"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload_files() -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(ROOT):
        directories[:] = sorted(
            name for name in directories if name not in EXCLUDED_DIRECTORIES
        )
        current_path = Path(current)
        for name in sorted(names):
            path = current_path / name
            if name in EXCLUDED_FILES or path.suffix == ".pyc":
                continue
            files.append(path)
    return sorted(files)


def refresh_a_manifest() -> None:
    sample = ROOT / "datasets/cadengbench_a_sample7_v2"
    manifest_path = sample / "sample_manifest.json"
    old = json.loads(manifest_path.read_text())
    files = sorted(
        path for path in sample.rglob("*") if path.is_file() and path != manifest_path
    )
    old["files"] = [
        {
            "path": path.relative_to(sample).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    old["total_bytes"] = sum(row["bytes"] for row in old["files"])
    write_json(manifest_path, old)


def refresh_package_manifest() -> None:
    manifest_path = ROOT / "package_manifest.json"
    files = [path for path in payload_files() if path != manifest_path]
    manifest = {
        "schema": "cadengbench_aaai27_code_data_supplement_v1",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    manifest["file_count"] = len(manifest["files"])
    manifest["total_uncompressed_bytes"] = sum(
        row["bytes"] for row in manifest["files"]
    )
    write_json(manifest_path, manifest)


def build(output: Path) -> None:
    members = payload_files()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in members:
            relative = Path(ROOT.name) / path.relative_to(ROOT)
            info = zipfile.ZipInfo(
                relative.as_posix(), date_time=(2026, 7, 31, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.suffix in {".sh", ".py"} else 0o644
            info.external_attr = mode << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    if output.stat().st_size >= MAX_BYTES:
        raise RuntimeError(
            f"archive is {output.stat().st_size} bytes; limit is {MAX_BYTES}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refresh_a_manifest()
    refresh_package_manifest()
    output = args.output.resolve()
    build(output)
    print(f"{output}: {output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
