#!/usr/bin/env python3
"""Verify a full Core-150 package or the compact submission sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected: dict[str, Any]) -> list[str]:
    errors = []
    if not path.is_file():
        return [f"missing: {path}"]
    if path.stat().st_size != expected["bytes"]:
        errors.append(f"size mismatch: {path}")
    if _sha256(path) != expected["sha256"]:
        errors.append(f"sha256 mismatch: {path}")
    return errors


def _verify_core150(
    root: Path,
    manifest: dict[str, Any],
    labels_root: Path | None = None,
) -> list[str]:
    errors = []
    for item in manifest["items"]:
        base = root / "public" / item["partition"] / item["assembly_id"]
        for record in item["public_files"]:
            errors.extend(_verify_file(base / record["relative_path"], record))
        if labels_root is not None:
            label_path = (
                labels_root
                / item["partition"]
                / item["assembly_id"]
                / "master.json"
            )
            errors.extend(
                _verify_file(
                    label_path,
                    {
                        "bytes": item["sealed_ground_truth_bytes"],
                        "sha256": item["sealed_ground_truth_sha256"],
                    },
                )
            )
    if len(manifest["items"]) != manifest["item_count"]:
        errors.append("item_count does not match manifest item records")
    return errors


def _verify_sample(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors = []
    for record in manifest["files"]:
        errors.extend(_verify_file(root / record["path"], record))
    return errors


def main() -> int:
    """Verify every byte declared by a supported release manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument(
        "--labels-root",
        type=Path,
        help="Optional evaluator-only labels root; verifies every sealed master.json.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.package_root / "package_manifest.json"
    if not manifest_path.is_file():
        manifest_path = args.package_root / "sample_manifest.json"
    if not manifest_path.is_file():
        parser.error("package has no package_manifest.json or sample_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] == "cadengbench_a_submission_sample_v2":
        errors = _verify_sample(args.package_root, manifest)
    else:
        errors = _verify_core150(args.package_root, manifest, args.labels_root)
    report = {
        "schema": "cadengbench_a_release_verification_v2",
        "package": str(args.package_root),
        "manifest": str(manifest_path),
        "pass": not errors,
        "error_count": len(errors),
        "errors": errors,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
