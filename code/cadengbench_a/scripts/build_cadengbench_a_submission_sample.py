#!/usr/bin/env python3
"""Build the compact seven-family CADEngBench-A submission sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVES = {
    "BallJointType": "joint_set_17859",
    "CylindricalJointType": "joint_set_02236",
    "PinSlotJointType": "joint_set_15485",
    "PlanarJointType": "joint_set_01993",
    "RevoluteJointType": "joint_set_09025",
    "RigidJointType": "joint_set_02495",
    "SliderJointType": "joint_set_18632",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """Materialize and hash the deterministic seven-family development sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "cadengbench_a_core150_public_v1",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "cadengbench_a_core150_v1"
        / "sealed_ground_truth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "submission" / "cadengbench_a_sample7_v2",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.force:
            parser.error(f"output exists: {args.output_dir}; pass --force to rebuild")
        shutil.rmtree(args.output_dir)

    for family, assembly_id in REPRESENTATIVES.items():
        source = args.public_root / "public" / "development" / assembly_id
        label = args.labels_root / "development" / assembly_id / "master.json"
        if not source.is_dir() or not label.is_file():
            parser.error(f"missing source or development label for {assembly_id}")
        target = args.output_dir / "public" / "development" / assembly_id
        shutil.copytree(source, target)
        label_target = (
            args.output_dir
            / "labels"
            / "development"
            / assembly_id
            / "master.json"
        )
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label, label_target)

    prompts_target = args.output_dir / "prompts"
    shutil.copytree(
        REPO_ROOT / "prompts" / "cadengbench_a" / "pair_track_b_v3",
        prompts_target / "a1_pair_track_b_v3",
    )
    shutil.copytree(
        REPO_ROOT / "prompts" / "cadengbench_a" / "a2_kinematics_v1",
        prompts_target / "a2_kinematics_v1",
    )

    files = sorted(path for path in args.output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema": "cadengbench_a_submission_sample_v2",
        "benchmark_layers": ["A0", "A1", "A2"],
        "split": "development",
        "item_count": len(REPRESENTATIVES),
        "joint_family_coverage": REPRESENTATIVES,
        "contains_public_development_labels": True,
        "contains_test_labels": False,
        "files": [
            {
                "path": path.relative_to(args.output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
    _write_json(args.output_dir / "sample_manifest.json", manifest)
    (args.output_dir / "README.md").write_text(
        "# CADEngBench-A Seven-Family Submission Sample\n\n"
        "This compact development-only sample contains one representative item for "
        "each supported joint family, the exact A1/A2 prompts, and development labels "
        "for evaluator reproduction. It contains no held-out test labels.\n\n"
        "Run a no-cost packaging check from the repository root:\n\n"
        "```bash\n"
        "python scripts/run_cadengbench_a_benchmark.py \\\n"
        "  --dataset-root submission/cadengbench_a_sample7_v2 \\\n"
        "  --labels-root submission/cadengbench_a_sample7_v2/labels \\\n"
        "  --provider openai --model YOUR_MODEL --dry-run \\\n"
        "  --output-dir outputs/sample7_dry_run\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"{args.output_dir}: {manifest['total_bytes']} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
