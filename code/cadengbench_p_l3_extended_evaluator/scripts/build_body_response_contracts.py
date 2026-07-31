#!/usr/bin/env python3
"""Precompile all promoted body-response contracts under the pinned HPC runtime."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cadengbench_p_l3_extended.body_response import calibrate_body_response
from cadengbench_p_l3_extended.io import write_json


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reference(release: Path, row: dict) -> Path:
    relative = (
        "source/generated_from_cadquery.step"
        if row["source"] == "benchcad"
        else "source/fusion360_occ_reconstruction.step"
    )
    return release / "items" / row["item_id"] / relative


def _compile(job: dict) -> dict:
    item_id = job["item_id"]
    try:
        contract = calibrate_body_response(
            profile_path=Path(job["profile"]),
            card_path=Path(job["card"]),
            reference_step=Path(job["reference"]),
            output_dir=Path(job["output"]),
            ccx_path=Path(job["ccx"]),
            timeout_seconds=job["timeout"],
            force=job["force"],
        )
        return {
            "item_id": item_id,
            "status": "ready",
            "contract_sha256": contract["contract_sha256"],
        }
    except Exception as exc:
        return {
            "item_id": item_id,
            "status": "error",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--applicability-index", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ccx", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = [
        row
        for row in _read_jsonl(args.applicability_index)
        if row.get("scenario_family") == "single_support_body_acceleration"
        and row.get("status") == "solver_qualified"
    ]
    jobs = []
    for row in rows:
        card = Path(row["physics_card_path"])
        if not card.is_absolute():
            card = args.repo_root / card
        jobs.append(
            {
                "item_id": row["item_id"],
                "profile": str(args.profile),
                "card": str(card),
                "reference": str(_reference(args.release, row)),
                "output": str(args.output / "items" / row["item_id"]),
                "ccx": str(args.ccx),
                "timeout": args.timeout_seconds,
                "force": args.force,
            }
        )
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_compile, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: result["item_id"])
    summary = {
        "schema_version": "cadengbench-p.l3-body-response-contract-build.v1",
        "expected_count": len(rows),
        "ready_count": sum(result["status"] == "ready" for result in results),
        "error_count": sum(result["status"] == "error" for result in results),
        "results": results,
    }
    write_json(args.output / "build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["error_count"] == 0 and summary["ready_count"] == 44 else 2


if __name__ == "__main__":
    raise SystemExit(main())
