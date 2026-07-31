#!/usr/bin/env python3
"""Precompile every frozen surface-response contract under the HPC runtime."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cadengbench_p_l3_response.io import write_json
from cadengbench_p_l3_response.runner import calibrate_item
from cadengbench_p_l3_response.schemas import load_contract


def _rows(path: Path) -> list[dict]:
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
    output = Path(job["output"])
    contract_path = output / "response_contract.json"
    try:
        if contract_path.is_file():
            contract = load_contract(contract_path)
            return {
                "item_id": item_id,
                "status": "ready",
                "mode": "resumed",
                "contract_sha256": contract["contract_sha256"],
            }
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("nonempty canonical output has no response contract")
        contract = calibrate_item(
            profile_path=Path(job["profile"]),
            card_path=Path(job["card"]),
            reference_step=Path(job["reference"]),
            output_dir=output,
            ccx_path=Path(job["ccx"]),
            timeout_seconds=job["timeout"],
        )
        return {
            "item_id": item_id,
            "status": "ready",
            "mode": "compiled",
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
    args = parser.parse_args()
    rows = [
        row
        for row in _rows(args.applicability_index)
        if row.get("status") == "solver_qualified"
        and row.get("scenario_family")
        in {"axial_tension", "cantilever_transverse"}
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
            }
        )
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_compile, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: result["item_id"])
    summary = {
        "schema_version": "cadengbench-p.l3-surface-response-contract-build.v1",
        "expected_count": len(rows),
        "ready_count": sum(result["status"] == "ready" for result in results),
        "compiled_count": sum(result.get("mode") == "compiled" for result in results),
        "resumed_count": sum(result.get("mode") == "resumed" for result in results),
        "error_count": sum(result["status"] == "error" for result in results),
        "results": results,
    }
    write_json(args.output / "build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ready_count"] == 120 and summary["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
