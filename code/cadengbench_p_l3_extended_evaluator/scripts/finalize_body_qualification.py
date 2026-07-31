from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cadengbench_p_l3_extended.contracts import validate_body_card
from cadengbench_p_l3_extended.io import (
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from cadengbench_p_l3_extended.qualification import _balanced_order


def _row(
    *,
    item_id: str,
    result: dict[str, Any] | None,
    missing_reason: str | None,
) -> dict[str, Any]:
    metrics = (result or {}).get("metrics") or {}
    failure = (result or {}).get("failure") or {}
    stress = metrics.get("regularized_von_mises_mpa") or {}
    passed = (result or {}).get("final_status") == "probe_pass"
    return {
        "item_id": item_id,
        "source": "benchcad" if item_id.startswith("p_benchcad_") else "fusion360",
        "status": "solver_qualified" if passed else "not_qualified",
        "final_status": (result or {}).get("final_status", "mesh_timeout"),
        "failure_stage": failure.get("stage", "mesh" if result is None else None),
        "failure_message": failure.get("message", missing_reason),
        "normalized_max_displacement": metrics.get("normalized_max_displacement"),
        "regularized_von_mises_p95_mpa": stress.get("p95"),
        "force_equilibrium_relative_error": metrics.get(
            "force_equilibrium_relative_error"
        ),
        "runtime_seconds": (result or {}).get("runtime_seconds"),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = args.audit_root.resolve()
    body_root = args.body_root.resolve()
    records = [
        json.loads(line)
        for line in (audit_root / "coverage_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    candidates = _balanced_order(
        [record for record in records if record["extension_candidate"]]
    )[: args.item_limit]
    rows = []
    promoted = []
    counts = Counter()
    for record in candidates:
        item_id = str(record["item_id"])
        result_path = body_root / "items" / item_id / "source_qualification/result.json"
        result = read_json(result_path) if result_path.is_file() else None
        missing_reason = (
            f"mesh_timeout_after_job_{args.source_job_id}"
            if result is None
            else None
        )
        row = _row(
            item_id=item_id,
            result=result,
            missing_reason=missing_reason,
        )
        rows.append(row)
        counts[row["status"]] += 1
        if row["failure_stage"]:
            counts[f"failure:{row['failure_stage']}"] += 1
        if row["status"] != "solver_qualified":
            continue
        card_path = body_root / "items" / item_id / "body_card.json"
        card = copy.deepcopy(read_json(card_path))
        card["status"] = "solver_qualified"
        card["qualification"] = {
            "promoted": True,
            "reference_step_sha256": result["stages"]["geometry"]["step_sha256"],
            "result_sha256": sha256_file(result_path),
            "source_probe_metrics": {
                "normalized_max_displacement": result["metrics"][
                    "normalized_max_displacement"
                ],
                "regularized_von_mises_p95_mpa": result["metrics"][
                    "regularized_von_mises_mpa"
                ]["p95"],
                "force_equilibrium_relative_error": result["metrics"][
                    "force_equilibrium_relative_error"
                ],
            },
        }
        validate_body_card(card)
        write_json(
            body_root / "items" / item_id / "body_card.solver_qualified.json",
            card,
        )
        promoted.append(item_id)
    write_csv(body_root / "body_qualification.csv", rows)
    source_qualified = Counter(
        row["source"] for row in rows if row["status"] == "solver_qualified"
    )
    summary = {
        "schema_version": "cadengbench-p.l3-body-qualification-summary.v1",
        "qualification_job_id": args.source_job_id,
        "predeclared_tranche_count": args.item_limit,
        "attempted_count": len(rows),
        "completed_result_count": sum(
            row["final_status"] != "mesh_timeout" for row in rows
        ),
        "mesh_timeout_count": sum(
            row["final_status"] == "mesh_timeout" for row in rows
        ),
        "solver_qualified_count": len(promoted),
        "solver_qualified_by_source": dict(sorted(source_qualified.items())),
        "coverage_with_frozen_v1_count": 120 + len(promoted),
        "coverage_with_frozen_v1_fraction": (120 + len(promoted)) / 300,
        "target_50_percent_met": 120 + len(promoted) >= 150,
        "status_counts": dict(sorted(counts.items())),
        "promoted_item_ids": promoted,
        "core_dataset_mutated": False,
    }
    write_json(body_root / "body_qualification_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--body-root", type=Path, required=True)
    parser.add_argument("--item-limit", type=int, default=60)
    parser.add_argument("--source-job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
