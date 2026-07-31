from __future__ import annotations

import argparse
import copy
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cadengbench_p_l3_extended.body_solver import run_body_card
from cadengbench_p_l3_extended.geometry import source_step
from cadengbench_p_l3_extended.io import read_json, write_csv, write_json

LEVELS = {
    "coarse": (18, 4),
    "production": (24, 5),
    "fine": (32, 7),
}


def _log_error(observed: float, expected: float) -> float:
    return abs(math.log(max(observed, 1e-30) / max(expected, 1e-30)))


def _select(body_root: Path, audit_root: Path) -> list[str]:
    summary = read_json(body_root / "body_qualification_summary.json")
    audit = {
        value["item_id"]: value
        for value in (
            json.loads(line)
            for line in (audit_root / "coverage_audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    values = sorted(
        (
            {
                "item_id": item_id,
                "source": audit[item_id]["source"],
                "size": float(audit[item_id]["bbox_diagonal_mm"]),
            }
            for item_id in summary["promoted_item_ids"]
        ),
        key=lambda value: (value["size"], value["item_id"]),
    )
    targets = (0.25, 0.5, 0.75)
    selected = []
    for index, quantile in enumerate(targets):
        target = values[round((len(values) - 1) * quantile)]["size"]
        required_source = "benchcad" if index != 1 else "fusion360"
        candidates = [
            value
            for value in values
            if value["source"] == required_source
            and value["item_id"] not in selected
        ]
        selected.append(
            min(
                candidates,
                key=lambda value: (
                    abs(math.log(value["size"] / target)),
                    value["item_id"],
                ),
            )["item_id"]
        )
    return selected


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    card = copy.deepcopy(read_json(Path(payload["card_path"])))
    card.pop("qualification", None)
    card["status"] = "geometry_qualified"
    level = str(payload["level"])
    long_axis, short_axis = LEVELS[level]
    card["mesh"]["target_long_axis_divisions"] = long_axis
    card["mesh"]["target_short_axis_divisions"] = short_axis
    card["card_id"] = f"{card['item_id']}:body-mesh-study:{level}"
    result = run_body_card(
        card,
        step_path=Path(payload["step_path"]),
        output_dir=Path(payload["output"]),
        ccx_path=Path(payload["ccx"]),
        timeout_seconds=int(payload["timeout_seconds"]),
    )
    metrics = result.get("metrics") or {}
    stress = metrics.get("regularized_von_mises_mpa") or {}
    mesh = (result.get("stages") or {}).get("mesh") or {}
    return {
        "item_id": card["item_id"],
        "source": payload["source"],
        "level": level,
        "final_status": result.get("final_status"),
        "element_count": mesh.get("element_count"),
        "minimum_sicn": mesh.get("minimum_quality"),
        "regularized_stress_p95_mpa": stress.get("p95"),
        "normalized_displacement": metrics.get("normalized_max_displacement"),
        "equilibrium_error": metrics.get("force_equilibrium_relative_error"),
        "runtime_seconds": result.get("runtime_seconds"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    body = args.body_root.resolve()
    audit = args.audit_root.resolve()
    selected = _select(body, audit)
    payloads = []
    for item_id in selected:
        source = "benchcad" if item_id.startswith("p_benchcad_") else "fusion360"
        for level in LEVELS:
            payloads.append(
                {
                    "item_id": item_id,
                    "source": source,
                    "level": level,
                    "card_path": str(
                        body / "items" / item_id / "body_card.solver_qualified.json"
                    ),
                    "step_path": str(source_step(repo, item_id)),
                    "output": str(args.output / "items" / item_id / level),
                    "ccx": str(args.ccx),
                    "timeout_seconds": args.timeout_seconds,
                }
            )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["item_id"], row["level"]))
    by_item = {
        item_id: {
            row["level"]: row for row in rows if row["item_id"] == item_id
        }
        for item_id in selected
    }
    comparisons = []
    for item_id in selected:
        production = by_item[item_id]["production"]
        fine = by_item[item_id]["fine"]
        stress_error = _log_error(
            float(production["regularized_stress_p95_mpa"]),
            float(fine["regularized_stress_p95_mpa"]),
        )
        displacement_error = _log_error(
            float(production["normalized_displacement"]),
            float(fine["normalized_displacement"]),
        )
        comparisons.append(
            {
                "item_id": item_id,
                "source": production["source"],
                "stress_p95_symmetric_log_error": stress_error,
                "normalized_displacement_symmetric_log_error": displacement_error,
                "stress_gate": stress_error <= math.log(1.25),
                "displacement_gate": displacement_error <= math.log(1.10),
            }
        )
    write_csv(args.output / "mesh_study_runs.csv", rows)
    write_csv(args.output / "mesh_study_comparisons.csv", comparisons)
    summary = {
        "schema_version": "cadengbench-p.l3-body-mesh-study.v1",
        "selected_item_ids": selected,
        "run_count": len(rows),
        "probe_pass_count": sum(row["final_status"] == "probe_pass" for row in rows),
        "comparison_count": len(comparisons),
        "stress_gate_pass_count": sum(row["stress_gate"] for row in comparisons),
        "displacement_gate_pass_count": sum(
            row["displacement_gate"] for row in comparisons
        ),
        "all_gates_passed": all(
            row["stress_gate"] and row["displacement_gate"] for row in comparisons
        )
        and all(row["final_status"] == "probe_pass" for row in rows),
    }
    write_json(args.output / "mesh_study_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--body-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ccx", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
