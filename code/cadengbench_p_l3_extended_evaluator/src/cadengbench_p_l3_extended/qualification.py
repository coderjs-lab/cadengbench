"""Bounded source qualification and promotion for body-acceleration cards."""

from __future__ import annotations

import copy
import json
import shutil
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cadengbench_p_l3_extended.annotations import compile_body_cards
from cadengbench_p_l3_extended.body_solver import run_body_card
from cadengbench_p_l3_extended.contracts import validate_body_card
from cadengbench_p_l3_extended.geometry import source_step
from cadengbench_p_l3_extended.io import read_json, sha256_file, write_csv, write_json
from cadengbench_p_l3_extended.provenance import evaluator_build_hash


def _balanced_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, deque[dict[str, Any]]] = {}
    for source in ("benchcad", "fusion360"):
        values = sorted(
            (record for record in records if record["source"] == source),
            key=lambda record: (
                float(record["bbox_diagonal_mm"]),
                str(record["item_id"]),
            ),
        )
        groups[source] = deque(values)
    ordered = []
    while any(groups.values()):
        for source in ("benchcad", "fusion360"):
            if groups[source]:
                ordered.append(groups[source].popleft())
    return ordered


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    card = read_json(Path(payload["card_path"]))
    run_dir = Path(payload["run_dir"])
    if run_dir.is_dir():
        shutil.rmtree(run_dir)
    result = run_body_card(
        card,
        step_path=Path(payload["step_path"]),
        output_dir=run_dir,
        ccx_path=Path(payload["ccx_path"]),
        timeout_seconds=int(payload["timeout_seconds"]),
        render=bool(payload["render"]),
    )
    return {"item_id": card["item_id"], "result": result}


def qualify_body_cards(
    *,
    repo_root: Path,
    profile_path: Path,
    audit_root: Path,
    output_root: Path,
    ccx_path: Path,
    workers: int,
    timeout_seconds: int,
    item_limit: int | None,
    render_count: int,
) -> dict[str, Any]:
    compile_body_cards(
        profile_path=profile_path,
        audit_root=audit_root,
        output_root=output_root,
    )
    records = [
        json.loads(line)
        for line in (audit_root / "coverage_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    candidates = _balanced_order(
        [record for record in records if record["extension_candidate"]]
    )
    if item_limit is not None:
        candidates = candidates[:item_limit]
    payloads = []
    rows = []
    for index, record in enumerate(candidates):
        item_id = record["item_id"]
        run_dir = output_root / "items" / item_id / "source_qualification"
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            if result.get("evaluator_build_hash") == evaluator_build_hash():
                rows.append({"item_id": item_id, "result": result, "resumed": True})
                continue
        payloads.append(
            {
                "card_path": str(output_root / "items" / item_id / "body_card.json"),
                "step_path": str(source_step(repo_root, item_id)),
                "run_dir": str(run_dir),
                "ccx_path": str(ccx_path),
                "timeout_seconds": timeout_seconds,
                "render": index < render_count,
            }
        )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, payload): payload for payload in payloads}
        for future in as_completed(futures):
            value = future.result()
            rows.append({**value, "resumed": False})

    output_rows = []
    counts = Counter()
    promoted = []
    for row in sorted(rows, key=lambda value: value["item_id"]):
        item_id = row["item_id"]
        result = row["result"]
        final = str(result["final_status"])
        failure_stage = (result.get("failure") or {}).get("stage")
        promoted_status = final == "probe_pass"
        counts["solver_qualified" if promoted_status else "not_qualified"] += 1
        if failure_stage:
            counts[f"failure:{failure_stage}"] += 1
        if promoted_status:
            card_path = output_root / "items" / item_id / "body_card.json"
            card = copy.deepcopy(read_json(card_path))
            card["status"] = "solver_qualified"
            card["qualification"] = {
                "promoted": True,
                "reference_step_sha256": result["stages"]["geometry"]["step_sha256"],
                "result_sha256": sha256_file(
                    output_root
                    / "items"
                    / item_id
                    / "source_qualification/result.json"
                ),
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
            promoted_path = (
                output_root / "items" / item_id / "body_card.solver_qualified.json"
            )
            write_json(promoted_path, card)
            promoted.append(item_id)
        output_rows.append(
            {
                "item_id": item_id,
                "source": (
                    "benchcad" if item_id.startswith("p_benchcad_") else "fusion360"
                ),
                "status": "solver_qualified" if promoted_status else "not_qualified",
                "final_status": final,
                "failure_stage": failure_stage,
                "failure_message": (result.get("failure") or {}).get("message"),
                "normalized_max_displacement": (
                    (result.get("metrics") or {}).get(
                        "normalized_max_displacement"
                    )
                ),
                "regularized_von_mises_p95_mpa": (
                    ((result.get("metrics") or {}).get(
                        "regularized_von_mises_mpa"
                    ) or {}).get("p95")
                ),
                "force_equilibrium_relative_error": (
                    (result.get("metrics") or {}).get(
                        "force_equilibrium_relative_error"
                    )
                ),
                "runtime_seconds": result.get("runtime_seconds"),
                "resumed": row["resumed"],
            }
        )
    write_csv(output_root / "body_qualification.csv", output_rows)
    summary = {
        "schema_version": "cadengbench-p.l3-body-qualification-summary.v1",
        "attempted_count": len(output_rows),
        "solver_qualified_count": len(promoted),
        "coverage_with_frozen_v1_count": 120 + len(promoted),
        "coverage_with_frozen_v1_fraction": (120 + len(promoted)) / 300,
        "target_50_percent_met": 120 + len(promoted) >= 150,
        "status_counts": dict(sorted(counts.items())),
        "promoted_item_ids": promoted,
        "core_dataset_mutated": False,
    }
    write_json(output_root / "body_qualification_summary.json", summary)
    return summary
