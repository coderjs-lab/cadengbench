"""Coverage and scale audit over the frozen 300-item CADEngBench-P release."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from cadengbench_p_l3_extended.geometry import (
    extension_candidate,
    inspect_geometry,
    source_step,
)
from cadengbench_p_l3_extended.io import read_json, sha256_file, write_csv, write_json


def _item_ids(repo_root: Path, profile: dict[str, Any]) -> list[str]:
    path = repo_root / profile["dataset"]["item_ids_path"]
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if len(values) != 300 or len(set(values)) != 300:
        raise ValueError("extended L3 audit requires the frozen unique 300-item roster")
    return values


def _existing_qualified(sidecar: Path) -> set[str]:
    return {
        path.parent.name
        for path in sidecar.glob("items/*/physics_card.solver_qualified.json")
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "p10": ordered[round((len(ordered) - 1) * 0.10)],
        "p25": ordered[round((len(ordered) - 1) * 0.25)],
        "median": statistics.median(ordered),
        "p75": ordered[round((len(ordered) - 1) * 0.75)],
        "p90": ordered[round((len(ordered) - 1) * 0.90)],
        "maximum": ordered[-1],
    }


def run_audit(
    *,
    repo_root: Path,
    profile_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    profile_path = profile_path.resolve()
    profile = read_json(profile_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar = repo_root / profile["dataset"]["frozen_l3_sidecar_root"]
    existing = _existing_qualified(sidecar)
    records = []
    disposition = Counter()
    sources = Counter()
    candidates_by_source = Counter()
    for item_id in _item_ids(repo_root, profile):
        step = source_step(repo_root, item_id)
        geometry = inspect_geometry(step, **profile["eligibility"])
        candidate, reason = extension_candidate(
            geometry,
            existing_solver_qualified=item_id in existing,
        )
        source = "benchcad" if item_id.startswith("p_benchcad_") else "fusion360"
        sources[source] += 1
        disposition[reason] += 1
        if candidate:
            candidates_by_source[source] += 1
        best = (
            geometry["qualified_single_support_regions"][0]
            if geometry["qualified_single_support_regions"]
            else None
        )
        record = {
            "item_id": item_id,
            "source": source,
            "source_step": step.relative_to(repo_root).as_posix(),
            "source_step_sha256": geometry["step_sha256"],
            "existing_solver_qualified_v1": item_id in existing,
            "extension_candidate": candidate,
            "disposition": reason,
            "solid_count": geometry["solid_count"],
            "face_count": geometry["face_count"],
            "bbox_diagonal_mm": geometry["bbox_mm"]["diagonal"],
            "volume_mm3": geometry["volume_mm3"],
            "qualified_support_region_count": len(
                geometry["qualified_single_support_regions"]
            ),
            "selected_support_region_id": (
                best["region_id"] if best is not None else None
            ),
            "selected_support_area_mm2": (
                best["area_mm2"] if best is not None else None
            ),
            "selected_support_area_fraction": (
                best["area_fraction"] if best is not None else None
            ),
            "geometry": geometry,
        }
        write_json(output_dir / "items" / item_id / "audit.json", record)
        records.append(record)

    compact = [
        {key: value for key, value in record.items() if key != "geometry"}
        for record in records
    ]
    write_csv(output_dir / "coverage_audit.csv", compact)
    with (output_dir / "coverage_audit.jsonl").open("w", encoding="utf-8") as stream:
        for record in compact:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    candidates = [record for record in compact if record["extension_candidate"]]
    summary = {
        "schema_version": "cadengbench-p.l3-extended-coverage-audit.v1",
        "core_dataset_mutated": False,
        "profile_path": profile_path.relative_to(repo_root).as_posix(),
        "profile_sha256": sha256_file(profile_path),
        "item_count": len(records),
        "source_counts": dict(sorted(sources.items())),
        "frozen_v1_solver_qualified_count": len(existing),
        "frozen_v1_coverage_fraction": len(existing) / len(records),
        "candidate_count": len(candidates),
        "candidate_counts_by_source": dict(sorted(candidates_by_source.items())),
        "maximum_coverage_if_all_candidates_qualify": len(existing) + len(candidates),
        "maximum_coverage_fraction_if_all_candidates_qualify": (
            len(existing) + len(candidates)
        )
        / len(records),
        "candidate_disposition_counts": dict(sorted(disposition.items())),
        "bbox_diagonal_mm": _quantiles(
            [float(record["bbox_diagonal_mm"]) for record in compact]
        ),
        "candidate_item_ids": [record["item_id"] for record in candidates],
        "audit_csv_sha256": sha256_file(output_dir / "coverage_audit.csv"),
        "audit_jsonl_sha256": sha256_file(output_dir / "coverage_audit.jsonl"),
    }
    write_json(output_dir / "coverage_audit_summary.json", summary)
    return summary
