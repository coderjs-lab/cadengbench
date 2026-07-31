#!/usr/bin/env python3
"""Verify that valid but materially mis-sized STEP controls fail L1."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cadquery as cq

from cadengbench_p_eval.case import load_case
from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.geometry.features import extract_cylindrical_features
from cadengbench_p_eval.geometry.requirements import evaluate_requirements
from cadengbench_p_eval.geometry.step import load_step
from cadengbench_p_eval.io import read_json


def _selected_cases(
    evalset_root: Path, item_ids_file: Path | None
) -> list[Path]:
    manifest = read_json(evalset_root / "zero_to_cad/manifest.json")
    records = {
        str(record["item_id"]): evalset_root
        / "zero_to_cad"
        / str(record["case_path"])
        for record in manifest["items"]
    }
    if item_ids_file is None:
        return list(records.values())
    item_ids = [
        line.strip()
        for line in item_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [item_id for item_id in item_ids if item_id not in records]
    if missing:
        raise ValueError(f"unknown item IDs: {missing}")
    return [records[item_id] for item_id in item_ids]


def _mutate_step(source: Path, output: Path, scale_factor: float) -> None:
    shape = cq.importers.importStep(str(source)).val()
    cq.exporters.export(shape.scale(scale_factor), str(output))


def _evaluate_case(
    case_path: Path,
    output_dir: Path,
    scale_factor: float,
    expected_outcome: str,
) -> dict[str, Any]:
    case = load_case(case_path, verify_artifacts=True)
    source_geometry = load_step(case.artifact_path("target_step"))
    candidate_step = output_dir / f"{case.item_id}.step"
    _mutate_step(case.artifact_path("target_step"), candidate_step, scale_factor)
    geometry = load_step(candidate_step)
    inventory = extract_cylindrical_features(geometry)
    checks = evaluate_requirements(
        case.data["requirements"],
        geometry,
        axis_tolerance_degrees=float(
            case.data["evaluator_profile"]["dfm"].get(
                "hole_axis_tolerance_degrees", 1.0
            )
        ),
        cylindrical_inventory=inventory,
    )
    required = [check for check in checks if check.required]
    failed = [check for check in required if check.status is Status.FAILED]
    errors = [
        check
        for check in required
        if check.status in {Status.EVALUATOR_ERROR, Status.NOT_SCOREABLE}
    ]
    bbox_failed = any(
        check.failure_code == "bbox_dimensions_mismatch" for check in failed
    )
    valid_control = (
        geometry.topology["is_valid"] is True
        and geometry.topology["solid_count"]
        == source_geometry.topology["solid_count"]
    )
    rejected_as_expected = valid_control and bbox_failed and not errors
    accepted_as_expected = (
        valid_control
        and not failed
        and not errors
        and all(check.status is Status.PASSED for check in required)
    )
    return {
        "item_id": case.item_id,
        "scale_factor": scale_factor,
        "valid_control": valid_control,
        "bbox_rejected": bbox_failed,
        "rejected_as_expected": rejected_as_expected,
        "accepted_as_expected": accepted_as_expected,
        "control_passed": (
            rejected_as_expected
            if expected_outcome == "reject"
            else accepted_as_expected
        ),
        "failed_required_checks": [
            {
                "check_id": check.check_id,
                "failure_code": check.failure_code,
            }
            for check in failed
        ],
        "error_check_ids": [check.check_id for check in errors],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--item-ids-file", type=Path)
    parser.add_argument("--scale-factor", type=float, default=1.05)
    parser.add_argument("--require-all-rejected", action="store_true")
    parser.add_argument("--require-all-accepted", action="store_true")
    arguments = parser.parse_args()
    if arguments.require_all_rejected and arguments.require_all_accepted:
        raise ValueError("choose one expected control outcome")
    if not 1.001 <= arguments.scale_factor <= 1.25:
        raise ValueError("scale factor must be between 1.001 and 1.25")
    expected_outcome = (
        "accept" if arguments.require_all_accepted else "reject"
    )

    case_paths = _selected_cases(
        arguments.evalset_root.resolve(),
        arguments.item_ids_file.resolve() if arguments.item_ids_file else None,
    )
    with tempfile.TemporaryDirectory(prefix="cadengbench-p-l1-controls-") as raw:
        temporary = Path(raw)
        records = [
            _evaluate_case(
                case_path,
                temporary,
                arguments.scale_factor,
                expected_outcome,
            )
            for case_path in case_paths
        ]

    outcomes = Counter(
        expected_outcome if record["control_passed"] else "unexpected"
        for record in records
    )
    summary = {
        "item_count": len(records),
        "scale_factor": arguments.scale_factor,
        "outcome_counts": dict(outcomes),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {"summary": summary, "items": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    required = (
        arguments.require_all_rejected or arguments.require_all_accepted
    )
    if required and outcomes != {expected_outcome: len(records)}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
