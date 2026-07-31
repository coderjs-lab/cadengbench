#!/usr/bin/env python3
"""Run frozen L1/DFM contracts against their own source and edit references."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from cadengbench_p_eval.case import load_case
from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.dfm.evaluator import evaluate_dfm
from cadengbench_p_eval.geometry.features import extract_cylindrical_features
from cadengbench_p_eval.geometry.requirements import evaluate_requirements
from cadengbench_p_eval.geometry.step import load_step
from cadengbench_p_eval.io import read_json, sha256_file


def _target_step(repo_root: Path, item_root: Path) -> Path:
    oracle = read_json(
        item_root / "completion/v2/parametric/edit_target_oracle.json"
    )
    artifact = oracle["reference_artifacts"]["target_step"]
    expected = str(artifact["sha256"])
    declared = Path(str(artifact["path"]))
    candidates = [
        declared if declared.is_absolute() else repo_root / declared,
        *sorted(
            (item_root / "domain/parametric_edits/v2/replay").glob("*/model.step")
        ),
    ]
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected:
            return candidate.resolve()
    raise FileNotFoundError(
        f"hidden functional target is unavailable for {item_root.name}"
    )


def _check_packet(checks: list[Any]) -> dict[str, Any]:
    required = [check for check in checks if check.required]
    failed = [
        check.failure_code or check.check_id
        for check in required
        if check.status is Status.FAILED
    ]
    errors = [
        check.check_id
        for check in required
        if check.status in {Status.EVALUATOR_ERROR, Status.NOT_SCOREABLE}
    ]
    return {
        "status": "error" if errors else "failed" if failed else "passed",
        "required_check_count": len(required),
        "failed": failed,
        "errors": errors,
        "checks": [check.as_dict() for check in checks],
    }


def _evaluate(
    case_data: dict[str, Any], step_path: Path, *, requirements: list[dict[str, Any]]
) -> dict[str, Any]:
    geometry = load_step(step_path)
    profile = case_data["evaluator_profile"]
    inventory = extract_cylindrical_features(geometry)
    requirement_checks = evaluate_requirements(
        requirements,
        geometry,
        axis_tolerance_degrees=float(
            profile["dfm"].get("hole_axis_tolerance_degrees", 1.0)
        ),
        cylindrical_inventory=inventory,
    )
    dfm_checks = evaluate_dfm(
        case_data["dfm"],
        geometry,
        dfm_profile=profile["dfm"],
        geometry_profile=profile["geometry"],
        cylindrical_inventory=inventory,
    )
    packet = _check_packet([*requirement_checks, *dfm_checks])
    packet["step_path"] = str(step_path)
    packet["step_sha256"] = sha256_file(step_path)
    return packet


def audit_item(
    repo_root: Path, zero_case_path: Path, functional_case_path: Path
) -> dict[str, Any]:
    zero_case = load_case(zero_case_path, verify_artifacts=True)
    functional_case = load_case(functional_case_path, verify_artifacts=True)
    if zero_case.item_id != functional_case.item_id:
        raise ValueError(
            "protocol case mismatch: "
            f"{zero_case.item_id} != {functional_case.item_id}"
        )

    source_step = zero_case.artifact_path("target_step")
    source = _evaluate(
        deepcopy(zero_case.data),
        source_step,
        requirements=deepcopy(zero_case.data["requirements"]),
    )
    functional_requirements = [
        deepcopy(requirement)
        for requirement in functional_case.data["requirements"]
        if requirement.get("category") == "solid_contract"
    ]
    target = _evaluate(
        deepcopy(functional_case.data),
        _target_step(repo_root, functional_case.item_root),
        requirements=functional_requirements,
    )
    return {
        "item_id": zero_case.item_id,
        "source_family": zero_case.data["source_family"],
        "source_reference": source,
        "functional_target_reference": target,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--evalset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-all-pass", action="store_true")
    parser.add_argument("--item-id", action="append", dest="item_ids")
    arguments = parser.parse_args()

    zero_manifest = read_json(
        arguments.evalset_root / "zero_to_cad/manifest.json"
    )
    functional_manifest = read_json(
        arguments.evalset_root / "functional_edit/manifest.json"
    )
    functional_cases = {
        str(record["item_id"]): (
            arguments.evalset_root
            / "functional_edit"
            / str(record["case_path"])
        )
        for record in functional_manifest["items"]
    }
    zero_records = list(zero_manifest["items"])
    if arguments.item_ids:
        requested = set(arguments.item_ids)
        available = {str(record["item_id"]) for record in zero_records}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown item IDs: {missing}")
        zero_records = [
            record
            for record in zero_records
            if str(record["item_id"]) in requested
        ]
    records = [
        audit_item(
            arguments.repo_root.resolve(),
            arguments.evalset_root
            / "zero_to_cad"
            / str(record["case_path"]),
            functional_cases[str(record["item_id"])],
        )
        for record in zero_records
    ]
    source_status = Counter(row["source_reference"]["status"] for row in records)
    target_status = Counter(
        row["functional_target_reference"]["status"] for row in records
    )
    source_failures = Counter(
        failure
        for row in records
        for failure in row["source_reference"]["failed"]
    )
    target_failures = Counter(
        failure
        for row in records
        for failure in row["functional_target_reference"]["failed"]
    )
    summary = {
        "item_count": len(records),
        "source_reference_status_counts": dict(source_status),
        "functional_target_status_counts": dict(target_status),
        "source_reference_failure_counts": dict(source_failures),
        "functional_target_failure_counts": dict(target_failures),
    }
    payload = {"summary": summary, "items": records}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if arguments.require_all_pass and (
        source_status != {"passed": len(records)}
        or target_status != {"passed": len(records)}
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
