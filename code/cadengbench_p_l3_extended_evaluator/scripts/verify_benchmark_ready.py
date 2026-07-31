#!/usr/bin/env python3
"""Fail closed unless the complete extended-v1 L3 release is benchmark-ready."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cadengbench_p_l3_extended.body_response import load_body_response_contract
from cadengbench_p_l3_extended.io import read_json, sha256_file, write_json


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root
    index_path = (
        repo
        / "benchmarks/cadengbench-p/evaluators/l3/applicability_extended_v1.jsonl"
    )
    rows = _jsonl(index_path)
    qualified = [row for row in rows if row["status"] == "solver_qualified"]
    family_counts = Counter(row["scenario_family"] for row in qualified)
    body_rows = [
        row
        for row in qualified
        if row["scenario_family"] == "single_support_body_acceleration"
    ]
    body_root = (
        repo
        / "data/derived/cadengbench_p_300_l3_extended_v1/body_response_contracts"
    )
    body_contracts = []
    body_bindings_valid = True
    for row in body_rows:
        item_id = row["item_id"]
        card = repo / row["physics_card_path"]
        contract_path = body_root / "items" / item_id / "response_contract.json"
        try:
            contract = load_body_response_contract(contract_path)
            body_contracts.append(contract)
            body_bindings_valid = body_bindings_valid and (
                contract["item_id"] == item_id
                and contract["bindings"]["base_card_sha256"] == sha256_file(card)
            )
        except Exception:
            body_bindings_valid = False
    surface_rows = [
        row
        for row in qualified
        if row["scenario_family"] != "single_support_body_acceleration"
    ]
    surface_contract_root = repo / "output/cadengbench_p_l3_response_contracts_v1"
    surface_contract_count = sum(
        (
            surface_contract_root
            / "items"
            / row["item_id"]
            / "response_contract.json"
        ).is_file()
        for row in surface_rows
    )
    build = read_json(body_root / "build_summary.json")
    acceptance = read_json(args.acceptance)
    checks = {
        "release_has_300_unique_rows": len(rows) == 300
        and len({row["item_id"] for row in rows}) == 300,
        "extended_profile_has_164_qualified_items": len(qualified) == 164,
        "family_counts_are_frozen": dict(family_counts)
        == {
            "axial_tension": 89,
            "cantilever_transverse": 31,
            "single_support_body_acceleration": 44,
        },
        "all_44_body_contracts_load": len(body_contracts) == 44,
        "all_body_card_bindings_match": body_bindings_valid,
        "all_120_surface_contracts_exist": surface_contract_count == 120,
        "body_contract_build_has_zero_errors": build["ready_count"] == 44
        and build["error_count"] == 0,
        "real_calculix_acceptance_passed": acceptance.get("passed") is True
        and acceptance.get("bindings_solver_qualified_count") == 164,
        "runtime_lock_exists": (
            repo / "configs/l3/cadengbench_p_l3_runtime_lock_v1.json"
        ).is_file(),
        "submission_entry_point_exists": (
            repo / "hpc/slurm/cadengbench_p_benchmark/submit_l3.sh"
        ).is_file(),
    }
    result = {
        "schema_version": "cadengbench-p.l3-extended-readiness.v1",
        "profile": "extended_v1",
        "item_count": len(rows),
        "solver_qualified_count": len(qualified),
        "family_counts": dict(family_counts),
        "body_contract_count": len(body_contracts),
        "surface_contract_count": surface_contract_count,
        "acceptance_path": str(args.acceptance.resolve()),
        "checks": checks,
        "benchmark_ready": all(checks.values()),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["benchmark_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
