from __future__ import annotations

import argparse
import json
from pathlib import Path

from cadengbench_p_l3_extended.contracts import validate_body_card
from cadengbench_p_l3_extended.io import read_json, write_json


def verify(args: argparse.Namespace) -> dict:
    repo = args.repo_root.resolve()
    profile = read_json(args.profile)
    root = repo / profile["dataset"]["extension_sidecar_root"]
    frozen = repo / profile["dataset"]["frozen_l3_sidecar_root"]
    body = read_json(root / "body/body_qualification_summary.json")
    scale = read_json(root / "scale_study/scale_study_summary.json")
    mesh = read_json(root / "body_mesh_study/mesh_study_summary.json")
    applicability_path = (
        repo
        / "benchmarks/cadengbench-p/evaluators/l3/applicability_extended_v1.jsonl"
    )
    applicability = [
        json.loads(line)
        for line in applicability_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frozen_cards = list(
        frozen.glob("items/*/physics_card.solver_qualified.json")
    )
    body_cards = list(
        (root / "body").glob("items/*/body_card.solver_qualified.json")
    )
    for path in body_cards:
        validate_body_card(read_json(path))
    item_ids = [record["item_id"] for record in applicability]
    qualified = [
        record for record in applicability if record["status"] == "solver_qualified"
    ]
    checks = {
        "frozen_card_count_is_120": len(frozen_cards) == 120,
        "body_card_count_is_44": len(body_cards) == 44,
        "body_target_met": bool(body["target_50_percent_met"]),
        "body_coverage_is_164": body["coverage_with_frozen_v1_count"] == 164,
        "scale_all_gates_passed": bool(scale["all_gates_passed"]),
        "scale_variant_count_is_24": scale["variant_count"] == 24,
        "mesh_all_gates_passed": bool(mesh["all_gates_passed"]),
        "mesh_run_count_is_9": mesh["run_count"] == 9,
        "applicability_has_300_unique_rows": (
            len(applicability) == 300 and len(set(item_ids)) == 300
        ),
        "applicability_qualified_count_is_164": len(qualified) == 164,
        "core_dataset_never_mutated": all(
            record["core_dataset_mutated"] is False for record in applicability
        ),
        "source_intent_never_claimed": all(
            record["source_intent_claimed"] is False for record in applicability
        ),
    }
    result = {
        "schema_version": "cadengbench-p.l3-extended-verification.v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    write_json(root / "verification.json", result)
    if not result["passed"]:
        raise SystemExit(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
