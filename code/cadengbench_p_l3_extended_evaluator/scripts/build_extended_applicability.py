from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cadengbench_p_l3_extended.io import read_json, write_json


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["item_id"]: row for row in csv.DictReader(stream)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    frozen = args.frozen_sidecar.resolve()
    body = args.body_root.resolve()
    body_rows = _read_csv(body / "body_qualification.csv")
    item_ids = [
        line.strip()
        for line in args.item_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = []
    families = Counter()
    sources = Counter()
    for item_id in item_ids:
        frozen_card = (
            frozen / "items" / item_id / "physics_card.solver_qualified.json"
        )
        body_card = body / "items" / item_id / "body_card.solver_qualified.json"
        source = "benchcad" if item_id.startswith("p_benchcad_") else "fusion360"
        if frozen_card.is_file():
            card = read_json(frozen_card)
            status = "solver_qualified"
            family = str(card["scenario_family"])
            profile = "frozen_l3_v1"
            card_path = frozen_card
            reason = None
        elif body_card.is_file():
            status = "solver_qualified"
            family = "single_support_body_acceleration"
            profile = "l3_extended_v1"
            card_path = body_card
            reason = None
        else:
            status = "not_applicable"
            family = None
            profile = None
            card_path = None
            body_row = body_rows.get(item_id)
            reason = (
                body_row.get("failure_message")
                if body_row is not None
                else "not_selected_or_not_eligible_for_extended_qualification"
            )
        if family is not None:
            families[family] += 1
            sources[source] += 1
        records.append(
            {
                "schema_version": "cadengbench-p.l3-extended-applicability.v1",
                "item_id": item_id,
                "source": source,
                "status": status,
                "scenario_family": family,
                "profile": profile,
                "physics_card_path": (
                    card_path.relative_to(repo).as_posix()
                    if card_path is not None
                    else None
                ),
                "reason": reason,
                "source_intent_claimed": False,
                "core_dataset_mutated": False,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    qualified = sum(record["status"] == "solver_qualified" for record in records)
    summary = {
        "schema_version": "cadengbench-p.l3-extended-applicability-summary.v1",
        "item_count": len(records),
        "solver_qualified_count": qualified,
        "not_applicable_count": len(records) - qualified,
        "coverage_fraction": qualified / len(records),
        "family_counts": dict(sorted(families.items())),
        "qualified_source_counts": dict(sorted(sources.items())),
        "frozen_v1_preserved": True,
        "core_dataset_mutated": False,
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--frozen-sidecar", type=Path, required=True)
    parser.add_argument("--body-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
