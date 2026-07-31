from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cadengbench_p_l3_response.io import write_json
from cadengbench_p_l3_response.runner import bind_family_manifest, run_family

ITEM_IDS = ("p_benchcad_000333", "p_fusion360_001312")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _reference_step(repo: Path, item_id: str) -> Path:
    item_root = (
        repo
        / "data/production/cadengbench_p_1800_v2_two_source_r25/dataset/items"
        / item_id
    )
    if item_id.startswith("p_fusion360_"):
        return item_root / "source/fusion360_occ_reconstruction.step"
    return item_root / "source/generated_from_cadquery.step"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--ccx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    summaries = []
    for item_id in ITEM_IDS:
        item_output = args.output / item_id
        source_manifest = _read_json(
            args.sidecar / "family_pilots/manifests" / f"{item_id}.json"
        )
        specification = {
            "item_id": item_id,
            "variants": [
                {
                    "variant_id": variant["variant_id"],
                    "upstream_status": variant["upstream_status"],
                    "reference_step_path": str(
                        _reference_step(args.repo_root, item_id)
                    ),
                    "candidate_step_path": variant["step_path"],
                }
                for variant in source_manifest["variants"]
            ],
        }
        specification_path = item_output / "variants_spec.json"
        write_json(specification_path, specification)
        manifest_path = item_output / "response_family_manifest.json"
        bind_family_manifest(
            contract_path=args.calibration_root
            / "items"
            / item_id
            / "calibration/response_contract.json",
            card_path=args.sidecar
            / "items"
            / item_id
            / "physics_card.solver_qualified.json",
            variants_spec_path=specification_path,
            output_path=manifest_path,
        )
        summaries.append(
            run_family(
                manifest_path=manifest_path,
                output_dir=item_output / "run",
                ccx_path=args.ccx,
                workers=args.workers,
                timeout_seconds=300,
            )
        )
    result = {
        "item_count": len(summaries),
        "variant_count": sum(value["variant_count"] for value in summaries),
        "scoreable_variant_count": sum(
            value["scoreable_variant_count"] for value in summaries
        ),
        "pass_count": sum(value["pass_count"] for value in summaries),
        "items": summaries,
    }
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
