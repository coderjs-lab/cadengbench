from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cadquery as cq
from cadengbench_p_eval.physics.calculix import run_probe_card

from cadengbench_p_l3_extended.geometry import source_step
from cadengbench_p_l3_extended.io import (
    read_json,
    sha256_file,
    write_csv,
    write_json,
)

METRICS = {
    "resultant_force": 2.0,
    "regularized_stress": 0.0,
    "max_displacement": 1.0,
    "normalized_displacement": 0.0,
    "normalized_compliance": 0.0,
}


def _magnitude(values: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _log_error(observed: float, expected: float) -> float:
    return abs(math.log(max(observed, 1e-30) / max(expected, 1e-30)))


def _select_items(sidecar: Path, count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for card_path in sorted(
        sidecar.glob("items/*/physics_card.solver_qualified.json")
    ):
        card = read_json(card_path)
        item_id = str(card["item_id"])
        context = read_json(sidecar / "items" / item_id / "context.json")
        source = (
            "benchcad" if item_id.startswith("p_benchcad_") else "fusion360"
        )
        groups[(source, str(card["scenario_family"]))].append(
            {
                "item_id": item_id,
                "source": source,
                "scenario_family": card["scenario_family"],
                "bbox_diagonal_mm": context["geometry"]["bbox_mm"]["diagonal"],
                "card_path": card_path,
            }
        )
    if count % len(groups) != 0:
        raise ValueError("scale-study item count must balance source/scenario groups")
    per_group = count // len(groups)
    selected = []
    for group, values in sorted(groups.items()):
        ordered = sorted(
            values,
            key=lambda value: (
                float(value["bbox_diagonal_mm"]),
                str(value["item_id"]),
            ),
        )
        if len(ordered) < per_group:
            raise ValueError(f"insufficient qualified items in group {group}")
        indices = [
            round((index + 1) * (len(ordered) - 1) / (per_group + 1))
            for index in range(per_group)
        ]
        selected.extend(ordered[index] for index in indices)
    return sorted(selected, key=lambda value: str(value["item_id"]))


def _write_scaled_step(source: Path, target: Path, factor: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shape = cq.importers.importStep(str(source)).val()
    if not math.isclose(factor, 1.0):
        shape = shape.scale(factor)
    cq.exporters.export(shape, str(target))


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    item_id = str(payload["item_id"])
    factor = float(payload["scale_factor"])
    output = Path(payload["output"])
    if output.is_dir():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    source = Path(payload["source_step"])
    scaled = output / "scaled.step"
    _write_scaled_step(source, scaled, factor)
    card = copy.deepcopy(read_json(Path(payload["card_path"])))
    card.pop("qualification", None)
    card["status"] = "geometry_qualified"
    card["card_id"] = f"{item_id}:scale-consistency:{factor:g}"
    result = run_probe_card(
        card,
        step_path=scaled,
        output_dir=output / "solver",
        ccx_path=Path(payload["ccx"]),
        timeout_seconds=int(payload["timeout_seconds"]),
        render=bool(payload["render"]),
    )
    metrics = result.get("metrics") or {}
    stress = metrics.get("regularized_von_mises_mpa") or {}
    return {
        "item_id": item_id,
        "source": payload["source"],
        "scenario_family": payload["scenario_family"],
        "scale_factor": factor,
        "variant_id": f"{item_id}::uniform_scale:{factor:g}",
        "derived_not_independent_item": True,
        "source_step_sha256": sha256_file(source),
        "scaled_step_sha256": sha256_file(scaled),
        "final_status": result.get("final_status"),
        "geometry": (result.get("stages") or {}).get("geometry", {}).get("status"),
        "selectors": (result.get("stages") or {}).get("selectors", {}).get("status"),
        "mesh": (result.get("stages") or {}).get("mesh", {}).get("status"),
        "solver": (result.get("stages") or {}).get("solver", {}).get("status"),
        "requirements": (result.get("stages") or {})
        .get("requirements", {})
        .get("status"),
        "resultant_force_n": (
            _magnitude(metrics["applied_force_n"])
            if "applied_force_n" in metrics
            else None
        ),
        "regularized_stress_p95_mpa": stress.get("p95"),
        "max_displacement_mm": metrics.get("max_displacement_mm"),
        "normalized_displacement": metrics.get("normalized_max_displacement"),
        "normalized_compliance": metrics.get("normalized_compliance"),
        "minimum_sicn": (result.get("stages") or {})
        .get("mesh", {})
        .get("minimum_quality"),
        "runtime_seconds": result.get("runtime_seconds"),
        "result_path": (output / "solver/result.json").as_posix(),
    }


def _annotate(
    *,
    output: Path,
    selected: list[dict[str, Any]],
    factors: list[float],
    profile: dict[str, Any],
) -> None:
    annotations = []
    for item in selected:
        annotations.append(
            {
                "schema_version": "cadengbench-p.l3-scale-annotation.v1",
                "topology_family_id": f"{item['item_id']}:uniform-scale:v1",
                "base_item_id": item["item_id"],
                "source": item["source"],
                "scenario_family": item["scenario_family"],
                "base_bbox_diagonal_mm": item["bbox_diagonal_mm"],
                "scale_factors": factors,
                "geometry_transform": "uniform_similarity_about_global_origin",
                "topology_expected_unchanged": True,
                "source_intent_claimed": False,
                "dataset_role": "derived_scale_ablation_not_independent_eval_item",
                "load_control": {
                    "within_scale_reference_candidate_pair": (
                        "same_resultant_force"
                    ),
                    "across_scale_contexts": (
                        "constant_surface_traction_resultant_scales_with_area"
                    ),
                },
                "expected_exponents": profile["scale_study"][
                    "surface_traction_laws"
                ],
                "manufacturing_interpretation": {
                    "l3_scored": False,
                    "statement": (
                        "Uniform scaling changes every absolute feature dimension; "
                        "process-specific DFM checks must be rerun with unchanged "
                        "manufacturing thresholds."
                    ),
                },
            }
        )
    path = output / "scale_annotations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for annotation in annotations:
            stream.write(json.dumps(annotation, sort_keys=True) + "\n")


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    profile = read_json(args.profile)
    sidecar = repo / profile["dataset"]["frozen_l3_sidecar_root"]
    output = args.output.resolve()
    selected = _select_items(sidecar, int(profile["scale_study"]["item_count"]))
    factors = [float(value) for value in profile["scale_study"]["factors"]]
    _annotate(
        output=output,
        selected=selected,
        factors=factors,
        profile=profile,
    )
    payloads = []
    exemplar = selected[0]["item_id"]
    for item in selected:
        source = source_step(repo, str(item["item_id"]))
        for factor in factors:
            payloads.append(
                {
                    **item,
                    "scale_factor": factor,
                    "source_step": str(source),
                    "output": str(
                        output
                        / "items"
                        / str(item["item_id"])
                        / f"scale_{factor:g}"
                    ),
                    "ccx": str(args.ccx),
                    "timeout_seconds": args.timeout_seconds,
                    "render": item["item_id"] == exemplar,
                }
            )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["item_id"], row["scale_factor"]))

    by_item: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_item[row["item_id"]][float(row["scale_factor"])] = row
    force_limit = float(
        profile["scale_study"]["maximum_force_law_symmetric_log_error"]
    )
    response_limit = float(
        profile["scale_study"]["maximum_response_law_symmetric_log_error"]
    )
    metric_fields = {
        "resultant_force": "resultant_force_n",
        "regularized_stress": "regularized_stress_p95_mpa",
        "max_displacement": "max_displacement_mm",
        "normalized_displacement": "normalized_displacement",
        "normalized_compliance": "normalized_compliance",
    }
    for row in rows:
        nominal = by_item[row["item_id"]][1.0]
        factor = float(row["scale_factor"])
        row["all_solver_stages_passed"] = row["final_status"] == "probe_pass"
        for metric, exponent in METRICS.items():
            field = metric_fields[metric]
            observed = (
                float(row[field]) / max(float(nominal[field]), 1e-30)
                if row[field] is not None and nominal[field] is not None
                else None
            )
            expected = factor**exponent
            error = (
                _log_error(observed, expected) if observed is not None else None
            )
            limit = force_limit if metric == "resultant_force" else response_limit
            row[f"{metric}_observed_ratio"] = observed
            row[f"{metric}_expected_ratio"] = expected
            row[f"{metric}_symmetric_log_error"] = error
            row[f"{metric}_law_passed"] = error is not None and error <= limit
    write_csv(output / "scale_results.csv", rows)
    ten_x = [row for row in rows if float(row["scale_factor"]) == 10.0]
    force_ratios = [
        float(row["resultant_force_observed_ratio"])
        for row in ten_x
        if row["resultant_force_observed_ratio"] is not None
    ]
    summary = {
        "schema_version": "cadengbench-p.l3-scale-study-summary.v1",
        "item_count": len(selected),
        "variant_count": len(rows),
        "selected_item_ids": [item["item_id"] for item in selected],
        "factors": factors,
        "probe_pass_count": sum(
            row["all_solver_stages_passed"] for row in rows
        ),
        "force_law_pass_count": sum(
            row["resultant_force_law_passed"] for row in rows
        ),
        "response_law_pass_count": sum(
            all(
                row[f"{metric}_law_passed"]
                for metric in METRICS
                if metric != "resultant_force"
            )
            for row in rows
        ),
        "ten_x_expected_resultant_force_ratio": 100.0,
        "ten_x_observed_resultant_force_ratio_median": (
            sorted(force_ratios)[len(force_ratios) // 2] if force_ratios else None
        ),
        "all_gates_passed": all(
            row["all_solver_stages_passed"]
            and all(row[f"{metric}_law_passed"] for metric in METRICS)
            for row in rows
        ),
        "core_dataset_mutated": False,
        "scaled_variants_are_independent_items": False,
        "manufacturing_scale_follow_up_required": True,
        "profile_sha256": sha256_file(args.profile),
        "results_csv_sha256": sha256_file(output / "scale_results.csv"),
    }
    write_json(output / "scale_study_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ccx", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    print(json.dumps(run_study(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
