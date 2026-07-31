from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cadquery as cq

from cadengbench_p_l3_response.io import write_json
from cadengbench_p_l3_response.runner import calibrate_item, evaluate_pair

METRICS = (
    "regularized_von_mises_p95_mpa",
    "normalized_max_displacement",
    "normalized_compliance",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source_step(repo: Path, item_id: str) -> Path:
    item_root = (
        repo
        / "data/production/cadengbench_p_1800_v2_two_source_r25/dataset/items"
        / item_id
    )
    consumer = _read_json(item_root / "benchmark/consumer_ready_item_v2.json")
    if consumer["source"]["source_key"] == "fusion360":
        source = item_root / "source/fusion360_occ_reconstruction.step"
    else:
        source = repo / consumer["source"]["canonical_geometry_artifact"]
    if not source.is_file():
        raise FileNotFoundError(f"missing canonical STEP: {source}")
    return source


def _write_variant(source: Path, target: Path, scale: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shape = cq.importers.importStep(str(source)).val()
    if scale != 1.0:
        shape = shape.scale(scale)
    cq.exporters.export(shape, str(target))


def _ratio(result: dict[str, Any], metric: str) -> float | None:
    value = (
        result.get("response_fidelity", {})
        .get("metrics", {})
        .get(metric, {})
        .get("candidate_reference_ratio")
    )
    return float(value) if value is not None else None


def _row(
    item: dict[str, Any],
    variant: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        **item,
        **variant,
        "scoreable": bool(result.get("scoreable")),
        "final_verdict": result.get("final_verdict"),
        "response_fidelity_score": result.get("margins", {}).get(
            "response_fidelity_score"
        ),
        "stress_ratio": _ratio(result, METRICS[0]),
        "displacement_ratio": _ratio(result, METRICS[1]),
        "compliance_ratio": _ratio(result, METRICS[2]),
        "rebuild": result.get("stages", {}).get("rebuild"),
        "selectors": result.get("stages", {}).get("selectors"),
        "mesh": result.get("stages", {}).get("mesh"),
        "solve": result.get("stages", {}).get("solve"),
        "numerical_validity": result.get("stages", {}).get(
            "numerical_validity"
        ),
        "response_fidelity": result.get("stages", {}).get("response_fidelity"),
        "engineering_requirements": result.get("stages", {}).get(
            "engineering_requirements"
        ),
        "runtime_seconds": result.get("runtime_seconds"),
        "error": None,
    }


def _error_rows(
    item: dict[str, Any], variants: list[dict[str, Any]], error: Exception
) -> list[dict[str, Any]]:
    return [
        {
            **item,
            **variant,
            "scoreable": False,
            "final_verdict": "evaluator_error",
            "error": f"{type(error).__name__}: {error}",
        }
        for variant in variants
    ]


def _run_item(
    *,
    repo: Path,
    sidecar: Path,
    profile: Path,
    ccx: Path,
    output: Path,
    item: dict[str, Any],
    variants: list[dict[str, Any]],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    item_id = item["item_id"]
    item_output = output / "items" / item_id
    try:
        source = _source_step(repo, item_id)
        card = sidecar / "items" / item_id / "physics_card.solver_qualified.json"
        mesh_study = (
            sidecar
            / "pilot_runs"
            / item_id
            / "mesh_study/mesh_sensitivity_summary.json"
        )
        calibration = item_output / "calibration"
        calibrate_item(
            profile_path=profile,
            card_path=card,
            reference_step=source,
            mesh_study_path=mesh_study if mesh_study.is_file() else None,
            output_dir=calibration,
            ccx_path=ccx,
            timeout_seconds=timeout_seconds,
        )
        rows = []
        for variant in variants:
            candidate = item_output / "candidates" / f"{variant['variant_id']}.step"
            _write_variant(source, candidate, float(variant["scale"]))
            result = evaluate_pair(
                contract_path=calibration / "response_contract.json",
                card_path=card,
                reference_step=source,
                candidate_step=candidate,
                variant_id=variant["variant_id"],
                output_dir=item_output / "pairs" / variant["variant_id"],
                ccx_path=ccx,
                timeout_seconds=timeout_seconds,
            )
            rows.append(_row(item, variant, result))
        return rows
    except Exception as error:  # Keep the study running and expose failed items.
        return _error_rows(item, variants, error)


def _rate(rows: list[dict[str, Any]], predicate: Any) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def _monotonic_rate(rows: list[dict[str, Any]]) -> float:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    checks = []
    for item_rows in by_item.values():
        ordered = sorted(item_rows, key=lambda row: float(row["scale"]), reverse=True)
        for key in ("stress_ratio", "displacement_ratio"):
            values = [row.get(key) for row in ordered]
            checks.append(
                all(value is not None for value in values)
                and all(
                    float(values[index + 1]) >= float(values[index]) * 0.98
                    for index in range(len(values) - 1)
                )
            )
    return sum(checks) / len(checks) if checks else 0.0


def _summarize(
    rows: list[dict[str, Any]], runtime_seconds: float, gate_mode: str = "lock_study"
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study_split"], row["variant_id"])].append(row)
    groups = []
    for (split, variant_id), values in sorted(grouped.items()):
        groups.append(
            {
                "study_split": split,
                "variant_id": variant_id,
                "count": len(values),
                "scoreable_rate": _rate(values, lambda row: row["scoreable"]),
                "pass_rate": _rate(
                    values, lambda row: row["final_verdict"] == "pass"
                ),
                "structural_fail_rate": _rate(
                    values,
                    lambda row: row["scoreable"]
                    and row["final_verdict"]
                    in {"response_fidelity_fail", "requirement_fail"},
                ),
                "pipeline_reject_rate": _rate(
                    values,
                    lambda row: row["scoreable"]
                    and row["final_verdict"] != "pass",
                ),
                "evaluator_error_count": sum(
                    row["final_verdict"] == "evaluator_error" for row in values
                ),
            }
        )

    if gate_mode == "final_holdout":
        held_out = [row for row in rows if row["study_split"] == "held_out"]
        by_variant = {
            variant_id: [row for row in held_out if row["variant_id"] == variant_id]
            for variant_id in {
                "equivalent_reexport",
                "uniform_scale_0p95",
                "uniform_scale_0p85",
            }
        }
        checks = {
            "all_variants_scoreable": _rate(
                held_out, lambda row: row["scoreable"]
            )
            == 1.0,
            "equivalent_pass_rate_1p00": _rate(
                by_variant["equivalent_reexport"],
                lambda row: row["final_verdict"] == "pass",
            )
            == 1.0,
            "mild_pass_rate_at_least_0p90": _rate(
                by_variant["uniform_scale_0p95"],
                lambda row: row["final_verdict"] == "pass",
            )
            >= 0.9,
            "moderate_pipeline_reject_rate_1p00": _rate(
                by_variant["uniform_scale_0p85"],
                lambda row: row["scoreable"]
                and row["final_verdict"] != "pass",
            )
            == 1.0,
            "moderate_response_fail_rate_at_least_0p80": _rate(
                by_variant["uniform_scale_0p85"],
                lambda row: row["scoreable"]
                and row["final_verdict"]
                in {"response_fidelity_fail", "requirement_fail"},
            )
            >= 0.8,
            "no_evaluator_errors": not any(
                row["final_verdict"] == "evaluator_error" for row in held_out
            ),
        }
        gates = [
            {
                "study_split": "held_out",
                "passed": all(checks.values()),
                "checks": checks,
            }
        ]
        return {
            "item_count": len({row["item_id"] for row in rows}),
            "pair_count": len(rows),
            "runtime_seconds": runtime_seconds,
            "groups": groups,
            "gates": gates,
            "lock_ready": gates[0]["passed"],
        }

    gates = []
    for split in ("calibration", "held_out"):
        split_rows = [row for row in rows if row["study_split"] == split]
        by_variant = {
            variant_id: [
                row for row in split_rows if row["variant_id"] == variant_id
            ]
            for variant_id in {
                "equivalent_reexport",
                "uniform_scale_0p95",
                "uniform_scale_0p85",
                "uniform_scale_0p75",
            }
        }
        checks = {
            "all_variants_scoreable": _rate(
                split_rows, lambda row: row["scoreable"]
            )
            == 1.0,
            "equivalent_pass_rate_1p00": _rate(
                by_variant["equivalent_reexport"],
                lambda row: row["final_verdict"] == "pass",
            )
            == 1.0,
            "mild_pass_rate_at_least_0p90": _rate(
                by_variant["uniform_scale_0p95"],
                lambda row: row["final_verdict"] == "pass",
            )
            >= 0.9,
            "moderate_fail_rate_at_least_0p80": _rate(
                by_variant["uniform_scale_0p85"],
                lambda row: row["scoreable"]
                and row["final_verdict"]
                in {"response_fidelity_fail", "requirement_fail"},
            )
            >= 0.8,
            "strong_fail_rate_at_least_0p90": _rate(
                by_variant["uniform_scale_0p75"],
                lambda row: row["scoreable"]
                and row["final_verdict"]
                in {"response_fidelity_fail", "requirement_fail"},
            )
            >= 0.9,
            "monotonic_stress_and_displacement_rate_at_least_0p90": _monotonic_rate(
                split_rows
            )
            >= 0.9,
            "no_evaluator_errors": not any(
                row["final_verdict"] == "evaluator_error" for row in split_rows
            ),
        }
        gates.append(
            {
                "study_split": split,
                "passed": all(checks.values()),
                "checks": checks,
                "monotonic_rate": _monotonic_rate(split_rows),
            }
        )
    return {
        "item_count": len({row["item_id"] for row in rows}),
        "pair_count": len(rows),
        "runtime_seconds": runtime_seconds,
        "groups": groups,
        "gates": gates,
        "lock_ready": len(gates) == 2 and all(gate["passed"] for gate in gates),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--ccx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    study = _read_json(args.study)
    profile = args.repo_root / study["profile"]
    started = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _run_item,
                repo=args.repo_root,
                sidecar=args.sidecar,
                profile=profile,
                ccx=args.ccx,
                output=args.output,
                item=item,
                variants=study["variants"],
                timeout_seconds=args.timeout_seconds,
            )
            for item in study["items"]
        ]
        for future in as_completed(futures):
            rows.extend(future.result())

    rows.sort(key=lambda row: (row["study_split"], row["item_id"], row["scale"]))
    _write_csv(args.output / "results.csv", rows)
    summary = {
        "study_id": study["study_id"],
        "profile": study["profile"],
        **_summarize(
            rows,
            time.monotonic() - started,
            gate_mode=study.get("gate_mode", "lock_study"),
        ),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["lock_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
