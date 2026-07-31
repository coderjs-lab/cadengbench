#!/usr/bin/env python3
"""Audit whether production parameters have source-neutral geometry bindings."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _bbox_lengths(geometry: dict[str, Any]) -> dict[str, float]:
    bbox = geometry["bbox"]
    return {
        axis: float(bbox[f"{axis}max"]) - float(bbox[f"{axis}min"])
        for axis in ("x", "y", "z")
    }


def _unique_hole_group(
    geometry: dict[str, Any], nominal: float
) -> dict[str, Any] | None:
    groups: Counter[tuple[float, float, tuple[float, float, float]]] = Counter()
    for hole in geometry["hole_inventory"]["holes"]:
        key = (
            round(float(hole["diameter"]), 8),
            round(float(hole["depth"]), 8),
            tuple(round(float(value), 8) for value in hole["axis_direction"]),
        )
        groups[key] += 1
    matches = [
        {
            "diameter_mm": key[0],
            "depth_mm": key[1],
            "axis": list(key[2]),
            "quantity": quantity,
        }
        for key, quantity in groups.items()
        if math.isclose(key[0], nominal, rel_tol=0.0, abs_tol=1e-6)
    ]
    return matches[0] if len(matches) == 1 else None


def _bbox_parameter_axes(
    selected: dict[str, Any], nominal_geometry: dict[str, Any], nominal: float
) -> list[str]:
    reports = selected.get("probe_reports")
    if not isinstance(reports, list) or len(reports) < 2:
        return []
    observations: list[tuple[float, dict[str, float]]] = []
    for report in reports:
        if not isinstance(report, dict):
            return []
        parameter = report.get("canonical_value_mm", report.get("value"))
        geometry = (report.get("step_roundtrip") or {}).get("geometry")
        if not isinstance(parameter, (int, float)) or not isinstance(geometry, dict):
            return []
        observations.append((float(parameter), _bbox_lengths(geometry)))
    nominal_bbox = _bbox_lengths(nominal_geometry)
    matches: list[str] = []
    for axis in ("x", "y", "z"):
        if not math.isclose(
            nominal_bbox[axis], nominal, rel_tol=0.0, abs_tol=max(1e-5, nominal * 1e-6)
        ):
            continue
        if all(
            math.isclose(
                lengths[axis],
                parameter,
                rel_tol=0.0,
                abs_tol=max(1e-5, abs(parameter) * 1e-6),
            )
            for parameter, lengths in observations
        ):
            matches.append(axis)
    return matches


def audit_item(item_root: Path) -> dict[str, Any]:
    consumer = _read(item_root / "benchmark/consumer_ready_item_v2.json")
    oracle = _read(item_root / "completion/v2/parametric/edit_target_oracle.json")
    binding_record = _read(item_root / "source/edit_binding_qualification.json")
    selected = binding_record.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"{item_root.name} has no selected edit binding")
    source_binding = selected.get("binding")
    if not isinstance(source_binding, dict):
        raise ValueError(f"{item_root.name} selected binding is invalid")
    inputs = consumer["evaluator_inputs"]
    geometry = inputs["geometry"]
    effect = inputs["functional_edit"]["effect_proxy"]
    parameter = oracle["parameter_contract"]
    nominal = float(parameter["base_value"])

    candidates: list[dict[str, Any]] = []
    hole = _unique_hole_group(geometry, nominal)
    if effect.get("metric_kind") == "hole_diameter" and hole is not None:
        candidates.append({"kind": "hole_diameter", **hole})
    for axis in _bbox_parameter_axes(selected, geometry, nominal):
        candidates.append({"kind": "bbox_axis", "axis": axis})

    source_family = (
        "benchcad" if item_root.name.startswith("p_benchcad_") else "fusion360"
    )
    return {
        "item_id": item_root.name,
        "source_family": source_family,
        "parameter_key": parameter.get("parameter_key"),
        "effect_metric_kind": effect.get("metric_kind"),
        "source_call": source_binding.get("call"),
        "edit_family": source_binding.get("edit_family"),
        "parameter_role": source_binding.get("parameter_role"),
        "extrusion_axis": source_binding.get("extrusion_axis"),
        "candidate_bindings": candidates,
        "candidate_binding_count": len(candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    roster = [
        line.strip()
        for line in (arguments.dataset_root / "item_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    records = [
        audit_item(arguments.dataset_root / "items" / item_id) for item_id in roster
    ]
    summary = {
        "item_count": len(records),
        "source_counts": dict(Counter(row["source_family"] for row in records)),
        "candidate_binding_counts": dict(
            Counter(str(row["candidate_binding_count"]) for row in records)
        ),
        "binding_kind_counts": dict(
            Counter(
                candidate["kind"]
                for row in records
                for candidate in row["candidate_bindings"]
            )
        ),
        "effect_metric_counts": dict(
            Counter(str(row["effect_metric_kind"]) for row in records)
        ),
        "benchcad_call_counts": dict(
            Counter(
                str(row["source_call"])
                for row in records
                if row["source_family"] == "benchcad"
            )
        ),
        "fusion_edit_family_counts": dict(
            Counter(
                str(row["edit_family"])
                for row in records
                if row["source_family"] == "fusion360"
            )
        ),
        "fusion_parameter_role_counts": dict(
            Counter(
                str(row["parameter_role"])
                for row in records
                if row["source_family"] == "fusion360"
            )
        ),
    }
    payload = {"summary": summary, "items": records}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
