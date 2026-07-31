"""Deterministic body-acceleration card construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cadengbench_p_l3_extended.contracts import validate_body_card
from cadengbench_p_l3_extended.io import read_json, sha256_file, write_json

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def compile_body_card(
    *,
    audit_record: dict[str, Any],
    audit_path: Path,
    profile: dict[str, Any],
    profile_path: Path,
) -> dict[str, Any]:
    if not audit_record["extension_candidate"]:
        raise ValueError(
            f"item is not an extension candidate: {audit_record['item_id']}"
        )
    geometry = audit_record["geometry"]
    support = geometry["qualified_single_support_regions"][0]
    axis = str(support["axis"])
    side = str(support["side"])
    vector = [0.0, 0.0, 0.0]
    vector[AXIS_INDEX[axis]] = (
        1.0 if side == "min" else -1.0
    ) * float(profile["body_acceleration"]["acceleration_magnitude_mm_s2"])
    eligibility = profile["eligibility"]
    body = profile["body_acceleration"]
    card = {
        "schema_version": "cadengbench-p.l3-body-acceleration-card.v1",
        "card_id": f"{audit_record['item_id']}:body-acceleration:v1",
        "item_id": audit_record["item_id"],
        "status": "geometry_qualified",
        "source_intent_claimed": False,
        "scenario_family": "single_support_body_acceleration",
        "geometry": {"units": "mm", "expected_solid_count": 1},
        "selector": {
            "kind": "axis_extreme_planar_surfaces_v2",
            "region_id": support["region_id"],
            "axis": axis,
            "side": side,
            "normal_alignment_min": eligibility["normal_alignment_min"],
            "plane_tolerance_relative": eligibility["plane_tolerance_relative"],
            "minimum_count": 1,
            "maximum_count": eligibility["maximum_region_surface_count"],
            "minimum_area_fraction": eligibility["minimum_region_area_fraction"],
        },
        "material": dict(body["material"]),
        "load": {
            "kind": "body_acceleration_probe",
            "acceleration_vector_mm_s2": vector,
            "distribution": "c3d10_consistent_nodal_body_force_v1",
            "control": "fixed_acceleration",
        },
        "mesh": dict(body["mesh"]),
        "analysis": dict(body["analysis"]),
        "scale_laws": dict(profile["scale_study"]["body_acceleration_laws"]),
        "provenance": {
            "audit_sha256": sha256_file(audit_path),
            "profile_sha256": sha256_file(profile_path),
            "annotation_method": (
                "deterministic_largest_axis_extreme_planar_support_v1"
            ),
        },
    }
    validate_body_card(card)
    return card


def compile_body_cards(
    *,
    profile_path: Path,
    audit_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    profile = read_json(profile_path)
    summary = read_json(audit_root / "coverage_audit_summary.json")
    compiled = []
    for item_id in summary["candidate_item_ids"]:
        audit_path = audit_root / "items" / item_id / "audit.json"
        audit = read_json(audit_path)
        card = compile_body_card(
            audit_record=audit,
            audit_path=audit_path,
            profile=profile,
            profile_path=profile_path,
        )
        path = output_root / "items" / item_id / "body_card.json"
        write_json(path, card)
        compiled.append(item_id)
    result = {
        "schema_version": "cadengbench-p.l3-body-card-compilation.v1",
        "compiled_count": len(compiled),
        "item_ids": compiled,
        "source_intent_claimed": False,
        "core_dataset_mutated": False,
    }
    write_json(output_root / "body_card_compilation_summary.json", result)
    return result
