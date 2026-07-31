from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cadquery as cq
from cadengbench_p_eval.io import write_json as write_l3_json
from cadengbench_p_eval.physics.runner import qualify_physics_card

from cadengbench_p_l3_response.io import write_evidence_commit, write_json
from cadengbench_p_l3_response.runner import calibrate_item, evaluate_pair
from cadengbench_p_l3_response.solver import prepare_output


def _canonical_card(item_id: str, scenario: str) -> dict[str, Any]:
    selector = {
        "kind": "axis_extreme_planar_surfaces_v2",
        "region_id": "axis:x:min",
        "axis": "x",
        "side": "min",
        "normal_alignment_min": 0.98,
        "plane_tolerance_relative": 1e-6,
        "minimum_count": 1,
        "maximum_count": 64,
        "minimum_area_fraction": 0.002,
    }
    load = [0.1, 0.0, 0.0] if scenario == "axial_tension" else [0.0, 0.0, -0.1]
    return {
        "schema_version": "cadengbench-p.l3-physics-card.v2",
        "card_id": f"{item_id}:canonical-response:v1",
        "item_id": item_id,
        "status": "geometry_qualified",
        "source_intent_claimed": False,
        "scenario_family": scenario,
        "geometry": {"units": "mm", "expected_solid_count": 1},
        "selectors": {
            "fixed": selector,
            "loaded": {**selector, "region_id": "axis:x:max", "side": "max"},
        },
        "material": {
            "profile_id": "probe-isotropic-70gpa-v1",
            "model": "linear_elastic_isotropic",
            "youngs_modulus_mpa": 70000.0,
            "poisson_ratio": 0.33,
        },
        "load": {
            "kind": "surface_traction_probe",
            "traction_vector_mpa": load,
            "distribution": "boundary_triangle_consistent_nodal_v1",
        },
        "mesh": {
            "element_type": "C3D10",
            "algorithm_3d": 10,
            "high_order_optimization": "none",
            "target_long_axis_divisions": 24,
            "target_short_axis_divisions": 5,
            "minimum_sicn": 0.02,
            "maximum_elements": 250000,
        },
        "analysis": {
            "type": "linear_static",
            "small_deformation": True,
            "max_equilibrium_relative_error": 1e-5,
            "max_normalized_displacement": 0.02,
            "stress_exclusion_mesh_lengths": 2.0,
            "regularized_stress_percentiles": [95.0, 99.0],
        },
        "checks": [
            "mesh_quality",
            "selector_stability",
            "solver_completion",
            "force_equilibrium",
            "finite_response",
            "small_deformation",
            "regularized_von_mises",
            "normalized_compliance",
        ],
        "provenance": {
            "annotation_candidate_sha256": "a" * 64,
            "context_sha256": "b" * 64,
            "profile_sha256": "c" * 64,
            "annotator_model": "deterministic-canonical-control",
            "annotator_revision": "v1",
        },
    }


def _run_canonical(
    *,
    root: Path,
    profile: Path,
    ccx: Path,
    item_id: str,
    scenario: str,
    reference_shape: cq.Shape,
    weak_shape: cq.Shape,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    reference = root / "reference.step"
    weak = root / "weak.step"
    cq.exporters.export(reference_shape, str(reference))
    cq.exporters.export(weak_shape, str(weak))
    card_path = root / "card.geometry_qualified.json"
    promoted_path = root / "card.solver_qualified.json"
    write_l3_json(card_path, _canonical_card(item_id, scenario))
    qualification = qualify_physics_card(
        card_path=card_path,
        reference_step=reference,
        output_dir=root / "source_qualification",
        promoted_card_path=promoted_path,
        ccx_path=ccx,
        timeout_seconds=180,
    )
    if not qualification["promoted"]:
        raise RuntimeError(f"canonical card did not qualify: {item_id}")
    calibration = root / "calibration"
    contract = calibrate_item(
        profile_path=profile,
        card_path=promoted_path,
        reference_step=reference,
        output_dir=calibration,
        ccx_path=ccx,
        timeout_seconds=180,
    )
    exact = evaluate_pair(
        contract_path=calibration / "response_contract.json",
        card_path=promoted_path,
        reference_step=reference,
        candidate_step=reference,
        variant_id="nominal",
        output_dir=root / "exact",
        ccx_path=ccx,
        timeout_seconds=180,
        render=True,
    )
    weak_result = evaluate_pair(
        contract_path=calibration / "response_contract.json",
        card_path=promoted_path,
        reference_step=reference,
        candidate_step=weak,
        variant_id="weakened",
        output_dir=root / "weakened",
        ccx_path=ccx,
        timeout_seconds=180,
        render=True,
    )
    return {
        "item_id": item_id,
        "scenario": scenario,
        "load_scale": contract["loading"]["load_scale"],
        "target_resultant_force_n": contract["loading"]["target_resultant_force_n"],
        "exact": exact,
        "weakened": weak_result,
    }


def _run_real_i_beam(
    *, repo: Path, root: Path, profile: Path, ccx: Path
) -> dict[str, Any]:
    item_id = "p_benchcad_000366"
    card = (
        repo
        / "data/derived/cadengbench_p_300_l3_probe_v1/items"
        / item_id
        / "physics_card.solver_qualified.json"
    )
    source = (
        repo
        / "data/production/cadengbench_p_1800_v2_two_source_r25/dataset/items"
        / item_id
        / "source/generated_from_cadquery.step"
    )
    mesh_study = (
        repo
        / "data/derived/cadengbench_p_300_l3_probe_v1/pilot_runs"
        / item_id
        / "mesh_study/mesh_sensitivity_summary.json"
    )
    root.mkdir(parents=True)
    weak = root / "weak_flange.step"
    weak_shape = (
        cq.Workplane("YZ")
        .moveTo(0.0, 0.0)
        .lineTo(25.0, 0.0)
        .lineTo(25.0, 6.49)
        .lineTo(2.21, 6.49)
        .lineTo(2.21, 125.11)
        .lineTo(25.0, 125.11)
        .lineTo(25.0, 131.6)
        .lineTo(0.0, 131.6)
        .mirrorY()
        .extrude(517.01)
        .val()
    )
    cq.exporters.export(weak_shape, str(weak))
    calibration = root / "calibration"
    contract = calibrate_item(
        profile_path=profile,
        card_path=card,
        reference_step=source,
        mesh_study_path=mesh_study if mesh_study.is_file() else None,
        output_dir=calibration,
        ccx_path=ccx,
        timeout_seconds=300,
    )
    exact = evaluate_pair(
        contract_path=calibration / "response_contract.json",
        card_path=card,
        reference_step=source,
        candidate_step=source,
        variant_id="nominal",
        output_dir=root / "exact",
        ccx_path=ccx,
        timeout_seconds=300,
        render=True,
    )
    weak_result = evaluate_pair(
        contract_path=calibration / "response_contract.json",
        card_path=card,
        reference_step=source,
        candidate_step=weak,
        variant_id="weak-flange-25mm",
        output_dir=root / "weakened",
        ccx_path=ccx,
        timeout_seconds=300,
        render=True,
    )
    return {
        "item_id": item_id,
        "scenario": contract["scenario"],
        "load_scale": contract["loading"]["load_scale"],
        "target_resultant_force_n": contract["loading"]["target_resultant_force_n"],
        "exact": exact,
        "weakened": weak_result,
    }


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result.get("candidate") or {}
    reference = result.get("reference") or {}
    return {
        "final_verdict": result["final_verdict"],
        "scoreable": result["scoreable"],
        "response_fidelity_score": result.get("margins", {}).get(
            "response_fidelity_score"
        ),
        "margins": result.get("margins"),
        "reference_metrics": reference.get("metrics"),
        "candidate_metrics": candidate.get("metrics"),
        "runtime_seconds": result["runtime_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ccx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    prepare_output(arguments.output, force=arguments.force)
    axial = _run_canonical(
        root=arguments.output / "canonical_axial",
        profile=arguments.profile,
        ccx=arguments.ccx,
        item_id="p_benchcad_900001",
        scenario="axial_tension",
        reference_shape=cq.Workplane("YZ").rect(10.0, 10.0).extrude(100.0).val(),
        weak_shape=cq.Workplane("YZ").rect(7.0, 7.0).extrude(100.0).val(),
    )
    cantilever = _run_canonical(
        root=arguments.output / "canonical_cantilever",
        profile=arguments.profile,
        ccx=arguments.ccx,
        item_id="p_benchcad_900002",
        scenario="cantilever_transverse",
        reference_shape=cq.Workplane("YZ").rect(10.0, 10.0).extrude(100.0).val(),
        weak_shape=cq.Workplane("YZ").rect(10.0, 7.0).extrude(100.0).val(),
    )
    real = _run_real_i_beam(
        repo=arguments.repo_root,
        root=arguments.output / "real_p_benchcad_000366",
        profile=arguments.profile,
        ccx=arguments.ccx,
    )
    summary = {
        "schema_version": "cadengbench-p.l3-response-pilot-summary.v1",
        "source_intent_claimed": False,
        "objects": [
            {
                "item_id": value["item_id"],
                "scenario": value["scenario"],
                "load_scale": value["load_scale"],
                "target_resultant_force_n": value["target_resultant_force_n"],
                "exact": _compact(value["exact"]),
                "weakened": _compact(value["weakened"]),
            }
            for value in (axial, cantilever, real)
        ],
    }
    write_json(arguments.output / "pilot_summary.json", summary)
    write_evidence_commit(arguments.output, arguments.output / "pilot_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
