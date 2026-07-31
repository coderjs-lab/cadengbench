"""Freeze evaluator-owned L0-L2 qualifications for a canonical dataset release."""

from __future__ import annotations

import keyword
import math
import os
import re
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .case import compile_evalset, load_case
from .io import read_json, sha256_file, sha256_json, write_json
from .profile import EvaluatorProfile

QUALIFICATION_SCHEMA = "cadengbench-p.evaluator-qualification.v3"
EDIT_ORACLE_SCHEMA = "cadengbench-p.edit-target-oracle.v2"
ACCEPTANCE_SCHEMA = "cadengbench-p.functional-edit-acceptance.v2"
RELEASE_SCHEMA = "cadengbench-p.l0-l2-release.v3"
RELEASE_SCHEMAS = {
    "cadengbench-p.l0-l2-release.v2",
    RELEASE_SCHEMA,
}
INTERFACE_VERSION = "cadengbench-p.cadquery-submission.v1"

ENGINEERING_ABSOLUTE_TOLERANCE_MM = 0.1
ENGINEERING_RELATIVE_TOLERANCE = 0.01

DFM_CAPABILITY = {
    "profile_id": "cadengbench-p.general-geometry-capability.v1",
    "scope": (
        "benchmark capability profile; geometry-only screening, not a claim of "
        "compliance with a material, machine, or manufacturing standard"
    ),
    "minimum_wall_thickness_mm": 1.0,
    "minimum_hole_diameter_mm": 2.0,
    "maximum_hole_depth_to_diameter": 8.0,
}


class QualificationError(ValueError):
    """A production item cannot be promoted into the frozen evaluator release."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{label} must be an object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise QualificationError(f"{label} must be finite")
    return result


def _engineering_name(raw: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(raw or "design_parameter"))
    if not name or name[0].isdigit():
        name = f"parameter_{name}"
    if keyword.iskeyword(name):
        name = f"{name}_parameter"
    return name


def _positive_predicate(parameter_id: str) -> dict[str, Any]:
    return {
        "type": "comparison",
        "op": "gt",
        "left": {"type": "variable", "source": "parameter", "key": parameter_id},
        "right": {"type": "constant", "value": 0.0},
        "tolerance": 0.0,
    }


def _requirements(geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    body_count = int(geometry.get("body_count", 0))
    if body_count < 1:
        raise QualificationError("geometry body_count must be positive")
    bbox = _mapping(geometry.get("bbox"), "geometry.bbox")
    envelope = {
        axis: {
            "value": _finite(bbox.get(f"{axis}len"), f"bbox.{axis}len"),
            "tolerance": max(
                ENGINEERING_ABSOLUTE_TOLERANCE_MM,
                abs(_finite(bbox.get(f"{axis}len"), f"bbox.{axis}len"))
                * ENGINEERING_RELATIVE_TOLERANCE,
            ),
        }
        for axis in ("x", "y", "z")
    }
    requirements: list[dict[str, Any]] = [
        {
            "requirement_id": "R-SOLID",
            "category": "solid_contract",
            "criticality": "required",
            "value_and_tolerance": {"body_count": body_count},
            "check_ids": ["hidden.check.solid_count.v1"],
        },
        {
            "requirement_id": "R-BBOX",
            "category": "global_envelope",
            "criticality": "required",
            "value_and_tolerance": envelope,
            "check_ids": ["hidden.check.bbox_dimensions.v1"],
        },
    ]
    inventory = _mapping(geometry.get("hole_inventory"), "geometry.hole_inventory")
    holes = inventory.get("holes")
    if not isinstance(holes, list):
        raise QualificationError("hole inventory must contain a list")
    groups: dict[tuple[float, float, tuple[float, float, float]], int] = {}
    for raw in holes:
        hole = _mapping(raw, "hole")
        axis = hole.get("axis_direction")
        if not isinstance(axis, list) or len(axis) != 3:
            raise QualificationError("hole axis must contain three components")
        key = (
            round(_finite(hole.get("diameter"), "hole diameter"), 8),
            round(_finite(hole.get("depth"), "hole depth"), 8),
            tuple(round(_finite(value, "hole axis"), 8) for value in axis),
        )
        groups[key] = groups.get(key, 0) + 1
    for index, ((diameter, depth, axis), quantity) in enumerate(
        sorted(groups.items()), 1
    ):
        requirements.append(
            {
                "requirement_id": f"R-HOLE-{index:03d}",
                "category": "cylindrical_opening_group",
                "criticality": "required",
                "value_and_tolerance": {
                    "quantity": quantity,
                    "diameter": diameter,
                    "diameter_tolerance": max(
                        ENGINEERING_ABSOLUTE_TOLERANCE_MM,
                        diameter * ENGINEERING_RELATIVE_TOLERANCE,
                    ),
                    "depth": depth,
                    "depth_tolerance": max(
                        ENGINEERING_ABSOLUTE_TOLERANCE_MM,
                        depth * ENGINEERING_RELATIVE_TOLERANCE,
                    ),
                    "axis_direction": list(axis),
                    "centers": [],
                    "position_tolerance": ENGINEERING_ABSOLUTE_TOLERANCE_MM,
                },
                "check_ids": ["hidden.check.cylindrical_opening_group.v1"],
            }
        )
    return requirements


def _bbox_lengths(geometry: Mapping[str, Any]) -> dict[str, float]:
    bbox = _mapping(geometry.get("bbox"), "geometry.bbox")
    return {
        axis: _finite(bbox.get(f"{axis}max"), f"bbox.{axis}max")
        - _finite(bbox.get(f"{axis}min"), f"bbox.{axis}min")
        for axis in ("x", "y", "z")
    }


def _source_neutral_binding(
    *,
    item_root: Path,
    evaluator_inputs: Mapping[str, Any],
    geometry: Mapping[str, Any],
    requirements: list[dict[str, Any]],
    nominal: float,
    fallback_name: str,
) -> tuple[str, list[str], dict[str, Any] | None]:
    """Resolve only observables proven by source replay and nominal geometry."""

    binding_record = _mapping(
        read_json(item_root / "source/edit_binding_qualification.json"),
        "edit binding qualification",
    )
    selected = _mapping(binding_record.get("selected"), "selected edit binding")
    source_binding = _mapping(selected.get("binding"), "selected source binding")
    functional = _mapping(
        evaluator_inputs.get("functional_edit"), "functional edit inputs"
    )
    effect = _mapping(functional.get("effect_proxy"), "functional effect proxy")

    if effect.get("metric_kind") == "hole_diameter":
        matches = [
            requirement
            for requirement in requirements
            if requirement.get("category") == "cylindrical_opening_group"
            and math.isclose(
                _finite(
                    _mapping(
                        requirement.get("value_and_tolerance"),
                        "opening requirement",
                    ).get("diameter"),
                    "opening diameter",
                ),
                nominal,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ]
        if len(matches) == 1:
            requirement_id = str(matches[0]["requirement_id"])
            group = int(requirement_id.rsplit("-", 1)[1])
            return (
                f"opening_group_{group:03d}_diameter",
                [requirement_id],
                {
                    "measurement_key": "P-001",
                    "measurement_kind": "cylindrical_opening_diameter",
                    "requirement_id": requirement_id,
                    "requirement_value_path": ["value_and_tolerance", "diameter"],
                    "opening_group": group,
                },
            )

    reports = selected.get("probe_reports")
    observations: list[tuple[float, dict[str, float]]] = []
    if isinstance(reports, list):
        for raw in reports:
            if not isinstance(raw, Mapping):
                observations = []
                break
            parameter_value = raw.get("canonical_value_mm", raw.get("value"))
            roundtrip = raw.get("step_roundtrip")
            probe_geometry = (
                roundtrip.get("geometry")
                if isinstance(roundtrip, Mapping)
                else None
            )
            if (
                isinstance(parameter_value, bool)
                or not isinstance(parameter_value, (int, float))
                or not isinstance(probe_geometry, Mapping)
            ):
                observations = []
                break
            observations.append(
                (float(parameter_value), _bbox_lengths(probe_geometry))
            )
    nominal_lengths = _bbox_lengths(geometry)
    axes = [
        axis
        for axis in ("x", "y", "z")
        if len(observations) >= 2
        and math.isclose(
            nominal_lengths[axis],
            nominal,
            rel_tol=0.0,
            abs_tol=max(1e-5, abs(nominal) * 1e-6),
        )
        and all(
            math.isclose(
                lengths[axis],
                parameter_value,
                rel_tol=0.0,
                abs_tol=max(1e-5, abs(parameter_value) * 1e-6),
            )
            for parameter_value, lengths in observations
        )
    ]
    if len(axes) == 1:
        axis = axes[0]
        name = {
            "x": "overall_length",
            "y": "overall_width",
            "z": "overall_height",
        }[axis]
        return (
            name,
            ["R-BBOX"],
            {
                "measurement_key": "P-001",
                "measurement_kind": "bounding_box_axis",
                "requirement_id": "R-BBOX",
                "requirement_value_path": [
                    "value_and_tolerance",
                    axis,
                    "value",
                ],
                "axis": axis,
            },
        )
    call = source_binding.get("call")
    argument_index = source_binding.get("argument_index")
    call_names = {
        ("box", 0): "box_length",
        ("box", 1): "box_width",
        ("box", 2): "box_height",
        ("chamfer", 0): "chamfer_size",
        ("circle", 0): "circular_profile_radius",
        ("cylinder", 0): "cylinder_height",
        ("cylinder", 1): "cylinder_radius",
        ("extrude", 0): "extrusion_distance",
        ("fillet", 0): "fillet_radius",
        ("hole", 0): "hole_diameter",
        ("rect", 0): "rectangle_width",
        ("rect", 1): "rectangle_height",
        ("revolve", 0): "revolve_angle",
    }
    return _engineering_name(
        source_binding.get("parameter_role")
        or call_names.get((call, argument_index))
        or fallback_name
    ), [], None


def _measurement_relation(*, evidence_ref: str, nominal: float) -> dict[str, Any]:
    return {
        "relation_id": "REL-P-001-MEASUREMENT",
        "predicate": {
            "type": "comparison",
            "op": "eq",
            "left": {
                "type": "variable",
                "source": "measurement",
                "key": "P-001",
            },
            "right": {
                "type": "variable",
                "source": "parameter",
                "key": "P-001",
            },
            "tolerance": max(0.05, abs(nominal) * 0.001),
        },
        "authority": "source_neutral_geometry_qualification",
        "evidence_ref": evidence_ref,
        "required": True,
    }


def _dfm(geometry: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _mapping(geometry.get("hole_inventory"), "geometry.hole_inventory")
    hole_count = int(inventory.get("hole_count", 0))
    evidence = {
        "profile_id": DFM_CAPABILITY["profile_id"],
        "scope": DFM_CAPABILITY["scope"],
        "authority": "benchmark_frozen_capability_profile",
    }
    families = [
        {
            "family": "minimum_wall_thickness",
            "applicability": "all_closed_solid_items",
            "required": True,
            "thresholds": {"minimum_mm": DFM_CAPABILITY["minimum_wall_thickness_mm"]},
            "threshold_evidence_ref": evidence,
            "extractor_certified": True,
            "scoreable": True,
            "unscoreable_reason": None,
        },
        {
            "family": "minimum_hole_geometry",
            "applicability": (
                "cylindrical_openings_present" if hole_count else "not_applicable"
            ),
            "required": hole_count > 0,
            "thresholds": {
                "minimum_diameter_mm": DFM_CAPABILITY["minimum_hole_diameter_mm"],
                "maximum_depth_to_diameter": DFM_CAPABILITY[
                    "maximum_hole_depth_to_diameter"
                ],
            },
            "threshold_evidence_ref": evidence,
            "extractor_certified": True,
            "expected_minimum_feature_count": hole_count,
            "scoreable": hole_count > 0,
            "unscoreable_reason": None if hole_count else "item has no openings",
        },
    ]
    return {
        "profile_origin": "benchmark_frozen_capability_profile",
        "process_profile": DFM_CAPABILITY["profile_id"],
        "material_card": None,
        "families": families,
        "scoreable_family_count": sum(value["scoreable"] for value in families),
    }


_DFM_CHECK_TO_FAMILY = {
    "L1.DFM.minimum_wall_thickness": "minimum_wall_thickness",
    "L1.DFM.minimum_hole_geometry": "minimum_hole_geometry",
}


def _reference_gated_dfm(
    dfm: Mapping[str, Any],
    packet: Mapping[str, Any],
    *,
    reference_kind: str,
) -> dict[str, Any]:
    """Keep a DFM gate scoreable only when its authoritative reference passes."""

    gated = deepcopy(dict(dfm))
    families = gated.get("families")
    checks = packet.get("checks")
    if not isinstance(families, list) or not isinstance(checks, list):
        raise QualificationError(f"{reference_kind} reference audit is incomplete")
    by_family = {
        family["family"]: family
        for family in families
        if isinstance(family, dict) and isinstance(family.get("family"), str)
    }
    for raw in checks:
        if not isinstance(raw, Mapping):
            continue
        check_id = str(raw.get("check_id"))
        status = str(raw.get("status"))
        family_name = _DFM_CHECK_TO_FAMILY.get(check_id)
        if family_name is None:
            continue
        family = by_family.get(family_name)
        if family is None or status in {"passed", "not_applicable"}:
            continue
        if status in {"evaluator_error", "not_scoreable", "not_reached"}:
            raise QualificationError(
                f"{reference_kind} DFM reference is not evaluable: "
                f"{check_id}={status}"
            )
        family["required"] = False
        family["scoreable"] = False
        family["applicability"] = f"{reference_kind}_reference_control_failed"
        family["unscoreable_reason"] = (
            f"the benchmark's own {reference_kind} reference does not satisfy "
            "this generic capability profile"
        )
    gated["scoreable_family_count"] = sum(
        isinstance(family, Mapping) and family.get("scoreable") is True
        for family in families
    )
    return gated


def build_qualification(
    item_root: Path, *, source_reference: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    item_id = item_root.name
    consumer_path = item_root / "benchmark/consumer_ready_item_v2.json"
    oracle_path = item_root / "completion/v2/parametric/edit_target_oracle.json"
    consumer = _mapping(read_json(consumer_path), "consumer-ready item")
    oracle = _mapping(read_json(oracle_path), "edit oracle")
    if consumer.get("item_id") != item_id or oracle.get("item_id") != item_id:
        raise QualificationError("item, consumer, and oracle IDs differ")
    qualification = _mapping(oracle.get("qualification"), "oracle qualification")
    if (
        oracle.get("schema_version") != EDIT_ORACLE_SCHEMA
        or oracle.get("status") != "verified"
        or oracle.get("llm_judge_used") is not False
        or qualification.get("all_probe_replays_verified") is not True
        or qualification.get("target_replay_verified") is not True
    ):
        raise QualificationError(f"edit oracle is not replay-qualified: {item_id}")
    evaluator_inputs = _mapping(consumer.get("evaluator_inputs"), "evaluator_inputs")
    geometry = _mapping(evaluator_inputs.get("geometry"), "evaluator_inputs.geometry")
    parameter = _mapping(oracle.get("parameter_contract"), "oracle parameter")
    domain = _mapping(parameter.get("domain"), "oracle parameter domain")
    lower = _finite(_mapping(domain.get("lower"), "domain lower").get("value"), "lower")
    upper = _finite(_mapping(domain.get("upper"), "domain upper").get("value"), "upper")
    nominal = _finite(parameter.get("base_value"), "base value")
    if not lower < nominal < upper:
        raise QualificationError("verified domain does not contain its nominal value")
    requirements = _requirements(geometry)
    evidence_ref = "completion/v2/parametric/edit_target_oracle.json"
    _observable_name, requirement_refs, measurement_binding = (
        _source_neutral_binding(
            item_root=item_root,
            evaluator_inputs=evaluator_inputs,
            geometry=geometry,
            requirements=requirements,
            nominal=nominal,
            fallback_name=str(parameter.get("parameter_key") or "design_parameter"),
        )
    )
    engineering_name = _engineering_name(
        parameter.get("parameter_key") or "design_parameter"
    )
    contract_kind = (
        "direct_measurement_only"
        if measurement_binding is not None
        else "geometry_response_only"
    )
    public_parameter = {
        "parameter_id": "P-001",
        "engineering_name": engineering_name,
        "nominal": nominal,
        "unit": str(parameter.get("canonical_unit") or "mm"),
        "role": "principal_dimension",
        "requirement_refs": requirement_refs,
        "domain": {
            "kind": "continuous",
            "minimum": lower,
            "maximum": upper,
            "minimum_inclusive": True,
            "maximum_inclusive": True,
            "authority": "verified_source_native_replay",
            "evidence_ref": evidence_ref,
        },
        "measurement_binding": measurement_binding,
        "variant_requirement_contract": {
            "kind": contract_kind,
            "authority": "verified_source_native_replay",
            "evidence_ref": evidence_ref,
        },
    }
    positive = _positive_predicate("P-001")
    dfm = _dfm(geometry)
    if source_reference is not None:
        dfm = _reference_gated_dfm(
            dfm, source_reference, reference_kind="source"
        )
    record: dict[str, Any] = {
        "schema_version": QUALIFICATION_SCHEMA,
        "item_id": item_id,
        "purpose": "frozen CADEngBench-P L0-L2 benchmark qualification",
        "source_evidence": {
            "consumer_item": "benchmark/consumer_ready_item_v2.json",
            "edit_oracle": evidence_ref,
            "oracle_id": oracle.get("oracle_id"),
            "source_parameter_id": parameter.get("parameter_id"),
            "source_parameter_key": parameter.get("parameter_key"),
        },
        "task_contract": {
            "task_kind": "zero_to_cad",
            "task_objective": "engineering_specification",
            "submission": {
                "code_submission_contract": {
                    "interface_version": INTERFACE_VERSION,
                    "build_signature": "build(params=None)",
                    "step_output_path": "model.step",
                    "required_top_level_symbols": [
                        "DEFAULT_PARAMS",
                        "PARAMETER_MANIFEST",
                        "build",
                        "result",
                    ],
                }
            },
            "solid_contract": {
                "required_body_count": int(geometry.get("body_count", 0))
            },
            "reference_frame": "source_canonical_frame",
            "fidelity_profile": "requirements_only",
        },
        "requirements": requirements,
        "parameter_system": {
            "interface_version": INTERFACE_VERSION,
            "parameters": [public_parameter],
            "relations": (
                [
                    _measurement_relation(
                        evidence_ref=evidence_ref,
                        nominal=nominal,
                    )
                ]
                if measurement_binding is not None
                else []
            ),
            "source_relations": list(
                _mapping(evaluator_inputs.get("parametric"), "parametric").get(
                    "constraint_relations", []
                )
            ),
            "normalized_source_relation_count": 0,
            "solver_obligations": [
                {
                    "obligation_id": "SOLVER-P-001-POSITIVE",
                    "assumptions": [],
                    "conclusion": positive,
                    "authority": "verified_positive_dimension_domain",
                    "evidence_ref": evidence_ref,
                    "required": True,
                }
            ],
            "feasibility_relations": [
                {
                    "relation_id": "FEASIBILITY-P-001-POSITIVE",
                    "predicate": positive,
                    "authority": "verified_positive_dimension_domain",
                    "evidence_ref": evidence_ref,
                    "required": True,
                }
            ],
            "invalid_probes": [],
            "qualified_parameter_count": 1,
            "measurement_bound_parameter_count": int(
                measurement_binding is not None
            ),
            "qualified_variant_parameter_count": 1,
            "complete_domains": True,
            "complete_measurement_bindings": measurement_binding is not None,
            "variant_contract_mode": contract_kind,
        },
        "dfm": dfm,
    }
    record["qualification_hash"] = sha256_json(record)
    return record


def build_functional_acceptance(item_root: Path) -> dict[str, Any]:
    oracle_path = item_root / "completion/v2/parametric/edit_target_oracle.json"
    oracle = _mapping(read_json(oracle_path), "edit oracle")
    parameter = _mapping(oracle.get("parameter_contract"), "oracle parameter")
    geometry = _mapping(oracle.get("geometry_contract"), "oracle geometry contract")
    preservation = _mapping(
        oracle.get("preservation_contract"), "oracle preservation contract"
    )
    if oracle.get("status") != "verified":
        raise QualificationError("functional-edit oracle is not verified")
    record = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "record_type": "deterministic_functional_edit_acceptance",
        "contract_id": f"{item_root.name}:functional-edit-acceptance:v2",
        "item_id": item_root.name,
        "candidate_id": None,
        "edit_oracle_id": oracle.get("oracle_id"),
        "source_key": oracle.get("source_key"),
        "status": "verified_compiled",
        "scoring_scope": "bounded_geometry_mediated_functional_edit",
        "parameter_contract": dict(parameter),
        "publication_projection": {
            "projection_version": "consumer_ready_functional_edit.v1",
            "candidate_declared_checks_are_scoring_authority": False,
            "critic_findings_are_scoring_authority": False,
        },
        "core_checks": [
            {
                "check_id": "core.target_parameter_assignment",
                "method": "source_bound_exact_parameter_assignment_v1",
                "expected_value": parameter.get("target_value"),
                "unit": parameter.get("canonical_unit"),
            },
            {
                "check_id": "core.parameter_domain",
                "method": "canonical_parameter_domain_membership_v1",
                "expected_value": True,
                "unit": None,
            },
            {
                "check_id": "core.valid_solid",
                "method": "occt_brep_validity",
                "expected_value": True,
                "unit": None,
            },
            {
                "check_id": "core.solid_count_preserved",
                "method": "canonical_step_solid_count_comparison",
                "expected_value": 0,
                "unit": "delta_count",
            },
            {
                "check_id": "core.non_target_source_preserved",
                "method": preservation.get("integrity_method"),
                "expected_value": True,
                "unit": None,
            },
            {
                "check_id": "core.target_brep_similarity",
                "method": _mapping(
                    geometry.get("comparison_protocol"), "comparison protocol"
                ).get("method"),
                "expected_value": "hidden_target_step",
                "unit": None,
            },
        ],
        "candidate_declared_checks": [],
        "candidate_effect_contracts": [],
        "candidate_preservation_contracts": [],
        "hidden_reference_artifacts": dict(
            _mapping(oracle.get("reference_artifacts"), "reference artifacts")
        ),
        "authority_boundary": dict(
            _mapping(oracle.get("authority_boundary"), "authority boundary")
        ),
        "engineering_effect_scope": (
            "The benchmark scores the deterministic geometry-mediated edit contract; "
            "it does not infer unmodeled service performance from prose."
        ),
        "llm_judge_used": False,
    }
    record["contract_hash"] = sha256_json(record)
    return record


def build_functional_dfm(
    item_id: str,
    nominal_dfm: Mapping[str, Any],
    target_reference: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": "cadengbench-p.functional-dfm.v1",
        "item_id": item_id,
        "dfm": _reference_gated_dfm(
            nominal_dfm,
            target_reference,
            reference_kind="functional_target",
        ),
        "authority": "model_independent_hidden_target_reference_control",
    }
    record["contract_hash"] = sha256_json(record)
    return record


def _link_item(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "evaluation":
            continue
        target = destination / child.name
        relative = os.path.relpath(child, target.parent)
        if target.is_symlink():
            if target.resolve() != child.resolve():
                raise QualificationError(f"stale evaluator item link: {target}")
            continue
        if target.exists():
            raise QualificationError(f"evaluator item path is not a symlink: {target}")
        target.symlink_to(relative, target_is_directory=child.is_dir())


def _roster(dataset_root: Path) -> list[str]:
    path = dataset_root / "item_ids.txt"
    item_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(item_ids) != 300 or len(set(item_ids)) != 300:
        raise QualificationError("canonical release must contain 300 unique item IDs")
    available = sorted(path.name for path in (dataset_root / "items").iterdir())
    if sorted(item_ids) != available:
        raise QualificationError("canonical item directory differs from item_ids.txt")
    return item_ids


def _reference_controls(
    path: Path, item_ids: list[str]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    audit = _mapping(read_json(path), "reference contract audit")
    records = audit.get("items")
    if not isinstance(records, list):
        raise QualificationError("reference contract audit has no item records")
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        record = _mapping(raw, "reference contract audit item")
        item_id = str(record.get("item_id"))
        if item_id in indexed:
            raise QualificationError(f"duplicate reference audit item: {item_id}")
        for field in ("source_reference", "functional_target_reference"):
            packet = _mapping(record.get(field), f"{item_id} {field}")
            if packet.get("status") == "error":
                raise QualificationError(f"{item_id} has an errored {field}")
        indexed[item_id] = record
    if set(indexed) != set(item_ids):
        raise QualificationError("reference contract audit roster differs from data")
    return indexed, dict(audit)


def _without_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    semantic = dict(value)
    semantic.pop(field, None)
    return semantic


def _safe_release_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise QualificationError(f"{label} must be a non-empty relative path")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise QualificationError(f"{label} escapes the frozen release")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise QualificationError(f"{label} escapes the frozen release") from exc
    return path


def validate_l0_l2_release(evalset_root: Path) -> dict[str, Any]:
    """Verify both frozen protocols, all 600 cases, and every bound artifact."""

    evalset_root = evalset_root.resolve()
    release_path = evalset_root / "release_manifest.json"
    release = _mapping(read_json(release_path), "release manifest")
    if release.get("schema_version") not in RELEASE_SCHEMAS:
        raise QualificationError("unsupported L0-L2 release schema")
    if release.get("release_hash") != sha256_json(
        _without_hash(release, "release_hash")
    ):
        raise QualificationError("release manifest hash is invalid")
    if "uncalibrated" in str(release.get("profile_id", "")).lower():
        raise QualificationError("publication release uses an uncalibrated profile")
    dataset_ref = release.get("dataset_root")
    if not isinstance(dataset_ref, str) or not dataset_ref:
        raise QualificationError("release dataset_root is invalid")
    raw_dataset = Path(dataset_ref)
    dataset_root = (
        raw_dataset.resolve()
        if raw_dataset.is_absolute()
        else (evalset_root / raw_dataset).resolve()
    )
    item_ids = _roster(dataset_root)
    if release.get("dataset_item_ids_sha256") != sha256_file(
        dataset_root / "item_ids.txt"
    ):
        raise QualificationError("canonical dataset roster hash differs")
    if release.get("item_count") != len(item_ids):
        raise QualificationError("release item count differs from dataset roster")
    audit_record = _mapping(
        release.get("reference_contract_audit"), "reference contract audit"
    )
    audit_path = _safe_release_path(
        evalset_root,
        audit_record.get("path"),
        "reference contract audit",
    )
    if sha256_file(audit_path) != audit_record.get("sha256"):
        raise QualificationError("reference contract audit binding is invalid")
    _reference_controls(audit_path, item_ids)

    protocol_reports: dict[str, Any] = {}
    protocol_rosters: dict[str, set[str]] = {}
    for protocol in ("zero_to_cad", "functional_edit"):
        protocol_record = _mapping(
            _mapping(release.get("protocols"), "release protocols").get(protocol),
            f"release protocol {protocol}",
        )
        manifest_path = _safe_release_path(
            evalset_root,
            protocol_record.get("manifest_path"),
            f"{protocol} manifest",
        )
        manifest = _mapping(read_json(manifest_path), f"{protocol} manifest")
        if manifest.get("schema_version") != "cadengbench-p.evalset-manifest.v1":
            raise QualificationError(f"unsupported {protocol} manifest schema")
        manifest_hash = sha256_json(_without_hash(manifest, "manifest_hash"))
        if (
            manifest.get("manifest_hash") != manifest_hash
            or protocol_record.get("manifest_hash") != manifest_hash
            or manifest.get("protocol") != protocol
            or manifest.get("profile_id") != release.get("profile_id")
            or manifest.get("profile_hash") != release.get("profile_hash")
        ):
            raise QualificationError(f"{protocol} manifest hash is invalid")
        records = manifest.get("items")
        if not isinstance(records, list) or len(records) != len(item_ids):
            raise QualificationError(f"{protocol} manifest has an invalid roster")
        seen: set[str] = set()
        sources: Counter[str] = Counter()
        dfm_slots = 0
        parameters = 0
        direct_measurement_parameters = 0
        geometry_response_parameters = 0
        exact_reconstruction_cases = 0
        for raw in records:
            record = _mapping(raw, f"{protocol} manifest item")
            item_id = str(record.get("item_id"))
            if item_id in seen:
                raise QualificationError(f"duplicate {protocol} item: {item_id}")
            seen.add(item_id)
            case_path = _safe_release_path(
                manifest_path.parent,
                record.get("case_path"),
                f"{protocol} case {item_id}",
            )
            loaded = load_case(case_path, verify_artifacts=True)
            if (
                loaded.item_id != item_id
                or loaded.data.get("protocol") != protocol
                or loaded.data.get("case_hash") != record.get("case_hash")
                or loaded.data.get("profile_hash") != release.get("profile_hash")
            ):
                raise QualificationError(f"{protocol} case binding failed: {item_id}")
            sources[str(loaded.data.get("source_family"))] += 1
            dfm_slots += int(loaded.data["dfm"]["scoreable_family_count"])
            parameters += int(
                loaded.data["parameter_system"]["qualified_parameter_count"]
            )
            for parameter in loaded.data["parameter_system"].get("parameters", []):
                contract = (
                    parameter.get("variant_requirement_contract")
                    if isinstance(parameter, Mapping)
                    else None
                )
                kind = contract.get("kind") if isinstance(contract, Mapping) else None
                direct_measurement_parameters += int(
                    kind == "direct_measurement_only"
                )
                geometry_response_parameters += int(
                    kind == "geometry_response_only"
                )
            exact_reconstruction_cases += int(
                loaded.data["task_contract"].get("task_objective")
                == "exact_reconstruction"
            )
        if seen != set(item_ids):
            raise QualificationError(f"{protocol} roster differs from canonical data")
        protocol_rosters[protocol] = seen
        protocol_reports[protocol] = {
            "valid": True,
            "item_count": len(seen),
            "manifest_hash": manifest_hash,
            "source_counts": dict(sorted(sources.items())),
            "scoreable_dfm_slots": dfm_slots,
            "qualified_parameter_slots": parameters,
            "direct_measurement_parameter_slots": direct_measurement_parameters,
            "geometry_response_parameter_slots": geometry_response_parameters,
            "exact_reconstruction_cases": exact_reconstruction_cases,
        }
        if protocol_record.get("item_count") != len(seen):
            raise QualificationError(f"{protocol} release item count is invalid")
        if dict(sources) != dict(release.get("source_counts") or {}):
            raise QualificationError(f"{protocol} source coverage is inconsistent")
        coverage = _mapping(release.get("coverage"), "release coverage")
        if parameters != coverage.get("qualified_parameter_slots_per_protocol"):
            raise QualificationError(f"{protocol} parameter coverage is incomplete")
        dfm_coverage_key = (
            "scoreable_dfm_slots_zero_to_cad"
            if protocol == "zero_to_cad"
            else "scoreable_dfm_slots_functional_edit"
        )
        if dfm_slots != coverage.get(dfm_coverage_key):
            raise QualificationError(f"{protocol} DFM coverage is inconsistent")
        if protocol == "zero_to_cad" and exact_reconstruction_cases != coverage.get(
            "exact_reconstruction_cases"
        ):
            raise QualificationError("exact-reconstruction coverage is inconsistent")
        if protocol == "zero_to_cad":
            if direct_measurement_parameters != coverage.get(
                "direct_measurement_parameter_slots"
            ):
                raise QualificationError(
                    "direct-measurement parameter coverage is inconsistent"
                )
            if geometry_response_parameters != coverage.get(
                "geometry_response_parameter_slots"
            ):
                raise QualificationError(
                    "geometry-response parameter coverage is inconsistent"
                )
            if (
                direct_measurement_parameters + geometry_response_parameters
                != parameters
            ):
                raise QualificationError(
                    "qualified parameters lack explicit L2 authority modes"
                )
    if protocol_rosters["zero_to_cad"] != protocol_rosters["functional_edit"]:
        raise QualificationError("zero-to-CAD and functional-edit rosters differ")
    if release.get("llm_judge_used") is not False:
        raise QualificationError("benchmark evaluator cannot use an LLM judge")
    return {
        "schema_version": "cadengbench-p.l0-l2-validation.v1",
        "valid": True,
        "release_hash": release["release_hash"],
        "dataset_release_id": release.get("dataset_release_id"),
        "item_count": len(item_ids),
        "case_count": len(item_ids) * 2,
        "protocols": protocol_reports,
        "llm_judge_used": False,
    }


def freeze_l0_l2_release(
    *,
    dataset_root: Path,
    profile: EvaluatorProfile,
    output_root: Path,
    reference_audit_path: Path,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    item_ids = _roster(dataset_root)
    reference_controls, reference_audit = _reference_controls(
        reference_audit_path.resolve(), item_ids
    )
    counts: Counter[str] = Counter()
    scoreable_dfm_slots = 0
    wall_dfm_items = 0
    hole_dfm_items = 0
    functional_dfm_slots = 0
    functional_wall_dfm_items = 0
    functional_hole_dfm_items = 0
    direct_measurement_slots = 0
    geometry_response_slots = 0
    for item_id in item_ids:
        source = dataset_root / "items" / item_id
        destination = output_root / "items" / item_id
        _link_item(source, destination)
        evaluation = destination / "evaluation"
        reference = reference_controls[item_id]
        qualification = build_qualification(
            source,
            source_reference=_mapping(
                reference.get("source_reference"), "source reference"
            ),
        )
        write_json(evaluation / "qualification.v2.json", qualification)
        write_json(
            evaluation / "functional_edit_acceptance.v2.json",
            build_functional_acceptance(source),
        )
        functional_dfm = build_functional_dfm(
            item_id,
            qualification["dfm"],
            _mapping(
                reference.get("functional_target_reference"),
                "functional target reference",
            ),
        )
        write_json(evaluation / "functional_dfm.v1.json", functional_dfm)
        counts[
            "benchcad"
            if item_id.startswith("p_benchcad_")
            else "fusion360_reconstruction"
        ] += 1
        scoreable_dfm_slots += int(qualification["dfm"]["scoreable_family_count"])
        wall_dfm_items += int(
            any(
                family["family"] == "minimum_wall_thickness"
                and family["scoreable"] is True
                for family in qualification["dfm"]["families"]
            )
        )
        contract_kind = qualification["parameter_system"]["variant_contract_mode"]
        direct_measurement_slots += int(contract_kind == "direct_measurement_only")
        geometry_response_slots += int(contract_kind == "geometry_response_only")
        functional_dfm_slots += int(
            functional_dfm["dfm"]["scoreable_family_count"]
        )
        functional_wall_dfm_items += int(
            any(
                family["family"] == "minimum_wall_thickness"
                and family["scoreable"] is True
                for family in functional_dfm["dfm"]["families"]
            )
        )
        functional_hole_dfm_items += int(
            any(
                family["family"] == "minimum_hole_geometry"
                and family["scoreable"] is True
                for family in functional_dfm["dfm"]["families"]
            )
        )
        hole_dfm_items += int(
            any(
                family["family"] == "minimum_hole_geometry"
                and family["scoreable"] is True
                for family in qualification["dfm"]["families"]
            )
        )
    zero_manifest = compile_evalset(
        output_root / "items",
        profile=profile,
        output_root=output_root / "zero_to_cad",
        protocol="zero_to_cad",
    )
    edit_manifest = compile_evalset(
        output_root / "items",
        profile=profile,
        output_root=output_root / "functional_edit",
        protocol="functional_edit",
    )
    frozen_reference_audit = output_root / "reference_contract_audit.json"
    write_json(frozen_reference_audit, reference_audit)
    release = {
        "schema_version": RELEASE_SCHEMA,
        "dataset_root": os.path.relpath(dataset_root, output_root),
        "dataset_root_path_semantics": "relative_to_release_root",
        "dataset_release_id": "cadengbench-p-300-v1-two-source",
        "dataset_item_ids_sha256": sha256_file(dataset_root / "item_ids.txt"),
        "item_count": len(item_ids),
        "source_counts": dict(sorted(counts.items())),
        "profile_id": profile.profile_id,
        "profile_hash": profile.digest,
        "dfm_capability_profile": DFM_CAPABILITY,
        "reference_contract_audit": {
            "path": "reference_contract_audit.json",
            "sha256": sha256_file(frozen_reference_audit),
        },
        "protocols": {
            "zero_to_cad": {
                "manifest_path": "zero_to_cad/manifest.json",
                "manifest_hash": zero_manifest["manifest_hash"],
                "item_count": zero_manifest["item_count"],
            },
            "functional_edit": {
                "manifest_path": "functional_edit/manifest.json",
                "manifest_hash": edit_manifest["manifest_hash"],
                "item_count": edit_manifest["item_count"],
            },
        },
        "coverage": {
            "zero_to_cad_cases": len(item_ids),
            "functional_edit_cases": len(item_ids),
            "qualified_parameter_slots_per_protocol": len(item_ids),
            "direct_measurement_parameter_slots": direct_measurement_slots,
            "geometry_response_parameter_slots": geometry_response_slots,
            "minimum_wall_thickness_items_zero_to_cad": wall_dfm_items,
            "minimum_hole_geometry_items_zero_to_cad": hole_dfm_items,
            "minimum_wall_thickness_items_functional_edit": (
                functional_wall_dfm_items
            ),
            "minimum_hole_geometry_items_functional_edit": (
                functional_hole_dfm_items
            ),
            "scoreable_dfm_slots_zero_to_cad": scoreable_dfm_slots,
            "scoreable_dfm_slots_functional_edit": functional_dfm_slots,
            "exact_reconstruction_cases": 0,
        },
        "llm_judge_used": False,
    }
    release["release_hash"] = sha256_json(release)
    write_json(output_root / "release_manifest.json", release)
    return release
