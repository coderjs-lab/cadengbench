"""Compile dataset records into small, immutable evaluator case manifests."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, sha256_json, write_json
from .profile import EvaluatorProfile

CASE_SCHEMA = "cadengbench-p.evaluator-case.v1"
DESIGN_SPEC_SCHEMA = "cadengbench-p.zero-to-cad-design-spec.v2"
CONSUMER_ITEM_SCHEMA = "cadengbench-p.consumer-ready-item.v2"
QUALIFICATION_SCHEMAS = {
    "cadengbench-p.evaluator-qualification.v1",
    "cadengbench-p.evaluator-qualification.v2",
    "cadengbench-p.evaluator-qualification.v3",
}


class CaseError(ValueError):
    """Dataset item cannot be compiled into an unambiguous evaluator case."""


@dataclass(frozen=True)
class LoadedCase:
    path: Path
    data: dict[str, Any]

    @property
    def item_id(self) -> str:
        return str(self.data["item_id"])

    @property
    def item_root(self) -> Path:
        raw = Path(str(self.data["source_item_root"]))
        if raw.is_absolute():
            return raw
        return (self.path.parent / raw).resolve()

    def artifact_path(self, key: str) -> Path:
        record = self.data.get("artifacts", {}).get(key)
        if not isinstance(record, Mapping):
            raise CaseError(f"case artifact {key!r} is absent")
        relative = record.get("item_relative_path")
        if not isinstance(relative, str) or not relative:
            raise CaseError(f"case artifact {key!r} has no relative path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CaseError(f"case artifact {key!r} escapes its item root")
        item_root = self.item_root.absolute()
        candidate = (item_root / relative_path).absolute()
        try:
            candidate.relative_to(item_root)
        except ValueError as exc:
            raise CaseError(f"case artifact {key!r} escapes its item root") from exc
        # The trusted evalset uses hash-bound symlinks into the immutable dataset.
        path = candidate.resolve()
        if not path.is_file():
            raise CaseError(f"case artifact {key!r} is missing: {path}")
        expected = record.get("sha256")
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise CaseError(f"case artifact {key!r} failed its SHA-256 binding")
        return path


def _source_family(item_id: str) -> str:
    if item_id.startswith("p_benchcad_"):
        return "benchcad"
    if item_id.startswith("p_fusion360_"):
        return "fusion360_reconstruction"
    raise CaseError(f"unsupported CADEngBench-P item ID: {item_id}")


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaseError(f"{name} must be an object")
    return value


def _as_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CaseError(f"{name} must be a list")
    return value


def _resolve_reference(
    item_root: Path,
    references: Iterable[Any],
    *,
    suffixes: tuple[str, ...],
    fallback_names: tuple[str, ...] = (),
) -> Path | None:
    for raw in references:
        if not isinstance(raw, str) or not raw.lower().endswith(suffixes):
            continue
        marker = f"/dataset/items/{item_root.name}/"
        if marker in raw:
            local = item_root / raw.split(marker, 1)[1]
            if local.is_file():
                return local.absolute()
        candidate = Path(raw)
        if candidate.is_file():
            return candidate.absolute()
        local = item_root / "source" / candidate.name
        if local.is_file():
            return local.absolute()
    for name in fallback_names:
        local = item_root / "source" / name
        if local.is_file():
            return local.absolute()
    return None


def _artifact_record(item_root: Path, path: Path) -> dict[str, Any]:
    try:
        relative = path.absolute().relative_to(item_root.absolute())
    except ValueError as exc:
        raise CaseError(f"authoritative artifact escapes item root: {path}") from exc
    return {
        "item_relative_path": relative.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _domain_for_parameter(record: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("qualified_domain", "domain"):
        raw = record.get(key)
        if isinstance(raw, Mapping):
            lower = raw.get("minimum", raw.get("min"))
            upper = raw.get("maximum", raw.get("max"))
            if (
                isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and float(lower) < float(upper)
            ):
                return {
                    "kind": "continuous",
                    "minimum": float(lower),
                    "maximum": float(upper),
                    "minimum_inclusive": bool(raw.get("minimum_inclusive", True)),
                    "maximum_inclusive": bool(raw.get("maximum_inclusive", True)),
                    "authority": raw.get("authority"),
                    "evidence_ref": raw.get("evidence_ref"),
                }
            values = raw.get("values")
            if (
                isinstance(values, list)
                and values
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values
                )
                and len({float(value) for value in values}) == len(values)
            ):
                return {
                    "kind": "discrete",
                    "values": [float(value) for value in values],
                    "authority": raw.get("authority"),
                    "evidence_ref": raw.get("evidence_ref"),
                }
    return None


def _measurement_binding(
    parameter_id: str,
    engineering_name: str,
    requirement_refs: list[str],
) -> dict[str, Any] | None:
    """Bind a public parameter to a source-neutral deterministic measurement."""

    envelope_axes = {
        "overall_length": "x",
        "overall_width": "y",
        "overall_height": "z",
    }
    axis = envelope_axes.get(engineering_name)
    if axis is not None and len(requirement_refs) == 1:
        return {
            "measurement_key": parameter_id,
            "measurement_kind": "bounding_box_axis",
            "requirement_id": requirement_refs[0],
            "requirement_value_path": [
                "value_and_tolerance",
                axis,
                "value",
            ],
            "axis": axis,
        }

    match = re.fullmatch(r"opening_group_(\d{3})_(diameter|depth)", engineering_name)
    if match is not None and len(requirement_refs) == 1:
        group_number, field = match.groups()
        return {
            "measurement_key": parameter_id,
            "measurement_kind": f"cylindrical_opening_{field}",
            "requirement_id": requirement_refs[0],
            "requirement_value_path": ["value_and_tolerance", field],
            "opening_group": int(group_number),
        }
    return None


def _variant_requirement_contract(record: Mapping[str, Any]) -> dict[str, str] | None:
    raw = record.get("variant_requirement_contract")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CaseError("variant_requirement_contract must be an object")
    allowed = {"kind", "authority", "evidence_ref"}
    extras = set(raw) - allowed
    if extras:
        raise CaseError(
            f"unsupported variant requirement contract fields: {sorted(extras)}"
        )
    kind = raw.get("kind")
    authority = raw.get("authority")
    evidence_ref = raw.get("evidence_ref")
    if kind not in {"direct_measurement_only", "geometry_response_only"}:
        raise CaseError(f"unsupported variant requirement contract kind: {kind!r}")
    if not isinstance(authority, str) or not authority:
        raise CaseError("variant requirement contract authority is required")
    if not isinstance(evidence_ref, str) or not evidence_ref:
        raise CaseError("variant requirement contract evidence_ref is required")
    return {
        "kind": kind,
        "authority": authority,
        "evidence_ref": evidence_ref,
    }


def _relation_tolerance(
    parameter: Mapping[str, Any],
    requirements: Mapping[str, Mapping[str, Any]],
) -> float:
    """Return the frozen tolerance for one public parameter measurement."""

    binding = parameter.get("measurement_binding")
    if not isinstance(binding, Mapping):
        raise CaseError(
            "relation parameter "
            f"{parameter.get('parameter_id')} has no measurement binding"
        )
    requirement = requirements.get(str(binding.get("requirement_id")))
    contract = (
        requirement.get("value_and_tolerance")
        if isinstance(requirement, Mapping)
        else None
    )
    if not isinstance(contract, Mapping):
        raise CaseError("relation measurement requirement is unresolved")
    kind = binding.get("measurement_kind")
    if kind == "cylindrical_opening_diameter":
        value = contract.get("diameter")
        tolerance = contract.get("diameter_tolerance")
    elif kind == "cylindrical_opening_depth":
        value = contract.get("depth")
        tolerance = contract.get("depth_tolerance")
    else:
        raise CaseError(f"unsupported relation measurement kind: {kind}")
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
        return float(tolerance)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CaseError("relation measurement has no numeric nominal or tolerance")
    return max(0.05, abs(float(value)) * 0.001)


def _normalize_source_relations(
    records: list[Any],
    parameters: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compile source relation records into the evaluator's typed relation DSL."""

    by_parameter = {str(value["parameter_id"]): value for value in parameters}
    by_requirement = {str(value.get("requirement_id")): value for value in requirements}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        record = _as_mapping(raw, "public constraint relation")
        relation_id = record.get("relation_id")
        relation_type = record.get("relation_type")
        parameter_ids = record.get("parameter_ids")
        if not isinstance(relation_id, str) or not relation_id or relation_id in seen:
            raise CaseError("public constraint relations require unique IDs")
        seen.add(relation_id)
        if relation_type != "equal_dimension_pattern":
            raise CaseError(
                f"unsupported public constraint relation type: {relation_type!r}"
            )
        if not isinstance(parameter_ids, list) or not parameter_ids:
            raise CaseError(f"relation {relation_id} has no parameter IDs")
        for parameter_id in parameter_ids:
            parameter = by_parameter.get(str(parameter_id))
            if parameter is None:
                raise CaseError(
                    f"relation {relation_id} references unknown parameter "
                    f"{parameter_id}"
                )
            normalized.append(
                {
                    "relation_id": f"{relation_id}:{parameter_id}",
                    "predicate": {
                        "type": "comparison",
                        "op": "eq",
                        "left": {
                            "type": "variable",
                            "source": "measurement",
                            "key": str(parameter_id),
                        },
                        "right": {
                            "type": "variable",
                            "source": "parameter",
                            "key": str(parameter_id),
                        },
                        "tolerance": _relation_tolerance(parameter, by_requirement),
                    },
                    "authority": "dataset_public_relation",
                    "evidence_ref": (
                        "zero_to_cad_design_spec.json#"
                        f"canonical_public_spec/public_constraint_relations/{index}"
                    ),
                    "required": True,
                }
            )
    return normalized


def _compile_parameter_system(
    canonical: Mapping[str, Any], requirements: list[dict[str, Any]]
) -> dict[str, Any]:
    raw_parameters = _as_list(canonical.get("public_parameters"), "public_parameters")
    parameters: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in raw_parameters:
        record = _as_mapping(raw, "public parameter")
        parameter_id = record.get("parameter_id")
        name = record.get("engineering_name")
        nominal = record.get("nominal")
        unit = record.get("unit")
        if (
            not isinstance(parameter_id, str)
            or not parameter_id
            or parameter_id in identifiers
            or not isinstance(name, str)
            or not name
            or not isinstance(nominal, (int, float))
            or isinstance(nominal, bool)
            or not isinstance(unit, str)
        ):
            raise CaseError(
                "public parameter records require unique IDs, names, units, "
                "and numeric nominals"
            )
        identifiers.add(parameter_id)
        requirement_refs = list(record.get("public_relation_refs") or [])
        parameter_kind = record.get("parameter_kind", "independent")
        if parameter_kind not in {"independent", "derived"}:
            raise CaseError(f"unsupported parameter_kind for {parameter_id}")
        dependencies = record.get("dependencies", [])
        derived_expression = record.get("derived_expression")
        if parameter_kind == "derived" and (
            not isinstance(dependencies, list)
            or not dependencies
            or not isinstance(derived_expression, Mapping)
        ):
            raise CaseError(
                f"derived parameter {parameter_id} requires dependencies and expression"
            )
        parameters.append(
            {
                "parameter_id": parameter_id,
                "engineering_name": name,
                "nominal": float(nominal),
                "unit": unit,
                "role": record.get("parameter_role"),
                "parameter_kind": parameter_kind,
                "dependencies": deepcopy(dependencies),
                "derived_expression": deepcopy(derived_expression),
                "requirement_refs": requirement_refs,
                "domain": (
                    None
                    if parameter_kind == "derived"
                    else _domain_for_parameter(record)
                ),
                "measurement_binding": _measurement_binding(
                    parameter_id,
                    name,
                    requirement_refs,
                ),
                "variant_requirement_contract": _variant_requirement_contract(record),
            }
        )
    source_relations = canonical.get("public_constraint_relations")
    if source_relations is None:
        source_relations = []
    if not isinstance(source_relations, list):
        raise CaseError("public_constraint_relations must be a list")
    relations = _normalize_source_relations(
        source_relations,
        parameters,
        requirements,
    )
    solver_obligations = canonical.get("solver_obligations")
    if solver_obligations is None:
        solver_obligations = []
    if not isinstance(solver_obligations, list):
        raise CaseError("solver_obligations must be a list")
    feasibility_relations = canonical.get("feasibility_relations")
    if feasibility_relations is None:
        feasibility_relations = []
    if not isinstance(feasibility_relations, list):
        raise CaseError("feasibility_relations must be a list")
    invalid_probes = canonical.get("invalid_probes")
    if invalid_probes is None:
        invalid_probes = []
    if not isinstance(invalid_probes, list):
        raise CaseError("invalid_probes must be a list")
    return {
        "interface_version": (
            canonical.get("parametric_submission_contract") or {}
        ).get("interface_version"),
        "parameters": parameters,
        "relations": relations,
        "source_relations": deepcopy(source_relations),
        "normalized_source_relation_count": len(relations),
        "solver_obligations": solver_obligations,
        "feasibility_relations": feasibility_relations,
        "invalid_probes": deepcopy(invalid_probes),
        "qualified_parameter_count": sum(
            parameter["parameter_kind"] == "independent"
            and parameter["domain"] is not None
            for parameter in parameters
        ),
        "measurement_bound_parameter_count": sum(
            parameter["measurement_binding"] is not None for parameter in parameters
        ),
        "qualified_variant_parameter_count": sum(
            parameter["parameter_kind"] == "independent"
            and parameter["domain"] is not None
            and parameter["variant_requirement_contract"] is not None
            for parameter in parameters
        ),
        "complete_domains": bool(parameters)
        and all(
            parameter["parameter_kind"] == "derived" or parameter["domain"] is not None
            for parameter in parameters
        ),
        "complete_measurement_bindings": bool(parameters)
        and all(
            parameter["measurement_binding"] is not None for parameter in parameters
        ),
    }


def _validate_measurement_bindings(
    parameter_system: Mapping[str, Any], requirements: list[dict[str, Any]]
) -> None:
    by_id = {
        str(requirement.get("requirement_id")): requirement
        for requirement in requirements
    }
    for parameter in parameter_system.get("parameters", []):
        if not isinstance(parameter, Mapping):
            continue
        binding = parameter.get("measurement_binding")
        if binding is None:
            continue
        if not isinstance(binding, Mapping):
            raise CaseError("measurement bindings must be objects or null")
        parameter_id = str(parameter.get("parameter_id"))
        if binding.get("measurement_key") != parameter_id:
            raise CaseError(f"measurement key mismatch for {parameter_id}")
        requirement = by_id.get(str(binding.get("requirement_id")))
        path = binding.get("requirement_value_path")
        if (
            not isinstance(requirement, Mapping)
            or not isinstance(path, list)
            or not path
        ):
            raise CaseError(f"measurement binding is unresolved for {parameter_id}")
        value: Any = requirement
        for component in path:
            if not isinstance(component, str) or not isinstance(value, Mapping):
                raise CaseError(f"measurement path is invalid for {parameter_id}")
            value = value.get(component)
        nominal = parameter.get("nominal")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isinstance(nominal, (int, float))
            or isinstance(nominal, bool)
            or not math.isclose(
                float(value),
                float(nominal),
                rel_tol=0.0,
                abs_tol=max(1e-5, abs(float(nominal)) * 1e-6),
            )
        ):
            raise CaseError(
                "public nominal and requirement measurement disagree for "
                f"{parameter_id}"
            )


def _apply_orientation_free_contract(
    task_contract: dict[str, Any],
    requirements: list[dict[str, Any]],
    parameter_system: dict[str, Any],
) -> None:
    """Remove hidden source-axis authority from zero-to-CAD scoring contracts."""

    task_contract["source_reference_frame"] = deepcopy(
        task_contract.get("reference_frame")
    )
    task_contract["reference_frame"] = "global_orientation_free"
    merged_requirements: list[dict[str, Any]] = []
    opening_groups: dict[float, dict[str, Any]] = {}
    for requirement in requirements:
        check_ids = requirement.get("check_ids")
        contract = requirement.get("value_and_tolerance")
        if (
            not isinstance(check_ids, list)
            or "hidden.check.cylindrical_opening_group.v1" not in check_ids
            or not isinstance(contract, dict)
            or isinstance(contract.get("diameter"), bool)
            or not isinstance(contract.get("diameter"), (int, float))
        ):
            merged_requirements.append(requirement)
            continue
        key = round(float(contract["diameter"]), 8)
        existing = opening_groups.get(key)
        if existing is None:
            existing = requirement
            existing["source_requirement_ids"] = [requirement["requirement_id"]]
            opening_groups[key] = existing
            merged_requirements.append(existing)
            continue
        existing_contract = existing["value_and_tolerance"]
        existing_contract["quantity"] = int(existing_contract["quantity"]) + int(
            contract["quantity"]
        )
        existing["source_requirement_ids"].append(requirement["requirement_id"])
    requirements[:] = merged_requirements
    for requirement in requirements:
        check_ids = requirement.get("check_ids")
        contract = requirement.get("value_and_tolerance")
        if not isinstance(check_ids, list) or not isinstance(contract, dict):
            continue
        if "hidden.check.bbox_dimensions.v1" in check_ids:
            contract["axis_assignment"] = "free_permutation"
        if "hidden.check.cylindrical_opening_group.v1" in check_ids:
            # Prompt v2 exposes opening count/diameter and visual placement, but
            # does not expose numeric depth, world axis, or center coordinates.
            contract["axis_assignment"] = "free"
            contract["depth_scoring"] = "not_disclosed"
            contract["position_scoring"] = "not_disclosed"
    for parameter in parameter_system.get("parameters", []):
        if not isinstance(parameter, dict):
            continue
        binding = parameter.get("measurement_binding")
        if not isinstance(binding, dict):
            continue
        if binding.get("measurement_kind") != "bounding_box_axis":
            continue
        source_axis = binding.pop("axis", None)
        binding["measurement_kind"] = "bounding_box_unordered_dimension"
        if source_axis in {"x", "y", "z"}:
            binding["source_axis"] = source_axis


def _compile_requirements(
    design_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ledger = _as_list(
        design_spec.get("public_requirement_ledger"), "public_requirement_ledger"
    )
    hidden = _as_mapping(design_spec.get("hidden_acceptance"), "hidden_acceptance")
    bindings = _as_mapping(
        hidden.get("requirement_to_check_refs"), "requirement_to_check_refs"
    )
    compiled: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in ledger:
        requirement = deepcopy(dict(_as_mapping(raw, "requirement")))
        requirement_id = requirement.get("requirement_id")
        if not isinstance(requirement_id, str) or requirement_id in seen:
            raise CaseError("requirements require unique string IDs")
        seen.add(requirement_id)
        check_ids = bindings.get(requirement_id)
        if not isinstance(check_ids, list) or not all(
            isinstance(check_id, str) for check_id in check_ids
        ):
            raise CaseError(
                f"requirement {requirement_id} lacks deterministic check bindings"
            )
        requirement["check_ids"] = list(check_ids)
        compiled.append(requirement)
    return compiled


def _compile_dfm(item_root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = item_root / "domain/manufacturing/v2/material_process_dfm_annotations.json"
    if not path.is_file():
        return {"families": [], "scoreable_family_count": 0}, None
    raw = _as_mapping(read_json(path), "DFM annotation")
    measurements = {
        str(value.get("check_family")): value
        for value in raw.get("deterministic_dfm_measurements", [])
        if isinstance(value, Mapping) and isinstance(value.get("check_family"), str)
    }
    families: list[dict[str, Any]] = []
    for contract in raw.get("dfm_check_contracts", []):
        if not isinstance(contract, Mapping):
            continue
        family = str(contract.get("check_id"))
        measurement = measurements.get(family, {})
        if family == "minimum_wall_thickness":
            thresholds = {"minimum_mm": measurement.get("threshold")}
            extractor_certified = (
                measurement.get("measurement_kind")
                == "certified_global_minimum_thickness"
            )
        elif family == "minimum_hole_geometry":
            thresholds = {
                "minimum_diameter_mm": measurement.get("diameter_threshold"),
                "maximum_depth_to_diameter": measurement.get("depth_ratio_threshold"),
            }
            extractor_certified = True
        else:
            continue
        threshold_values = list(thresholds.values())
        thresholds_complete = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in threshold_values
        )
        evidence_backed = contract.get("threshold_ref") is not None
        scoreable = thresholds_complete and evidence_backed and extractor_certified
        families.append(
            {
                "family": family,
                "applicability": contract.get("applicability"),
                "required": bool(contract.get("required", False)),
                "thresholds": thresholds,
                "threshold_evidence_ref": contract.get("threshold_ref"),
                "extractor_certified": extractor_certified,
                "expected_minimum_feature_count": (
                    int(measurement["hole_count"])
                    if family == "minimum_hole_geometry"
                    and isinstance(measurement.get("hole_count"), int)
                    and not isinstance(measurement.get("hole_count"), bool)
                    and int(measurement["hole_count"]) >= 0
                    else None
                ),
                "scoreable": scoreable,
                "unscoreable_reason": None
                if scoreable
                else (
                    "certified public threshold and extractor authority are incomplete"
                ),
            }
        )
    return {
        "profile_origin": "dataset_annotation",
        "process_profile": raw.get("process_profile"),
        "material_card": raw.get("material_card"),
        "families": families,
        "scoreable_family_count": sum(value["scoreable"] for value in families),
    }, _artifact_record(item_root, path)


def _optional_artifact(item_root: Path, relative: str) -> dict[str, Any] | None:
    path = item_root / relative
    return _artifact_record(item_root, path) if path.is_file() else None


def _semantic_case_payload(case: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(case))
    value.pop("case_hash", None)
    value.pop("source_item_root", None)
    return value


def _source_item_root_record(item_root: Path, output_path: Path | None) -> str:
    """Record a relocatable root without changing the semantic case hash."""

    if output_path is None:
        return str(item_root)
    return os.path.relpath(item_root, output_path.parent)


def _validate_qualified_parameter_system(
    parameter_system: Mapping[str, Any], requirements: list[dict[str, Any]]
) -> None:
    if (
        parameter_system.get("interface_version")
        != "cadengbench-p.cadquery-submission.v1"
    ):
        raise CaseError("qualification uses an unsupported submission interface")
    parameters = _as_list(
        parameter_system.get("parameters"), "qualification parameters"
    )
    if not parameters:
        raise CaseError("qualification must expose at least one public parameter")
    identifiers: set[str] = set()
    for raw in parameters:
        parameter = _as_mapping(raw, "qualification parameter")
        parameter_id = parameter.get("parameter_id")
        if (
            not isinstance(parameter_id, str)
            or not parameter_id
            or parameter_id in identifiers
        ):
            raise CaseError("qualification parameter IDs must be unique strings")
        identifiers.add(parameter_id)
        for field in ("engineering_name", "unit", "role"):
            if not isinstance(parameter.get(field), str) or not parameter.get(field):
                raise CaseError(f"qualification parameter {parameter_id} lacks {field}")
        nominal = parameter.get("nominal")
        if (
            isinstance(nominal, bool)
            or not isinstance(nominal, (int, float))
            or not math.isfinite(float(nominal))
        ):
            raise CaseError(f"qualification parameter {parameter_id} lacks a nominal")
        parameter_kind = parameter.get("parameter_kind", "independent")
        if parameter_kind not in {"independent", "derived"}:
            raise CaseError(
                f"qualification parameter {parameter_id} has invalid parameter_kind"
            )
        if parameter_kind == "independent" and _domain_for_parameter(parameter) is None:
            raise CaseError(
                f"qualification parameter {parameter_id} lacks a bounded domain"
            )
        if parameter_kind == "derived":
            dependencies = parameter.get("dependencies")
            expression = parameter.get("derived_expression")
            if (
                not isinstance(dependencies, list)
                or not dependencies
                or not isinstance(expression, Mapping)
                or parameter.get("domain") is not None
            ):
                raise CaseError(
                    f"qualification derived parameter {parameter_id} is incomplete"
                )
        variant_contract = _variant_requirement_contract(parameter)
        if parameter_kind == "independent" and variant_contract is None:
            raise CaseError(
                f"qualification parameter {parameter_id} lacks variant semantics"
            )
        if parameter_kind == "independent" and variant_contract is not None:
            binding = parameter.get("measurement_binding")
            kind = variant_contract["kind"]
            if kind == "direct_measurement_only" and not isinstance(binding, Mapping):
                raise CaseError(
                    f"qualification parameter {parameter_id} lacks its direct "
                    "measurement binding"
                )
            if kind == "geometry_response_only" and binding is not None:
                raise CaseError(
                    f"qualification parameter {parameter_id} mixes response-only "
                    "and direct-measurement authority"
                )
    for field in ("relations", "solver_obligations", "feasibility_relations"):
        if not isinstance(parameter_system.get(field), list):
            raise CaseError(f"qualification parameter_system.{field} must be a list")
    if parameter_system.get("invalid_probes") is not None and not isinstance(
        parameter_system.get("invalid_probes"), list
    ):
        raise CaseError("qualification parameter_system.invalid_probes must be a list")
    _validate_measurement_bindings(parameter_system, requirements)


def _qualified_optional_artifacts(item_root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for key, relative in (
        (
            "parametric_edit_annotation",
            "domain/parametric_edits/v2/parametric_edit_annotations.json",
        ),
        (
            "functional_edit_contract",
            "completion/v2/parametric/functional_edit_contract.json",
        ),
        (
            "functional_edit_target_oracle",
            "completion/v2/parametric/edit_target_oracle.json",
        ),
        (
            "functional_edit_acceptance_contract",
            "completion/v2/task_prompts/deterministic_acceptance_contract.json",
        ),
        ("functional_dfm_contract", "evaluation/functional_dfm.v1.json"),
        (
            "functional_edit_code_only_prompt",
            "completion/v2/task_prompts/code_only.json",
        ),
        (
            "functional_edit_agentic_prompt",
            "completion/v2/task_prompts/agentic_python_cad.json",
        ),
        ("replay_adapter", "completion/v2/parametric/replay_adapter.json"),
    ):
        record = _optional_artifact(item_root, relative)
        if record is not None:
            artifacts[key] = record
    if "functional_edit_acceptance_contract" not in artifacts:
        record = _optional_artifact(
            item_root,
            "evaluation/functional_edit_acceptance.v2.json",
        )
        if record is not None:
            artifacts["functional_edit_acceptance_contract"] = record
    return artifacts


def _configure_nominal_task_contract(
    task_contract: dict[str, Any],
    *,
    source_fidelity_profile: str,
) -> None:
    objective = task_contract.get("task_objective", "engineering_specification")
    if objective not in {"engineering_specification", "exact_reconstruction"}:
        raise CaseError(f"unsupported nominal task objective: {objective!r}")
    task_contract["task_objective"] = objective
    task_contract["source_fidelity_profile"] = source_fidelity_profile
    if objective == "exact_reconstruction":
        if not isinstance(
            task_contract.get("reference_similarity_thresholds"), Mapping
        ):
            raise CaseError(
                "exact_reconstruction requires reference_similarity_thresholds"
            )
        task_contract["fidelity_profile"] = "exact_reference"
        task_contract["l1_scoring_authority"] = (
            "typed_requirements_applicable_dfm_and_reference_similarity"
        )
        task_contract["reference_geometry_role"] = (
            "score_bearing_exact_reconstruction_target"
        )
        return
    if "reference_similarity_thresholds" in task_contract:
        raise CaseError(
            "reference_similarity_thresholds are reserved for exact_reconstruction"
        )
    task_contract["fidelity_profile"] = "requirements_only"
    task_contract["l1_scoring_authority"] = (
        "typed_engineering_requirements_and_applicable_dfm"
    )
    task_contract["reference_geometry_role"] = "provenance_and_non_scoring_diagnostics"


def _compile_qualified_production_case(
    item_root: Path,
    *,
    profile: EvaluatorProfile,
    output_path: Path | None,
) -> dict[str, Any]:
    """Compile a consumer-ready production item plus a frozen qualification.

    Production dataset records intentionally do not contain evaluator thresholds.
    The qualification is therefore a separate, evaluator-owned and hash-bound
    artifact. This keeps benchmark policy out of the immutable source dataset.
    """

    item_id = item_root.name
    consumer_path = item_root / "benchmark/consumer_ready_item_v2.json"
    qualification_path = item_root / "evaluation/qualification.v2.json"
    if not qualification_path.is_file():
        qualification_path = item_root / "evaluation/qualification.v1.json"
    if not consumer_path.is_file():
        raise CaseError(f"consumer-ready item is missing: {consumer_path}")
    if not qualification_path.is_file():
        raise CaseError(
            f"production item requires an evaluator qualification: {qualification_path}"
        )
    consumer = _as_mapping(read_json(consumer_path), "consumer-ready item")
    qualification = _as_mapping(
        read_json(qualification_path), "evaluator qualification"
    )
    if consumer.get("schema_version") != CONSUMER_ITEM_SCHEMA:
        raise CaseError(
            f"unsupported consumer-ready item: {consumer.get('schema_version')}"
        )
    if qualification.get("schema_version") not in QUALIFICATION_SCHEMAS:
        raise CaseError(
            "unsupported evaluator qualification: "
            f"{qualification.get('schema_version')}"
        )
    if consumer.get("item_id") != item_id or qualification.get("item_id") != item_id:
        raise CaseError("item directory, consumer record, and qualification IDs differ")

    source = _as_mapping(consumer.get("source"), "consumer source")
    geometry = _resolve_reference(
        item_root,
        [source.get("canonical_geometry_artifact")],
        suffixes=(".step", ".stp"),
        fallback_names=(
            "generated_from_cadquery.step",
            "fusion360_occ_reconstruction.step",
        ),
    )
    if geometry is None:
        raise CaseError("authoritative target STEP could not be resolved")
    history = _resolve_reference(
        item_root,
        [source.get("native_history_artifact")],
        suffixes=(".py", ".json"),
        fallback_names=("cadquery.py",),
    )

    requirements = deepcopy(
        _as_list(qualification.get("requirements"), "qualification requirements")
    )
    parameter_system = deepcopy(
        dict(
            _as_mapping(
                qualification.get("parameter_system"),
                "qualification parameter_system",
            )
        )
    )
    _validate_qualified_parameter_system(parameter_system, requirements)
    task_contract = deepcopy(
        dict(
            _as_mapping(
                qualification.get("task_contract"), "qualification task_contract"
            )
        )
    )
    source_fidelity_profile = task_contract.get("fidelity_profile")
    if source_fidelity_profile not in {"exact_reference", "requirements_only"}:
        raise CaseError(f"unsupported fidelity profile: {source_fidelity_profile!r}")
    _configure_nominal_task_contract(
        task_contract,
        source_fidelity_profile=str(source_fidelity_profile),
    )
    _apply_orientation_free_contract(task_contract, requirements, parameter_system)
    _validate_measurement_bindings(parameter_system, requirements)
    dfm = deepcopy(dict(_as_mapping(qualification.get("dfm"), "qualification DFM")))
    families = dfm.get("families")
    if not isinstance(families, list):
        raise CaseError("qualification DFM families must be a list")
    scoreable_count = sum(
        isinstance(value, Mapping) and value.get("scoreable") is True
        for value in families
    )
    if dfm.get("scoreable_family_count") != scoreable_count:
        raise CaseError("qualification DFM scoreable-family count is inconsistent")

    artifacts: dict[str, Any] = {
        "consumer_item": _artifact_record(item_root, consumer_path),
        "evaluator_qualification": _artifact_record(item_root, qualification_path),
        "target_step": _artifact_record(item_root, geometry),
    }
    if history is not None:
        artifacts["reference_history"] = _artifact_record(item_root, history)
    artifacts.update(_qualified_optional_artifacts(item_root))

    case: dict[str, Any] = {
        "schema_version": CASE_SCHEMA,
        "item_id": item_id,
        "source_family": _source_family(item_id),
        "protocol": "zero_to_cad",
        "source_item_root": _source_item_root_record(item_root, output_path),
        "task_contract": task_contract,
        "requirements": requirements,
        "parameter_system": parameter_system,
        "dfm": dfm,
        "functional_edit": {
            "available": False,
            "dataset_contract_available": "functional_edit_contract" in artifacts,
            "separate_protocol_available": all(
                key in artifacts
                for key in (
                    "functional_edit_contract",
                    "functional_edit_target_oracle",
                    "functional_edit_acceptance_contract",
                )
            ),
            "zero_to_cad_layer_status": "not_applicable_separate_protocol",
            "adapter_kind": "cadquery_submission"
            if item_id.startswith("p_benchcad_")
            else "fusion360_json_patch_occ_replay",
        },
        "artifacts": artifacts,
        "evaluator_profile": deepcopy(profile.data),
        "profile_hash": profile.digest,
    }
    case["case_hash"] = sha256_json(_semantic_case_payload(case))
    if output_path is not None:
        write_json(output_path, case)
    return case


def compile_case(
    item_root: Path,
    *,
    profile: EvaluatorProfile,
    output_path: Path | None = None,
) -> dict[str, Any]:
    item_root = item_root.resolve()
    if not item_root.is_dir():
        raise CaseError(f"item root is not a directory: {item_root}")
    item_id = item_root.name
    qualification_path = item_root / "evaluation/qualification.v2.json"
    if not qualification_path.is_file():
        qualification_path = item_root / "evaluation/qualification.v1.json"
    if qualification_path.is_file():
        return _compile_qualified_production_case(
            item_root,
            profile=profile,
            output_path=output_path,
        )
    design_path = item_root / "zero_to_cad/compiled/zero_to_cad_design_spec.json"
    if not design_path.is_file():
        return _compile_qualified_production_case(
            item_root,
            profile=profile,
            output_path=output_path,
        )
    design = _as_mapping(read_json(design_path), "zero-to-CAD design specification")
    if design.get("schema_version") != DESIGN_SPEC_SCHEMA:
        raise CaseError(
            f"unsupported design specification: {design.get('schema_version')}"
        )
    if design.get("item_id") != item_id:
        raise CaseError("item directory and design-spec item IDs differ")
    canonical = _as_mapping(
        design.get("canonical_public_spec"), "canonical_public_spec"
    )
    hidden = _as_mapping(design.get("hidden_acceptance"), "hidden_acceptance")
    geometry = _resolve_reference(
        item_root,
        hidden.get("reference_geometry_refs", []),
        suffixes=(".step", ".stp"),
        fallback_names=(
            "generated_from_cadquery.step",
            "fusion360_occ_reconstruction.step",
        ),
    )
    if geometry is None:
        raise CaseError("authoritative target STEP could not be resolved")
    history = _resolve_reference(
        item_root,
        hidden.get("reference_history_refs", []),
        suffixes=(".py", ".json"),
        fallback_names=("cadquery.py",),
    )
    dfm, dfm_artifact = _compile_dfm(item_root)
    artifacts: dict[str, Any] = {
        "design_spec": _artifact_record(item_root, design_path),
        "target_step": _artifact_record(item_root, geometry),
    }
    if history is not None:
        artifacts["reference_history"] = _artifact_record(item_root, history)
    if dfm_artifact is not None:
        artifacts["dfm_annotation"] = dfm_artifact
    for key, relative in (
        (
            "parametric_edit_annotation",
            "domain/parametric_edits/v2/parametric_edit_annotations.json",
        ),
        (
            "functional_edit_contract",
            "completion/v2/parametric/functional_edit_contract.json",
        ),
        (
            "functional_edit_target_oracle",
            "completion/v2/parametric/edit_target_oracle.json",
        ),
        (
            "functional_edit_acceptance_contract",
            "completion/v2/task_prompts/deterministic_acceptance_contract.json",
        ),
        (
            "functional_edit_code_only_prompt",
            "completion/v2/task_prompts/code_only.json",
        ),
        (
            "functional_edit_agentic_prompt",
            "completion/v2/task_prompts/agentic_python_cad.json",
        ),
        ("replay_adapter", "completion/v2/parametric/replay_adapter.json"),
    ):
        record = _optional_artifact(item_root, relative)
        if record is not None:
            artifacts[key] = record
    requirements = _compile_requirements(design)
    parameter_system = _compile_parameter_system(canonical, requirements)
    task_contract = {
        "task_kind": design.get("task_kind"),
        "task_objective": canonical.get(
            "task_objective", design.get("task_objective", "engineering_specification")
        ),
        "submission": design.get("model_environment_contract"),
        "solid_contract": canonical.get("solid_contract"),
        "reference_frame": canonical.get("reference_frame"),
        "required_features": deepcopy(canonical.get("required_features", [])),
        "allowed_design_freedom": deepcopy(canonical.get("allowed_design_freedom", {})),
        "source_fidelity_profile": (
            canonical.get("fidelity_profile") or "exact_reference"
        ),
    }
    if task_contract["task_objective"] == "exact_reconstruction":
        task_contract["reference_similarity_thresholds"] = deepcopy(
            hidden.get("reference_similarity_thresholds")
        )
    _apply_orientation_free_contract(task_contract, requirements, parameter_system)
    _validate_measurement_bindings(parameter_system, requirements)
    source_fidelity_profile = canonical.get("fidelity_profile") or "exact_reference"
    if source_fidelity_profile not in {"exact_reference", "requirements_only"}:
        raise CaseError(f"unsupported fidelity profile: {source_fidelity_profile!r}")
    _configure_nominal_task_contract(
        task_contract,
        source_fidelity_profile=str(source_fidelity_profile),
    )
    case: dict[str, Any] = {
        "schema_version": CASE_SCHEMA,
        "item_id": item_id,
        "source_family": _source_family(item_id),
        "protocol": "zero_to_cad",
        "source_item_root": _source_item_root_record(item_root, output_path),
        "task_contract": task_contract,
        "requirements": requirements,
        "parameter_system": parameter_system,
        "dfm": dfm,
        "functional_edit": {
            "available": False,
            "dataset_contract_available": "functional_edit_contract" in artifacts,
            "separate_protocol_available": all(
                key in artifacts
                for key in (
                    "functional_edit_contract",
                    "functional_edit_target_oracle",
                    "functional_edit_acceptance_contract",
                )
            ),
            "zero_to_cad_layer_status": "not_applicable_separate_protocol",
            "adapter_kind": "cadquery_submission"
            if item_id.startswith("p_benchcad_")
            else "fusion360_json_patch_occ_replay",
        },
        "artifacts": artifacts,
        "evaluator_profile": deepcopy(profile.data),
        "profile_hash": profile.digest,
    }
    case["case_hash"] = sha256_json(_semantic_case_payload(case))
    if output_path is not None:
        write_json(output_path, case)
    return case


def _acceptance_quantity(raw: Any, field: str) -> float:
    value = raw.get("value") if isinstance(raw, Mapping) else raw
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CaseError(f"functional-edit acceptance {field} must be finite")
    return float(value)


def compile_functional_case(
    item_root: Path,
    *,
    profile: EvaluatorProfile,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compile the separate source-native Protocol-E evaluator case."""

    case = compile_case(item_root, profile=profile)
    artifacts = case["artifacts"]
    acceptance_record = artifacts.get("functional_edit_acceptance_contract")
    if not isinstance(acceptance_record, Mapping):
        raise CaseError(
            "functional-edit protocol requires a deterministic acceptance contract"
        )
    acceptance_path = item_root / str(acceptance_record["item_relative_path"])
    acceptance = _as_mapping(
        read_json(acceptance_path), "functional-edit acceptance contract"
    )
    if acceptance.get("schema_version") != (
        "cadengbench-p.functional-edit-acceptance.v2"
    ):
        raise CaseError("unsupported functional-edit acceptance contract")
    if acceptance.get("status") != "verified_compiled":
        raise CaseError("functional-edit acceptance contract is not verified")
    if acceptance.get("item_id") != case["item_id"]:
        raise CaseError("functional-edit acceptance item ID differs")
    source_key = acceptance.get("source_key")
    expected_source = "benchcad" if case["source_family"] == "benchcad" else "fusion360"
    if source_key != expected_source:
        raise CaseError("functional-edit acceptance source differs from item source")

    parameter = _as_mapping(
        acceptance.get("parameter_contract"),
        "functional-edit parameter contract",
    )
    domain = _as_mapping(parameter.get("domain"), "functional-edit domain")
    lower = _acceptance_quantity(domain.get("lower"), "domain lower")
    upper = _acceptance_quantity(domain.get("upper"), "domain upper")
    target = _acceptance_quantity(parameter.get("target_value"), "target value")
    base = _acceptance_quantity(parameter.get("base_value"), "base value")
    if not lower < upper or not lower <= target <= upper:
        raise CaseError("functional-edit target is outside its qualified domain")
    unit = parameter.get("canonical_unit")
    key = parameter.get("parameter_key")
    if not isinstance(unit, str) or not unit or not isinstance(key, str) or not key:
        raise CaseError("functional-edit parameter identity is incomplete")

    prompt_artifact = artifacts.get("functional_edit_code_only_prompt")
    prompt_contract: Mapping[str, Any] = {}
    if isinstance(prompt_artifact, Mapping):
        prompt_path = item_root / str(prompt_artifact["item_relative_path"])
        prompt = _as_mapping(read_json(prompt_path), "functional-edit prompt")
        prompt_contract = _as_mapping(
            (prompt.get("public_task") or {}).get("expected_submission_contract"),
            "functional-edit submission contract",
        )

    case["protocol"] = "functional_edit"
    functional_dfm_record = artifacts.get("functional_dfm_contract")
    if isinstance(functional_dfm_record, Mapping):
        functional_dfm_path = item_root / str(
            functional_dfm_record["item_relative_path"]
        )
        functional_dfm = _as_mapping(
            read_json(functional_dfm_path), "functional-edit DFM contract"
        )
        if (
            functional_dfm.get("schema_version")
            != "cadengbench-p.functional-dfm.v1"
            or functional_dfm.get("item_id") != case["item_id"]
        ):
            raise CaseError("functional-edit DFM contract is invalid")
        case["dfm"] = deepcopy(
            dict(_as_mapping(functional_dfm.get("dfm"), "functional-edit DFM"))
        )
    case["task_contract"] = {
        "task_kind": "functional_edit",
        "task_objective": "functional_edit",
        "submission": deepcopy(dict(prompt_contract)),
        "fidelity_profile": "local_edit_target",
        "l1_scoring_authority": (
            "edited_artifact_validity_applicable_dfm_and_declared_invariants"
        ),
        "target_shape_policy": "exact_unique_source_bound_edit",
    }
    case["parameter_system"] = {
        "interface_version": "cadengbench-p.functional-edit-parameter.v1",
        "parameters": [
            {
                "parameter_id": "E-P-001",
                "engineering_name": key,
                "unit": unit,
                "role": "functional_edit_target",
                "nominal": target,
                "base_value": base,
                "source_parameter_id": parameter.get("parameter_id"),
                "source_pointer": parameter.get("source_pointer"),
                "domain": {
                    "kind": "continuous",
                    "minimum": lower,
                    "maximum": upper,
                    "minimum_inclusive": True,
                    "maximum_inclusive": True,
                    "authority": "verified_functional_edit_replay",
                    "evidence_ref": (
                        "functional_edit_acceptance_contract#/parameter_contract/domain"
                    ),
                },
            }
        ],
        "relations": [],
        "solver_obligations": [],
        "feasibility_relations": [],
        "invalid_probes": [],
        "qualified_parameter_count": 1,
        "complete_domains": True,
    }
    case["functional_edit"] = {
        "available": True,
        "contract_id": acceptance.get("contract_id"),
        "adapter_kind": (
            "cadquery_source_edit"
            if source_key == "benchcad"
            else "fusion360_bounded_json_patch_occ_replay"
        ),
        "target_parameter_id": "E-P-001",
        "target_value": target,
        "base_value": base,
        "unit": unit,
        "target_shape_policy": "exact_unique_source_bound_edit",
        "post_edit_replay_required": True,
        "llm_judge_used": False,
    }
    case["source_item_root"] = _source_item_root_record(item_root, output_path)
    case["case_hash"] = sha256_json(_semantic_case_payload(case))
    if output_path is not None:
        write_json(output_path, case)
    return case


def load_case(path: Path, *, verify_artifacts: bool = True) -> LoadedCase:
    raw = read_json(path)
    if not isinstance(raw, dict) or raw.get("schema_version") != CASE_SCHEMA:
        raise CaseError(f"{path} is not a {CASE_SCHEMA} case")
    expected = raw.get("case_hash")
    if (
        not isinstance(expected, str)
        or sha256_json(_semantic_case_payload(raw)) != expected
    ):
        raise CaseError("case manifest hash is invalid")
    loaded = LoadedCase(path=path.resolve(), data=raw)
    if verify_artifacts:
        for key in raw.get("artifacts", {}):
            loaded.artifact_path(str(key))
    return loaded


def compile_evalset(
    items_root: Path,
    *,
    profile: EvaluatorProfile,
    output_root: Path,
    protocol: str = "zero_to_cad",
) -> dict[str, Any]:
    if protocol not in {"zero_to_cad", "functional_edit"}:
        raise CaseError(f"unsupported evalset protocol: {protocol}")
    compiler = (
        compile_functional_case if protocol == "functional_edit" else compile_case
    )
    item_roots = sorted(path for path in items_root.iterdir() if path.is_dir())
    records: list[dict[str, Any]] = []
    for item_root in item_roots:
        output = output_root / "cases" / f"{item_root.name}.case.json"
        case = compiler(item_root, profile=profile, output_path=output)
        records.append(
            {
                "item_id": case["item_id"],
                "protocol": case["protocol"],
                "source_family": case["source_family"],
                "case_path": output.relative_to(output_root).as_posix(),
                "case_hash": case["case_hash"],
                "scoreable_dfm_families": case["dfm"]["scoreable_family_count"],
                "separate_functional_edit_protocol_available": bool(
                    case["functional_edit"].get(
                        "separate_protocol_available",
                        case["functional_edit"].get("available", False),
                    )
                ),
                "qualified_parameter_count": case["parameter_system"][
                    "qualified_parameter_count"
                ],
            }
        )
    manifest = {
        "schema_version": "cadengbench-p.evalset-manifest.v1",
        "profile_id": profile.profile_id,
        "profile_hash": profile.digest,
        "protocol": protocol,
        "item_count": len(records),
        "items": records,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    write_json(output_root / "manifest.json", manifest)
    return manifest
