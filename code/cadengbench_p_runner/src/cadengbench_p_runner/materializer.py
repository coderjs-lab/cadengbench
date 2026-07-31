"""Materialize one synchronized dataset item into a canonical model request."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import (
    copy_file,
    read_json,
    semantic_hash,
    sha256_file,
    sha256_text,
    write_json,
    write_text,
)
from .schema import validate_canonical_request

REQUEST_SCHEMA = "cadengbench-p.canonical-model-request.v1"
REQUEST_V2_SCHEMA = "cadengbench-p.canonical-model-request.v2"
DESIGN_SPEC_SCHEMA = "cadengbench-p.zero-to-cad-design-spec.v2"
CONSUMER_READY_SCHEMA = "cadengbench-p.consumer-ready-item.v2"
RUNNER_DESIGN_SOURCE_SCHEMA = "cadengbench-p.runner-design-source.v1"
EVALUATOR_QUALIFICATION_SCHEMAS = {
    "cadengbench-p.evaluator-qualification.v1",
    "cadengbench-p.evaluator-qualification.v2",
    "cadengbench-p.evaluator-qualification.v3",
}
EDIT_ORACLE_SCHEMA = "cadengbench-p.edit-target-oracle.v2"


@dataclass(frozen=True)
class InputProfile:
    user_prompt: str
    images: tuple[tuple[str, str], ...]


INPUT_PROFILES = {
    "text": InputProfile("zero_to_cad.text.v2.user.txt", ()),
    "text_iso": InputProfile(
        "zero_to_cad.text_iso.v2.user.txt",
        (
            (
                "visuals/v2/core/iso_front_right_top.shaded.png",
                "isometric_front_right_top",
            ),
            (
                "visuals/v2/core/iso_rear_left_top.shaded.png",
                "isometric_rear_left_top",
            ),
        ),
    ),
    "text_engineering_views": InputProfile(
        "zero_to_cad.text_engineering_views.v2.user.txt",
        (
            (
                "visuals/v2/engineering_4view_visible_hidden.png",
                "engineering_four_view_visible_hidden",
            ),
        ),
    ),
}


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def _text_part(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"prompt is empty: {path}")
    return {"type": "text", "text": text, "sha256": sha256_text(text)}


def _resolve_project_relative_file(item_root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ValueError(f"consumer system prompt path is unsafe: {relative}")
    for root in (item_root, *item_root.parents):
        candidate = root / candidate_relative
        if candidate.is_file():
            return candidate
    raise ValueError(f"consumer system prompt is missing: {relative}")


def _load_public_qualification(
    item_root: Path, item_id: str
) -> tuple[Path, dict[str, Any]]:
    candidates = (
        item_root / "evaluation/qualification.v2.json",
        item_root / "evaluation/qualification.v1.json",
    )
    qualification_path = next((path for path in candidates if path.is_file()), None)
    if qualification_path is None:
        raise ValueError(f"public evaluator qualification is missing: {candidates[0]}")
    qualification = read_json(qualification_path)
    if (
        not isinstance(qualification, dict)
        or qualification.get("schema_version")
        not in EVALUATOR_QUALIFICATION_SCHEMAS
        or qualification.get("item_id") != item_id
    ):
        raise ValueError(f"evaluator qualification is malformed: {qualification_path}")
    return qualification_path, qualification


def _public_parameter_interface(
    item_root: Path, item_id: str
) -> tuple[str, dict[str, Any]]:
    qualification_path, qualification = _load_public_qualification(item_root, item_id)
    parameter_system = qualification.get("parameter_system")
    parameters = (
        parameter_system.get("parameters")
        if isinstance(parameter_system, dict)
        else None
    )
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("evaluator qualification has no public parameters")

    public_parameters: list[dict[str, Any]] = []
    for record in parameters:
        if not isinstance(record, dict):
            raise ValueError("evaluator qualification parameter is malformed")
        public = {
            key: record.get(key)
            for key in ("parameter_id", "engineering_name", "nominal", "unit", "role")
        }
        if (
            not isinstance(public["parameter_id"], str)
            or not isinstance(public["engineering_name"], str)
            or isinstance(public["nominal"], bool)
            or not isinstance(public["nominal"], (int, float))
            or not isinstance(public["unit"], str)
            or not isinstance(public["role"], str)
        ):
            raise ValueError("evaluator qualification public parameter is incomplete")
        public_parameters.append(public)

    default_lines = [
        f'    "{record["engineering_name"]}": {record["nominal"]!r},'
        for record in public_parameters
    ]
    manifest_lines: list[str] = []
    for record in public_parameters:
        manifest_lines.extend(
            [
                f'    "{record["parameter_id"]}": {{',
                f'        "name": "{record["engineering_name"]}",',
                f'        "unit": "{record["unit"]}",',
                f'        "role": "{record["role"]}",',
                "    },",
            ]
        )
    requirement_lines = ["Public deterministic acceptance requirements:"]
    public_requirements: list[dict[str, Any]] = []
    requirements = qualification.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("evaluator qualification has no public requirements")
    for requirement in requirements:
        if not isinstance(requirement, dict):
            raise ValueError("evaluator qualification requirement is malformed")
        requirement_id = str(requirement.get("requirement_id"))
        category = requirement.get("category")
        values = requirement.get("value_and_tolerance")
        if not isinstance(values, dict):
            raise ValueError(f"public requirement is incomplete: {requirement_id}")
        if category == "solid_contract":
            requirement_lines.append(
                f"- {requirement_id}: exactly {values['body_count']} solid body/bodies."
            )
        elif category == "global_envelope":
            dimensions = []
            for axis in ("x", "y", "z"):
                record = values.get(axis)
                if not isinstance(record, dict):
                    raise ValueError(f"public envelope requirement lacks {axis}")
                dimensions.append(
                    f"{axis.upper()}={record['value']}+/-{record['tolerance']} mm"
                )
            requirement_lines.append(
                f"- {requirement_id}: overall envelope " + ", ".join(dimensions) + "."
            )
        elif category == "cylindrical_opening_group":
            axis = values.get("axis_direction")
            requirement_lines.append(
                f"- {requirement_id}: {values['quantity']} cylindrical opening(s), "
                f"diameter {values['diameter']}+/-{values['diameter_tolerance']} mm, "
                f"depth {values['depth']}+/-{values['depth_tolerance']} mm, "
                f"axis direction {axis}."
            )
        else:
            raise ValueError(f"unsupported public requirement category: {category}")
        public_requirements.append(
            {
                "requirement_id": requirement_id,
                "category": category,
                "value_and_tolerance": values,
            }
        )
    dfm_lines, public_dfm = _public_dfm_projection(qualification)
    interface = "\n".join(
        [
            *requirement_lines,
            *dfm_lines,
            "",
            "Public parametric submission interface (use these entries exactly):",
            "DEFAULT_PARAMS = {",
            *default_lines,
            "}",
            "PARAMETER_MANIFEST = {",
            *manifest_lines,
            "}",
            "Every DEFAULT_PARAMS value must control its named geometric dimension",
            "when build(params) is replayed. Do not add private construction values",
            "DEFAULT_PARAMS or PARAMETER_MANIFEST.",
        ]
    )
    evidence = {
        "schema_version": qualification["schema_version"],
        "path": qualification_path.relative_to(item_root).as_posix(),
        "sha256": sha256_file(qualification_path),
        "public_parameters": public_parameters,
        "public_requirements": public_requirements,
        "public_dfm": public_dfm,
    }
    return interface, evidence


def _public_dfm_projection(
    qualification: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    dfm = qualification.get("dfm")
    families = dfm.get("families") if isinstance(dfm, dict) else None
    if not isinstance(families, list):
        raise ValueError("evaluator qualification has no DFM families")
    lines = [
        "Public geometry-capability requirements "
        "(not a material/process certification):"
    ]
    records: list[dict[str, Any]] = []
    for family in families:
        if not isinstance(family, dict) or family.get("scoreable") is not True:
            continue
        name = family.get("family")
        thresholds = family.get("thresholds")
        if not isinstance(thresholds, dict):
            raise ValueError(f"scoreable DFM family lacks thresholds: {name}")
        if name == "minimum_wall_thickness":
            lines.append(
                f"- Minimum wall thickness: {thresholds['minimum_mm']} mm."
            )
        elif name == "minimum_hole_geometry":
            lines.append(
                "- Each recognized cylindrical opening: diameter at least "
                f"{thresholds['minimum_diameter_mm']} mm and depth/diameter at "
                f"most {thresholds['maximum_depth_to_diameter']}."
            )
        else:
            raise ValueError(f"unsupported public DFM family: {name}")
        records.append({"family": name, "thresholds": thresholds})
    return lines, records


def _public_functional_dfm_interface(item_root: Path, item_id: str) -> str:
    _, qualification = _load_public_qualification(item_root, item_id)
    lines, _ = _public_dfm_projection(qualification)
    return "\n".join(lines)


def _materialize_consumer_prompt_sources(
    item_root: Path,
    *,
    run_root: Path,
    item_id: str,
    user_prompt_filename: str,
) -> tuple[Path, Path, Path]:
    consumer_path = _required_file(
        item_root / "benchmark/consumer_ready_item_v2.json",
        "consumer-ready item",
    )
    consumer = read_json(consumer_path)
    if (
        not isinstance(consumer, dict)
        or consumer.get("schema_version") != CONSUMER_READY_SCHEMA
        or consumer.get("item_id") != item_id
    ):
        raise ValueError(f"consumer-ready item is malformed: {consumer_path}")
    model_facing = consumer.get("model_facing")
    zero_to_cad = (
        model_facing.get("zero_to_cad") if isinstance(model_facing, dict) else None
    )
    if not isinstance(zero_to_cad, dict):
        raise ValueError("consumer-ready item has no model-facing zero-to-CAD block")
    system_prompt_ref = zero_to_cad.get("system_prompt")
    user_prompt = zero_to_cad.get("user_prompt")
    if not isinstance(system_prompt_ref, str) or not system_prompt_ref:
        raise ValueError("consumer-ready zero-to-CAD system prompt reference is absent")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("consumer-ready zero-to-CAD user prompt is absent")
    public_parameter_interface, qualification_evidence = _public_parameter_interface(
        item_root, item_id
    )
    materialized_user_prompt = (
        f"{user_prompt.rstrip()}\n\n{public_parameter_interface}\n"
    )

    source_root = run_root / "request_sources" / item_id
    system_source = _resolve_project_relative_file(item_root, system_prompt_ref)
    system_path = source_root / "model.system.txt"
    user_path = source_root / user_prompt_filename
    design_path = source_root / "zero_to_cad_design_source.json"
    copy_file(system_source, system_path)
    write_text(user_path, materialized_user_prompt)
    write_json(
        design_path,
        {
            "schema_version": RUNNER_DESIGN_SOURCE_SCHEMA,
            "item_id": item_id,
            "task_kind": "zero_to_cad",
            "source": {
                "schema_version": CONSUMER_READY_SCHEMA,
                "path": "benchmark/consumer_ready_item_v2.json",
                "sha256": sha256_file(consumer_path),
            },
            "model_facing_zero_to_cad": zero_to_cad,
            "public_parameter_interface": qualification_evidence,
            "resolved_system_prompt_sha256": sha256_file(system_path),
            "resolved_user_prompt_sha256": sha256_file(user_path),
        },
    )
    return design_path, system_path, user_path


def _prompt_source_paths(
    item_root: Path,
    *,
    run_root: Path,
    item_id: str,
    user_prompt_filename: str,
) -> tuple[Path, Path, Path]:
    compiled_root = item_root / "zero_to_cad/compiled"
    design_path = compiled_root / "zero_to_cad_design_spec.json"
    system_path = compiled_root / "model.system.txt"
    user_path = compiled_root / user_prompt_filename
    compiled_paths = (design_path, system_path, user_path)
    if all(path.is_file() for path in compiled_paths):
        design = read_json(design_path)
        if (
            not isinstance(design, dict)
            or design.get("schema_version") != DESIGN_SPEC_SCHEMA
            or design.get("item_id") != item_id
            or design.get("task_kind") != "zero_to_cad"
        ):
            raise ValueError(
                f"compiled zero-to-CAD design spec is malformed: {design_path}"
            )
        return design_path, system_path, user_path
    if any(path.exists() for path in compiled_paths):
        raise ValueError(
            f"compiled zero-to-CAD prompt bundle is incomplete: {compiled_root}"
        )
    return _materialize_consumer_prompt_sources(
        item_root,
        run_root=run_root,
        item_id=item_id,
        user_prompt_filename=user_prompt_filename,
    )


def _image_part(
    item_root: Path,
    run_root: Path,
    item_id: str,
    relative: str,
    role: str,
) -> dict[str, Any]:
    source = _required_file(item_root / relative, f"{role} image")
    destination_relative = f"request_assets/{item_id}/{source.name}"
    destination = run_root / destination_relative
    copy_file(source, destination)
    copied_hash = sha256_file(destination)
    source_hash = sha256_file(source)
    if copied_hash != source_hash:
        raise OSError(f"copied image failed digest verification: {source}")
    media_type = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
    return {
        "type": "image",
        "asset_path": destination_relative,
        "media_type": media_type,
        "sha256": copied_hash,
        "view_role": role,
    }


def _materialize_zero_to_cad_request(
    item_root: Path,
    *,
    run_root: Path,
    input_profile: str,
) -> dict[str, Any]:
    """Freeze prompts and ordered visual evidence for one item."""

    profile = INPUT_PROFILES.get(input_profile)
    if profile is None:
        raise ValueError(f"unsupported input profile: {input_profile}")
    item_root = item_root.resolve()
    item_id = item_root.name
    item_manifest_path = _required_file(item_root / "item.json", "item manifest")
    design_path, system_path, user_path = _prompt_source_paths(
        item_root,
        run_root=run_root,
        item_id=item_id,
        user_prompt_filename=profile.user_prompt,
    )
    item_manifest = read_json(item_manifest_path)
    if (
        not isinstance(item_manifest, dict)
        or item_manifest.get("dataset_version") is None
    ):
        raise ValueError(f"item manifest has no dataset_version: {item_manifest_path}")

    user_parts = [_text_part(user_path)]
    for image_relative, view_role in profile.images:
        user_parts.append(
            _image_part(
                item_root,
                run_root,
                item_id,
                image_relative,
                view_role,
            )
        )
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": f"{item_id}:{input_profile}:v1",
        "item_id": item_id,
        "task_kind": "zero_to_cad",
        "input_profile": input_profile,
        "dataset": {
            "release_id": str(item_manifest["dataset_version"]),
            "item_manifest_sha256": sha256_file(item_manifest_path),
            "design_spec_sha256": sha256_file(design_path),
        },
        "messages": [
            {"role": "system", "parts": [_text_part(system_path)]},
            {"role": "user", "parts": user_parts},
        ],
        "response_contract": {
            "artifact_kind": "python_source",
            "filename": "submission.py",
            "media_type": "text/x-python",
            "interface_version": "cadengbench-p.cadquery-submission.v1",
            "extraction_policy": "single_python_artifact_no_repair.v1",
        },
    }
    request["request_hash"] = semantic_hash(request, "request_hash")
    validate_canonical_request(request)
    return request


def _native_source_path(item_root: Path, consumer: dict[str, Any]) -> Path:
    source = consumer.get("source")
    reference = (
        source.get("native_history_artifact") if isinstance(source, dict) else None
    )
    if not isinstance(reference, str) or not reference:
        raise ValueError("consumer-ready item has no native history artifact")
    candidate = _required_file(
        item_root / "source" / Path(reference).name,
        "source-native editable history",
    )
    return candidate


def _functional_response_contract(source_key: str) -> dict[str, str]:
    if source_key == "benchcad":
        return {
            "artifact_kind": "python_source",
            "filename": "submission.py",
            "media_type": "text/x-python",
            "interface_version": "cadengbench-p.benchcad-source-edit.v1",
            "extraction_policy": "single_python_artifact_no_repair.v1",
        }
    if source_key == "fusion360":
        return {
            "artifact_kind": "json_document",
            "filename": "submission.json",
            "media_type": "application/json",
            "interface_version": "cadengbench-p.fusion360-patch-submission.v1",
            "extraction_policy": "single_json_artifact_no_repair.v1",
        }
    raise ValueError(f"unsupported functional-edit source: {source_key!r}")


def _functional_system_prompt(source_key: str) -> str:
    common = (
        "You are an AI CAD co-engineer completing a source-native engineering edit. "
        "Use only the supplied task evidence. Implement the requested change while "
        "preserving all non-target construction data and public invariants. Do not "
        "redesign, simplify, or approximate the part. The benchmark evaluates the "
        "returned artifact deterministically and does not use an LLM judge. "
    )
    if source_key == "benchcad":
        return common + (
            "Return only the complete edited executable Python source, with no "
            "Markdown fences or prose. Preserve the original source except for the "
            "smallest source-native edit needed to satisfy the instruction.\n"
        )
    return common + (
        "Return only one JSON object matching the supplied bounded Fusion patch "
        "schema, with no Markdown fences or prose. Identify the source JSON pointer "
        "from the supplied editable-parameter index and express its replacement in "
        "the declared source-native unit. The index contains exact pointers and "
        "values into the hash-bound full native history; it is not a reconstructed "
        "or approximate CAD representation.\n"
    )


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _vector_projection(value: object) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        number = _finite_number(value.get(axis))
        if number is None:
            return None
        result[axis] = number
    return result


def _fusion_editable_parameter_index(
    source_path: Path, oracle: dict[str, Any]
) -> dict[str, Any]:
    """Project every replay-supported numeric binding without exposing the oracle."""

    source = read_json(source_path)
    if not isinstance(source, dict):
        raise ValueError(f"Fusion native history is not an object: {source_path}")
    entities = source.get("entities")
    timeline = source.get("timeline")
    if not isinstance(entities, dict) or not isinstance(timeline, list):
        raise ValueError(
            f"Fusion native history lacks entities/timeline: {source_path}"
        )

    timeline_indices: dict[str, int] = {}
    for record in timeline:
        if not isinstance(record, dict):
            continue
        entity_id = record.get("entity")
        index = record.get("index")
        if isinstance(entity_id, str) and isinstance(index, int):
            timeline_indices[entity_id] = index

    editable: list[dict[str, Any]] = []
    for entity_id, entity in entities.items():
        if not isinstance(entity_id, str) or not isinstance(entity, dict):
            continue
        entity_token = _json_pointer_token(entity_id)
        owner = {
            "entity_id": entity_id,
            "name": str(entity.get("name", "")),
            "type": str(entity.get("type", "")),
            "timeline_index": timeline_indices.get(entity_id),
        }

        if entity.get("type") == "ExtrudeFeature":
            for extent_key in ("extent_one", "extent_two"):
                extent = entity.get(extent_key)
                distance = extent.get("distance") if isinstance(extent, dict) else None
                value = (
                    _finite_number(distance.get("value"))
                    if isinstance(distance, dict)
                    else None
                )
                if value is None:
                    continue
                editable.append(
                    {
                        "parameter_role": "extrude_distance",
                        "json_pointer": (
                            f"/entities/{entity_token}/{extent_key}/distance/value"
                        ),
                        "value": value,
                        "source_unit": "cm",
                        "owner": owner,
                        "operation": entity.get("operation"),
                        "extent": extent_key,
                    }
                )

        profiles = entity.get("profiles")
        if not isinstance(profiles, dict):
            continue
        for profile_id, profile in profiles.items():
            if not isinstance(profile_id, str) or not isinstance(profile, dict):
                continue
            loops = profile.get("loops")
            if not isinstance(loops, list):
                continue
            for loop_index, loop in enumerate(loops):
                if not isinstance(loop, dict):
                    continue
                curves = loop.get("profile_curves")
                if not isinstance(curves, list):
                    continue
                for curve_index, curve in enumerate(curves):
                    if not isinstance(curve, dict) or curve.get("type") != "Circle3D":
                        continue
                    radius = _finite_number(curve.get("radius"))
                    if radius is None:
                        continue
                    pointer = (
                        f"/entities/{entity_token}/profiles/"
                        f"{_json_pointer_token(profile_id)}/loops/{loop_index}/"
                        f"profile_curves/{curve_index}/radius"
                    )
                    editable.append(
                        {
                            "parameter_role": "sketch_circle_radius",
                            "json_pointer": pointer,
                            "value": radius,
                            "source_unit": "cm",
                            "owner": owner,
                            "profile_id": profile_id,
                            "loop_index": loop_index,
                            "curve_index": curve_index,
                            "is_outer_loop": loop.get("is_outer"),
                            "center": _vector_projection(curve.get("center_point")),
                            "normal": _vector_projection(curve.get("normal")),
                        }
                    )

    editable.sort(
        key=lambda record: (
            record["owner"]["timeline_index"]
            if isinstance(record["owner"]["timeline_index"], int)
            else 2**31,
            record["parameter_role"],
            record["json_pointer"],
        )
    )
    pointers = [record["json_pointer"] for record in editable]
    if not pointers or len(pointers) != len(set(pointers)):
        raise ValueError("Fusion editable-parameter index is empty or has duplicates")

    parameter = oracle.get("parameter_contract")
    if not isinstance(parameter, dict):
        raise ValueError("verified edit oracle has no parameter contract")
    target_pointer = parameter.get("source_pointer")
    if not isinstance(target_pointer, str) or target_pointer not in set(pointers):
        raise ValueError("verified Fusion edit pointer is absent from public index")
    target_record = editable[pointers.index(target_pointer)]
    scale = _finite_number(parameter.get("scale_to_canonical"))
    base_value = _finite_number(parameter.get("base_value"))
    if scale is None or scale <= 0 or base_value is None:
        raise ValueError("verified Fusion parameter has invalid unit conversion")
    if not math.isclose(
        target_record["value"] * scale,
        base_value,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("public Fusion index disagrees with verified base value")

    return {
        "schema_version": "cadengbench-p.fusion360-edit-interface.v1",
        "source_filename": source_path.name,
        "base_history_sha256": f"sha256:{sha256_file(source_path)}",
        "source_unit": "cm",
        "representation": (
            "complete_index_of_replay_supported_numeric_bindings_into_hash_bound_"
            "native_history"
        ),
        "supported_parameter_roles": [
            "extrude_distance",
            "sketch_circle_radius",
        ],
        "editable_parameter_count": len(editable),
        "editable_parameters": editable,
    }


def _functional_user_prompt(
    *,
    source_key: str,
    task_prompt: str,
    source_path: Path,
    oracle: dict[str, Any],
    public_dfm_interface: str,
) -> tuple[str, dict[str, Any]]:
    sections = [
        task_prompt.strip(),
        public_dfm_interface,
        "Hash-bound editable source artifact",
        f"Filename: {source_path.name}",
        f"SHA-256: {sha256_file(source_path)}",
    ]
    if source_key == "benchcad":
        source_text = source_path.read_text(encoding="utf-8")
        sections.extend(
            [
                "BEGIN EDITABLE SOURCE",
                source_text.rstrip(),
                "END EDITABLE SOURCE",
            ]
        )
        interface_evidence = {
            "representation": "complete_native_source",
            "source_sha256": sha256_file(source_path),
        }
    else:
        interface = _fusion_editable_parameter_index(source_path, oracle)
        interface_text = json.dumps(interface, indent=2, sort_keys=True)
        sections.extend(
            [
                "Editable parameter index",
                (
                    "The full native JSON is evaluator-held. This deterministic "
                    "index lists every numeric binding supported by the bounded "
                    "Fusion replay contract. Choose exactly one listed pointer; "
                    "all non-target source data remain unchanged by construction."
                ),
                "BEGIN EDITABLE PARAMETER INDEX",
                interface_text,
                "END EDITABLE PARAMETER INDEX",
                "Required response schema",
                "{",
                '  "schema_version": "cadengbench-p.fusion360-patch-submission.v1",',
                '  "submission_id": "<non-empty identifier>",',
                f'  "base_history_sha256": "sha256:{sha256_file(source_path)}",',
                '  "edits": [{',
                '    "operation": "replace_numeric",',
                '    "json_pointer": "<source JSON pointer>",',
                '    "value": <finite number in source-native units>,',
                '    "source_unit": "cm"',
                "  }]",
                "}",
            ]
        )
        interface_evidence = {
            "representation": interface["representation"],
            "schema_version": interface["schema_version"],
            "editable_parameter_count": interface["editable_parameter_count"],
            "interface_sha256": sha256_text(interface_text),
            "source_sha256": sha256_file(source_path),
        }
    parameter = oracle.get("parameter_contract")
    if not isinstance(parameter, dict):
        raise ValueError("verified edit oracle has no parameter contract")
    return "\n\n".join(sections).rstrip() + "\n", interface_evidence


def _materialize_functional_edit_request(
    item_root: Path,
    *,
    run_root: Path,
    input_profile: str,
) -> dict[str, Any]:
    profile = INPUT_PROFILES.get(input_profile)
    if profile is None:
        raise ValueError(f"unsupported input profile: {input_profile}")
    item_root = item_root.resolve()
    item_id = item_root.name
    item_manifest_path = _required_file(item_root / "item.json", "item manifest")
    consumer_path = _required_file(
        item_root / "benchmark/consumer_ready_item_v2.json", "consumer-ready item"
    )
    oracle_path = _required_file(
        item_root / "completion/v2/parametric/edit_target_oracle.json",
        "verified functional-edit oracle",
    )
    item_manifest = read_json(item_manifest_path)
    consumer = read_json(consumer_path)
    oracle = read_json(oracle_path)
    if (
        not isinstance(item_manifest, dict)
        or item_manifest.get("dataset_version") is None
    ):
        raise ValueError(f"item manifest has no dataset_version: {item_manifest_path}")
    if (
        not isinstance(consumer, dict)
        or consumer.get("schema_version") != CONSUMER_READY_SCHEMA
        or consumer.get("item_id") != item_id
    ):
        raise ValueError(f"consumer-ready item is malformed: {consumer_path}")
    if (
        not isinstance(oracle, dict)
        or oracle.get("schema_version") != EDIT_ORACLE_SCHEMA
        or oracle.get("item_id") != item_id
        or oracle.get("status") != "verified"
        or oracle.get("llm_judge_used") is not False
    ):
        raise ValueError(f"functional-edit oracle is not verified: {oracle_path}")
    source_key = str(oracle.get("source_key"))
    model_facing = consumer.get("model_facing")
    functional = (
        model_facing.get("functional_edit") if isinstance(model_facing, dict) else None
    )
    task_prompt = (
        functional.get("user_prompt") if isinstance(functional, dict) else None
    )
    if not isinstance(task_prompt, str) or not task_prompt.strip():
        raise ValueError("consumer-ready item has no functional-edit user prompt")
    source_path = _native_source_path(item_root, consumer)
    source_root = run_root / "request_sources" / item_id
    system_path = source_root / "model.system.txt"
    user_path = source_root / "functional_edit.user.txt"
    design_path = source_root / "functional_edit_design_source.json"
    write_text(system_path, _functional_system_prompt(source_key))
    user_prompt, source_interface = _functional_user_prompt(
        source_key=source_key,
        task_prompt=task_prompt,
        source_path=source_path,
        oracle=oracle,
        public_dfm_interface=_public_functional_dfm_interface(
            item_root, item_id
        ),
    )
    write_text(user_path, user_prompt)
    write_json(
        design_path,
        {
            "schema_version": "cadengbench-p.runner-functional-edit-source.v1",
            "item_id": item_id,
            "task_kind": "functional_edit",
            "source_key": source_key,
            "consumer_sha256": sha256_file(consumer_path),
            "oracle_sha256": sha256_file(oracle_path),
            "native_history_sha256": sha256_file(source_path),
            "model_facing_source_interface": source_interface,
            "system_prompt_sha256": sha256_file(system_path),
            "user_prompt_sha256": sha256_file(user_path),
        },
    )
    user_parts = [_text_part(user_path)]
    for image_relative, view_role in profile.images:
        user_parts.append(
            _image_part(item_root, run_root, item_id, image_relative, view_role)
        )
    request: dict[str, Any] = {
        "schema_version": REQUEST_V2_SCHEMA,
        "request_id": f"{item_id}:functional_edit:{input_profile}:v2",
        "item_id": item_id,
        "task_kind": "functional_edit",
        "input_profile": input_profile,
        "dataset": {
            "release_id": str(item_manifest["dataset_version"]),
            "item_manifest_sha256": sha256_file(item_manifest_path),
            "design_spec_sha256": sha256_file(design_path),
        },
        "messages": [
            {"role": "system", "parts": [_text_part(system_path)]},
            {"role": "user", "parts": user_parts},
        ],
        "response_contract": _functional_response_contract(source_key),
    }
    request["request_hash"] = semantic_hash(request, "request_hash")
    validate_canonical_request(request)
    return request


def materialize_request(
    item_root: Path,
    *,
    run_root: Path,
    input_profile: str,
    task_kind: str = "zero_to_cad",
) -> dict[str, Any]:
    if task_kind == "zero_to_cad":
        return _materialize_zero_to_cad_request(
            item_root, run_root=run_root, input_profile=input_profile
        )
    if task_kind == "functional_edit":
        return _materialize_functional_edit_request(
            item_root, run_root=run_root, input_profile=input_profile
        )
    raise ValueError(f"unsupported task kind: {task_kind!r}")
