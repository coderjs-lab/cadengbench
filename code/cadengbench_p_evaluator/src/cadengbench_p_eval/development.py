"""Build reproducible real-item controls for evaluator development.

This module is deliberately separate from public scoring. Its generated
submissions are calibration controls, not model baselines or dataset labels.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, sha256_json, write_json, write_text

QUALIFICATION_SCHEMA = "cadengbench-p.evaluator-qualification.v1"
SLICE_PLAN_SCHEMA = "cadengbench-p.evaluator-dev-slice-plan.v1"
INTERFACE_VERSION = "cadengbench-p.cadquery-submission.v1"
SOURCE_TO_MM = 10.0


class DevelopmentSliceError(ValueError):
    """A development control cannot be generated without ambiguity."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentSliceError(f"{label} must be an object")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DevelopmentSliceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DevelopmentSliceError(f"{label} must be finite")
    return result


def _consumer(item_root: Path) -> dict[str, Any]:
    path = item_root / "benchmark/consumer_ready_item_v2.json"
    value = read_json(path)
    if not isinstance(value, dict) or value.get("item_id") != item_root.name:
        raise DevelopmentSliceError(f"invalid consumer record: {path}")
    return value


def _submission_contract() -> dict[str, Any]:
    return {
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
    }


def _bbox_requirement(geometry: Mapping[str, Any]) -> dict[str, Any]:
    bbox = _mapping(geometry.get("bbox"), "geometry.bbox")
    values: dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        value = _number(bbox.get(f"{axis}len"), f"bbox.{axis}len")
        values[axis] = {
            "value": value,
            "tolerance": max(0.05, abs(value) * 0.001),
        }
    return {
        "requirement_id": "R-BBOX",
        "category": "global_envelope",
        "criticality": "required",
        "value_and_tolerance": values,
        "check_ids": ["hidden.check.bbox_dimensions.v1"],
    }


def _hole_groups(geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = _mapping(geometry.get("hole_inventory"), "geometry.hole_inventory")
    holes = inventory.get("holes")
    if not isinstance(holes, list):
        raise DevelopmentSliceError("hole inventory must contain a list")
    groups: dict[
        tuple[float, float, tuple[float, float, float]],
        list[Mapping[str, Any]],
    ] = {}
    for raw in holes:
        hole = _mapping(raw, "hole")
        axis_raw = hole.get("axis_direction")
        if not isinstance(axis_raw, list) or len(axis_raw) != 3:
            raise DevelopmentSliceError("hole axis must contain three components")
        axis = tuple(round(_number(value, "hole axis"), 8) for value in axis_raw)
        key = (
            round(_number(hole.get("diameter"), "hole diameter"), 8),
            round(_number(hole.get("depth"), "hole depth"), 8),
            axis,
        )
        groups.setdefault(key, []).append(hole)
    requirements: list[dict[str, Any]] = []
    for index, ((diameter, depth, axis), members) in enumerate(
        sorted(groups.items()), 1
    ):
        requirements.append(
            {
                "requirement_id": f"R-HOLE-{index:03d}",
                "category": "cylindrical_opening_group",
                "criticality": "required",
                "value_and_tolerance": {
                    "quantity": len(members),
                    "diameter": diameter,
                    "diameter_tolerance": max(0.05, diameter * 0.001),
                    "depth": depth,
                    "depth_tolerance": max(0.05, depth * 0.001),
                    "axis_direction": list(axis),
                    "centers": [],
                    "position_tolerance": 0.05,
                },
                "check_ids": ["hidden.check.cylindrical_opening_group.v1"],
            }
        )
    return requirements


def _parameter_binding(
    selection: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[str, float, str, dict[str, Any]]:
    kind = selection.get("measurement_kind")
    if kind == "bbox_axis":
        axis = selection.get("axis")
        if axis not in {"x", "y", "z"}:
            raise DevelopmentSliceError("bbox-axis selection requires x, y, or z")
        requirement = next(
            value for value in requirements if value["requirement_id"] == "R-BBOX"
        )
        nominal = _number(
            requirement["value_and_tolerance"][axis]["value"], "bbox nominal"
        )
        name = {
            "x": "overall_length",
            "y": "overall_width",
            "z": "overall_height",
        }[str(axis)]
        binding = {
            "measurement_key": "P-001",
            "measurement_kind": "bounding_box_axis",
            "requirement_id": "R-BBOX",
            "requirement_value_path": [
                "value_and_tolerance",
                str(axis),
                "value",
            ],
            "axis": axis,
        }
        return name, nominal, "R-BBOX", binding
    if kind == "hole_diameter":
        base = _number(target.get("base_value"), "functional target base value")
        matches = [
            value
            for value in requirements
            if value.get("category") == "cylindrical_opening_group"
            and math.isclose(
                _number(value["value_and_tolerance"]["diameter"], "hole diameter"),
                base,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ]
        if len(matches) != 1:
            raise DevelopmentSliceError(
                "hole parameter does not resolve to exactly one requirement group"
            )
        requirement = matches[0]
        number = int(str(requirement["requirement_id"]).rsplit("-", 1)[1])
        name = f"opening_group_{number:03d}_diameter"
        binding = {
            "measurement_key": "P-001",
            "measurement_kind": "cylindrical_opening_diameter",
            "requirement_id": requirement["requirement_id"],
            "requirement_value_path": ["value_and_tolerance", "diameter"],
            "opening_group": number,
        }
        return name, base, str(requirement["requirement_id"]), binding
    raise DevelopmentSliceError(f"unsupported measurement selection: {kind!r}")


def _relation(
    *, parameter_id: str, tolerance: float, evidence_ref: str
) -> dict[str, Any]:
    return {
        "relation_id": "REL-P-001-MEASUREMENT",
        "predicate": {
            "type": "comparison",
            "op": "eq",
            "left": {
                "type": "variable",
                "source": "measurement",
                "key": parameter_id,
            },
            "right": {
                "type": "variable",
                "source": "parameter",
                "key": parameter_id,
            },
            "tolerance": tolerance,
        },
        "authority": "real_item_geometry_qualification",
        "evidence_ref": evidence_ref,
        "required": True,
    }


def _positive_predicate(parameter_id: str) -> dict[str, Any]:
    return {
        "type": "comparison",
        "op": "gt",
        "left": {"type": "variable", "source": "parameter", "key": parameter_id},
        "right": {"type": "constant", "value": 0.0},
        "tolerance": 0.0,
    }


def build_qualification(
    item_root: Path, selection: Mapping[str, Any]
) -> dict[str, Any]:
    consumer = _consumer(item_root)
    evaluator_inputs = _mapping(consumer.get("evaluator_inputs"), "evaluator_inputs")
    geometry = _mapping(evaluator_inputs.get("geometry"), "evaluator_inputs.geometry")
    functional = _mapping(
        evaluator_inputs.get("functional_edit"), "evaluator_inputs.functional_edit"
    )
    target = _mapping(functional.get("target"), "functional edit target")
    requirements = [
        {
            "requirement_id": "R-SOLID",
            "category": "solid_contract",
            "criticality": "required",
            "value_and_tolerance": {"body_count": int(geometry.get("body_count"))},
            "check_ids": ["hidden.check.solid_count.v1"],
        },
        _bbox_requirement(geometry),
        *_hole_groups(geometry),
    ]
    name, nominal, requirement_id, measurement_binding = _parameter_binding(
        selection, requirements, target
    )
    source_nominal = _number(target.get("base_value"), "target base value")
    if not math.isclose(nominal, source_nominal, rel_tol=0.0, abs_tol=1e-5):
        raise DevelopmentSliceError(
            f"selected observable {nominal} does not match source parameter "
            f"{source_nominal}"
        )
    raw_domain = _mapping(target.get("valid_range"), "target valid range")
    lower = _number(raw_domain.get("lower"), "domain lower")
    upper = _number(raw_domain.get("upper"), "domain upper")
    if not lower < nominal < upper:
        raise DevelopmentSliceError("source-qualified domain does not contain nominal")
    evidence_ref = "evaluation/qualification.v1.json#/parameter_system/parameters/0"
    domain = {
        "kind": "continuous",
        "minimum": lower,
        "maximum": upper,
        "minimum_inclusive": True,
        "maximum_inclusive": True,
        "authority": "source_two_sided_replay_qualification",
        "evidence_ref": (
            "benchmark/consumer_ready_item_v2.json#/evaluator_inputs/"
            "functional_edit/target/valid_range"
        ),
    }
    parameter = {
        "parameter_id": "P-001",
        "engineering_name": name,
        "nominal": nominal,
        "unit": "mm",
        "role": "principal_dimension",
        "requirement_refs": [requirement_id],
        "domain": domain,
        "measurement_binding": measurement_binding,
        "variant_requirement_contract": {
            "kind": "direct_measurement_only",
            "authority": "real_item_geometry_qualification",
            "evidence_ref": evidence_ref,
        },
    }
    tolerance = max(0.05, abs(nominal) * 0.001)
    relation = _relation(
        parameter_id="P-001", tolerance=tolerance, evidence_ref=evidence_ref
    )
    solver_obligation = {
        "obligation_id": "SOLVER-P-001-POSITIVE",
        "assumptions": [],
        "conclusion": _positive_predicate("P-001"),
        "authority": "qualified_positive_dimension_domain",
        "evidence_ref": evidence_ref,
        "required": True,
    }
    hole_count = int(
        _mapping(geometry.get("hole_inventory"), "holes").get("hole_count", 0)
    )
    dfm_families: list[dict[str, Any]] = []
    if hole_count > 0:
        dfm_families.append(
            {
                "family": "minimum_hole_geometry",
                "applicability": "development_capability_profile",
                "required": True,
                "thresholds": {
                    "minimum_diameter_mm": 2.0,
                    "maximum_depth_to_diameter": 8.0,
                },
                "threshold_evidence_ref": {
                    "profile_id": "cadengbench-p.dev-machining-capability.v1",
                    "scope": "evaluator development only; not a universal standard",
                    "source": "evaluation/qualification.v1.json#/dfm/families/0",
                },
                "extractor_certified": True,
                "expected_minimum_feature_count": hole_count,
                "scoreable": True,
                "unscoreable_reason": None,
            }
        )
    qualification = {
        "schema_version": QUALIFICATION_SCHEMA,
        "item_id": item_root.name,
        "purpose": "real-item evaluator development and mutation calibration",
        "selection": deepcopy(dict(selection)),
        "source_evidence": {
            "consumer_item": "benchmark/consumer_ready_item_v2.json",
            "source_parameter_id": target.get("parameter_id"),
            "source_parameter_key": target.get("parameter_key"),
            "source_oracle_id": target.get("oracle_id"),
        },
        "task_contract": {
            "task_kind": "zero_to_cad",
            "submission": _submission_contract(),
            "solid_contract": {"required_body_count": int(geometry.get("body_count"))},
            "reference_frame": "source_canonical_frame",
            "fidelity_profile": "exact_reference",
        },
        "requirements": requirements,
        "parameter_system": {
            "interface_version": INTERFACE_VERSION,
            "parameters": [parameter],
            "relations": [relation],
            "source_relations": [],
            "normalized_source_relation_count": 1,
            "solver_obligations": [solver_obligation],
            "feasibility_relations": [
                {
                    "relation_id": "FEASIBILITY-P-001-POSITIVE",
                    "predicate": _positive_predicate("P-001"),
                    "authority": "qualified_positive_dimension_domain",
                    "evidence_ref": evidence_ref,
                    "required": True,
                }
            ],
            "qualified_parameter_count": 1,
            "measurement_bound_parameter_count": 1,
            "qualified_variant_parameter_count": 1,
            "complete_domains": True,
            "complete_measurement_bindings": True,
        },
        "dfm": {
            "profile_origin": "evaluator_development_qualification",
            "process_profile": "cadengbench-p.dev-machining-capability.v1"
            if hole_count
            else None,
            "material_card": None,
            "families": dfm_families,
            "scoreable_family_count": len(dfm_families),
        },
    }
    qualification["qualification_hash"] = sha256_json(qualification)
    return qualification


class _SourceParameter(ast.NodeTransformer):
    def __init__(self, *, line: int, column: int, name: str, scale: float) -> None:
        self.line = line
        self.column = column
        self.name = name
        self.scale = scale
        self.replacements = 0

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if node.lineno == self.line and node.col_offset == self.column:
            value: ast.expr = ast.Subscript(
                value=ast.Name(id="values", ctx=ast.Load()),
                slice=ast.Constant(value=self.name),
                ctx=ast.Load(),
            )
            if not math.isclose(self.scale, 1.0):
                value = ast.BinOp(
                    left=value,
                    op=ast.Mult(),
                    right=ast.Constant(value=self.scale),
                )
            self.replacements += 1
            return ast.copy_location(value, node)
        return node


def _submission_header(name: str, nominal: float) -> list[str]:
    return [
        "import cadquery as cq",
        "",
        f"DEFAULT_PARAMS = {json.dumps({name: nominal}, sort_keys=True)}",
        "PARAMETER_MANIFEST = {",
        '    "P-001": {',
        f'        "name": {name!r},',
        '        "unit": "mm",',
        '        "role": "principal_dimension",',
        "    }",
        "}",
        "",
    ]


def _submission_footer() -> list[str]:
    return [
        "",
        "result = build()",
        'cq.exporters.export(result, "model.step")',
        "",
    ]


def _benchcad_submission(
    item_root: Path,
    qualification: Mapping[str, Any],
    *,
    scale: float,
) -> str:
    source_path = item_root / "source/cadquery.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    result_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "result"
            for target in node.targets
        )
    ]
    if len(result_nodes) != 1:
        raise DevelopmentSliceError("BenchCAD source must assign result exactly once")
    target = _mapping(
        _mapping(qualification.get("source_evidence"), "source evidence"),
        "source evidence",
    )
    source_parameter = target.get("source_parameter_id")
    match = re.search(r"line:(\d+):col:(\d+)$", str(source_parameter))
    if match is None:
        raise DevelopmentSliceError("BenchCAD source parameter has no AST location")
    parameter = _mapping(
        _mapping(qualification.get("parameter_system"), "parameter system")[
            "parameters"
        ][0],
        "parameter",
    )
    name = str(parameter["engineering_name"])
    transformer = _SourceParameter(
        line=int(match.group(1)), column=int(match.group(2)), name=name, scale=scale
    )
    expression = transformer.visit(deepcopy(result_nodes[0].value))
    if transformer.replacements != 1:
        raise DevelopmentSliceError(
            f"expected one source-parameter replacement, got {transformer.replacements}"
        )
    ast.fix_missing_locations(expression)
    lines = _submission_header(name, _number(parameter["nominal"], "nominal"))
    lines.extend(
        [
            "def build(params=None):",
            "    values = dict(DEFAULT_PARAMS)",
            "    if params is not None:",
            "        values.update(params)",
            "    return " + ast.unparse(expression),
        ]
    )
    lines.extend(_submission_footer())
    return "\n".join(lines)


def _components(value: Mapping[str, Any]) -> tuple[float, float, float]:
    return tuple(_number(value.get(key), f"vector.{key}") for key in ("x", "y", "z"))


def _frame(sketch: Mapping[str, Any]) -> tuple[tuple[float, float, float], ...]:
    transform = _mapping(sketch.get("transform"), "sketch transform")
    return tuple(
        _components(_mapping(transform.get(key), f"transform.{key}"))
        for key in ("origin", "x_axis", "y_axis", "z_axis")
    )


def _world_point(
    value: Mapping[str, Any], frame: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float]:
    origin, x_axis, y_axis, z_axis = frame
    x, y, z = _components(value)
    return tuple(
        SOURCE_TO_MM
        * (origin[index] + x * x_axis[index] + y * y_axis[index] + z * z_axis[index])
        for index in range(3)
    )


def _world_vector(
    value: Mapping[str, Any], frame: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float]:
    _, x_axis, y_axis, z_axis = frame
    x, y, z = _components(value)
    result = tuple(
        x * x_axis[index] + y * y_axis[index] + z * z_axis[index] for index in range(3)
    )
    magnitude = math.sqrt(sum(component * component for component in result))
    if magnitude <= 1e-12:
        raise DevelopmentSliceError("transformed direction has zero length")
    return tuple(component / magnitude for component in result)


def _rotate(
    vector: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle: float,
) -> tuple[float, float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    dot = sum(left * right for left, right in zip(vector, axis, strict=True))
    cross = (
        axis[1] * vector[2] - axis[2] * vector[1],
        axis[2] * vector[0] - axis[0] * vector[2],
        axis[0] * vector[1] - axis[1] * vector[0],
    )
    return tuple(
        vector[index] * cosine
        + cross[index] * sine
        + axis[index] * dot * (1.0 - cosine)
        for index in range(3)
    )


def _point_literal(value: Sequence[float]) -> str:
    return repr(tuple(round(float(component), 12) for component in value))


def _curve_source(
    curve: Mapping[str, Any], frame: tuple[tuple[float, float, float], ...]
) -> str:
    kind = curve.get("type")
    if kind == "Line3D":
        start = _world_point(_mapping(curve.get("start_point"), "line start"), frame)
        end = _world_point(_mapping(curve.get("end_point"), "line end"), frame)
        return f"cq.Edge.makeLine({_point_literal(start)}, {_point_literal(end)})"
    if kind == "Circle3D":
        center = _world_point(
            _mapping(curve.get("center_point"), "circle center"), frame
        )
        normal = _world_vector(
            _mapping(curve.get("normal") or {"x": 0, "y": 0, "z": 1}, "circle normal"),
            frame,
        )
        radius = _number(curve.get("radius"), "circle radius") * SOURCE_TO_MM
        return (
            f"cq.Edge.makeCircle({radius!r}, {_point_literal(center)}, "
            f"{_point_literal(normal)})"
        )
    if kind == "Arc3D":
        center = _world_point(_mapping(curve.get("center_point"), "arc center"), frame)
        normal = _world_vector(
            _mapping(curve.get("normal") or {"x": 0, "y": 0, "z": 1}, "arc normal"),
            frame,
        )
        reference = _world_vector(
            _mapping(curve.get("reference_vector"), "arc reference"), frame
        )
        radius = _number(curve.get("radius"), "arc radius") * SOURCE_TO_MM
        start_angle = _number(curve.get("start_angle"), "arc start angle")
        end_angle = _number(curve.get("end_angle"), "arc end angle")
        points = []
        for angle in (start_angle, (start_angle + end_angle) / 2.0, end_angle):
            radial = _rotate(reference, normal, angle)
            points.append(
                tuple(center[index] + radius * radial[index] for index in range(3))
            )
        return (
            f"cq.Edge.makeThreePointArc({_point_literal(points[0])}, "
            f"{_point_literal(points[1])}, {_point_literal(points[2])})"
        )
    raise DevelopmentSliceError(f"unsupported Fusion control curve: {kind!r}")


def _native_history_path(item_root: Path, consumer: Mapping[str, Any]) -> Path:
    source = _mapping(consumer.get("source"), "source")
    raw = Path(str(source.get("native_history_artifact")))
    local = item_root / "source" / raw.name
    if not local.is_file():
        raise DevelopmentSliceError(f"native history is missing: {local}")
    return local


def _fusion_submission(
    item_root: Path,
    qualification: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    scale: float,
) -> str:
    consumer = _consumer(item_root)
    payload = _mapping(
        read_json(_native_history_path(item_root, consumer)), "Fusion history"
    )
    entities = _mapping(payload.get("entities"), "Fusion entities")
    timeline = payload.get("timeline")
    if not isinstance(timeline, list):
        raise DevelopmentSliceError("Fusion timeline must be a list")
    parameter = _mapping(
        qualification["parameter_system"]["parameters"][0], "parameter"
    )
    name = str(parameter["engineering_name"])
    nominal = _number(parameter["nominal"], "nominal")
    distance_expressions = _mapping(
        selection.get("fusion_distance_expressions"), "Fusion distance expressions"
    )
    lines = _submission_header(name, nominal)
    lines.extend(
        [
            "def build(params=None):",
            "    values = dict(DEFAULT_PARAMS)",
            "    if params is not None:",
            "        values.update(params)",
            "    current = None",
        ]
    )
    operation_index = 0
    for timeline_record in sorted(timeline, key=lambda value: int(value["index"])):
        entity_id = str(timeline_record.get("entity"))
        feature = _mapping(entities.get(entity_id), f"Fusion entity {entity_id}")
        if feature.get("type") != "ExtrudeFeature":
            continue
        expression_spec = distance_expressions.get(entity_id)
        extent = _mapping(feature.get("extent_one"), "extrude extent")
        source_distance = _number(
            _mapping(extent.get("distance"), "extrude distance").get("value"),
            "extrude distance",
        )
        if expression_spec is None:
            distance_expression = repr(source_distance * SOURCE_TO_MM)
        else:
            spec = _mapping(expression_spec, f"distance expression {entity_id}")
            parameter_scale = _number(spec.get("scale", 1.0), "distance scale")
            offset = _number(spec.get("offset_mm", 0.0), "distance offset")
            distance_expression = f"values[{name!r}] * {scale * parameter_scale!r}"
            if offset:
                distance_expression += f" + {offset!r}"
        profile_solids: list[str] = []
        for profile_index, profile_ref_raw in enumerate(feature.get("profiles") or []):
            profile_ref = _mapping(profile_ref_raw, "Fusion profile reference")
            sketch = _mapping(
                entities.get(str(profile_ref.get("sketch"))), "Fusion sketch"
            )
            profiles = _mapping(sketch.get("profiles"), "Fusion profiles")
            profile = _mapping(
                profiles.get(str(profile_ref.get("profile"))), "Fusion profile"
            )
            frame = _frame(sketch)
            loops = profile.get("loops")
            if not isinstance(loops, list):
                raise DevelopmentSliceError("Fusion profile loops must be a list")
            outer_name: str | None = None
            inner_names: list[str] = []
            for loop_index, loop_raw in enumerate(loops):
                loop = _mapping(loop_raw, "Fusion profile loop")
                edges = loop.get("profile_curves")
                if not isinstance(edges, list) or not edges:
                    raise DevelopmentSliceError("Fusion profile loop has no curves")
                wire_name = f"wire_{operation_index}_{profile_index}_{loop_index}"
                edge_source = ",\n            ".join(
                    _curve_source(_mapping(curve, "Fusion curve"), frame)
                    for curve in edges
                )
                lines.extend(
                    [
                        f"    {wire_name} = cq.Wire.assembleEdges([",
                        f"            {edge_source}",
                        "    ])",
                    ]
                )
                if bool(loop.get("is_outer")):
                    if outer_name is not None:
                        raise DevelopmentSliceError(
                            "Fusion profile has multiple outer loops"
                        )
                    outer_name = wire_name
                else:
                    inner_names.append(wire_name)
            if outer_name is None:
                raise DevelopmentSliceError("Fusion profile has no outer loop")
            normal = frame[3]
            extent_type = str(feature.get("extent_type"))
            if extent_type == "OneSideFeatureExtentType":
                vector = (
                    f"cq.Vector(*tuple(component * ({distance_expression}) "
                    f"for component in {_point_literal(normal)}))"
                )
            elif extent_type == "SymmetricFeatureExtentType":
                raise DevelopmentSliceError(
                    "symmetric Fusion controls are outside this development slice"
                )
            else:
                raise DevelopmentSliceError(f"unsupported Fusion extent: {extent_type}")
            solid_name = f"solid_{operation_index}_{profile_index}"
            lines.append(
                f"    {solid_name} = cq.Solid.extrudeLinear({outer_name}, "
                f"[{', '.join(inner_names)}], {vector})"
            )
            profile_solids.append(solid_name)
        if not profile_solids:
            raise DevelopmentSliceError("Fusion extrusion has no profiles")
        tool_name = f"tool_{operation_index}"
        lines.append(f"    {tool_name} = {profile_solids[0]}")
        for solid_name in profile_solids[1:]:
            lines.append(f"    {tool_name} = {tool_name}.fuse({solid_name})")
        operation = feature.get("operation")
        if operation == "NewBodyFeatureOperation":
            lines.extend(
                [
                    "    if current is None:",
                    f"        current = {tool_name}",
                    "    else:",
                    "        current = cq.Compound.makeCompound("
                    f"[current, {tool_name}])",
                ]
            )
        elif operation == "JoinFeatureOperation":
            lines.append(f"    current = current.fuse({tool_name})")
        elif operation == "CutFeatureOperation":
            lines.append(f"    current = current.cut({tool_name})")
        else:
            raise DevelopmentSliceError(f"unsupported Fusion operation: {operation!r}")
        operation_index += 1
    lines.append("    return current")
    lines.extend(_submission_footer())
    return "\n".join(lines)


def build_development_slice(
    *, items_root: Path, plan_path: Path, output_root: Path
) -> dict[str, Any]:
    plan = _mapping(read_json(plan_path), "development slice plan")
    if plan.get("schema_version") != SLICE_PLAN_SCHEMA:
        raise DevelopmentSliceError("unsupported development slice plan")
    raw_items = plan.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise DevelopmentSliceError("development slice plan has no items")
    records: list[dict[str, Any]] = []
    for raw in raw_items:
        selection = _mapping(raw, "slice item")
        item_id = str(selection.get("item_id"))
        item_root = items_root / item_id
        if not item_root.is_dir():
            raise DevelopmentSliceError(f"slice item is missing: {item_id}")
        qualification = build_qualification(item_root, selection)
        qualification_path = item_root / "evaluation/qualification.v1.json"
        write_json(qualification_path, qualification)
        source_key = "benchcad" if item_id.startswith("p_benchcad_") else "fusion360"
        positive = (
            _benchcad_submission(item_root, qualification, scale=1.0)
            if source_key == "benchcad"
            else _fusion_submission(item_root, qualification, selection, scale=1.0)
        )
        mutant = (
            _benchcad_submission(item_root, qualification, scale=1.1)
            if source_key == "benchcad"
            else _fusion_submission(item_root, qualification, selection, scale=1.1)
        )
        positive_path = output_root / "submissions/positive" / f"{item_id}.py"
        mutant_path = (
            output_root / "submissions/mutant_parameter_scale" / f"{item_id}.py"
        )
        write_text(positive_path, positive)
        write_text(mutant_path, mutant)
        records.append(
            {
                "item_id": item_id,
                "source_key": source_key,
                "measurement_kind": selection.get("measurement_kind"),
                "qualification_path": qualification_path.relative_to(
                    item_root
                ).as_posix(),
                "qualification_hash": qualification["qualification_hash"],
                "positive_submission": positive_path.relative_to(
                    output_root
                ).as_posix(),
                "positive_sha256": sha256_file(positive_path),
                "mutant_submission": mutant_path.relative_to(output_root).as_posix(),
                "mutant_sha256": sha256_file(mutant_path),
                "expected_positive_status": "passed",
                "expected_mutant_status": "failed",
            }
        )
    manifest = {
        "schema_version": "cadengbench-p.evaluator-dev-slice.v1",
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "item_count": len(records),
        "items": records,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    write_json(output_root / "development_slice_manifest.json", manifest)
    return manifest


def _first_model_failure(result: Mapping[str, Any]) -> dict[str, Any] | None:
    layers = result.get("layers")
    if not isinstance(layers, list):
        return None
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        checks = layer.get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if isinstance(check, Mapping) and check.get("status") == "failed":
                return {
                    "layer": layer.get("layer"),
                    "check_id": check.get("check_id"),
                    "failure_code": check.get("failure_code"),
                    "summary": check.get("summary"),
                }
    return None


def summarize_development_slice(
    *,
    manifest_path: Path,
    positive_run: Path,
    mutant_run: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify the positive/mutant separation and write a compact report."""

    from .aggregation import aggregate_run

    manifest = _mapping(read_json(manifest_path), "development slice manifest")
    records = manifest.get("items")
    if not isinstance(records, list) or not records:
        raise DevelopmentSliceError("development slice manifest has no items")
    expected_ids = [str(record["item_id"]) for record in records]
    if len(expected_ids) != len(set(expected_ids)):
        raise DevelopmentSliceError("development slice item IDs are duplicated")
    positive_summary = aggregate_run(positive_run)
    mutant_summary = aggregate_run(mutant_run)
    rows: list[dict[str, Any]] = []
    unexpected: list[dict[str, str]] = []
    for item_id in expected_ids:
        positive_result = _mapping(
            read_json(positive_run / "items" / item_id / "result.json"),
            f"positive result {item_id}",
        )
        mutant_result = _mapping(
            read_json(mutant_run / "items" / item_id / "result.json"),
            f"mutant result {item_id}",
        )
        positive_status = str(positive_result.get("overall_status"))
        mutant_status = str(mutant_result.get("overall_status"))
        if positive_status != "passed":
            unexpected.append(
                {
                    "item_id": item_id,
                    "control": "positive",
                    "status": positive_status,
                }
            )
        if mutant_status != "failed":
            unexpected.append(
                {
                    "item_id": item_id,
                    "control": "mutant_parameter_scale",
                    "status": mutant_status,
                }
            )
        rows.append(
            {
                "item_id": item_id,
                "positive_status": positive_status,
                "mutant_status": mutant_status,
                "mutant_first_failure": _first_model_failure(mutant_result),
            }
        )
    report = {
        "schema_version": "cadengbench-p.evaluator-dev-slice-report.v1",
        "slice_manifest_sha256": sha256_file(manifest_path),
        "item_count": len(expected_ids),
        "positive_pass_count": sum(row["positive_status"] == "passed" for row in rows),
        "mutant_fail_count": sum(row["mutant_status"] == "failed" for row in rows),
        "unexpected_outcomes": unexpected,
        "positive_run": positive_summary,
        "mutant_run": mutant_summary,
        "items": rows,
    }
    report["valid"] = (
        not unexpected
        and (positive_summary.get("overall") or {}).get("publication_valid") is True
        and (mutant_summary.get("overall") or {}).get("publication_valid") is True
    )
    report["report_hash"] = sha256_json(report)
    write_json(output_path, report)
    return report
