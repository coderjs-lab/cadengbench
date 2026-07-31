from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_IDS = (
    "joint_set_00000",
    "joint_set_00002",
    "joint_set_00003",
    "joint_set_00004",
    "joint_set_00005",
    "joint_set_00008",
    "joint_set_00009",
    "joint_set_00012",
)

FUSION_TO_URDF_JOINT = {
    "RigidJointType": "fixed",
    "RevoluteJointType": "revolute",
    "SliderJointType": "prismatic",
    # URDF has no native 2-DOF cylindrical joint. The pilot keeps the source type
    # in metadata and emits a revolute approximation for loadability.
    "CylindricalJointType": "continuous",
    # URDF has no native pin-slot joint. Prismatic preserves the translational DOF.
    "PinSlotJointType": "prismatic",
    "PlanarJointType": "planar",
    # URDF's portable joint vocabulary has no ball joint. Floating is closest.
    "BallJointType": "floating",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _vector3(data: dict[str, Any] | None) -> list[float]:
    data = data or {}
    return [float(data.get(axis, 0.0)) for axis in ("x", "y", "z")]


def _normalise(vec: list[float]) -> list[float]:
    length = math.sqrt(sum(v * v for v in vec))
    if length <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [v / length for v in vec]


def _fmt(values: list[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _axis_alignment(a: list[float], b: list[float]) -> float:
    a_norm = _normalise(a)
    b_norm = _normalise(b)
    return abs(_dot(a_norm, b_norm))


def _transform_to_matrix(transform: dict[str, Any] | None) -> list[list[float]] | None:
    if not transform:
        return None
    origin = _vector3(transform.get("origin"))
    x_axis = _vector3(transform.get("x_axis"))
    y_axis = _vector3(transform.get("y_axis"))
    z_axis = _vector3(transform.get("z_axis"))
    return [
        [x_axis[0], y_axis[0], z_axis[0], origin[0]],
        [x_axis[1], y_axis[1], z_axis[1], origin[1]],
        [x_axis[2], y_axis[2], z_axis[2], origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _read_body_properties(graph_path: Path) -> dict[str, Any]:
    if not graph_path.exists():
        return {}
    try:
        data = _load_json(graph_path)
    except Exception:
        return {}
    props = data.get("properties")
    return props if isinstance(props, dict) else {}


def _body_record(dataset_dir: Path, body_id: str) -> dict[str, Any]:
    files = {
        "graph_json": dataset_dir / f"{body_id}.json",
        "step": dataset_dir / f"{body_id}.step",
        "obj": dataset_dir / f"{body_id}.obj",
        "smt": dataset_dir / f"{body_id}.smt",
    }
    properties = _read_body_properties(files["graph_json"])
    return {
        "id": body_id,
        "source_files": {key: str(path) for key, path in files.items()},
        "file_exists": {key: path.exists() for key, path in files.items()},
        "physical_properties": {
            key: properties.get(key)
            for key in ("area", "volume", "density", "mass", "center_of_mass")
            if key in properties
        },
        "topology": {
            key: properties.get(key)
            for key in ("body_count", "shell_count", "face_count", "edge_count", "vertex_count")
            if key in properties
        },
        "geometry_quality": _body_geometry_quality(properties, files),
    }


def _body_geometry_quality(properties: dict[str, Any], files: dict[str, Path]) -> dict[str, Any]:
    bbox = properties.get("bounding_box") or {}
    min_point = _vector3(bbox.get("min_point")) if bbox else None
    max_point = _vector3(bbox.get("max_point")) if bbox else None
    extents = [abs(max_point[i] - min_point[i]) for i in range(3)] if min_point and max_point else None
    min_extent = min(extents) if extents else None
    max_extent = max(extents) if extents else None
    thin_ratio = (min_extent / max_extent) if min_extent is not None and max_extent and max_extent > 1e-12 else None
    missing_files = [key for key, path in files.items() if not path.exists()]
    warnings = []
    if missing_files:
        warnings.append("missing_files")
    if properties.get("body_count") not in (None, 1):
        warnings.append("body_count_not_one")
    if properties.get("shell_count") not in (None, 1):
        warnings.append("shell_count_not_one")
    if thin_ratio is not None and thin_ratio < 0.01:
        warnings.append("thin_body")
    return {
        "bounding_box": {"min": min_point, "max": max_point, "extents": extents} if extents else None,
        "thin_ratio": thin_ratio,
        "missing_files": missing_files,
        "warnings": warnings,
    }


def _geometry_record(geometry: dict[str, Any]) -> dict[str, Any]:
    entity = geometry.get("entity_one", {})
    axis_line = geometry.get("axis_line", {})
    return {
        "geometry_type": geometry.get("geometry_type"),
        "key_point_type": geometry.get("key_point_type"),
        "entity_one": {
            "type": entity.get("type"),
            "index": entity.get("index"),
            "body": entity.get("body"),
            "surface_type": entity.get("surface_type"),
            "curve_type": entity.get("curve_type"),
        },
        "entity_one_equivalents": [
            {
                "type": equivalent.get("type"),
                "index": equivalent.get("index"),
                "surface_type": equivalent.get("surface_type"),
                "curve_type": equivalent.get("curve_type"),
            }
            for equivalent in geometry.get("entity_one_equivalents", [])
        ],
        "axis_line": {
            "origin": _vector3(axis_line.get("origin")),
            "direction": _normalise(_vector3(axis_line.get("direction"))),
        },
        "transform": geometry.get("transform"),
        "transform_matrix": _transform_to_matrix(geometry.get("transform")),
    }


def _infer_origin_type(geometry: dict[str, Any]) -> str:
    key_point_type = geometry.get("key_point_type")
    entity = geometry.get("entity_one") or {}
    if key_point_type == "CenterKeyPoint":
        if entity.get("curve_type") in {"Circle3DCurveType", "Ellipse3DCurveType"}:
            return "center"
        if entity.get("surface_type") in {"CylinderSurfaceType", "ConeSurfaceType", "TorusSurfaceType"}:
            return "mid_axis_point"
        return "centroid_or_center"
    return str(key_point_type or "unknown")


def _fallback_x_axis(z_axis: list[float]) -> list[float]:
    candidates = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    for candidate in candidates:
        if _axis_alignment(candidate, z_axis) < 0.95:
            return candidate
    return [1.0, 0.0, 0.0]


def _mating_coordinate_frame(geometry: dict[str, Any], body_id: str) -> dict[str, Any]:
    axis_line = geometry.get("axis_line") or {}
    transform = geometry.get("transform") or {}
    z_axis = _normalise(axis_line.get("direction") or [0.0, 0.0, 1.0])
    raw_x_axis = _normalise(_vector3(transform.get("x_axis")) if transform else _fallback_x_axis(z_axis))
    if _axis_alignment(raw_x_axis, z_axis) > 0.98:
        raw_x_axis = _fallback_x_axis(z_axis)
    y_axis = _normalise(_cross(z_axis, raw_x_axis))
    x_axis = _normalise(_cross(y_axis, z_axis))
    entity = geometry.get("entity_one") or {}
    return {
        "body": body_id,
        "origin": axis_line.get("origin"),
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "origin_type": _infer_origin_type(geometry),
        "origin_reference_entity": entity,
        "orientation_reference_entity": entity,
        "source": "fusion_joint_geometry",
    }


def parse_joint_set(joint_set_path: str | Path) -> dict[str, Any]:
    joint_set_path = Path(joint_set_path)
    dataset_dir = joint_set_path.parent
    source = _load_json(joint_set_path)
    body_ids = [source["body_one"], source["body_two"]]
    joints = []
    for index, joint in enumerate(source.get("joints", [])):
        joint_motion = joint.get("joint_motion", {})
        joint_type = joint_motion.get("joint_type", "UnknownJointType")
        geometry_one = _geometry_record(joint.get("geometry_or_origin_one", {}))
        geometry_two = _geometry_record(joint.get("geometry_or_origin_two", {}))
        mating_coordinate_frames = {
            "body_a": _mating_coordinate_frame(geometry_one, body_ids[0]),
            "body_b": _mating_coordinate_frame(geometry_two, body_ids[1]),
        }
        joints.append(
            {
                "id": f"{joint_set_path.stem}_joint_{index}",
                "source_index": index,
                "name": joint.get("name"),
                "source_type": joint.get("type"),
                "fusion_joint_type": joint_type,
                "urdf_joint_type": FUSION_TO_URDF_JOINT.get(joint_type, "fixed"),
                "urdf_mapping_note": _joint_mapping_note(joint_type),
                "body_a": body_ids[0],
                "body_b": body_ids[1],
                "geometry_or_origin_one": geometry_one,
                "geometry_or_origin_two": geometry_two,
                "mating_coordinate_frames": mating_coordinate_frames,
                "axis": geometry_one["axis_line"],
                "joint_equivalence_signature": _joint_equivalence_signature(joint_type, body_ids[0], body_ids[1], geometry_one),
                "limits": _limits_from_joint_motion(joint_motion),
                "motion_semantics": _motion_semantics(
                    joint_motion, mating_coordinate_frames["body_a"]
                ),
                "offset": joint.get("offset", {}).get("value"),
                "angle": joint.get("angle", {}).get("value"),
                "is_flipped": bool(joint.get("is_flipped", False)),
            }
        )
    return {
        "schema": "cadengbench_a_master_v1",
        "source_joint_set": str(joint_set_path),
        "joint_set_id": joint_set_path.stem,
        "bodies": [_body_record(dataset_dir, body_id) for body_id in body_ids],
        "joints": joints,
        "contacts_count": len(source.get("contacts", [])),
        "holes_count": len(source.get("holes", [])),
    }


def _canonical_axis_direction(direction: list[float]) -> list[float]:
    direction = _normalise(direction)
    for value in direction:
        if abs(value) > 1e-9:
            return [-component for component in direction] if value < 0 else direction
    return direction


def _rounded(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _joint_equivalence_signature(joint_type: str, body_a: str, body_b: str, geometry: dict[str, Any]) -> dict[str, Any]:
    axis = geometry.get("axis_line") or {}
    entity = geometry.get("entity_one") or {}
    return {
        "body_pair": [body_a, body_b],
        "joint_type": joint_type,
        "axis_origin_rounded": _rounded(axis.get("origin") or [0.0, 0.0, 0.0]),
        "axis_direction_canonical_rounded": _rounded(_canonical_axis_direction(axis.get("direction") or [0.0, 0.0, 1.0])),
        "entity": {
            "type": entity.get("type"),
            "index": entity.get("index"),
            "surface_type": entity.get("surface_type"),
            "curve_type": entity.get("curve_type"),
        },
    }


def _joint_mapping_note(joint_type: str) -> str:
    if joint_type in {"RigidJointType", "RevoluteJointType", "SliderJointType"}:
        return "direct"
    if joint_type == "CylindricalJointType":
        return "approximation: URDF cannot represent coupled rotation and translation in one joint"
    if joint_type == "PinSlotJointType":
        return "approximation: URDF cannot represent pin-slot as one portable joint"
    if joint_type == "BallJointType":
        return "approximation: URDF has no standard spherical joint"
    if joint_type == "PlanarJointType":
        return "approximation: simulator support for URDF planar joints varies"
    return "unknown joint type; emitted as fixed"


def _limits_from_joint_motion(joint_motion: dict[str, Any]) -> dict[str, Any]:
    limit_keys = ("rotation_limits", "slide_limits")
    for key in limit_keys:
        limits = joint_motion.get(key)
        if isinstance(limits, dict):
            return {
                "source_key": key,
                "lower": limits.get("minimum_value"),
                "upper": limits.get("maximum_value"),
                "rest": limits.get("rest_value"),
                "lower_enabled": bool(limits.get("is_minimum_value_enabled", False)),
                "upper_enabled": bool(limits.get("is_maximum_value_enabled", False)),
                "rest_enabled": bool(limits.get("is_rest_value_enabled", False)),
            }
    return {}


def _normalised_limits(limits: dict[str, Any]) -> dict[str, Any]:
    return {
        "lower": limits.get("minimum_value"),
        "upper": limits.get("maximum_value"),
        "rest": limits.get("rest_value"),
        "lower_enabled": bool(limits.get("is_minimum_value_enabled", False)),
        "upper_enabled": bool(limits.get("is_maximum_value_enabled", False)),
        "rest_enabled": bool(limits.get("is_rest_value_enabled", False)),
    }


def _direction_from_joint_frame(
    label: str | None, frame: dict[str, Any]
) -> list[float] | None:
    frame_key = {
        "XAxisJointDirection": "x_axis",
        "YAxisJointDirection": "y_axis",
        "ZAxisJointDirection": "z_axis",
    }.get(str(label))
    if frame_key is None:
        return None
    value = frame.get(frame_key)
    return _normalise(value) if isinstance(value, list) and len(value) == 3 else None


def _motion_semantics(
    joint_motion: dict[str, Any], frame: dict[str, Any]
) -> dict[str, Any]:
    direction_keys = (
        "rotation_axis",
        "slide_direction",
        "normal_direction",
        "primary_slide_direction",
        "pitch_direction",
        "yaw_direction",
    )
    direction_labels = {
        key: joint_motion[key]
        for key in direction_keys
        if isinstance(joint_motion.get(key), str)
    }
    resolved_directions = {}
    unresolved_direction_labels = {}
    for key, label in direction_labels.items():
        direction = _direction_from_joint_frame(label, frame)
        if direction is None:
            unresolved_direction_labels[key] = label
        else:
            resolved_directions[key] = direction

    joint_type = joint_motion.get("joint_type", "UnknownJointType")
    if joint_type == "CylindricalJointType" and "rotation_axis" in resolved_directions:
        resolved_directions["slide_direction"] = list(
            resolved_directions["rotation_axis"]
        )
    if joint_type == "PlanarJointType":
        normal = resolved_directions.get("normal_direction")
        primary = resolved_directions.get("primary_slide_direction")
        if normal and primary:
            resolved_directions["secondary_slide_direction"] = _normalise(
                _cross(normal, primary)
            )
    if joint_type == "BallJointType":
        pitch = resolved_directions.get("pitch_direction")
        yaw = resolved_directions.get("yaw_direction")
        if pitch and yaw:
            resolved_directions["roll_direction"] = _normalise(_cross(yaw, pitch))

    limit_groups = {
        key.removesuffix("_limits"): _normalised_limits(value)
        for key, value in joint_motion.items()
        if key.endswith("_limits") and isinstance(value, dict)
    }
    return {
        "joint_type": joint_type,
        "joint_frame_source": "body_a_fusion_joint_geometry",
        "direction_labels": direction_labels,
        "resolved_directions": resolved_directions,
        "unresolved_direction_labels": unresolved_direction_labels,
        "limit_groups": limit_groups,
    }


def select_joint_sets(dataset_dir: str | Path, sample_ids: list[str] | None = None, limit: int = 8) -> list[Path]:
    dataset_dir = Path(dataset_dir).expanduser()
    if sample_ids:
        paths = []
        for sample_id in sample_ids:
            stem = sample_id[:-5] if sample_id.endswith(".json") else sample_id
            paths.append(dataset_dir / f"{stem}.json")
        return paths

    selected: list[Path] = []
    for stem in DEFAULT_SAMPLE_IDS:
        path = dataset_dir / f"{stem}.json"
        if path.exists():
            selected.append(path)
    if selected:
        return selected[:limit]
    return sorted(dataset_dir.glob("joint_set_*.json"))[:limit]


def write_urdf(master: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    robot = ET.Element("robot", {"name": master["joint_set_id"]})
    skipped: list[dict[str, Any]] = []
    emitted_joint_ids: list[str] = []

    for body in master["bodies"]:
        link = ET.SubElement(robot, "link", {"name": body["id"]})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        mass = body.get("physical_properties", {}).get("mass") or 1.0
        try:
            mass = max(float(mass), 1e-6)
        except Exception:
            mass = 1.0
        ET.SubElement(inertial, "mass", {"value": f"{mass:.9g}"})
        ET.SubElement(
            inertial,
            "inertia",
            {"ixx": "1", "ixy": "0", "ixz": "0", "iyy": "1", "iyz": "0", "izz": "1"},
        )
        mesh_path = body["source_files"].get("obj")
        if mesh_path:
            for tag in ("visual", "collision"):
                node = ET.SubElement(link, tag)
                ET.SubElement(node, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
                geometry = ET.SubElement(node, "geometry")
                ET.SubElement(geometry, "mesh", {"filename": mesh_path})

    used_children: set[str] = set()
    emitted_by_child: dict[str, dict[str, Any]] = {}
    for joint in master["joints"]:
        child = joint["body_b"]
        if child in used_children:
            kept_joint = emitted_by_child.get(child)
            skipped.append(
                {
                    "joint_id": joint["id"],
                    "joint": joint,
                    "kept_joint": _joint_summary(kept_joint) if kept_joint else None,
                    "equivalence": _parallel_joint_equivalence(kept_joint, joint) if kept_joint else None,
                    "reason": "URDF tree cannot attach the same child link through multiple joints",
                }
            )
            continue
        used_children.add(child)
        emitted_by_child[child] = joint
        emitted_joint_ids.append(joint["id"])
        joint_type = joint["urdf_joint_type"]
        joint_node = ET.SubElement(robot, "joint", {"name": joint["id"], "type": joint_type})
        ET.SubElement(joint_node, "parent", {"link": joint["body_a"]})
        ET.SubElement(joint_node, "child", {"link": child})
        axis = joint["axis"]
        ET.SubElement(joint_node, "origin", {"xyz": _fmt(axis["origin"]), "rpy": "0 0 0"})
        if joint_type != "fixed":
            ET.SubElement(joint_node, "axis", {"xyz": _fmt(axis["direction"])})
        if joint_type in {"revolute", "prismatic"}:
            limits = _urdf_limits(joint)
            ET.SubElement(joint_node, "limit", limits)

    ET.indent(robot)
    tree = ET.ElementTree(robot)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    urdf_info = {"path": str(output_path), "emitted_joint_ids": emitted_joint_ids, "skipped_joints": skipped}
    urdf_info["export_status"] = classify_urdf_export_status(master, urdf_info)
    return urdf_info


def classify_urdf_export_status(master: dict[str, Any], urdf_info: dict[str, Any]) -> dict[str, Any]:
    """Classify how lossy the URDF export is relative to CAD-native joints."""
    skipped_joints = urdf_info.get("skipped_joints", [])
    skipped_classes = [
        (skipped_joint.get("equivalence") or {}).get("classification", "unknown")
        for skipped_joint in skipped_joints
    ]
    unsupported_joints = [
        {
            "id": joint.get("id"),
            "name": joint.get("name"),
            "fusion_joint_type": joint.get("fusion_joint_type"),
            "urdf_joint_type": joint.get("urdf_joint_type"),
            "urdf_mapping_note": joint.get("urdf_mapping_note"),
        }
        for joint in master.get("joints", [])
        if joint.get("urdf_mapping_note") != "direct"
    ]
    potentially_meaningful = [
        skipped_joint
        for skipped_joint in skipped_joints
        if _is_potentially_meaningful_skipped_joint(skipped_joint)
    ]
    likely_redundant = [
        skipped_joint
        for skipped_joint in skipped_joints
        if _is_likely_redundant_skipped_joint(skipped_joint)
    ]

    if potentially_meaningful:
        label = "potentially_meaningful_loss"
        explanation = "URDF skipped one or more CAD joints that appear to add distinct constraints."
    elif unsupported_joints:
        label = "unsupported_by_urdf"
        explanation = "At least one Fusion joint type has no exact portable URDF equivalent."
    elif skipped_joints:
        label = "likely_redundant_loss"
        explanation = "URDF skipped parallel joints, but current equivalence checks classify them as redundant or already locked."
    else:
        label = "no_loss"
        explanation = "Every source joint was represented directly in the URDF export."

    return {
        "lossiness_label": label,
        "explanation": explanation,
        "source_joint_count": len(master.get("joints", [])),
        "emitted_urdf_joint_count": len(urdf_info.get("emitted_joint_ids", [])),
        "skipped_parallel_joint_count": len(skipped_joints),
        "unsupported_joint_count": len(unsupported_joints),
        "potentially_meaningful_skipped_joint_count": len(potentially_meaningful),
        "likely_redundant_skipped_joint_count": len(likely_redundant),
        "skipped_equivalence_class_counts": {
            classification: skipped_classes.count(classification)
            for classification in sorted(set(skipped_classes))
        },
        "unsupported_joints": unsupported_joints,
    }


def _is_likely_redundant_skipped_joint(skipped_joint: dict[str, Any]) -> bool:
    classification = (skipped_joint.get("equivalence") or {}).get("classification", "unknown")
    return classification.startswith("likely_redundant") or classification == "additional_rigid_locating_constraint_urdf_fixed_already_locks"


def _is_potentially_meaningful_skipped_joint(skipped_joint: dict[str, Any]) -> bool:
    classification = (skipped_joint.get("equivalence") or {}).get("classification", "unknown")
    if classification.startswith("potentially_meaningful"):
        return True
    return not _is_likely_redundant_skipped_joint(skipped_joint)


def _joint_summary(joint: dict[str, Any] | None) -> dict[str, Any] | None:
    if not joint:
        return None
    return {
        "id": joint.get("id"),
        "name": joint.get("name"),
        "fusion_joint_type": joint.get("fusion_joint_type"),
        "body_a": joint.get("body_a"),
        "body_b": joint.get("body_b"),
        "axis": joint.get("axis"),
        "geometry_or_origin_one_entity": (joint.get("geometry_or_origin_one") or {}).get("entity_one"),
        "geometry_or_origin_two_entity": (joint.get("geometry_or_origin_two") or {}).get("entity_one"),
        "joint_equivalence_signature": joint.get("joint_equivalence_signature"),
    }


def _point_to_axis_distance(point: list[float], axis_origin: list[float], axis_direction: list[float]) -> float:
    direction = _normalise(axis_direction)
    return _norm(_cross(_sub(point, axis_origin), direction))


def _same_entity(entity_a: dict[str, Any] | None, entity_b: dict[str, Any] | None) -> bool:
    entity_a = entity_a or {}
    entity_b = entity_b or {}
    keys = ("type", "index", "body", "surface_type", "curve_type")
    return all(entity_a.get(key) == entity_b.get(key) for key in keys)


def _parallel_joint_equivalence(kept: dict[str, Any] | None, skipped: dict[str, Any]) -> dict[str, Any]:
    if not kept:
        return {"classification": "unknown_no_kept_joint"}
    kept_axis = kept.get("axis") or {}
    skipped_axis = skipped.get("axis") or {}
    kept_origin = kept_axis.get("origin") or [0.0, 0.0, 0.0]
    skipped_origin = skipped_axis.get("origin") or [0.0, 0.0, 0.0]
    kept_direction = kept_axis.get("direction") or [0.0, 0.0, 1.0]
    skipped_direction = skipped_axis.get("direction") or [0.0, 0.0, 1.0]
    axis_alignment = _axis_alignment(kept_direction, skipped_direction)
    origin_distance_to_kept_axis = _point_to_axis_distance(skipped_origin, kept_origin, kept_direction)
    same_axis = axis_alignment > 0.999 and origin_distance_to_kept_axis < 1e-5
    same_type = kept.get("fusion_joint_type") == skipped.get("fusion_joint_type")
    kept_geo1 = (kept.get("geometry_or_origin_one") or {}).get("entity_one")
    kept_geo2 = (kept.get("geometry_or_origin_two") or {}).get("entity_one")
    skipped_geo1 = (skipped.get("geometry_or_origin_one") or {}).get("entity_one")
    skipped_geo2 = (skipped.get("geometry_or_origin_two") or {}).get("entity_one")
    same_entities = _same_entity(kept_geo1, skipped_geo1) and _same_entity(kept_geo2, skipped_geo2)

    if same_type and same_axis and same_entities:
        classification = "likely_redundant_duplicate_constraint"
    elif same_type and same_axis:
        classification = "likely_redundant_same_dof_same_axis"
    elif same_type and kept.get("fusion_joint_type") == "RigidJointType":
        classification = "additional_rigid_locating_constraint_urdf_fixed_already_locks"
    elif same_type:
        classification = "potentially_meaningful_same_type_different_axis_or_entities"
    else:
        classification = "potentially_meaningful_mixed_joint_types"

    return {
        "classification": classification,
        "same_joint_type": same_type,
        "same_axis": same_axis,
        "same_selected_entities": same_entities,
        "axis_alignment_abs_dot": axis_alignment,
        "skipped_origin_distance_to_kept_axis": origin_distance_to_kept_axis,
        "kept_joint_id": kept.get("id"),
        "skipped_joint_id": skipped.get("id"),
    }


def _urdf_limits(joint: dict[str, Any]) -> dict[str, str]:
    limits = joint.get("limits") or {}
    lower = limits.get("lower") if limits.get("lower_enabled") else None
    upper = limits.get("upper") if limits.get("upper_enabled") else None
    if lower is None or upper is None or float(lower) == float(upper):
        if joint["urdf_joint_type"] == "prismatic":
            lower, upper = -1.0, 1.0
        else:
            lower, upper = -math.pi, math.pi
    return {
        "lower": f"{float(lower):.9g}",
        "upper": f"{float(upper):.9g}",
        "effort": "100",
        "velocity": "1",
    }


def validate_with_pybullet(
    urdf_path: str | Path,
    master: dict[str, Any] | None = None,
    emitted_joint_ids: list[str] | None = None,
) -> dict[str, Any]:
    try:
        import pybullet as p  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"pybullet unavailable: {type(exc).__name__}: {exc}"}

    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0.0, 0.0, -9.81)
        body_id = p.loadURDF(str(urdf_path), useFixedBase=True)
        expected_by_name = {}
        if master is not None:
            emitted = set(emitted_joint_ids or [])
            expected_by_name = {
                joint["id"]: joint
                for joint in master.get("joints", [])
                if not emitted or joint["id"] in emitted
            }
        joints = []
        for index in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, index)
            name = info[1].decode("utf-8", errors="replace")
            joint_type_code = int(info[2])
            axis = [float(value) for value in info[13]]
            expected = expected_by_name.get(name, {})
            behavior = _validate_pybullet_joint_behavior(p, body_id, index, joint_type_code, axis, expected)
            joints.append(
                {
                    "index": index,
                    "name": name,
                    "type_code": joint_type_code,
                    "type_name": _pybullet_joint_type_name(p, joint_type_code),
                    "axis": axis,
                    "expected_fusion_joint_type": expected.get("fusion_joint_type"),
                    "expected_urdf_joint_type": expected.get("urdf_joint_type"),
                    "mapping_note": expected.get("urdf_mapping_note"),
                    "expected_axis": (expected.get("axis") or {}).get("direction"),
                    "behavior": behavior,
                }
            )
        behavior_ok = all(joint["behavior"]["status"] in {"ok", "approximation"} for joint in joints)
        return {"status": "ok" if behavior_ok else "failed", "num_joints": len(joints), "joints": joints}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        p.disconnect(client)


def _pybullet_joint_type_name(p: Any, joint_type_code: int) -> str:
    mapping = {
        p.JOINT_REVOLUTE: "JOINT_REVOLUTE",
        p.JOINT_PRISMATIC: "JOINT_PRISMATIC",
        p.JOINT_SPHERICAL: "JOINT_SPHERICAL",
        p.JOINT_PLANAR: "JOINT_PLANAR",
        p.JOINT_FIXED: "JOINT_FIXED",
    }
    return mapping.get(joint_type_code, f"UNKNOWN_{joint_type_code}")


def _quat_angle_delta(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _validate_pybullet_joint_behavior(
    p: Any,
    body_id: int,
    joint_index: int,
    joint_type_code: int,
    pybullet_axis: list[float],
    expected: dict[str, Any],
) -> dict[str, Any]:
    expected_urdf = expected.get("urdf_joint_type")
    expected_fusion = expected.get("fusion_joint_type")
    expected_axis = (expected.get("axis") or {}).get("direction") or [0.0, 0.0, 1.0]
    axis_alignment = _axis_alignment(pybullet_axis, expected_axis)

    if expected_urdf == "fixed" or expected_fusion == "RigidJointType":
        return {
            "status": "ok" if joint_type_code == p.JOINT_FIXED else "failed",
            "check": "rigid_locked",
            "is_fixed_joint": joint_type_code == p.JOINT_FIXED,
            "message": "RigidJointType is represented as a PyBullet fixed joint with no movable DOF.",
        }

    if expected_urdf in {"revolute", "continuous"}:
        p.resetJointState(body_id, joint_index, 0.0)
        before = p.getLinkState(body_id, joint_index, computeForwardKinematics=True)[5]
        target_position = 0.4
        p.resetJointState(body_id, joint_index, target_position)
        state = p.getJointState(body_id, joint_index)
        after = p.getLinkState(body_id, joint_index, computeForwardKinematics=True)[5]
        position_error = abs(float(state[0]) - target_position)
        orientation_delta = _quat_angle_delta(before, after)
        type_ok = joint_type_code == p.JOINT_REVOLUTE
        axis_ok = axis_alignment > 0.999
        moved_ok = position_error < 1e-6 and orientation_delta > 1e-3
        status = "ok" if type_ok and axis_ok and moved_ok else "failed"
        if expected_fusion == "CylindricalJointType" and status == "ok":
            status = "approximation"
        return {
            "status": status,
            "check": "revolute_motion_about_declared_axis",
            "is_revolute_joint": type_ok,
            "axis_alignment_abs_dot": axis_alignment,
            "target_position_rad": target_position,
            "reported_position_rad": float(state[0]),
            "position_error": position_error,
            "orientation_delta_rad": orientation_delta,
            "message": (
                "Joint accepts revolute motion about the declared axis."
                if status in {"ok", "approximation"}
                else "Revolute behavior check failed."
            ),
        }

    if expected_urdf == "prismatic":
        p.resetJointState(body_id, joint_index, 0.0)
        target_position = 0.25
        p.resetJointState(body_id, joint_index, target_position)
        state = p.getJointState(body_id, joint_index)
        position_error = abs(float(state[0]) - target_position)
        type_ok = joint_type_code == p.JOINT_PRISMATIC
        axis_ok = axis_alignment > 0.999
        status = "ok" if type_ok and axis_ok and position_error < 1e-6 else "failed"
        if expected_fusion == "PinSlotJointType" and status == "ok":
            status = "approximation"
        return {
            "status": status,
            "check": "prismatic_motion_along_declared_axis",
            "is_prismatic_joint": type_ok,
            "axis_alignment_abs_dot": axis_alignment,
            "target_position": target_position,
            "reported_position": float(state[0]),
            "position_error": position_error,
            "message": (
                "Joint accepts prismatic motion along the declared axis."
                if status in {"ok", "approximation"}
                else "Prismatic behavior check failed."
            ),
        }

    if expected_urdf in {"planar", "floating"}:
        return {
            "status": "approximation",
            "check": "unsupported_complex_joint_loaded",
            "message": f"{expected_fusion} maps to URDF {expected_urdf}; behavior needs a richer simulator/model check.",
        }

    return {"status": "failed", "check": "unknown_joint_mapping", "message": f"No validator for {expected_urdf}"}


def validate_step_import(step_path: str | Path) -> dict[str, Any]:
    step_path = Path(step_path)
    if not step_path.exists():
        return {"status": "failed", "reason": "STEP file not found"}
    try:
        import cadquery as cq  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"cadquery unavailable: {type(exc).__name__}: {exc}"}
    try:
        shape = cq.importers.importStep(str(step_path))
        solids = shape.solids().size() if hasattr(shape, "solids") else None
        valid = None
        try:
            valid = bool(shape.val().isValid())
        except Exception:
            valid = None
        return {"status": "ok", "solids": solids, "valid": valid}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def validate_step_imports(master: dict[str, Any]) -> dict[str, Any]:
    bodies = []
    for body in master.get("bodies", []):
        step_path = body["source_files"]["step"]
        check = validate_step_import(step_path)
        bodies.append({"body_id": body["id"], "step_path": step_path, **check})
    if not bodies:
        status = "failed"
    elif all(body["status"] == "ok" for body in bodies):
        status = "ok"
    elif any(body["status"] == "failed" for body in bodies):
        status = "failed"
    elif all(body["status"] == "skipped" for body in bodies):
        status = "skipped"
    else:
        status = "partial"
    return {"status": status, "bodies": bodies}


def run_pilot(
    dataset_dir: str | Path,
    output_dir: str | Path,
    sample_ids: list[str] | None = None,
    extra_sample_ids: list[str] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).expanduser()
    output_dir = Path(output_dir)
    masters_dir = output_dir / "master"
    urdf_dir = output_dir / "urdf"
    primary_selected = select_joint_sets(dataset_dir, sample_ids=sample_ids, limit=limit)
    extra_selected = select_joint_sets(dataset_dir, sample_ids=extra_sample_ids, limit=len(extra_sample_ids or [])) if extra_sample_ids else []
    selected = primary_selected + [path for path in extra_selected if path not in primary_selected]
    results = []

    for joint_set_path in selected:
        cohort = "primary" if joint_set_path in primary_selected else "additional_joint_type"
        result: dict[str, Any] = {
            "joint_set": joint_set_path.stem,
            "cohort": cohort,
            "source": str(joint_set_path),
            "parse": "not_run",
            "urdf": "not_run",
            "pybullet": {"status": "not_run"},
            "step_check": {"status": "not_run"},
            "warnings": [],
        }
        if not joint_set_path.exists():
            result["parse"] = "failed"
            result["warnings"].append("joint_set JSON missing")
            results.append(result)
            continue
        try:
            master = parse_joint_set(joint_set_path)
            result["parse"] = "ok"
            result["joint_count"] = len(master["joints"])
            result["joint_types"] = [joint["fusion_joint_type"] for joint in master["joints"]]
            result["mating_coordinate_frames"] = _mcf_summary(master)
            result["geometry_quality"] = _assembly_geometry_quality(master)
            for warning in result["geometry_quality"].get("warnings", []):
                result["warnings"].append(f"geometry QA: {warning}")
            master_path = masters_dir / f"{joint_set_path.stem}.json"
            _write_json(master_path, master)
            result["master_path"] = str(master_path)
        except Exception as exc:
            result["parse"] = "failed"
            result["warnings"].append(f"parse failed: {type(exc).__name__}: {exc}")
            results.append(result)
            continue

        try:
            urdf_info = write_urdf(master, urdf_dir / f"{joint_set_path.stem}.urdf")
            result["urdf"] = "ok"
            result["urdf_path"] = urdf_info["path"]
            if urdf_info["skipped_joints"]:
                result["warnings"].append(f"skipped {len(urdf_info['skipped_joints'])} parallel URDF joint(s)")
                result["skipped_parallel_joints"] = urdf_info["skipped_joints"]
            result["pybullet"] = validate_with_pybullet(
                urdf_info["path"],
                master=master,
                emitted_joint_ids=urdf_info["emitted_joint_ids"],
            )
        except Exception as exc:
            result["urdf"] = "failed"
            result["warnings"].append(f"URDF failed: {type(exc).__name__}: {exc}")

        result["step_check"] = validate_step_imports(master)
        results.append(result)

    report = {
        "schema": "cadengbench_a_pilot_report_v0",
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "primary_joint_sets": [path.stem for path in primary_selected],
        "additional_joint_type_joint_sets": [path.stem for path in extra_selected],
        "selected_joint_sets": [path.stem for path in selected],
        "environment": _environment_info(),
        "results": results,
        "summary": _summarise(results),
        "recommendation": _recommend(results),
    }
    _write_json(output_dir / "pilot_report.json", report)
    (output_dir / "pilot_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": _summarise_group(results),
        "primary": _summarise_group([r for r in results if r.get("cohort") == "primary"]),
        "additional_joint_type": _summarise_group([r for r in results if r.get("cohort") == "additional_joint_type"]),
    }


def _environment_info() -> dict[str, Any]:
    packages = {}
    for package in ("pybullet", "cadquery", "cadquery-ocp", "ocp"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "packages": packages,
    }


def _summarise_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    pybullet_statuses = [r.get("pybullet", {}).get("status") for r in results]
    step_statuses = [r.get("step_check", {}).get("status") for r in results]
    skipped_parallel = [
        skipped_joint
        for result in results
        for skipped_joint in result.get("skipped_parallel_joints", [])
    ]
    equivalence_classes = [
        (skipped_joint.get("equivalence") or {}).get("classification", "unknown")
        for skipped_joint in skipped_parallel
    ]
    return {
        "total": len(results),
        "parse_ok": sum(1 for r in results if r.get("parse") == "ok"),
        "urdf_ok": sum(1 for r in results if r.get("urdf") == "ok"),
        "pybullet_ok": pybullet_statuses.count("ok"),
        "pybullet_failed": pybullet_statuses.count("failed"),
        "pybullet_skipped": pybullet_statuses.count("skipped"),
        "step_ok": step_statuses.count("ok"),
        "step_failed": step_statuses.count("failed"),
        "step_skipped": step_statuses.count("skipped"),
        "skipped_parallel_joints": len(skipped_parallel),
        "likely_redundant_parallel_joints": sum(1 for value in equivalence_classes if value.startswith("likely_redundant")),
        "potentially_meaningful_parallel_joints": sum(
            1
            for value in equivalence_classes
            if value.startswith("potentially_meaningful") or value.startswith("additional_rigid")
        ),
    }


def _mcf_summary(master: dict[str, Any]) -> dict[str, Any]:
    frame_count = 0
    origin_types: dict[str, int] = {}
    for joint in master.get("joints", []):
        for frame in (joint.get("mating_coordinate_frames") or {}).values():
            frame_count += 1
            origin_type = frame.get("origin_type", "unknown")
            origin_types[origin_type] = origin_types.get(origin_type, 0) + 1
    return {"frame_count": frame_count, "origin_types": origin_types}


def _assembly_geometry_quality(master: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    bodies = []
    for body in master.get("bodies", []):
        quality = body.get("geometry_quality") or {}
        body_warnings = quality.get("warnings", [])
        if body_warnings:
            warnings.append(f"{body.get('id')}:{','.join(body_warnings)}")
        bodies.append(
            {
                "body_id": body.get("id"),
                "topology": body.get("topology", {}),
                "physical_properties": body.get("physical_properties", {}),
                "quality": quality,
            }
        )
    return {"status": "ok" if not warnings else "warnings", "warnings": warnings, "bodies": bodies}


def _recommend(results: list[dict[str, Any]]) -> str:
    primary_results = [result for result in results if result.get("cohort") == "primary"]
    if not primary_results:
        return "no-go: no assemblies were selected"
    parse_ok = sum(1 for r in primary_results if r.get("parse") == "ok")
    urdf_ok = sum(1 for r in primary_results if r.get("urdf") == "ok")
    pybullet_ok = sum(1 for r in primary_results if r.get("pybullet", {}).get("status") == "ok")
    step_ok = sum(1 for r in primary_results if r.get("step_check", {}).get("status") == "ok")
    if parse_ok < max(1, len(primary_results) // 2):
        return "no-go: parsing failed for too many selected joint sets"
    if urdf_ok < max(1, len(primary_results) // 2):
        return "revise: source parsing works, but URDF generation needs rethinking"
    if pybullet_ok < len(primary_results) or step_ok < len(primary_results):
        return (
            "hold: do not scale yet; primary samples need confirmed PyBullet behavior "
            "and STEP imports before upgrading the recommendation"
        )
    return "go for a broader pilot: primary samples passed parse, URDF, PyBullet behavior, and STEP import checks"


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# CADEngBench-A Pilot Report",
        "",
        f"Dataset: `{report['dataset_dir']}`",
        f"Output: `{report['output_dir']}`",
        "",
        "## Summary",
        "",
    ]
    lines.append(
        "| cohort | total | parse ok | URDF ok | PyBullet ok | STEP ok | skipped parallel | likely redundant | potentially meaningful |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, value in report["summary"].items():
        lines.append(
            "| {cohort} | {total} | {parse_ok} | {urdf_ok} | {pybullet_ok} | {step_ok} | {skipped} | {redundant} | {meaningful} |".format(
                cohort=key,
                total=value.get("total", 0),
                parse_ok=value.get("parse_ok", 0),
                urdf_ok=value.get("urdf_ok", 0),
                pybullet_ok=value.get("pybullet_ok", 0),
                step_ok=value.get("step_ok", 0),
                skipped=value.get("skipped_parallel_joints", 0),
                redundant=value.get("likely_redundant_parallel_joints", 0),
                meaningful=value.get("potentially_meaningful_parallel_joints", 0),
            )
        )
    environment = report.get("environment", {})
    packages = environment.get("packages", {})
    lines.extend(
        [
            "",
            "## Runner Environment",
            "",
            f"- Python: `{environment.get('python_version')}` at `{environment.get('python_executable')}`",
            f"- pybullet: `{packages.get('pybullet')}`",
            f"- cadquery: `{packages.get('cadquery')}`",
            f"- cadquery-ocp: `{packages.get('cadquery-ocp')}`",
        ]
    )
    lines.extend(["", f"Recommendation: **{report['recommendation']}**", "", "## Assemblies", ""])
    lines.append("| cohort | joint set | parse | URDF | PyBullet | STEP | joint types | warnings |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in report["results"]:
        lines.append(
            "| {cohort} | {joint_set} | {parse} | {urdf} | {pybullet} | {step} | {types} | {warnings} |".format(
                cohort=result.get("cohort", ""),
                joint_set=result["joint_set"],
                parse=result.get("parse"),
                urdf=result.get("urdf"),
                pybullet=result.get("pybullet", {}).get("status"),
                step=result.get("step_check", {}).get("status"),
                types=", ".join(result.get("joint_types", [])),
                warnings="<br>".join(result.get("warnings", [])),
            )
        )
    lines.extend(_schema_enhancements_markdown(report))
    lines.extend(_pybullet_details_markdown(report))
    lines.extend(_step_details_markdown(report))
    lines.extend(_parallel_joint_markdown(report))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Master JSON preserves every source joint, including multiple joints between the same two bodies.",
            "- Master JSON now includes AutoMate-inspired mating coordinate frames for each side of every source joint.",
            "- URDF is a tree format; parallel joints attaching the same child link are skipped in the generated URDF and logged above.",
            "- STEP and PyBullet checks are optional dependency checks in this pilot runner.",
        ]
    )
    return "\n".join(lines) + "\n"


def _schema_enhancements_markdown(report: dict[str, Any]) -> list[str]:
    lines = ["", "## CAD-Native Schema Enhancements", ""]
    lines.append(
        "This run adds AutoMate-inspired mating coordinate frames (MCFs) to the master JSON. "
        "These preserve CAD-local mate frames separately from the URDF approximation."
    )
    lines.append("")
    lines.append("| joint set | MCFs | origin types | geometry QA |")
    lines.append("| --- | ---: | --- | --- |")
    for result in report["results"]:
        mcf = result.get("mating_coordinate_frames") or {}
        origin_types = ", ".join(f"{key}:{value}" for key, value in sorted((mcf.get("origin_types") or {}).items()))
        geometry_quality = result.get("geometry_quality") or {}
        warnings = geometry_quality.get("warnings") or []
        lines.append(
            "| {joint_set} | {frame_count} | {origin_types} | {geometry_status}{warnings} |".format(
                joint_set=result.get("joint_set"),
                frame_count=mcf.get("frame_count", 0),
                origin_types=origin_types,
                geometry_status=geometry_quality.get("status", "not_run"),
                warnings=(": " + "; ".join(warnings)) if warnings else "",
            )
        )
    return lines


def _pybullet_details_markdown(report: dict[str, Any]) -> list[str]:
    lines = ["", "## PyBullet Behavior Checks", ""]
    for result in report["results"]:
        pybullet = result.get("pybullet", {})
        if pybullet.get("status") == "skipped":
            lines.append(f"- `{result['joint_set']}`: skipped - {pybullet.get('reason')}")
            continue
        if pybullet.get("status") == "failed" and "reason" in pybullet:
            lines.append(f"- `{result['joint_set']}`: failed - {pybullet.get('reason')}")
            continue
        lines.append(f"- `{result['joint_set']}`: {pybullet.get('status')} ({pybullet.get('num_joints', 0)} emitted URDF joint(s))")
        for joint in pybullet.get("joints", []):
            behavior = joint.get("behavior", {})
            detail = behavior.get("message", behavior.get("check", ""))
            mapping_note = joint.get("mapping_note")
            if mapping_note and mapping_note != "direct":
                detail = f"{detail} Mapping note: {mapping_note}."
            if behavior.get("axis_alignment_abs_dot") is not None:
                detail = f"{detail} Axis alignment abs-dot={behavior.get('axis_alignment_abs_dot'):.6f}."
            lines.append(
                "  - `{name}`: {fusion} -> {type_name}, behavior `{status}`; {detail}".format(
                    name=joint.get("name"),
                    fusion=joint.get("expected_fusion_joint_type"),
                    type_name=joint.get("type_name"),
                    status=behavior.get("status"),
                    detail=detail,
                )
            )
    return lines


def _step_details_markdown(report: dict[str, Any]) -> list[str]:
    lines = ["", "## STEP Import Checks", ""]
    for result in report["results"]:
        check = result.get("step_check", {})
        lines.append(f"- `{result['joint_set']}`: {check.get('status')}")
        for body in check.get("bodies", []):
            suffix = ""
            if body.get("status") == "ok":
                suffix = f", solids={body.get('solids')}, valid={body.get('valid')}"
            elif body.get("reason"):
                suffix = f", reason={body.get('reason')}"
            lines.append(f"  - `{body.get('body_id')}`: {body.get('status')}{suffix}")
    return lines


def _parallel_joint_markdown(report: dict[str, Any]) -> list[str]:
    skipped_results = [result for result in report["results"] if result.get("skipped_parallel_joints")]
    if not skipped_results:
        return []
    lines = ["", "## Parallel Joints Dropped From URDF", ""]
    lines.append(
        "URDF generation still drops extra joints that attach the same child link. "
        "The full source joint definitions below are preserved in master JSON and reported for human review."
    )
    for result in skipped_results:
        lines.append("")
        lines.append(f"### {result['joint_set']}")
        for skipped in result.get("skipped_parallel_joints", []):
            joint = skipped.get("joint", {})
            geo1 = joint.get("geometry_or_origin_one", {})
            geo2 = joint.get("geometry_or_origin_two", {})
            lines.append(f"- Dropped `{joint.get('id')}` (`{joint.get('name')}`), `{joint.get('fusion_joint_type')}`")
            lines.append(f"  - Reason: {skipped.get('reason')}")
            equivalence = skipped.get("equivalence") or {}
            kept = skipped.get("kept_joint") or {}
            if kept:
                lines.append(
                    "  - Kept comparison joint: `{id}` (`{name}`), `{joint_type}`".format(
                        id=kept.get("id"),
                        name=kept.get("name"),
                        joint_type=kept.get("fusion_joint_type"),
                    )
                )
                kept_axis = kept.get("axis") or {}
                lines.append(
                    "  - Kept axis origin `{origin}`, direction `{direction}`".format(
                        origin=kept_axis.get("origin"),
                        direction=kept_axis.get("direction"),
                    )
                )
            if equivalence:
                lines.append(
                    "  - Equivalence classification: `{classification}`; same_axis=`{same_axis}`, same_entities=`{same_entities}`, axis_alignment=`{axis_alignment:.6f}`, origin_to_axis_distance=`{distance:.6g}`".format(
                        classification=equivalence.get("classification"),
                        same_axis=equivalence.get("same_axis"),
                        same_entities=equivalence.get("same_selected_entities"),
                        axis_alignment=float(equivalence.get("axis_alignment_abs_dot") or 0.0),
                        distance=float(equivalence.get("skipped_origin_distance_to_kept_axis") or 0.0),
                    )
                )
            lines.append(
                "  - Body A `{body}` entity: `{etype}` index `{index}` `{subtype}`".format(
                    body=joint.get("body_a"),
                    etype=(geo1.get("entity_one") or {}).get("type"),
                    index=(geo1.get("entity_one") or {}).get("index"),
                    subtype=(geo1.get("entity_one") or {}).get("surface_type")
                    or (geo1.get("entity_one") or {}).get("curve_type"),
                )
            )
            lines.append(
                "  - Body B `{body}` entity: `{etype}` index `{index}` `{subtype}`".format(
                    body=joint.get("body_b"),
                    etype=(geo2.get("entity_one") or {}).get("type"),
                    index=(geo2.get("entity_one") or {}).get("index"),
                    subtype=(geo2.get("entity_one") or {}).get("surface_type")
                    or (geo2.get("entity_one") or {}).get("curve_type"),
                )
            )
            lines.append(
                "  - Axis origin `{origin}`, direction `{direction}`, offset `{offset}`, angle `{angle}`, flipped `{flipped}`".format(
                    origin=joint.get("axis", {}).get("origin"),
                    direction=joint.get("axis", {}).get("direction"),
                    offset=joint.get("offset"),
                    angle=joint.get("angle"),
                    flipped=joint.get("is_flipped"),
                )
            )
            lines.append(
                "  - Equivalent counts: body A `{a}`, body B `{b}`".format(
                    a=len(geo1.get("entity_one_equivalents", [])),
                    b=len(geo2.get("entity_one_equivalents", [])),
                )
            )
            lines.append(f"  - Constraint interpretation: {_constraint_interpretation(joint)}")
    return lines


def _constraint_interpretation(joint: dict[str, Any]) -> str:
    joint_type = joint.get("fusion_joint_type")
    body_a = joint.get("body_a")
    body_b = joint.get("body_b")
    axis = joint.get("axis", {})
    origin = axis.get("origin")
    direction = axis.get("direction")
    if joint_type == "RevoluteJointType":
        return (
            f"`{body_b}` is constrained to one rotational DOF relative to `{body_a}` "
            f"about axis origin `{origin}` and direction `{direction}`."
        )
    if joint_type == "RigidJointType":
        return (
            f"`{body_b}` is fully locked to `{body_a}` at the selected joint geometry "
            f"with alignment origin `{origin}` and axis direction `{direction}`."
        )
    return (
        f"`{body_b}` is constrained to `{body_a}` as `{joint_type}` using axis origin "
        f"`{origin}` and direction `{direction}`."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CADEngBench-A Assembly-Joint pilot pipeline.")
    parser.add_argument("--dataset-dir", default="~/Downloads/j1.0.0/joint")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "cadengbench_a_pilot"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--extra-sample-id", action="append", dest="extra_sample_ids")
    args = parser.parse_args(argv)

    report = run_pilot(
        args.dataset_dir,
        args.output_dir,
        sample_ids=args.sample_ids,
        extra_sample_ids=args.extra_sample_ids,
        limit=args.limit,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(report["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
