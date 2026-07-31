"""Compound PyBullet representations for multi-DOF A2 joint families."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pipeline.cadengbench_a_a2_pybullet import (
    _append_inertial as _append_cad_inertial,
    _append_mesh,
    _rpy_from_matrix,
    _transform_point,
)


COMPOUND_TYPES = {"CylindricalJointType", "PlanarJointType", "PinSlotJointType"}


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: list[float]) -> list[float]:
    norm = _norm(vector)
    if norm <= 1e-12:
        raise ValueError("motion direction must be non-zero")
    return [float(value) / norm for value in vector]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _scale(vector: list[float], scalar: float) -> list[float]:
    return [value * scalar for value in vector]


def _fmt(vector: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in vector)


def _perpendicular(axis: list[float]) -> list[float]:
    reference = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.8 else [0.0, 1.0, 0.0]
    return _unit(_cross(axis, reference))


def _validate_orthogonal(a: list[float], b: list[float], names: str) -> None:
    if abs(_dot(_unit(a), _unit(b))) > 1e-6:
        raise ValueError(f"{names} must be orthogonal")


def _joint_specs(joint_type: str, directions: dict[str, list[float]]) -> list[dict[str, Any]]:
    if joint_type == "CylindricalJointType":
        rotation = _unit(directions["rotation_axis"])
        slide = _unit(directions["slide_direction"])
        if abs(abs(_dot(rotation, slide)) - 1.0) > 1e-6:
            raise ValueError("cylindrical rotation and slide axes must be coaxial")
        return [
            {"name": "rotation", "urdf_type": "revolute", "axis": rotation, "motion": "rotation"},
            {"name": "slide", "urdf_type": "prismatic", "axis": slide, "motion": "translation"},
        ]
    if joint_type == "PlanarJointType":
        normal = _unit(directions["normal_direction"])
        primary = _unit(directions["primary_slide_direction"])
        secondary = _unit(directions["secondary_slide_direction"])
        _validate_orthogonal(normal, primary, "planar normal and primary directions")
        _validate_orthogonal(normal, secondary, "planar normal and secondary directions")
        _validate_orthogonal(primary, secondary, "planar slide directions")
        return [
            {"name": "primary_slide", "urdf_type": "prismatic", "axis": primary, "motion": "translation"},
            {"name": "secondary_slide", "urdf_type": "prismatic", "axis": secondary, "motion": "translation"},
            {"name": "normal_rotation", "urdf_type": "revolute", "axis": normal, "motion": "rotation"},
        ]
    if joint_type == "PinSlotJointType":
        rotation = _unit(directions["rotation_axis"])
        slide = _unit(directions["slide_direction"])
        _validate_orthogonal(rotation, slide, "pin-slot rotation and slide directions")
        return [
            {"name": "slot_slide", "urdf_type": "prismatic", "axis": slide, "motion": "translation"},
            {"name": "pin_rotation", "urdf_type": "revolute", "axis": rotation, "motion": "rotation"},
        ]
    raise ValueError(f"no compound fixture for {joint_type}")


def _append_inertial(link: ET.Element, mass: float, origin: list[float] | None = None) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": _fmt(origin or [0.0, 0.0, 0.0]), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass:.12g}"})
    diagonal = max(mass * 0.01, 1e-8)
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{diagonal:.12g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{diagonal:.12g}",
            "iyz": "0",
            "izz": f"{diagonal:.12g}",
        },
    )


def write_compound_fixture_urdf(
    joint_type: str,
    directions: dict[str, list[float]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a deterministic compound-joint fixture URDF."""
    specs = _joint_specs(joint_type, directions)
    rotation_axis = next((spec["axis"] for spec in specs if spec["motion"] == "rotation"), specs[0]["axis"])
    child_offset = _scale(_perpendicular(rotation_axis), 0.75)
    robot = ET.Element("robot", {"name": f"a2_compound_fixture_{joint_type}"})
    parent = ET.SubElement(robot, "link", {"name": "parent"})
    _append_inertial(parent, 1.0)
    parent_collision = ET.SubElement(parent, "collision")
    parent_geometry = ET.SubElement(parent_collision, "geometry")
    ET.SubElement(parent_geometry, "sphere", {"radius": "0.1"})

    previous_link = "parent"
    for index, spec in enumerate(specs):
        is_final = index == len(specs) - 1
        link_name = "child" if is_final else f"virtual_{index}"
        link = ET.SubElement(robot, "link", {"name": link_name})
        # Keep the fixture COM at the joint center so a force applied at the COM
        # is a pure force and cannot introduce an accidental moment.
        _append_inertial(link, 1.0 if is_final else 1e-4)
        if is_final:
            for tag in ("visual", "collision"):
                node = ET.SubElement(link, tag)
                ET.SubElement(node, "origin", {"xyz": _fmt(child_offset), "rpy": "0 0 0"})
                geometry = ET.SubElement(node, "geometry")
                ET.SubElement(geometry, "box", {"size": "0.2 0.2 0.2"})
        joint = ET.SubElement(robot, "joint", {"name": spec["name"], "type": spec["urdf_type"]})
        ET.SubElement(joint, "parent", {"link": previous_link})
        ET.SubElement(joint, "child", {"link": link_name})
        ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(joint, "axis", {"xyz": _fmt(spec["axis"])})
        span = math.pi if spec["motion"] == "rotation" else 1.0
        ET.SubElement(
            joint,
            "limit",
            {"lower": f"{-span:.12g}", "upper": f"{span:.12g}", "effort": "100", "velocity": "10"},
        )
        previous_link = link_name

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "path": str(output_path),
        "joint_type": joint_type,
        "joint_specs": specs,
        "final_link_index": len(specs) - 1,
    }


def _source_limit(
    joint: dict[str, Any],
    spec: dict[str, Any],
    mesh_scale: float,
) -> tuple[float, float, float, float, str]:
    group_name = {
        "rotation": "rotation",
        "slide": "slide",
        "primary_slide": "primary_slide",
        "secondary_slide": "secondary_slide",
        "normal_rotation": "rotation",
        "slot_slide": "slide",
        "pin_rotation": "rotation",
    }[spec["name"]]
    group = ((joint.get("motion_semantics") or {}).get("limit_groups") or {}).get(group_name, {})
    is_translation = spec["motion"] == "translation"
    scale = mesh_scale if is_translation else 1.0
    fallback = (-0.1, 0.1) if is_translation else (-math.pi, math.pi)
    default_target = 0.01 if is_translation else 0.25
    lower = float(group.get("lower")) * scale if group.get("lower_enabled") else fallback[0]
    upper = float(group.get("upper")) * scale if group.get("upper_enabled") else fallback[1]
    if upper - lower <= 1e-9:
        return lower, upper, lower, lower, "source_limits_lock_motion"
    baseline = 0.0 if lower <= 0.0 <= upper else lower + 0.5 * (upper - lower)
    target = min(max(default_target, lower + 0.25 * (upper - lower)), upper - 0.25 * (upper - lower))
    if abs(target - baseline) <= 1e-9:
        target = lower + 0.75 * (upper - lower)
    return lower, upper, baseline, target, "motion_available"


def write_selected_compound_cad_urdf(
    master: dict[str, Any],
    joint_id: str,
    output_path: str | Path,
    mesh_scale: float = 0.01,
) -> dict[str, Any]:
    """Write a compound URDF using the selected source CAD geometry."""
    joint = next((item for item in master.get("joints", []) if item.get("id") == joint_id), None)
    if joint is None:
        raise ValueError(f"joint not found: {joint_id}")
    joint_type = str(joint.get("fusion_joint_type"))
    if joint_type not in COMPOUND_TYPES:
        raise ValueError(f"no compound CAD representation for {joint_type}")
    directions = (joint.get("motion_semantics") or {}).get("resolved_directions") or {}
    specs = _joint_specs(joint_type, directions)
    bodies = {body["id"]: body for body in master.get("bodies", [])}
    parent_body = bodies[joint["body_a"]]
    child_body = bodies[joint["body_b"]]
    parent_transform = joint["geometry_or_origin_one"]["transform_matrix"]
    child_transform = joint["geometry_or_origin_two"]["transform_matrix"]
    if not parent_transform or not child_transform:
        raise ValueError("selected joint is missing occurrence placement transforms")

    joint_origin = _scale([float(value) for value in joint["axis"]["origin"]], mesh_scale)
    parent_mesh_origin = _scale([float(parent_transform[row][3]) for row in range(3)], mesh_scale)
    child_world_origin = _scale([float(child_transform[row][3]) for row in range(3)], mesh_scale)
    child_mesh_origin = _sub(child_world_origin, joint_origin)
    parent_rpy = _rpy_from_matrix(parent_transform)
    child_rpy = _rpy_from_matrix(child_transform)

    robot = ET.Element("robot", {"name": f"a2_compound_{master['joint_set_id']}_{joint_id}"})
    parent = ET.SubElement(robot, "link", {"name": parent_body["id"]})
    _append_cad_inertial(parent, [0.0, 0.0, 0.0], 1.0, [0.1, 0.1, 0.1])
    _append_mesh(parent, parent_body["source_files"]["obj"], parent_mesh_origin, parent_rpy, mesh_scale)

    previous_link = parent_body["id"]
    targets = []
    baselines = []
    limit_statuses = []
    for index, spec in enumerate(specs):
        is_final = index == len(specs) - 1
        link_name = child_body["id"] if is_final else f"virtual_{index}"
        link = ET.SubElement(robot, "link", {"name": link_name})
        if is_final:
            center_of_mass = child_body.get("physical_properties", {}).get("center_of_mass") or {}
            local_center = [float(center_of_mass.get(name, 0.0)) for name in ("x", "y", "z")]
            child_com_world = _scale(_transform_point(child_transform, local_center), mesh_scale)
            child_com_joint = _sub(child_com_world, joint_origin)
            extents_cm = ((child_body.get("geometry_quality") or {}).get("bounding_box") or {}).get("extents") or [1.0, 1.0, 1.0]
            extents = _scale([float(value) for value in extents_cm], mesh_scale)
            mass = float(child_body.get("physical_properties", {}).get("mass") or 1.0)
            _append_cad_inertial(link, child_com_joint, mass, extents)
            _append_mesh(link, child_body["source_files"]["obj"], child_mesh_origin, child_rpy, mesh_scale)
        else:
            _append_inertial(link, 1e-4)

        node = ET.SubElement(robot, "joint", {"name": spec["name"], "type": spec["urdf_type"]})
        ET.SubElement(node, "parent", {"link": previous_link})
        ET.SubElement(node, "child", {"link": link_name})
        ET.SubElement(node, "origin", {"xyz": _fmt(joint_origin if index == 0 else [0.0, 0.0, 0.0]), "rpy": "0 0 0"})
        ET.SubElement(node, "axis", {"xyz": _fmt(spec["axis"])})
        lower, upper, baseline, target, limit_status = _source_limit(joint, spec, mesh_scale)
        ET.SubElement(
            node,
            "limit",
            {"lower": f"{lower:.12g}", "upper": f"{upper:.12g}", "effort": "100", "velocity": "10"},
        )
        targets.append(target)
        baselines.append(baseline)
        limit_statuses.append(limit_status)
        previous_link = link_name

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "path": str(output_path),
        "joint_id": joint_id,
        "joint_type": joint_type,
        "directions": directions,
        "joint_specs": specs,
        "target_coordinates": targets,
        "baseline_coordinates": baselines,
        "source_limit_statuses": limit_statuses,
        "mesh_scale_cm_to_m": mesh_scale,
        "placement_policy": "fusion_occurrence_transforms_with_compound_chain_at_body_a_axis_origin",
    }


def _quat_delta(p: Any, before: list[float], after: list[float]) -> tuple[float, list[float]]:
    inverse_before = p.invertTransform([0.0, 0.0, 0.0], before)[1]
    delta = p.multiplyTransforms([0.0, 0.0, 0.0], after, [0.0, 0.0, 0.0], inverse_before)[1]
    x, y, z, w = (float(value) for value in delta)
    vector_norm = math.sqrt(x * x + y * y + z * z)
    angle = 2.0 * math.atan2(vector_norm, abs(w))
    axis = [x / vector_norm, y / vector_norm, z / vector_norm] if vector_norm > 1e-12 else [0.0, 0.0, 1.0]
    return angle, axis


def _link_frame(p: Any, body_id: int, link_index: int) -> tuple[list[float], list[float]]:
    state = p.getLinkState(body_id, link_index, computeForwardKinematics=True)
    return list(state[4]), list(state[5])


def _reset_all(p: Any, body_id: int, count: int, values: list[float] | None = None) -> None:
    values = values or [0.0] * count
    for index, value in enumerate(values):
        p.resetJointState(body_id, index, value, 0.0)


def _forbidden_wrench(joint_type: str, specs: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    if joint_type == "CylindricalJointType":
        perpendicular = _perpendicular(specs[0]["axis"])
        return perpendicular, perpendicular
    if joint_type == "PlanarJointType":
        normal = specs[2]["axis"]
        primary = specs[0]["axis"]
        return normal, primary
    rotation = specs[1]["axis"]
    slide = specs[0]["axis"]
    normal = _unit(_cross(rotation, slide))
    return normal, slide


def run_compound_fixture(
    urdf_path: str | Path,
    joint_type: str,
    directions: dict[str, list[float]],
    steps: int = 240,
    target_coordinates: list[float] | None = None,
    baseline_coordinates: list[float] | None = None,
    limit_statuses: list[str] | None = None,
    scope: str = "adapter_fixture_not_model_evaluation",
) -> dict[str, Any]:
    """Verify allowed and forbidden motion for a compound joint contract."""
    try:
        import pybullet as p  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"pybullet unavailable: {type(exc).__name__}: {exc}"}

    specs = _joint_specs(joint_type, directions)
    baselines = baseline_coordinates or [0.0] * len(specs)
    statuses = limit_statuses or ["motion_available"] * len(specs)
    final_link = len(specs) - 1
    client = p.connect(p.DIRECT)
    checks: dict[str, Any] = {}
    try:
        p.setGravity(0.0, 0.0, 0.0)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)
        body_id = p.loadURDF(str(urdf_path), useFixedBase=True)
        loaded = []
        representation_passed = p.getNumJoints(body_id) == len(specs)
        for index, spec in enumerate(specs):
            info = p.getJointInfo(body_id, index)
            expected_code = p.JOINT_REVOLUTE if spec["urdf_type"] == "revolute" else p.JOINT_PRISMATIC
            loaded_axis = _unit([float(value) for value in info[13]])
            axis_alignment = abs(_dot(loaded_axis, spec["axis"]))
            passed = int(info[2]) == expected_code and axis_alignment > 0.999999
            representation_passed = representation_passed and passed
            loaded.append({"name": spec["name"], "passed": passed, "axis_alignment_abs_dot": axis_alignment})
        checks["representation"] = {"passed": representation_passed, "joints": loaded}

        allowed_results = []
        for index, spec in enumerate(specs):
            _reset_all(p, body_id, len(specs), baselines)
            before_position, before_orientation = _link_frame(p, body_id, final_link)
            target = target_coordinates[index] if target_coordinates is not None else (0.25 if spec["motion"] == "rotation" else 0.2)
            if statuses[index] == "source_limits_lock_motion":
                info = p.getJointInfo(body_id, index)
                passed = abs(float(info[8]) - float(info[9])) <= 1e-9
                allowed_results.append(
                    {
                        "joint": spec["name"],
                        "motion": "source_limit_lock",
                        "passed": passed,
                        "locked_coordinate": baselines[index],
                    }
                )
                continue
            p.resetJointState(body_id, index, target, 0.0)
            after_position, after_orientation = _link_frame(p, body_id, final_link)
            displacement = _sub(after_position, before_position)
            rotation, rotation_axis = _quat_delta(p, before_orientation, after_orientation)
            coordinate_delta = abs(target - baselines[index])
            if spec["motion"] == "rotation":
                passed = (
                    _norm(displacement) < 1e-6
                    and abs(rotation - coordinate_delta) < 1e-6
                    and abs(_dot(rotation_axis, spec["axis"])) > 0.999999
                )
            else:
                passed = (
                    abs(_norm(displacement) - coordinate_delta) < 1e-6
                    and abs(_dot(_unit(displacement), spec["axis"])) > 0.999999
                    and rotation < 1e-7
                )
            allowed_results.append(
                {
                    "joint": spec["name"],
                    "motion": spec["motion"],
                    "passed": passed,
                    "displacement": displacement,
                    "orientation_delta_rad": rotation,
                }
            )
        checks["allowed_motion"] = {
            "passed": all(result["passed"] for result in allowed_results),
            "motions": allowed_results,
        }

        targets = target_coordinates or [0.12] * len(specs)
        _reset_all(p, body_id, len(specs), targets)
        for index in range(len(specs)):
            p.setJointMotorControl2(body_id, index, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
        for _ in range(steps):
            p.stepSimulation()
        drifts = [abs(float(p.getJointState(body_id, index)[0]) - targets[index]) for index in range(len(specs))]
        checks["stability"] = {"passed": max(drifts) < 1e-5, "coordinate_drifts": drifts, "steps": steps}

        forbidden_baseline = target_coordinates or baselines
        _reset_all(p, body_id, len(specs), forbidden_baseline)
        for index in range(len(specs)):
            p.setJointMotorControl2(body_id, index, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
        forbidden_before_position, forbidden_before_orientation = _link_frame(p, body_id, final_link)
        force_direction, torque_direction = _forbidden_wrench(joint_type, specs)
        dynamics = p.getDynamicsInfo(body_id, final_link)
        mass = max(float(dynamics[0]), 1e-8)
        inertia_diagonal = [max(float(value), 1e-10) for value in dynamics[2]]
        force_magnitude = mass
        torque_magnitude = min(inertia_diagonal)
        for _ in range(steps):
            joint_center = p.getLinkState(body_id, final_link, computeForwardKinematics=True)[4]
            p.applyExternalForce(body_id, final_link, _scale(force_direction, force_magnitude), joint_center, p.WORLD_FRAME)
            p.applyExternalTorque(body_id, final_link, _scale(torque_direction, torque_magnitude), p.WORLD_FRAME)
            p.stepSimulation()
        forbidden_after_position, forbidden_after_orientation = _link_frame(p, body_id, final_link)
        forbidden_displacement = _norm(_sub(forbidden_after_position, forbidden_before_position))
        forbidden_rotation, _ = _quat_delta(p, forbidden_before_orientation, forbidden_after_orientation)
        coordinate_drifts = [
            abs(float(p.getJointState(body_id, index)[0]) - forbidden_baseline[index])
            for index in range(len(specs))
        ]
        duration = steps / 240.0
        unconstrained_reference_motion = 0.5 * duration * duration
        maximum_leakage = max(forbidden_displacement, forbidden_rotation, max(coordinate_drifts))
        leakage_ratio = maximum_leakage / unconstrained_reference_motion
        checks["forbidden_motion"] = {
            "passed": leakage_ratio < 1e-3,
            "link_frame_displacement_norm": forbidden_displacement,
            "orientation_delta_rad": forbidden_rotation,
            "coordinate_drifts": coordinate_drifts,
            "normalization": "one_unit_unconstrained_linear_and_angular_acceleration",
            "force_magnitude": force_magnitude,
            "torque_magnitude": torque_magnitude,
            "unconstrained_reference_motion": unconstrained_reference_motion,
            "maximum_leakage_ratio": leakage_ratio,
            "maximum_allowed_leakage_ratio": 1e-3,
        }

        lock_results = []
        for locked_index, status in enumerate(statuses):
            if status != "source_limits_lock_motion":
                continue
            _reset_all(p, body_id, len(specs), baselines)
            for index in range(len(specs)):
                p.setJointMotorControl2(body_id, index, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
            dynamics = p.getDynamicsInfo(body_id, final_link)
            mass = max(float(dynamics[0]), 1e-8)
            inertia_diagonal = [max(float(value), 1e-10) for value in dynamics[2]]
            spec = specs[locked_index]
            for _ in range(steps):
                joint_center = p.getLinkState(body_id, final_link, computeForwardKinematics=True)[4]
                if spec["motion"] == "translation":
                    p.applyExternalForce(body_id, final_link, _scale(spec["axis"], mass), joint_center, p.WORLD_FRAME)
                else:
                    p.applyExternalTorque(body_id, final_link, _scale(spec["axis"], min(inertia_diagonal)), p.WORLD_FRAME)
                p.stepSimulation()
            drift = abs(float(p.getJointState(body_id, locked_index)[0]) - baselines[locked_index])
            leakage_ratio = drift / unconstrained_reference_motion
            lock_results.append(
                {
                    "joint": spec["name"],
                    "passed": leakage_ratio < 1e-3,
                    "coordinate_drift": drift,
                    "maximum_leakage_ratio": leakage_ratio,
                    "maximum_allowed_leakage_ratio": 1e-3,
                }
            )
        if lock_results:
            checks["source_limit_locks"] = {
                "passed": all(result["passed"] for result in lock_results),
                "coordinates": lock_results,
            }
        contacts = len(
            p.getClosestPoints(
                bodyA=body_id,
                bodyB=body_id,
                distance=0.0,
                linkIndexA=-1,
                linkIndexB=final_link,
            )
        )
        checks["collision_query"] = {
            "passed": True,
            "nominal_self_contact_count": contacts,
            "interpretation": "diagnostic_only_for_fixture",
        }
        status = "passed" if all(check["passed"] for check in checks.values()) else "failed"
        return {
            "schema": "cadengbench_a_a2_compound_fixture_result_v1",
            "scope": scope,
            "status": status,
            "joint_type": joint_type,
            "checks": checks,
        }
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "checks": checks}
    finally:
        p.disconnect(client)
