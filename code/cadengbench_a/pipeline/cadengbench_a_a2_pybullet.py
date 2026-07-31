"""Direct rigid, revolute, and slider PyBullet adapters for A2."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DIRECT_JOINT_TYPES = {
    "RigidJointType": "fixed",
    "RevoluteJointType": "revolute",
    "SliderJointType": "prismatic",
}


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm <= 1e-12:
        raise ValueError("joint axis must be non-zero")
    return [float(value) / norm for value in vector]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _scale(vector: list[float], scalar: float) -> list[float]:
    return [value * scalar for value in vector]


def _perpendicular(axis: list[float]) -> list[float]:
    reference = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.8 else [0.0, 1.0, 0.0]
    return _unit(_cross(axis, reference))


def _fmt(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _rpy_from_matrix(matrix: list[list[float]]) -> list[float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(matrix[2][0]))))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(matrix[2][1]), float(matrix[2][2]))
        yaw = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
    else:
        roll = math.atan2(-float(matrix[1][2]), float(matrix[1][1]))
        yaw = 0.0
    return [roll, pitch, yaw]


def _transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    return [
        sum(float(matrix[row][column]) * float(point[column]) for column in range(3))
        + float(matrix[row][3])
        for row in range(3)
    ]


def _append_inertial(link: ET.Element, origin: list[float], mass: float, extents: list[float]) -> None:
    mass = max(float(mass), 1e-6)
    x, y, z = (max(float(value), 1e-4) for value in extents)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": _fmt(origin), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass:.12g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{mass * (y * y + z * z) / 12.0:.12g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{mass * (x * x + z * z) / 12.0:.12g}",
            "iyz": "0",
            "izz": f"{mass * (x * x + y * y) / 12.0:.12g}",
        },
    )


def _append_mesh(link: ET.Element, mesh_path: str, origin: list[float], rpy: list[float], scale: float) -> None:
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", {"xyz": _fmt(origin), "rpy": _fmt(rpy)})
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {"filename": mesh_path, "scale": _fmt([scale, scale, scale])},
        )


def _selected_axis(joint: dict[str, Any]) -> list[float]:
    directions = (joint.get("motion_semantics") or {}).get("resolved_directions") or {}
    joint_type = joint.get("fusion_joint_type")
    if joint_type == "RevoluteJointType":
        return _unit(directions["rotation_axis"])
    if joint_type == "SliderJointType":
        return _unit(directions["slide_direction"])
    return _unit((joint.get("axis") or {}).get("direction") or [0.0, 0.0, 1.0])


def _real_joint_limits(joint: dict[str, Any], mesh_scale: float) -> tuple[float, float, float, str]:
    joint_type = joint["fusion_joint_type"]
    if joint_type == "RevoluteJointType":
        group = ((joint.get("motion_semantics") or {}).get("limit_groups") or {}).get("rotation", {})
        fallback = (-math.pi, math.pi)
        target = 0.25
    else:
        group = ((joint.get("motion_semantics") or {}).get("limit_groups") or {}).get("slide", {})
        fallback = (-0.1, 0.1)
        target = 0.01
    lower = float(group.get("lower")) * (mesh_scale if joint_type == "SliderJointType" else 1.0) if group.get("lower_enabled") else fallback[0]
    upper = float(group.get("upper")) * (mesh_scale if joint_type == "SliderJointType" else 1.0) if group.get("upper_enabled") else fallback[1]
    if upper - lower <= 1e-9:
        return lower, upper, lower, "source_limits_lock_motion"
    target = min(max(target, lower + 0.25 * (upper - lower)), upper - 0.25 * (upper - lower))
    return lower, upper, target, "motion_available"


def write_selected_cad_urdf(
    master: dict[str, Any],
    joint_id: str,
    output_path: str | Path,
    mesh_scale: float = 0.01,
) -> dict[str, Any]:
    """Write one selected direct hypothesis with nominal Fusion occurrence placement."""
    joint = next((item for item in master.get("joints", []) if item.get("id") == joint_id), None)
    if joint is None:
        raise ValueError(f"joint not found: {joint_id}")
    joint_type = joint.get("fusion_joint_type")
    urdf_type = DIRECT_JOINT_TYPES.get(str(joint_type))
    if urdf_type is None:
        raise ValueError(f"no direct PyBullet CAD representation for {joint_type}")
    bodies = {body["id"]: body for body in master.get("bodies", [])}
    parent_body = bodies[joint["body_a"]]
    child_body = bodies[joint["body_b"]]
    parent_transform = joint["geometry_or_origin_one"]["transform_matrix"]
    child_transform = joint["geometry_or_origin_two"]["transform_matrix"]
    if not parent_transform or not child_transform:
        raise ValueError("selected joint is missing occurrence placement transforms")

    joint_origin_cm = [float(value) for value in joint["axis"]["origin"]]
    joint_origin = _scale(joint_origin_cm, mesh_scale)
    parent_mesh_origin = _scale([float(parent_transform[row][3]) for row in range(3)], mesh_scale)
    child_world_origin = _scale([float(child_transform[row][3]) for row in range(3)], mesh_scale)
    child_mesh_origin = _sub(child_world_origin, joint_origin)
    parent_rpy = _rpy_from_matrix(parent_transform)
    child_rpy = _rpy_from_matrix(child_transform)
    axis = _selected_axis(joint)

    robot = ET.Element("robot", {"name": f"a2_{master['joint_set_id']}_{joint_id}"})
    parent = ET.SubElement(robot, "link", {"name": parent_body["id"]})
    _append_inertial(parent, [0.0, 0.0, 0.0], 1.0, [0.1, 0.1, 0.1])
    _append_mesh(parent, parent_body["source_files"]["obj"], parent_mesh_origin, parent_rpy, mesh_scale)

    child = ET.SubElement(robot, "link", {"name": child_body["id"]})
    center_of_mass = child_body.get("physical_properties", {}).get("center_of_mass") or {}
    local_center = [float(center_of_mass.get(name, 0.0)) for name in ("x", "y", "z")]
    child_com_world = _scale(_transform_point(child_transform, local_center), mesh_scale)
    child_com_joint = _sub(child_com_world, joint_origin)
    extents_cm = ((child_body.get("geometry_quality") or {}).get("bounding_box") or {}).get("extents") or [1.0, 1.0, 1.0]
    extents = _scale([float(value) for value in extents_cm], mesh_scale)
    mass = child_body.get("physical_properties", {}).get("mass") or 1.0
    _append_inertial(child, child_com_joint, float(mass), extents)
    _append_mesh(child, child_body["source_files"]["obj"], child_mesh_origin, child_rpy, mesh_scale)

    joint_node = ET.SubElement(robot, "joint", {"name": joint_id, "type": urdf_type})
    ET.SubElement(joint_node, "parent", {"link": parent_body["id"]})
    ET.SubElement(joint_node, "child", {"link": child_body["id"]})
    ET.SubElement(joint_node, "origin", {"xyz": _fmt(joint_origin), "rpy": "0 0 0"})
    target = 0.0
    limit_status = "not_applicable"
    if urdf_type != "fixed":
        ET.SubElement(joint_node, "axis", {"xyz": _fmt(axis)})
        lower, upper, target, limit_status = _real_joint_limits(joint, mesh_scale)
        ET.SubElement(
            joint_node,
            "limit",
            {"lower": f"{lower:.12g}", "upper": f"{upper:.12g}", "effort": "100", "velocity": "10"},
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "path": str(output_path),
        "joint_id": joint_id,
        "fusion_joint_type": joint_type,
        "axis": axis,
        "target_coordinate": target,
        "source_limit_status": limit_status,
        "mesh_scale_cm_to_m": mesh_scale,
        "placement_policy": "fusion_occurrence_transforms_with_joint_frame_at_body_a_axis_origin",
    }


def write_direct_fixture_urdf(
    fusion_joint_type: str,
    axis: list[float],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a controlled two-link fixture for one direct A2 joint contract."""
    urdf_type = DIRECT_JOINT_TYPES.get(fusion_joint_type)
    if urdf_type is None:
        raise ValueError(f"no direct PyBullet fixture for {fusion_joint_type}")
    axis = _unit(axis)
    perpendicular = _perpendicular(axis)
    child_offset = _scale(perpendicular, 0.75)

    robot = ET.Element("robot", {"name": f"a2_fixture_{fusion_joint_type}"})
    parent = ET.SubElement(robot, "link", {"name": "parent"})
    parent_inertial = ET.SubElement(parent, "inertial")
    ET.SubElement(parent_inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(parent_inertial, "mass", {"value": "1"})
    ET.SubElement(
        parent_inertial,
        "inertia",
        {"ixx": "0.01", "ixy": "0", "ixz": "0", "iyy": "0.01", "iyz": "0", "izz": "0.01"},
    )
    parent_collision = ET.SubElement(parent, "collision")
    parent_geometry = ET.SubElement(parent_collision, "geometry")
    ET.SubElement(parent_geometry, "sphere", {"radius": "0.1"})

    child = ET.SubElement(robot, "link", {"name": "child"})
    inertial = ET.SubElement(child, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": _fmt(child_offset), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "1"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "0.02", "ixy": "0", "ixz": "0", "iyy": "0.02", "iyz": "0", "izz": "0.02"},
    )
    for tag in ("visual", "collision"):
        node = ET.SubElement(child, tag)
        ET.SubElement(node, "origin", {"xyz": _fmt(child_offset), "rpy": "0 0 0"})
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(geometry, "box", {"size": "0.2 0.2 0.2"})

    joint = ET.SubElement(robot, "joint", {"name": "selected_joint", "type": urdf_type})
    ET.SubElement(joint, "parent", {"link": "parent"})
    ET.SubElement(joint, "child", {"link": "child"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    if urdf_type != "fixed":
        ET.SubElement(joint, "axis", {"xyz": _fmt(axis)})
        span = math.pi if urdf_type == "revolute" else 1.0
        ET.SubElement(
            joint,
            "limit",
            {"lower": f"{-span:.12g}", "upper": f"{span:.12g}", "effort": "100", "velocity": "10"},
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "path": str(output_path),
        "fusion_joint_type": fusion_joint_type,
        "urdf_joint_type": urdf_type,
        "axis": axis,
        "fixture_child_offset": child_offset,
    }


def _quat_delta(p: Any, before: list[float] | tuple[float, ...], after: list[float] | tuple[float, ...]) -> tuple[float, list[float]]:
    inverse_before = p.invertTransform([0.0, 0.0, 0.0], before)[1]
    delta = p.multiplyTransforms([0.0, 0.0, 0.0], after, [0.0, 0.0, 0.0], inverse_before)[1]
    x, y, z, w = (float(value) for value in delta)
    vector_norm = math.sqrt(x * x + y * y + z * z)
    angle = 2.0 * math.atan2(vector_norm, abs(w))
    rotation_axis = [x / vector_norm, y / vector_norm, z / vector_norm] if vector_norm > 1e-12 else [0.0, 0.0, 1.0]
    return angle, rotation_axis


def _link_frame(p: Any, body_id: int) -> tuple[list[float], list[float]]:
    state = p.getLinkState(body_id, 0, computeForwardKinematics=True)
    return list(state[4]), list(state[5])


def _collision_count(p: Any, body_id: int) -> int:
    return len(
        p.getClosestPoints(
            bodyA=body_id,
            bodyB=body_id,
            distance=0.0,
            linkIndexA=-1,
            linkIndexB=0,
        )
    )


def run_direct_fixture(
    urdf_path: str | Path,
    fusion_joint_type: str,
    expected_axis: list[float],
    steps: int = 240,
    target_coordinate: float | None = None,
    scope: str = "adapter_fixture_not_model_evaluation",
) -> dict[str, Any]:
    """Execute the controlled behavior checks required before real-CAD A2 runs."""
    try:
        import pybullet as p  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"pybullet unavailable: {type(exc).__name__}: {exc}"}

    axis = _unit(expected_axis)
    perpendicular = _perpendicular(axis)
    client = p.connect(p.DIRECT)
    checks: dict[str, Any] = {}
    try:
        p.setGravity(0.0, 0.0, 0.0)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        body_id = p.loadURDF(str(urdf_path), useFixedBase=True)
        if p.getNumJoints(body_id) != 1:
            return {"status": "failed", "reason": "fixture must contain exactly one joint"}
        info = p.getJointInfo(body_id, 0)
        loaded_axis = _unit([float(value) for value in info[13]]) if fusion_joint_type != "RigidJointType" else axis
        axis_alignment = abs(_dot(axis, loaded_axis))
        expected_type = {
            "RigidJointType": p.JOINT_FIXED,
            "RevoluteJointType": p.JOINT_REVOLUTE,
            "SliderJointType": p.JOINT_PRISMATIC,
        }[fusion_joint_type]
        checks["representation"] = {
            "passed": int(info[2]) == expected_type and axis_alignment > 0.999999,
            "expected_joint_type_code": expected_type,
            "actual_joint_type_code": int(info[2]),
            "axis_alignment_abs_dot": axis_alignment,
        }

        before_position, before_orientation = _link_frame(p, body_id)
        nominal_contacts = _collision_count(p, body_id)
        if fusion_joint_type == "RigidJointType":
            checks["allowed_motion"] = {
                "passed": True,
                "expected": "no_allowed_motion",
            }
            target = 0.0
            after_position, after_orientation = before_position, before_orientation
        else:
            target = target_coordinate if target_coordinate is not None else (0.35 if fusion_joint_type == "RevoluteJointType" else 0.2)
            p.resetJointState(body_id, 0, target, 0.0)
            after_position, after_orientation = _link_frame(p, body_id)
            reported = float(p.getJointState(body_id, 0)[0])
            displacement = _sub(after_position, before_position)
            orientation_delta, rotation_axis = _quat_delta(p, before_orientation, after_orientation)
            if fusion_joint_type == "RevoluteJointType":
                expected_orientation_delta = abs((target + math.pi) % (2.0 * math.pi) - math.pi)
                passed = (
                    abs(reported - target) < 1e-7
                    and _norm(displacement) < 1e-6
                    and abs(orientation_delta - expected_orientation_delta) < 1e-6
                    and abs(_dot(axis, rotation_axis)) > 0.999999
                )
            else:
                passed = (
                    abs(reported - target) < 1e-7
                    and abs(_norm(displacement) - abs(target)) < 1e-6
                    and abs(_dot(_unit(displacement), axis)) > 0.999999
                    and orientation_delta < 1e-7
                )
            checks["allowed_motion"] = {
                "passed": passed,
                "target_coordinate": target,
                "reported_coordinate": reported,
                "link_frame_displacement": displacement,
                "orientation_delta_rad": orientation_delta,
                "rotation_axis_alignment_abs_dot": abs(_dot(axis, rotation_axis)),
            }
        perturbed_contacts = _collision_count(p, body_id)
        checks["collision_query"] = {
            "passed": True,
            "nominal_self_contact_count": nominal_contacts,
            "perturbed_self_contact_count": perturbed_contacts,
            "interpretation": "diagnostic_only_for_fixture",
        }

        if fusion_joint_type != "RigidJointType":
            p.resetJointState(body_id, 0, target, 0.0)
            p.setJointMotorControl2(body_id, 0, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
            for _ in range(steps):
                p.stepSimulation()
            stable_coordinate = float(p.getJointState(body_id, 0)[0])
            checks["stability"] = {
                "passed": abs(stable_coordinate - target) < 1e-5,
                "steps": steps,
                "coordinate_drift": abs(stable_coordinate - target),
            }
        else:
            checks["stability"] = {"passed": True, "steps": steps, "coordinate_drift": 0.0}

        p.resetSimulation()
        p.setGravity(0.0, 0.0, 0.0)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        body_id = p.loadURDF(str(urdf_path), useFixedBase=True)
        forbidden_baseline = target if fusion_joint_type != "RigidJointType" else 0.0
        if fusion_joint_type != "RigidJointType":
            p.resetJointState(body_id, 0, forbidden_baseline, 0.0)
            p.setJointMotorControl2(body_id, 0, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
        forbidden_before_position, forbidden_before_orientation = _link_frame(p, body_id)
        for _ in range(steps):
            center_of_mass = p.getLinkState(body_id, 0, computeForwardKinematics=True)[0]
            if fusion_joint_type in {"RigidJointType", "SliderJointType"}:
                p.applyExternalForce(
                    body_id,
                    0,
                    _scale(perpendicular, 20.0),
                    center_of_mass,
                    p.WORLD_FRAME,
                )
            if fusion_joint_type in {"RigidJointType", "RevoluteJointType", "SliderJointType"}:
                p.applyExternalTorque(body_id, 0, _scale(perpendicular, 10.0), p.WORLD_FRAME)
            p.stepSimulation()
        forbidden_after_position, forbidden_after_orientation = _link_frame(p, body_id)
        forbidden_displacement = _sub(forbidden_after_position, forbidden_before_position)
        forbidden_rotation, _ = _quat_delta(p, forbidden_before_orientation, forbidden_after_orientation)
        coordinate = float(p.getJointState(body_id, 0)[0]) if fusion_joint_type != "RigidJointType" else 0.0
        checks["forbidden_motion"] = {
            "passed": _norm(forbidden_displacement) < 1e-4 and forbidden_rotation < 1e-4 and abs(coordinate - forbidden_baseline) < 1e-4,
            "link_frame_displacement_norm": _norm(forbidden_displacement),
            "orientation_delta_rad": forbidden_rotation,
            "joint_coordinate_drift": abs(coordinate - forbidden_baseline),
            "baseline_coordinate": forbidden_baseline,
            "perturbation": "off_axis_force_and_torque",
        }
        status = "passed" if all(check["passed"] for check in checks.values()) else "failed"
        return {
            "schema": "cadengbench_a_a2_direct_fixture_result_v1",
            "scope": scope,
            "status": status,
            "fusion_joint_type": fusion_joint_type,
            "expected_axis": axis,
            "checks": checks,
        }
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "checks": checks}
    finally:
        p.disconnect(client)


def run_source_limit_lock(
    urdf_path: str | Path,
    fusion_joint_type: str,
    expected_axis: list[float],
    locked_coordinate: float,
    steps: int = 240,
) -> dict[str, Any]:
    """Verify that an enabled zero-span source limit behaves as an effective lock."""
    try:
        import pybullet as p  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"pybullet unavailable: {type(exc).__name__}: {exc}"}

    axis = _unit(expected_axis)
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0.0, 0.0, 0.0)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        body_id = p.loadURDF(str(urdf_path), useFixedBase=True)
        p.resetJointState(body_id, 0, locked_coordinate, 0.0)
        p.setJointMotorControl2(body_id, 0, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
        before_position, before_orientation = _link_frame(p, body_id)
        nominal_contacts = _collision_count(p, body_id)
        for _ in range(steps):
            center_of_mass = p.getLinkState(body_id, 0, computeForwardKinematics=True)[0]
            p.applyExternalForce(body_id, 0, _scale(axis, 20.0), center_of_mass, p.WORLD_FRAME)
            p.applyExternalTorque(body_id, 0, _scale(_perpendicular(axis), 10.0), p.WORLD_FRAME)
            p.stepSimulation()
        after_position, after_orientation = _link_frame(p, body_id)
        coordinate = float(p.getJointState(body_id, 0)[0])
        displacement = _norm(_sub(after_position, before_position))
        rotation, _ = _quat_delta(p, before_orientation, after_orientation)
        coordinate_drift = abs(coordinate - locked_coordinate)
        passed = coordinate_drift < 1e-4 and displacement < 1e-4 and rotation < 1e-4
        return {
            "schema": "cadengbench_a_a2_source_limit_lock_result_v1",
            "scope": "real_cad_construction_validation_not_model_evaluation",
            "status": "passed" if passed else "failed",
            "fusion_joint_type": fusion_joint_type,
            "expected_axis": axis,
            "checks": {
                "source_limit_lock": {
                    "passed": passed,
                    "locked_coordinate": locked_coordinate,
                    "coordinate_drift": coordinate_drift,
                    "link_frame_displacement_norm": displacement,
                    "orientation_delta_rad": rotation,
                    "axis_aligned_force_applied": True,
                },
                "collision_query": {
                    "passed": True,
                    "nominal_self_contact_count": nominal_contacts,
                    "interpretation": "diagnostic_only",
                },
            },
        }
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        p.disconnect(client)
