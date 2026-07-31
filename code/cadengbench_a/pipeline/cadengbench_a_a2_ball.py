"""PyBullet spherical-joint fixtures and real-CAD checks for A2."""

from __future__ import annotations

import math
from typing import Any

from pipeline.cadengbench_a_a2_pybullet import _rpy_from_matrix, _transform_point


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
        raise ValueError("ball frame direction must be non-zero")
    return [float(value) / norm for value in vector]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _scale(vector: list[float], scalar: float) -> list[float]:
    return [value * scalar for value in vector]


def _validate_frame(frame: dict[str, list[float]]) -> dict[str, list[float]]:
    result = {name: _unit(frame[name]) for name in ("pitch_direction", "yaw_direction", "roll_direction")}
    pairs = (("pitch_direction", "yaw_direction"), ("pitch_direction", "roll_direction"), ("yaw_direction", "roll_direction"))
    if any(abs(_dot(result[first], result[second])) > 1e-6 for first, second in pairs):
        raise ValueError("ball frame directions must be orthogonal")
    return result


def _axis_angle_quaternion(axis: list[float], angle: float) -> list[float]:
    axis = _unit(axis)
    sine = math.sin(angle / 2.0)
    return [axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(angle / 2.0)]


def _quat_delta(p: Any, before: list[float], after: list[float]) -> tuple[float, list[float]]:
    inverse_before = p.invertTransform([0.0, 0.0, 0.0], before)[1]
    delta = p.multiplyTransforms([0.0, 0.0, 0.0], after, [0.0, 0.0, 0.0], inverse_before)[1]
    x, y, z, w = (float(value) for value in delta)
    vector_norm = math.sqrt(x * x + y * y + z * z)
    angle = 2.0 * math.atan2(vector_norm, abs(w))
    axis = [x / vector_norm, y / vector_norm, z / vector_norm] if vector_norm > 1e-12 else [0.0, 0.0, 1.0]
    return angle, axis


def _link_frame(p: Any, body_id: int) -> tuple[list[float], list[float]]:
    state = p.getLinkState(body_id, 0, computeForwardKinematics=True)
    return list(state[4]), list(state[5])


def _disable_motor(p: Any, body_id: int, quaternion: list[float]) -> None:
    p.setJointMotorControlMultiDof(
        body_id,
        0,
        p.POSITION_CONTROL,
        targetPosition=quaternion,
        positionGain=0.0,
        velocityGain=0.0,
        force=[0.0, 0.0, 0.0],
    )


def _evaluate_ball_body(
    p: Any,
    body_id: int,
    frame: dict[str, list[float]],
    steps: int,
    scope: str,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    info = p.getJointInfo(body_id, 0)
    checks["representation"] = {
        "passed": p.getNumJoints(body_id) == 1 and int(info[2]) == p.JOINT_SPHERICAL,
        "actual_joint_type_code": int(info[2]),
        "expected_joint_type_code": p.JOINT_SPHERICAL,
    }
    allowed = []
    identity = [0.0, 0.0, 0.0, 1.0]
    for name, axis in frame.items():
        p.resetJointStateMultiDof(body_id, 0, identity, [0.0, 0.0, 0.0])
        before_position, before_orientation = _link_frame(p, body_id)
        target = _axis_angle_quaternion(axis, 0.25)
        p.resetJointStateMultiDof(body_id, 0, target, [0.0, 0.0, 0.0])
        after_position, after_orientation = _link_frame(p, body_id)
        angle, measured_axis = _quat_delta(p, before_orientation, after_orientation)
        passed = (
            _norm(_sub(after_position, before_position)) < 1e-6
            and abs(angle - 0.25) < 1e-6
            and abs(_dot(axis, measured_axis)) > 0.999999
        )
        allowed.append(
            {
                "direction": name,
                "passed": passed,
                "orientation_delta_rad": angle,
                "axis_alignment_abs_dot": abs(_dot(axis, measured_axis)),
            }
        )
    checks["allowed_motion"] = {"passed": all(item["passed"] for item in allowed), "rotations": allowed}

    stable_target = _axis_angle_quaternion(frame["roll_direction"], 0.2)
    p.resetJointStateMultiDof(body_id, 0, stable_target, [0.0, 0.0, 0.0])
    _disable_motor(p, body_id, stable_target)
    before_position, before_orientation = _link_frame(p, body_id)
    for _ in range(steps):
        p.stepSimulation()
    after_position, after_orientation = _link_frame(p, body_id)
    stability_rotation, _ = _quat_delta(p, before_orientation, after_orientation)
    checks["stability"] = {
        "passed": _norm(_sub(after_position, before_position)) < 1e-5 and stability_rotation < 1e-5,
        "translation_drift": _norm(_sub(after_position, before_position)),
        "orientation_drift_rad": stability_rotation,
    }

    p.resetJointStateMultiDof(body_id, 0, identity, [0.0, 0.0, 0.0])
    _disable_motor(p, body_id, identity)
    before_position, _ = _link_frame(p, body_id)
    dynamics = p.getDynamicsInfo(body_id, 0)
    force_magnitude = max(float(dynamics[0]), 1e-8)
    for _ in range(steps):
        joint_center = p.getLinkState(body_id, 0, computeForwardKinematics=True)[4]
        p.applyExternalForce(body_id, 0, _scale(frame["pitch_direction"], force_magnitude), joint_center, p.WORLD_FRAME)
        p.stepSimulation()
    after_position, _ = _link_frame(p, body_id)
    translation_leakage = _norm(_sub(after_position, before_position))
    reference_motion = 0.5 * (steps / 240.0) ** 2
    checks["forbidden_translation"] = {
        "passed": translation_leakage / reference_motion < 1e-3,
        "translation_leakage": translation_leakage,
        "maximum_leakage_ratio": translation_leakage / reference_motion,
        "maximum_allowed_leakage_ratio": 1e-3,
    }
    contacts = len(
        p.getClosestPoints(bodyA=body_id, bodyB=body_id, distance=0.0, linkIndexA=-1, linkIndexB=0)
    )
    checks["collision_query"] = {
        "passed": True,
        "nominal_self_contact_count": contacts,
        "interpretation": "diagnostic_only",
    }
    status = "passed" if all(check["passed"] for check in checks.values()) else "failed"
    return {
        "schema": "cadengbench_a_a2_ball_result_v1",
        "scope": scope,
        "status": status,
        "frame": frame,
        "checks": checks,
    }


def run_ball_fixture(frame: dict[str, list[float]], steps: int = 240) -> dict[str, Any]:
    """Exercise allowed rotations and forbidden translations on a ball fixture."""
    try:
        import pybullet as p  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"pybullet unavailable: {type(exc).__name__}: {exc}"}

    frame = _validate_frame(frame)
    client = p.connect(p.DIRECT)
    checks: dict[str, Any] = {}
    try:
        p.setGravity(0.0, 0.0, 0.0)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)
        parent_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        child_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], collisionFramePosition=[0.75, 0.0, 0.0])
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=parent_collision,
            linkMasses=[1.0],
            linkCollisionShapeIndices=[child_collision],
            linkVisualShapeIndices=[-1],
            linkPositions=[[0.0, 0.0, 0.0]],
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkInertialFramePositions=[[0.0, 0.0, 0.0]],
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_SPHERICAL],
            linkJointAxis=[[0.0, 0.0, 0.0]],
        )
        return _evaluate_ball_body(p, body_id, frame, steps, "native_adapter_fixture_not_model_evaluation")
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "checks": checks}
    finally:
        p.disconnect(client)


def run_selected_ball_cad(
    master: dict[str, Any],
    joint_id: str,
    mesh_scale: float = 0.01,
    steps: int = 240,
) -> dict[str, Any]:
    """Run the A2 spherical-motion contract on selected source CAD bodies."""
    joint = next((item for item in master.get("joints", []) if item.get("id") == joint_id), None)
    if joint is None or joint.get("fusion_joint_type") != "BallJointType":
        return {"status": "failed", "reason": f"ball joint not found: {joint_id}"}
    motion = joint.get("motion_semantics") or {}
    frame = _validate_frame(motion.get("resolved_directions") or {})
    enabled_limits = {
        name: group
        for name, group in (motion.get("limit_groups") or {}).items()
        if group.get("lower_enabled") or group.get("upper_enabled")
    }
    if enabled_limits:
        return {
            "status": "unsupported_source_limits",
            "reason": "PyBullet JOINT_SPHERICAL has no independent pitch/yaw/roll limit enforcement",
            "enabled_limit_groups": enabled_limits,
        }
    try:
        import pybullet as p  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"pybullet unavailable: {type(exc).__name__}: {exc}"}

    bodies = {body["id"]: body for body in master.get("bodies", [])}
    parent_body = bodies[joint["body_a"]]
    child_body = bodies[joint["body_b"]]
    parent_transform = joint["geometry_or_origin_one"]["transform_matrix"]
    child_transform = joint["geometry_or_origin_two"]["transform_matrix"]
    joint_origin = _scale([float(value) for value in joint["axis"]["origin"]], mesh_scale)
    parent_origin = _scale([float(parent_transform[row][3]) for row in range(3)], mesh_scale)
    child_world_origin = _scale([float(child_transform[row][3]) for row in range(3)], mesh_scale)
    child_mesh_origin = _sub(child_world_origin, joint_origin)
    center_of_mass = child_body.get("physical_properties", {}).get("center_of_mass") or {}
    local_center = [float(center_of_mass.get(name, 0.0)) for name in ("x", "y", "z")]
    child_com_world = _scale(_transform_point(child_transform, local_center), mesh_scale)
    child_com_joint = _sub(child_com_world, joint_origin)
    parent_rpy = _rpy_from_matrix(parent_transform)
    child_rpy = _rpy_from_matrix(child_transform)
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0.0, 0.0, 0.0)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)
        scale = [mesh_scale, mesh_scale, mesh_scale]
        parent_collision = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=parent_body["source_files"]["obj"],
            meshScale=scale,
            collisionFramePosition=parent_origin,
            collisionFrameOrientation=p.getQuaternionFromEuler(parent_rpy),
        )
        child_collision = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=child_body["source_files"]["obj"],
            meshScale=scale,
            collisionFramePosition=child_mesh_origin,
            collisionFrameOrientation=p.getQuaternionFromEuler(child_rpy),
        )
        mass = max(float(child_body.get("physical_properties", {}).get("mass") or 1.0), 1e-6)
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=parent_collision,
            linkMasses=[mass],
            linkCollisionShapeIndices=[child_collision],
            linkVisualShapeIndices=[-1],
            linkPositions=[joint_origin],
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkInertialFramePositions=[child_com_joint],
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_SPHERICAL],
            linkJointAxis=[[0.0, 0.0, 0.0]],
        )
        result = _evaluate_ball_body(
            p,
            body_id,
            frame,
            steps,
            "real_cad_native_ball_construction_validation_not_model_evaluation",
        )
        result["joint_id"] = joint_id
        result["placement_policy"] = "fusion_occurrence_transforms_with_native_spherical_joint_at_body_a_axis_origin"
        return result
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        p.disconnect(client)
