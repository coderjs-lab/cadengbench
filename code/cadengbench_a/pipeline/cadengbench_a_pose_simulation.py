"""Fail-closed PyBullet execution of an authorized A2 joint-frame prediction."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from pipeline.cadengbench_a_a2_ball import run_selected_ball_cad
from pipeline.cadengbench_a_a2_compound import (
    COMPOUND_TYPES,
    run_compound_fixture,
    write_selected_compound_cad_urdf,
)
from pipeline.cadengbench_a_a2_pybullet import (
    DIRECT_JOINT_TYPES,
    run_direct_fixture,
    write_selected_cad_urdf,
)


POSE_JOINT_ID = "a2_kinematic_prediction"


def _unit(values: list[float]) -> list[float]:
    length = math.sqrt(sum(float(value) ** 2 for value in values))
    if length <= 1e-12:
        raise ValueError("pose direction must be non-zero")
    return [float(value) / length for value in values]


def _quaternion_matrix(quaternion: list[float], translation: list[float]) -> list[list[float]]:
    x, y, z, w = _unit(quaternion)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(translation[0])],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(translation[1])],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(translation[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _primary_axis(joint_type: str, directions: dict[str, list[float]]) -> list[float]:
    preferred = {
        "RevoluteJointType": "rotation_axis",
        "SliderJointType": "slide_direction",
        "CylindricalJointType": "rotation_axis",
        "PlanarJointType": "normal_direction",
        "PinSlotJointType": "rotation_axis",
        "BallJointType": "pitch_direction",
    }.get(joint_type)
    return _unit(directions[preferred]) if preferred else [0.0, 0.0, 1.0]


def materialize_pose_master(master: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an authorized body-A-local pose into the PyBullet adapter contract."""
    if payload.get("coordinate_frame") != "body_a_local_cm":
        raise ValueError("pose payload must use body_a_local_cm")
    body_ids = {body["id"] for body in master.get("bodies", [])}
    body_a = payload.get("body_a_id")
    body_b = payload.get("body_b_id")
    if body_a not in body_ids or body_b not in body_ids or body_a == body_b:
        raise ValueError("pose payload body IDs do not identify two available bodies")
    joint_type = str(payload["joint_type"])
    directions = {
        name: _unit(vector)
        for name, vector in payload.get("motion_directions_body_a", {}).items()
    }
    origin = payload.get("joint_origin_body_a_cm") or [0.0, 0.0, 0.0]
    relative = payload["body_b_to_body_a"]
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    child_transform = _quaternion_matrix(
        relative["rotation_quaternion_xyzw"], relative["translation_cm"]
    )
    synthetic = copy.deepcopy(master)
    synthetic["joints"] = [
        {
            "id": POSE_JOINT_ID,
            "name": POSE_JOINT_ID,
            "body_a": body_a,
            "body_b": body_b,
            "fusion_joint_type": joint_type,
            "axis": {"origin": [float(value) for value in origin], "direction": _primary_axis(joint_type, directions)},
            "geometry_or_origin_one": {"transform_matrix": identity},
            "geometry_or_origin_two": {"transform_matrix": child_transform},
            "motion_semantics": {
                "resolved_directions": directions,
                "limit_groups": {},
                "source": "model_predicted_a2_kinematic_payload",
            },
        }
    ]
    return synthetic


def run_authorized_pose_simulation(
    master: dict[str, Any], evaluation: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run real-CAD physics only after the pose evaluator explicitly authorizes execution."""
    if not evaluation.get("kinematic_execution_authorized") or not evaluation.get("simulation_payload"):
        return {
            "schema": "cadengbench_a_a2_kinematic_simulation_result_v1",
            "status": "blocked",
            "reason": "pose_evaluator_did_not_authorize_kinematic_execution",
            "physics_executed": False,
        }

    payload = evaluation["simulation_payload"]
    synthetic = materialize_pose_master(master, payload)
    joint_type = payload["joint_type"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if joint_type in DIRECT_JOINT_TYPES:
        metadata = write_selected_cad_urdf(
            synthetic, POSE_JOINT_ID, output_dir / "predicted_pose.urdf"
        )
        result = run_direct_fixture(
            metadata["path"],
            joint_type,
            metadata["axis"],
            target_coordinate=metadata["target_coordinate"],
            scope="model_predicted_pose_real_cad_validation",
        )
    elif joint_type in COMPOUND_TYPES:
        metadata = write_selected_compound_cad_urdf(
            synthetic, POSE_JOINT_ID, output_dir / "predicted_pose.urdf"
        )
        result = run_compound_fixture(
            metadata["path"],
            joint_type,
            metadata["directions"],
            target_coordinates=metadata["target_coordinates"],
            baseline_coordinates=metadata["baseline_coordinates"],
            limit_statuses=metadata["source_limit_statuses"],
            scope="model_predicted_pose_real_cad_validation",
        )
    elif joint_type == "BallJointType":
        metadata = {"representation": "native_pybullet_spherical_joint"}
        result = run_selected_ball_cad(synthetic, POSE_JOINT_ID)
        result["scope"] = "model_predicted_pose_real_cad_validation"
    else:
        raise ValueError(f"unsupported A2 kinematic joint type: {joint_type}")

    return {
        "schema": "cadengbench_a_a2_kinematic_simulation_result_v1",
        "status": result.get("status"),
        "physics_executed": result.get("status") != "skipped",
        "joint_type": joint_type,
        "body_a_id": payload["body_a_id"],
        "body_b_id": payload["body_b_id"],
        "adapter_metadata": metadata,
        "physics_result": result,
    }
