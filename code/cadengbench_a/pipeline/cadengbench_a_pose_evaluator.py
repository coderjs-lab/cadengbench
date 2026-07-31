"""Conditional A2 joint-frame evaluator and metric implementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from pipeline.cadengbench_a_dev20 import read_json
from pipeline.cadengbench_a_hypothesis_evaluator import evaluate_hypothesis_prediction
from pipeline.cadengbench_a_kinematics import JOINT_KINEMATIC_CONTRACTS


CANARY_THRESHOLDS = {
    "origin_normalized_error_max": 0.01,
    "direction_error_deg_max": 5.0,
    "relative_translation_normalized_error_max": 0.01,
    "relative_rotation_error_deg_max": 5.0,
}


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _unit(values: list[float]) -> list[float]:
    length = _norm(values)
    if length <= 1e-12:
        raise ValueError("zero-length vector")
    return [float(value) / length for value in values]


def _rigid_inverse(matrix: list[list[float]]) -> list[list[float]]:
    rotation = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    translation = [float(matrix[row][3]) for row in range(3)]
    rotation_t = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(rotation_t[row][column] * translation[column] for column in range(3))
        for row in range(3)
    ]
    return [
        [*rotation_t[row], inverse_translation[row]] for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _matmul4(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
    return [
        [sum(float(first[row][k]) * float(second[k][column]) for k in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    return [
        sum(float(matrix[row][column]) * float(point[column]) for column in range(3))
        + float(matrix[row][3])
        for row in range(3)
    ]


def _transform_direction(matrix: list[list[float]], direction: list[float]) -> list[float]:
    return _unit(
        [
            sum(float(matrix[row][column]) * float(direction[column]) for column in range(3))
            for row in range(3)
        ]
    )


def _rotation_to_quaternion_xyzw(rotation: list[list[float]]) -> list[float]:
    r00, r01, r02 = rotation[0]
    r10, r11, r12 = rotation[1]
    r20, r21, r22 = rotation[2]
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x, y, z, w = (r21 - r12) / scale, (r02 - r20) / scale, (r10 - r01) / scale, 0.25 * scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        x, y, z, w = 0.25 * scale, (r01 + r10) / scale, (r02 + r20) / scale, (r21 - r12) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        x, y, z, w = (r01 + r10) / scale, 0.25 * scale, (r12 + r21) / scale, (r02 - r20) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        x, y, z, w = (r02 + r20) / scale, (r12 + r21) / scale, 0.25 * scale, (r10 - r01) / scale
    quaternion = _unit([x, y, z, w])
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    return quaternion


def ground_truth_pose_from_joint(joint: dict[str, Any]) -> dict[str, Any]:
    """Express a recorded source joint in body A's local coordinate frame."""
    transform_a = joint["geometry_or_origin_one"]["transform_matrix"]
    transform_b = joint["geometry_or_origin_two"]["transform_matrix"]
    world_to_a = _rigid_inverse(transform_a)
    body_b_to_body_a = _matmul4(world_to_a, transform_b)
    joint_type = joint["fusion_joint_type"]
    required = JOINT_KINEMATIC_CONTRACTS[joint_type]["required_directions"]
    directions_world = (joint.get("motion_semantics") or {}).get("resolved_directions") or {}
    directions_body_a = {
        name: _transform_direction(world_to_a, directions_world[name]) for name in required
    }
    origin = None
    if joint_type != "RigidJointType":
        origin = _transform_point(world_to_a, joint["axis"]["origin"])
    return {
        "joint_id": joint["id"],
        "joint_type": joint_type,
        "joint_origin_body_a_cm": origin,
        "motion_directions_body_a": directions_body_a,
        "body_b_to_body_a": {
            "translation_cm": [body_b_to_body_a[row][3] for row in range(3)],
            "rotation_quaternion_xyzw": _rotation_to_quaternion_xyzw(
                [row[:3] for row in body_b_to_body_a[:3]]
            ),
        },
    }


def _body_scale(public_geometry: dict[str, Any], role: str) -> float:
    body = next(body for body in public_geometry["bodies"] if body["body_role"] == role)
    extents = body["bounding_box"]["extents"]
    return max(_norm(extents), 1e-9)


def _distance(first: list[float], second: list[float]) -> float:
    return _norm([float(a) - float(b) for a, b in zip(first, second)])


def _direction_error_deg(first: list[float], second: list[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(_unit(first), _unit(second))))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _rotation_error_deg(first: list[float], second: list[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(_unit(first), _unit(second))))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _score_pose(
    predicted: dict[str, Any], truth: dict[str, Any], body_a_scale: float, placement_scale: float
) -> dict[str, Any]:
    if truth["joint_origin_body_a_cm"] is None:
        origin_error = None
        origin_normalized = None
    else:
        origin_error = _distance(
            predicted["joint_origin_body_a_cm"], truth["joint_origin_body_a_cm"]
        )
        origin_normalized = origin_error / body_a_scale
    direction_errors = {
        name: _direction_error_deg(predicted["motion_directions_body_a"][name], direction)
        for name, direction in truth["motion_directions_body_a"].items()
    }
    translation_error = _distance(
        predicted["body_b_to_body_a"]["translation_cm"],
        truth["body_b_to_body_a"]["translation_cm"],
    )
    rotation_error = _rotation_error_deg(
        predicted["body_b_to_body_a"]["rotation_quaternion_xyzw"],
        truth["body_b_to_body_a"]["rotation_quaternion_xyzw"],
    )
    return {
        "truth_joint_id": truth["joint_id"],
        "origin_error_cm": origin_error,
        "origin_normalized_error": origin_normalized,
        "direction_errors_deg": direction_errors,
        "max_direction_error_deg": max(direction_errors.values(), default=0.0),
        "relative_translation_error_cm": translation_error,
        "relative_translation_normalized_error": translation_error / placement_scale,
        "relative_rotation_error_deg": rotation_error,
    }


def _metric_pass(metrics: dict[str, Any], thresholds: dict[str, float]) -> bool:
    origin = metrics["origin_normalized_error"]
    return (
        (origin is None or origin <= thresholds["origin_normalized_error_max"])
        and metrics["max_direction_error_deg"] <= thresholds["direction_error_deg_max"]
        and metrics["relative_translation_normalized_error"]
        <= thresholds["relative_translation_normalized_error_max"]
        and metrics["relative_rotation_error_deg"]
        <= thresholds["relative_rotation_error_deg_max"]
    )


def _validation_errors(
    pose_prediction: dict[str, Any], schema: dict[str, Any], joint_type: str | None
) -> list[str]:
    errors = [error.message for error in Draft202012Validator(schema).iter_errors(pose_prediction)]
    pose = pose_prediction.get("pose")
    abstention = pose_prediction.get("abstention")
    if pose is None and abstention is None:
        errors.append("pose null requires an abstention")
    if pose is not None and abstention is not None:
        errors.append("pose and abstention cannot both be populated")
    if not isinstance(pose, dict) or joint_type not in JOINT_KINEMATIC_CONTRACTS:
        return errors
    required = set(JOINT_KINEMATIC_CONTRACTS[joint_type]["required_directions"])
    supplied = set((pose.get("motion_directions_body_a") or {}).keys())
    if supplied != required:
        errors.append(f"motion direction keys must be exactly {sorted(required)}")
    origin = pose.get("joint_origin_body_a_cm")
    if joint_type == "RigidJointType" and origin is not None:
        errors.append("rigid joint origin must be null")
    if joint_type != "RigidJointType" and origin is None:
        errors.append("non-rigid joint origin is required")
    vectors = list((pose.get("motion_directions_body_a") or {}).values())
    transform = pose.get("body_b_to_body_a") or {}
    quaternion = transform.get("rotation_quaternion_xyzw")
    for vector in vectors:
        if isinstance(vector, list) and len(vector) == 3 and abs(_norm(vector) - 1.0) > 1e-3:
            errors.append("motion directions must be unit length")
    if isinstance(quaternion, list) and len(quaternion) == 4 and abs(_norm(quaternion) - 1.0) > 1e-3:
        errors.append("rotation quaternion must be unit length")
    return errors


def evaluate_a2_kinematic_prediction(
    master: dict[str, Any],
    candidate_registry: dict[str, Any],
    public_geometry: dict[str, Any],
    a1_prediction: dict[str, Any],
    pose_prediction: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score A2 only when the selected A1 relation maps to a known positive."""
    schema = schema or read_json(
        Path("prompts/cadengbench_a/a2_kinematics_v1/output_schema.json")
    )
    thresholds = thresholds or CANARY_THRESHOLDS
    a1 = evaluate_hypothesis_prediction(master, candidate_registry, a1_prediction)
    selected_rank = pose_prediction.get("selected_rank")
    selected = next((row for row in a1["hypotheses"] if row["rank"] == selected_rank), None)
    errors = _validation_errors(
        pose_prediction, schema, selected.get("joint_type") if selected else None
    )
    if pose_prediction.get("assembly_id") != master.get("joint_set_id"):
        errors.append("assembly_id does not match task")
    if not a1["l0_valid"]:
        errors.append("A1 prediction is invalid")
    if selected is None:
        errors.append("selected_rank does not reference a parsed A1 hypothesis")
    typed_ids = selected.get("typed_entity_matched_source_joint_ids", []) if selected else []
    if not typed_ids:
        status = "not_scored_no_typed_known_positive_mapping"
    elif pose_prediction.get("pose") is None:
        status = "abstained"
    elif errors:
        status = "invalid_pose_prediction"
    else:
        status = "scored"

    alternative_metrics: list[dict[str, Any]] = []
    best = None
    passed = False
    if status == "scored":
        joints = {joint["id"]: joint for joint in master.get("joints", [])}
        truths = [ground_truth_pose_from_joint(joints[joint_id]) for joint_id in typed_ids]
        scale_a = _body_scale(public_geometry, "body_a")
        placement_scale = max(scale_a, _body_scale(public_geometry, "body_b"))
        alternative_metrics = [
            _score_pose(pose_prediction["pose"], truth, scale_a, placement_scale)
            for truth in truths
        ]
        best = min(
            alternative_metrics,
            key=lambda row: (
                row["origin_normalized_error"] or 0.0,
                row["max_direction_error_deg"],
                row["relative_translation_normalized_error"],
                row["relative_rotation_error_deg"],
            ),
        )
        passed = _metric_pass(best, thresholds)

    return {
        "schema": "cadengbench_a_a2_kinematic_evaluation_v1",
        "assembly_id": master.get("joint_set_id"),
        "scope": "conditional_pose_track_canary_not_headline_a1_score",
        "status": status,
        "l0_valid": not errors,
        "validation_errors": errors,
        "selected_rank": selected_rank,
        "selected_joint_type": selected.get("joint_type") if selected else None,
        "typed_known_positive_source_joint_ids": typed_ids,
        "pose_alternative_count": len(typed_ids),
        "best_match": best,
        "alternative_metrics": alternative_metrics,
        "threshold_profile": "a2_kinematics_v1",
        "thresholds": thresholds,
        "pose_pass": passed,
        "kinematic_execution_authorized": status == "scored" and passed,
        "simulation_payload": (
            {
                "joint_type": selected["joint_type"],
                "body_a_id": next(
                    joint["body_a"]
                    for joint in master.get("joints", [])
                    if joint["id"] == best["truth_joint_id"]
                ),
                "body_b_id": next(
                    joint["body_b"]
                    for joint in master.get("joints", [])
                    if joint["id"] == best["truth_joint_id"]
                ),
                **pose_prediction["pose"],
                "coordinate_frame": "body_a_local_cm",
            }
            if status == "scored" and passed
            else None
        ),
        "interpretation": (
            "Pose is scored only conditional on recovering a recorded typed A1 known positive. "
            "Best-of-source alternatives handles recorded pose ambiguity without leaking a "
            "variant into the model request. Canary thresholds require calibration before freeze."
        ),
    }
