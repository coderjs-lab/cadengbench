from __future__ import annotations

from copy import deepcopy

from pipeline.cadengbench_a_pose_evaluator import (
    evaluate_a2_kinematic_prediction,
    ground_truth_pose_from_joint,
)


IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _joint(joint_id: str, *, origin_world: list[float] | None = None) -> dict:
    origin_world = origin_world or [0.0, 0.0, 0.0]
    return {
        "id": joint_id,
        "source_index": int(joint_id.rsplit("_", 1)[-1]),
        "name": joint_id,
        "source_type": "Joint",
        "body_a": "part_a",
        "body_b": "part_b",
        "fusion_joint_type": "RevoluteJointType",
        "geometry_or_origin_one": {
            "entity_one": {"body": "part_a", "type": "BRepFace", "index": 0},
            "transform_matrix": IDENTITY,
            "axis_line": {"origin": origin_world, "direction": [0.0, 0.0, 1.0]},
        },
        "geometry_or_origin_two": {
            "entity_one": {"body": "part_b", "type": "BRepFace", "index": 0},
            "transform_matrix": [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "axis_line": {"origin": origin_world, "direction": [0.0, 0.0, 1.0]},
        },
        "axis": {"origin": origin_world, "direction": [0.0, 0.0, 1.0]},
        "motion_semantics": {
            "resolved_directions": {"rotation_axis": [0.0, 0.0, 1.0]},
            "unresolved_direction_labels": {},
            "limit_groups": {},
        },
    }


def _registry() -> dict:
    return {
        "bodies": [
            {
                "candidates": [
                    {
                        "candidate_id": "body_a_face_0000",
                        "body_role": "body_a",
                        "body_id": "part_a",
                        "entity_type": "BRepFace",
                        "entity_index": 0,
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "candidate_id": "body_b_face_0000",
                        "body_role": "body_b",
                        "body_id": "part_b",
                        "entity_type": "BRepFace",
                        "entity_index": 0,
                    }
                ]
            },
        ]
    }


def _public_geometry() -> dict:
    body = lambda role: {
        "body_role": role,
        "bounding_box": {"extents": [10.0, 10.0, 10.0]},
    }
    return {"bodies": [body("body_a"), body("body_b")]}


def _a1(joint_type: str = "RevoluteJointType") -> dict:
    return {
        "assembly_id": "joint_set_00001",
        "ranked_hypotheses": [
            {
                "rank": 1,
                "joint_type": joint_type,
                "body_a_candidate_id": "body_a_face_0000",
                "body_b_candidate_id": "body_b_face_0000",
                "confidence": 0.9,
            }
        ],
        "abstention": None,
    }


def _pose(joint: dict) -> dict:
    truth = ground_truth_pose_from_joint(joint)
    return {
        "assembly_id": "joint_set_00001",
        "selected_rank": 1,
        "pose": {
            "joint_origin_body_a_cm": truth["joint_origin_body_a_cm"],
            "motion_directions_body_a": truth["motion_directions_body_a"],
            "body_b_to_body_a": truth["body_b_to_body_a"],
            "confidence": 1.0,
        },
        "abstention": None,
    }


def _evaluate(master: dict, prediction: dict) -> dict:
    return evaluate_a2_kinematic_prediction(
        master, _registry(), _public_geometry(), _a1(), prediction
    )


def test_exact_pose_authorizes_kinematic_execution() -> None:
    joint = _joint("joint_set_00001_joint_0")
    result = _evaluate({"joint_set_id": "joint_set_00001", "joints": [joint]}, _pose(joint))

    assert result["l0_valid"] is True
    assert result["pose_pass"] is True
    assert result["kinematic_execution_authorized"] is True
    assert result["best_match"]["relative_rotation_error_deg"] == 0.0


def test_axis_sign_is_equivalent() -> None:
    joint = _joint("joint_set_00001_joint_0")
    prediction = _pose(joint)
    prediction["pose"]["motion_directions_body_a"]["rotation_axis"] = [0.0, 0.0, -1.0]

    result = _evaluate({"joint_set_id": "joint_set_00001", "joints": [joint]}, prediction)

    assert result["pose_pass"] is True
    assert result["best_match"]["max_direction_error_deg"] == 0.0


def test_wrong_direction_fails_pose_threshold() -> None:
    joint = _joint("joint_set_00001_joint_0")
    prediction = _pose(joint)
    prediction["pose"]["motion_directions_body_a"]["rotation_axis"] = [1.0, 0.0, 0.0]

    result = _evaluate({"joint_set_id": "joint_set_00001", "joints": [joint]}, prediction)

    assert result["pose_pass"] is False
    assert result["best_match"]["max_direction_error_deg"] == 90.0


def test_wrong_relative_translation_fails_pose_threshold() -> None:
    joint = _joint("joint_set_00001_joint_0")
    prediction = _pose(joint)
    prediction["pose"]["body_b_to_body_a"]["translation_cm"] = [5.0, 0.0, 0.0]

    result = _evaluate({"joint_set_id": "joint_set_00001", "joints": [joint]}, prediction)

    assert result["pose_pass"] is False
    assert result["best_match"]["relative_translation_error_cm"] == 3.0


def test_quaternion_sign_is_equivalent() -> None:
    joint = _joint("joint_set_00001_joint_0")
    prediction = _pose(joint)
    quaternion = prediction["pose"]["body_b_to_body_a"]["rotation_quaternion_xyzw"]
    prediction["pose"]["body_b_to_body_a"]["rotation_quaternion_xyzw"] = [-v for v in quaternion]

    result = _evaluate({"joint_set_id": "joint_set_00001", "joints": [joint]}, prediction)

    assert result["pose_pass"] is True


def test_pose_selects_best_recorded_alternative_without_hidden_variant_hint() -> None:
    first = _joint("joint_set_00001_joint_0")
    second = _joint("joint_set_00001_joint_1", origin_world=[3.0, 0.0, 0.0])
    master = {"joint_set_id": "joint_set_00001", "joints": [first, second]}

    result = _evaluate(master, _pose(second))

    assert result["pose_alternative_count"] == 2
    assert result["best_match"]["truth_joint_id"] == second["id"]
    assert result["pose_pass"] is True


def test_missing_family_direction_is_l0_invalid() -> None:
    joint = _joint("joint_set_00001_joint_0")
    prediction = _pose(joint)
    prediction["pose"]["motion_directions_body_a"] = {}

    result = _evaluate({"joint_set_id": "joint_set_00001", "joints": [joint]}, prediction)

    assert result["l0_valid"] is False
    assert result["status"] == "invalid_pose_prediction"


def test_pose_abstention_is_valid_and_not_executed() -> None:
    joint = _joint("joint_set_00001_joint_0")
    prediction = {
        "assembly_id": "joint_set_00001",
        "selected_rank": 1,
        "pose": None,
        "abstention": {"reason_code": "ambiguous_pose"},
    }

    result = _evaluate({"joint_set_id": "joint_set_00001", "joints": [joint]}, prediction)

    assert result["l0_valid"] is True
    assert result["status"] == "abstained"
    assert result["kinematic_execution_authorized"] is False
