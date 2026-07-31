from __future__ import annotations

import math

import pytest

from pipeline.cadengbench_a_pose_simulation import (
    POSE_JOINT_ID,
    materialize_pose_master,
    run_authorized_pose_simulation,
)


def _master() -> dict:
    return {
        "joint_set_id": "joint_set_test",
        "bodies": [{"id": "a"}, {"id": "b"}],
        "joints": [{"id": "source"}],
    }


def _payload() -> dict:
    angle = math.pi / 2
    return {
        "joint_type": "RevoluteJointType",
        "body_a_id": "a",
        "body_b_id": "b",
        "joint_origin_body_a_cm": [1.0, 2.0, 3.0],
        "motion_directions_body_a": {"rotation_axis": [0.0, 0.0, 1.0]},
        "body_b_to_body_a": {
            "translation_cm": [4.0, 5.0, 6.0],
            "rotation_quaternion_xyzw": [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)],
        },
        "coordinate_frame": "body_a_local_cm",
    }


def test_materializes_model_pose_as_adapter_joint() -> None:
    synthetic = materialize_pose_master(_master(), _payload())
    joint = synthetic["joints"][0]
    assert joint["id"] == POSE_JOINT_ID
    assert joint["body_a"] == "a"
    assert joint["axis"]["origin"] == [1.0, 2.0, 3.0]
    assert joint["motion_semantics"]["source"] == "model_predicted_a2_kinematic_payload"
    transform = joint["geometry_or_origin_two"]["transform_matrix"]
    assert [transform[row][3] for row in range(3)] == [4.0, 5.0, 6.0]
    assert transform[0][0] == pytest.approx(0.0, abs=1e-12)
    assert transform[1][0] == pytest.approx(1.0)


def test_rejects_unknown_body_id() -> None:
    payload = _payload()
    payload["body_b_id"] = "missing"
    with pytest.raises(ValueError, match="body IDs"):
        materialize_pose_master(_master(), payload)


def test_blocks_unauthorized_pose_before_physics(tmp_path) -> None:
    result = run_authorized_pose_simulation(
        _master(),
        {"kinematic_execution_authorized": False, "simulation_payload": None},
        tmp_path,
    )
    assert result["status"] == "blocked"
    assert result["physics_executed"] is False
    assert not list(tmp_path.iterdir())
