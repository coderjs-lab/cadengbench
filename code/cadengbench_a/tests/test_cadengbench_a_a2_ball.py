from __future__ import annotations

import pytest

from pipeline.cadengbench_a_a2_ball import _validate_frame, run_selected_ball_cad


def test_ball_frame_rejects_nonorthogonal_directions() -> None:
    with pytest.raises(ValueError, match="must be orthogonal"):
        _validate_frame(
            {
                "pitch_direction": [1.0, 0.0, 0.0],
                "yaw_direction": [1.0, 0.0, 0.0],
                "roll_direction": [0.0, 0.0, 1.0],
            }
        )


def test_enabled_ball_limits_are_explicitly_unsupported() -> None:
    master = {
        "joints": [
            {
                "id": "ball_0",
                "fusion_joint_type": "BallJointType",
                "motion_semantics": {
                    "resolved_directions": {
                        "pitch_direction": [1.0, 0.0, 0.0],
                        "yaw_direction": [0.0, 1.0, 0.0],
                        "roll_direction": [0.0, 0.0, 1.0],
                    },
                    "limit_groups": {
                        "pitch": {
                            "lower": -0.5,
                            "upper": 0.5,
                            "lower_enabled": True,
                            "upper_enabled": True,
                        }
                    },
                },
            }
        ]
    }

    result = run_selected_ball_cad(master, "ball_0")

    assert result["status"] == "unsupported_source_limits"
    assert "pitch" in result["enabled_limit_groups"]
