"""Canonical degrees-of-freedom contracts for CADEngBench-A A2."""

from __future__ import annotations

from typing import Any


JOINT_KINEMATIC_CONTRACTS: dict[str, dict[str, Any]] = {
    "RigidJointType": {
        "dof": 0,
        "allowed_motion": [],
        "required_directions": [],
        "qualification": "direct_single_joint_ready",
        "representation": "fixed_joint",
        "candidate_simulators": ["pybullet_urdf", "mujoco", "sdf"],
        "behavior_checks": [
            "translation_locked_all_axes",
            "rotation_locked_all_axes",
            "constraint_drift_below_tolerance",
        ],
    },
    "RevoluteJointType": {
        "dof": 1,
        "allowed_motion": ["rotation_about_axis"],
        "required_directions": ["rotation_axis"],
        "qualification": "direct_single_joint_ready",
        "representation": "revolute_joint",
        "candidate_simulators": ["pybullet_urdf", "mujoco", "sdf"],
        "behavior_checks": [
            "rotation_about_axis_allowed",
            "translation_along_axis_locked",
            "translation_perpendicular_to_axis_locked",
            "rotation_perpendicular_to_axis_locked",
        ],
    },
    "SliderJointType": {
        "dof": 1,
        "allowed_motion": ["translation_along_axis"],
        "required_directions": ["slide_direction"],
        "qualification": "direct_single_joint_ready",
        "representation": "prismatic_joint",
        "candidate_simulators": ["pybullet_urdf", "mujoco", "sdf"],
        "behavior_checks": [
            "translation_along_axis_allowed",
            "translation_perpendicular_to_axis_locked",
            "rotation_all_axes_locked",
        ],
    },
    "CylindricalJointType": {
        "dof": 2,
        "allowed_motion": ["rotation_about_axis", "translation_along_axis"],
        "required_directions": ["rotation_axis", "slide_direction"],
        "qualification": "compound_joint_ready",
        "representation": "coaxial_revolute_prismatic_compound",
        "candidate_simulators": ["mujoco", "sdf", "compound_urdf"],
        "behavior_checks": [
            "rotation_about_axis_allowed",
            "translation_along_axis_allowed",
            "translation_perpendicular_to_axis_locked",
            "rotation_perpendicular_to_axis_locked",
        ],
    },
    "PlanarJointType": {
        "dof": 3,
        "allowed_motion": [
            "translation_in_plane_primary",
            "translation_in_plane_secondary",
            "rotation_about_plane_normal",
        ],
        "required_directions": [
            "normal_direction",
            "primary_slide_direction",
            "secondary_slide_direction",
        ],
        "qualification": "compound_joint_ready",
        "representation": "two_prismatic_one_revolute_compound",
        "candidate_simulators": ["mujoco", "sdf", "compound_urdf"],
        "behavior_checks": [
            "two_independent_in_plane_translations_allowed",
            "rotation_about_normal_allowed",
            "translation_normal_to_plane_locked",
            "rotation_in_plane_axes_locked",
        ],
    },
    "BallJointType": {
        "dof": 3,
        "allowed_motion": [
            "rotation_about_frame_x",
            "rotation_about_frame_y",
            "rotation_about_frame_z",
        ],
        "required_directions": [
            "pitch_direction",
            "yaw_direction",
            "roll_direction",
        ],
        "qualification": "richer_simulator_ready",
        "representation": "spherical_ball_joint",
        "candidate_simulators": ["mujoco", "sdf"],
        "behavior_checks": [
            "three_independent_rotations_allowed",
            "translation_all_axes_locked",
            "joint_center_drift_below_tolerance",
        ],
    },
    "PinSlotJointType": {
        "dof": 2,
        "allowed_motion": ["rotation_about_pin_axis", "translation_along_slot"],
        "required_directions": ["rotation_axis", "slide_direction"],
        "qualification": "compound_joint_ready",
        "representation": "prismatic_revolute_pin_slot_compound",
        "candidate_simulators": ["mujoco", "sdf", "compound_urdf"],
        "behavior_checks": [
            "rotation_about_pin_axis_allowed",
            "translation_along_slot_allowed",
            "translation_perpendicular_to_slot_locked",
            "off_axis_rotation_locked",
        ],
    },
}
