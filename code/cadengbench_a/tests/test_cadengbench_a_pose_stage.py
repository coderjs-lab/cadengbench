from __future__ import annotations

import json
from pathlib import Path

from pipeline.cadengbench_a_pose_stage import (
    aggregate_pose_stage_results,
    build_pose_stage_package,
)


def test_build_pose_stage_package_is_ground_truth_free(tmp_path: Path) -> None:
    item = tmp_path / "item"
    prompts = tmp_path / "prompts"
    (item / "views").mkdir(parents=True)
    prompts.mkdir()
    (item / "views/a.png").write_bytes(b"a")
    (item / "views/b.png").write_bytes(b"b")
    (item / "task.json").write_text(
        json.dumps(
            {
                "assembly_id": "joint_set_00001",
                "assets": {"views": {"body_a": ["views/a.png"], "body_b": ["views/b.png"]}},
            }
        )
    )
    (item / "public_geometry.json").write_text("{}")
    (item / "candidates.json").write_text("{}")
    (prompts / "system_prompt.txt").write_text("system")
    (prompts / "user_template.txt").write_text(
        "{{assembly_id}} {{selected_rank}} {{joint_type}} {{body_a_candidate_id}} {{body_b_candidate_id}}"
    )
    (prompts / "output_schema.json").write_text("{}")
    a1 = {
        "ranked_hypotheses": [
            {
                "rank": 1,
                "joint_type": "RevoluteJointType",
                "body_a_candidate_id": "body_a_edge_0001",
                "body_b_candidate_id": "body_b_edge_0002",
            }
        ]
    }
    package = build_pose_stage_package(item, prompts, a1)
    assert package["ground_truth_accessed"] is False
    assert package["selected_rank"] == 1
    assert package["selected_hypothesis"]["joint_type"] == "RevoluteJointType"
    assert "ground_truth" not in package["system_prompt"].lower()
    assert "ground_truth" not in package["user_prompt"].lower()
    assert set(package["evidence"]) == {
        "public_geometry_sha256",
        "candidate_registry_sha256",
    }
    assert len(package["images"]) == 2


def test_aggregate_keeps_pose_separate_from_headline() -> None:
    records = [
        {
            "pose_evaluation": {
                "status": "scored",
                "typed_known_positive_source_joint_ids": ["j0"],
                "pose_pass": True,
                "kinematic_execution_authorized": True,
            },
            "simulation": {"physics_executed": True, "status": "passed"},
        },
        {
            "pose_evaluation": {
                "status": "not_scored_no_typed_known_positive_mapping",
                "typed_known_positive_source_joint_ids": [],
                "pose_pass": False,
                "kinematic_execution_authorized": False,
            },
            "simulation": {"physics_executed": False, "status": "blocked"},
        },
    ]
    report = aggregate_pose_stage_results(records)
    assert report["typed_a1_eligible_count"] == 1
    assert report["conditional_pose_pass_rate"] == 1.0
    assert report["end_to_end_authorization_rate"] == 0.5
    assert report["headline_a1_score_modified"] is False
