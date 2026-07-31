from __future__ import annotations

from pipeline.cadengbench_a_hypothesis_evaluator import (
    evaluate_hypothesis_prediction,
    source_variant_groups,
)


def _candidate(candidate_id: str, role: str, body: str, entity_type: str, index: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "body_role": role,
        "body_id": body,
        "entity_type": entity_type,
        "entity_index": index,
    }


def _fixture() -> tuple[dict, dict]:
    geometry_a = {
        "entity_one": {"body": "part_a", "type": "BRepFace", "index": 0},
        "entity_one_equivalents": [{"type": "BRepEdge", "index": 0}],
    }
    geometry_b = {
        "entity_one": {"body": "part_b", "type": "BRepFace", "index": 0},
        "entity_one_equivalents": [{"type": "BRepEdge", "index": 0}],
    }
    joint = {
        "id": "joint_set_00001_joint_0",
        "source_index": 0,
        "name": "Rev1",
        "source_type": "Joint",
        "body_a": "part_a",
        "body_b": "part_b",
        "fusion_joint_type": "RevoluteJointType",
        "geometry_or_origin_one": geometry_a,
        "geometry_or_origin_two": geometry_b,
        "axis": {"origin": [0, 0, 0], "direction": [0, 0, 1]},
    }
    duplicate = {
        **joint,
        "id": "joint_set_00001_joint_1",
        "name": "Rev2",
        "source_index": 1,
    }
    master = {"joint_set_id": "joint_set_00001", "joints": [joint, duplicate]}
    registry = {
        "bodies": [
            {
                "candidates": [
                    _candidate("body_a_face_0000", "body_a", "part_a", "BRepFace", 0),
                    _candidate("body_a_edge_0000", "body_a", "part_a", "BRepEdge", 0),
                ]
            },
            {
                "candidates": [
                    _candidate("body_b_face_0000", "body_b", "part_b", "BRepFace", 0),
                    _candidate("body_b_edge_0000", "body_b", "part_b", "BRepEdge", 0),
                ]
            },
        ]
    }
    return master, registry


def test_source_variant_groups_collapse_exact_duplicate_records() -> None:
    master, _ = _fixture()

    groups = source_variant_groups(master)

    assert len(groups) == 1
    assert groups[0]["source_record_count"] == 2


def test_known_positive_equivalent_entity_and_type_is_a_hit() -> None:
    master, registry = _fixture()
    prediction = {
        "assembly_id": "joint_set_00001",
        "ranked_hypotheses": [
            {
                "rank": 1,
                "joint_type": "RevoluteJointType",
                "body_a_candidate_id": "body_a_edge_0000",
                "body_b_candidate_id": "body_b_edge_0000",
                "confidence": 0.8,
            }
        ],
        "abstention": None,
    }

    result = evaluate_hypothesis_prediction(master, registry, prediction)

    assert result["l0_valid"] is True
    assert result["source_record_count"] == 2
    assert result["exact_unique_source_variant_count"] == 1
    assert result["top1_entity_hit"] is True
    assert result["top1_typed_entity_hit"] is True


def test_known_entity_with_unseen_type_is_not_a_typed_hit() -> None:
    master, registry = _fixture()
    prediction = {
        "assembly_id": "joint_set_00001",
        "ranked_hypotheses": [
            {
                "rank": 1,
                "joint_type": "SliderJointType",
                "body_a_candidate_id": "body_a_face_0000",
                "body_b_candidate_id": "body_b_face_0000",
                "confidence": 0.7,
            }
        ],
        "abstention": None,
    }

    result = evaluate_hypothesis_prediction(master, registry, prediction)

    assert result["top1_entity_hit"] is True
    assert result["top1_type_hit"] is False
    assert result["top1_typed_entity_hit"] is False


def test_rank_order_and_duplicate_hypotheses_are_l0_errors() -> None:
    master, registry = _fixture()
    hypothesis = {
        "rank": 2,
        "joint_type": "RevoluteJointType",
        "body_a_candidate_id": "body_a_face_0000",
        "body_b_candidate_id": "body_b_face_0000",
        "confidence": 0.7,
    }
    prediction = {
        "assembly_id": "joint_set_00001",
        "ranked_hypotheses": [hypothesis, hypothesis],
        "abstention": None,
    }

    result = evaluate_hypothesis_prediction(master, registry, prediction)

    assert result["l0_valid"] is False
    assert any("rank must be 1" in error for error in result["validation_errors"])
    assert any("duplicates" in error for error in result["validation_errors"])


def test_confidence_must_be_finite_and_in_range() -> None:
    master, registry = _fixture()
    prediction = {
        "assembly_id": "joint_set_00001",
        "ranked_hypotheses": [
            {
                "rank": 1,
                "joint_type": "RevoluteJointType",
                "body_a_candidate_id": "body_a_face_0000",
                "body_b_candidate_id": "body_b_face_0000",
                "confidence": float("nan"),
            }
        ],
        "abstention": None,
    }

    result = evaluate_hypothesis_prediction(master, registry, prediction)

    assert result["l0_valid"] is False
    assert any("confidence must be finite" in error for error in result["validation_errors"])
