"""Known-positive A1 evaluator for ranked joint and B-Rep hypotheses."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

from pipeline.cadengbench_a_dev20 import JOINT_TYPES, audit_candidate_coverage


def source_variant_key(joint: dict[str, Any]) -> str:
    """Return a stable key that collapses exact duplicate source relations."""
    value = copy.deepcopy(joint)
    for key in (
        "id",
        "name",
        "source_index",
        "joint_equivalence_signature",
        "urdf_joint_type",
        "urdf_mapping_note",
    ):
        value.pop(key, None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# Kept for compatibility with earlier internal callers.
_source_variant_key = source_variant_key


def source_variant_groups(master: dict[str, Any]) -> list[dict[str, Any]]:
    """Group source joints into exact known-positive alternatives."""
    groups: dict[str, dict[str, Any]] = {}
    for joint in master.get("joints", []):
        key = source_variant_key(joint)
        if key not in groups:
            groups[key] = {
                "joint_type": joint.get("fusion_joint_type"),
                "source_joint_ids": [],
                "source_names": [],
                "source_record_count": 0,
            }
        group = groups[key]
        group["source_joint_ids"].append(joint.get("id"))
        group["source_names"].append(joint.get("name"))
        group["source_record_count"] += 1
    return list(groups.values())


def _known_positive_variants(
    master: dict[str, Any], candidate_registry: dict[str, Any]
) -> list[dict[str, Any]]:
    coverage = audit_candidate_coverage(master, candidate_registry)
    sides_by_joint: dict[str, dict[str, dict[str, Any]]] = {}
    for side in coverage["sides"]:
        sides_by_joint.setdefault(side["joint_id"], {})[side["side"]] = side
    variants = []
    for joint in master.get("joints", []):
        sides = sides_by_joint.get(joint["id"], {})
        if "body_a" not in sides or "body_b" not in sides:
            continue
        accepted = {}
        for role in ("body_a", "body_b"):
            side = sides[role]
            ids = set(side["equivalent_candidate_ids"])
            if side["primary_candidate_id"]:
                ids.add(side["primary_candidate_id"])
            accepted[role] = sorted(ids)
        variants.append(
            {
                "source_joint_id": joint["id"],
                "joint_type": joint["fusion_joint_type"],
                "body_a_candidate_ids": accepted["body_a"],
                "body_b_candidate_ids": accepted["body_b"],
                "source_variant_key": source_variant_key(joint),
            }
        )
    return variants


def evaluate_hypothesis_prediction(
    master: dict[str, Any],
    candidate_registry: dict[str, Any],
    prediction: dict[str, Any] | Any,
) -> dict[str, Any]:
    """Score one A1 prediction against role-aware known-positive alternatives."""
    errors = []
    if not isinstance(prediction, dict):
        prediction = {}
        errors.append("prediction root must be an object")
    assembly_id = master.get("joint_set_id")
    if prediction.get("assembly_id") != assembly_id:
        errors.append("assembly_id does not match task")
    raw_hypotheses = prediction.get("ranked_hypotheses")
    if not isinstance(raw_hypotheses, list):
        errors.append("ranked_hypotheses must be a list")
        raw_hypotheses = []
    if len(raw_hypotheses) > 3:
        errors.append("ranked_hypotheses must contain at most 3 entries")

    candidate_by_id = {
        candidate["candidate_id"]: candidate
        for body in candidate_registry.get("bodies", [])
        for candidate in body.get("candidates", [])
    }
    parsed = []
    seen = set()
    candidate_id_reference_count = 0
    hallucinated_candidate_id_count = 0
    for index, raw in enumerate(raw_hypotheses):
        if not isinstance(raw, dict):
            errors.append(f"ranked_hypotheses[{index}] must be an object")
            continue
        expected_rank = index + 1
        if raw.get("rank") != expected_rank:
            errors.append(f"ranked_hypotheses[{index}] rank must be {expected_rank}")
        joint_type = raw.get("joint_type")
        if joint_type not in JOINT_TYPES:
            errors.append(f"ranked_hypotheses[{index}] has unknown joint_type")
        confidence = raw.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            errors.append(
                f"ranked_hypotheses[{index}] confidence must be finite and between 0 and 1"
            )
        candidate_id_reference_count += 2
        candidate_a = candidate_by_id.get(raw.get("body_a_candidate_id"))
        candidate_b = candidate_by_id.get(raw.get("body_b_candidate_id"))
        if not candidate_a or candidate_a.get("body_role") != "body_a":
            errors.append(f"ranked_hypotheses[{index}] body_a candidate is invalid")
            hallucinated_candidate_id_count += 1
        if not candidate_b or candidate_b.get("body_role") != "body_b":
            errors.append(f"ranked_hypotheses[{index}] body_b candidate is invalid")
            hallucinated_candidate_id_count += 1
        signature = (
            joint_type,
            raw.get("body_a_candidate_id"),
            raw.get("body_b_candidate_id"),
        )
        if signature in seen:
            errors.append(f"ranked_hypotheses[{index}] duplicates an earlier hypothesis")
        seen.add(signature)
        if joint_type in JOINT_TYPES and candidate_a and candidate_b:
            parsed.append(
                {
                    "rank": expected_rank,
                    "joint_type": joint_type,
                    "body_a_candidate_id": candidate_a["candidate_id"],
                    "body_b_candidate_id": candidate_b["candidate_id"],
                    "confidence": confidence,
                }
            )

    abstention = prediction.get("abstention")
    if raw_hypotheses and abstention is not None:
        errors.append("abstention must be null when hypotheses are present")
    if not raw_hypotheses and abstention is None:
        errors.append("empty hypotheses require an abstention reason")

    variants = _known_positive_variants(master, candidate_registry)
    known_types = sorted({variant["joint_type"] for variant in variants})
    hypothesis_results = []
    for hypothesis in parsed:
        entity_matches = [
            variant
            for variant in variants
            if hypothesis["body_a_candidate_id"] in variant["body_a_candidate_ids"]
            and hypothesis["body_b_candidate_id"] in variant["body_b_candidate_ids"]
        ]
        typed_entity_matches = [
            variant
            for variant in entity_matches
            if hypothesis["joint_type"] == variant["joint_type"]
        ]
        hypothesis_results.append(
            {
                **hypothesis,
                "known_joint_type": hypothesis["joint_type"] in known_types,
                "known_positive_entity_pair": bool(entity_matches),
                "known_positive_typed_entity_pair": bool(typed_entity_matches),
                "entity_matched_source_joint_ids": [
                    variant["source_joint_id"] for variant in entity_matches
                ],
                "typed_entity_matched_source_joint_ids": [
                    variant["source_joint_id"] for variant in typed_entity_matches
                ],
            }
        )

    def hit_at(k: int, key: str) -> bool:
        return any(row[key] for row in hypothesis_results[:k])

    typed_hit_rank = next(
        (
            row["rank"]
            for row in hypothesis_results
            if row["known_positive_typed_entity_pair"]
        ),
        None,
    )
    groups = source_variant_groups(master)
    duplicate_extras = sum(group["source_record_count"] - 1 for group in groups)
    return {
        "schema": "cadengbench_a_known_joint_hypothesis_evaluation_v1",
        "assembly_id": assembly_id,
        "l0_valid": not errors,
        "validation_errors": errors,
        "abstained": not parsed and abstention is not None,
        "hypothesis_count": len(parsed),
        "raw_hypothesis_count": len(raw_hypotheses),
        "candidate_id_reference_count": candidate_id_reference_count,
        "hallucinated_candidate_id_count": hallucinated_candidate_id_count,
        "hallucinated_candidate_id_rate": (
            hallucinated_candidate_id_count / candidate_id_reference_count
            if candidate_id_reference_count
            else 0.0
        ),
        "known_joint_types": known_types,
        "source_record_count": len(master.get("joints", [])),
        "exact_unique_source_variant_count": len(groups),
        "exact_duplicate_extra_count": duplicate_extras,
        "source_variant_groups": groups,
        "top1_type_hit": hit_at(1, "known_joint_type"),
        "top1_entity_hit": hit_at(1, "known_positive_entity_pair"),
        "top1_typed_entity_hit": hit_at(1, "known_positive_typed_entity_pair"),
        "top3_type_hit": hit_at(3, "known_joint_type"),
        "top3_entity_hit": hit_at(3, "known_positive_entity_pair"),
        "top3_typed_entity_hit": hit_at(3, "known_positive_typed_entity_pair"),
        "typed_entity_reciprocal_rank": 1.0 / typed_hit_rank if typed_hit_rank else 0.0,
        "hypotheses": hypothesis_results,
        "positive_unlabeled_warning": (
            "A non-matching hypothesis is not proven mechanically invalid; the joint_set contains "
            "known positives consolidated from observed assemblies, not exhaustive negatives."
        ),
    }
