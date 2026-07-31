from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


RECOGNIZED_JOINT_TYPES = {
    "RigidJointType",
    "RevoluteJointType",
    "SliderJointType",
    "CylindricalJointType",
    "PlanarJointType",
    "BallJointType",
    "PinSlotJointType",
}

DOF_SIGNATURES = {
    "RigidJointType": (0, 0),
    "RevoluteJointType": (0, 1),
    "SliderJointType": (1, 0),
    "CylindricalJointType": (1, 1),
    "PlanarJointType": (2, 1),
    "BallJointType": (0, 3),
    "PinSlotJointType": (1, 1),
}

AXIS_DEFINED_JOINT_TYPES = {
    "RevoluteJointType",
    "SliderJointType",
    "CylindricalJointType",
    "PlanarJointType",
    "PinSlotJointType",
}
UNRESOLVED_AXIS_STATUSES = {
    "unresolved_no_shared_assembly_frame",
    "not_applicable_single_axis",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def make_oracle_prediction(master: dict[str, Any]) -> dict[str, Any]:
    """Create a prediction object from ground truth for evaluator smoke tests."""
    return {
        "schema": "cadengbench_a_pair_prediction_v1",
        "assembly_id": master.get("joint_set_id"),
        "predicted_joints": [
            {
                "body_a": joint.get("body_a"),
                "body_b": joint.get("body_b"),
                "joint_type": joint.get("fusion_joint_type"),
                "axis": joint.get("axis"),
                "mating_coordinate_frames": joint.get("mating_coordinate_frames"),
                "selected_entities": {
                    "body_a": (joint.get("geometry_or_origin_one") or {}).get("entity_one"),
                    "body_b": (joint.get("geometry_or_origin_two") or {}).get("entity_one"),
                },
                "limits": joint.get("limits"),
                "confidence": 1.0,
            }
            for joint in master.get("joints", [])
        ],
    }


def evaluate_master_prediction(master: dict[str, Any], prediction: dict[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(prediction, dict):
        prediction = {
            "predicted_joints": [],
            "_load_errors": ["prediction JSON root must be an object"],
        }
    truth_joints = _truth_joints(master)
    pred_joints, validation_errors = _prediction_joints(master, prediction)
    pair_scores = [_score_joint_pair(truth, pred) for truth in truth_joints for pred in pred_joints]
    assignments = _best_assignments(pair_scores, len(truth_joints), len(pred_joints))
    matched_scores = [pair_scores[index] for index in assignments]
    body_pair_matched_truth = {
        score["truth_index"] for score in matched_scores if score["body_pair_match"]
    }
    body_pair_matched_pred = {
        score["prediction_index"] for score in matched_scores if score["body_pair_match"]
    }
    type_correct = [
        score
        for score in matched_scores
        if score["body_pair_match"] and score["joint_type_match"]
    ]
    typed_matched_truth = {score["truth_index"] for score in type_correct}
    typed_matched_pred = {score["prediction_index"] for score in type_correct}
    dof_correct = [
        score for score in matched_scores if score["body_pair_match"] and score["dof_signature_match"]
    ]
    axis_scores = [
        score
        for score in matched_scores
        if score["body_pair_match"] and score.get("axis_angular_error_deg") is not None
    ]
    primary_axis_required_scores = [
        score
        for score in matched_scores
        if score["body_pair_match"]
        and score["joint_type_match"]
        and score["truth_joint_type"] in AXIS_DEFINED_JOINT_TYPES
    ]
    primary_axis_scores = [
        score
        for score in primary_axis_required_scores
        if score.get("axis_angular_error_deg") is not None
    ]
    entity_sides = [
        (score, side)
        for score in matched_scores
        if score["body_pair_match"]
        for side in ("body_a", "body_b")
        if score.get(f"entity_{side}_truth_available")
    ]

    true_count = len(truth_joints)
    pred_count = len(pred_joints)
    body_pair_match_count = len(body_pair_matched_truth)
    typed_match_count = len(typed_matched_truth)
    body_pair_precision = _safe_div(body_pair_match_count, pred_count)
    body_pair_recall = _safe_div(body_pair_match_count, true_count)
    joint_instance_precision = _safe_div(typed_match_count, pred_count)
    joint_instance_recall = _safe_div(typed_match_count, true_count)
    type_accuracy = _safe_div(typed_match_count, body_pair_match_count)

    result = {
        "assembly_id": master.get("joint_set_id"),
        "l0_valid": not validation_errors,
        "validation_errors": validation_errors,
        "truth_joint_count": true_count,
        "predicted_joint_count": pred_count,
        "matched_joint_count": typed_match_count,
        "body_pair_matched_joint_count": body_pair_match_count,
        "joint_count_exact": true_count == pred_count,
        "body_pair_precision": body_pair_precision,
        "body_pair_recall": body_pair_recall,
        "body_pair_f1": _f1(body_pair_precision, body_pair_recall),
        "body_pair_multiplicity_precision": body_pair_precision,
        "body_pair_multiplicity_recall": body_pair_recall,
        "body_pair_multiplicity_f1": _f1(body_pair_precision, body_pair_recall),
        "joint_instance_precision": joint_instance_precision,
        "joint_instance_recall": joint_instance_recall,
        "joint_instance_f1": _f1(joint_instance_precision, joint_instance_recall),
        "joint_type_accuracy_on_matched": type_accuracy,
        "dof_compatibility_accuracy_on_matched": _safe_div(
            len(dof_correct), body_pair_match_count
        ),
        "joint_type_precision_recall_f1": _type_prf(truth_joints, pred_joints, matched_scores),
        "axis_mean_angular_error_deg": _mean([score["axis_angular_error_deg"] for score in axis_scores]),
        "axis_median_angular_error_deg": _median([score["axis_angular_error_deg"] for score in axis_scores]),
        "axis_mean_distance_error": _mean([score["axis_distance_error"] for score in axis_scores]),
        "axis_median_distance_error": _median([score["axis_distance_error"] for score in axis_scores]),
        "axis_mean_normalized_distance_error": _mean(
            [score["axis_distance_error_normalized"] for score in axis_scores]
        ),
        "axis_median_normalized_distance_error": _median(
            [score["axis_distance_error_normalized"] for score in axis_scores]
        ),
        "primary_axis_mean_angular_error_deg": _mean(
            [score["axis_angular_error_deg"] for score in primary_axis_scores]
        ),
        "primary_axis_median_angular_error_deg": _median(
            [score["axis_angular_error_deg"] for score in primary_axis_scores]
        ),
        "primary_axis_mean_distance_error": _mean(
            [score["axis_distance_error"] for score in primary_axis_scores]
        ),
        "primary_axis_median_distance_error": _median(
            [score["axis_distance_error"] for score in primary_axis_scores]
        ),
        "primary_axis_mean_normalized_distance_error": _mean(
            [score["axis_distance_error_normalized"] for score in primary_axis_scores]
        ),
        "primary_axis_median_normalized_distance_error": _median(
            [score["axis_distance_error_normalized"] for score in primary_axis_scores]
        ),
        "primary_axis_required_count": len(primary_axis_required_scores),
        "primary_axis_prediction_count": len(primary_axis_scores),
        "primary_axis_prediction_rate": _safe_div(
            len(primary_axis_scores), len(primary_axis_required_scores)
        ),
        "entity_reference_prediction_rate": _safe_div(
            sum(bool(score.get(f"entity_{side}_predicted")) for score, side in entity_sides),
            len(entity_sides),
        ),
        "entity_reference_exact_accuracy": _safe_div(
            sum(bool(score.get(f"entity_{side}_exact_match")) for score, side in entity_sides),
            len(entity_sides),
        ),
        "entity_reference_equivalence_accuracy": _safe_div(
            sum(bool(score.get(f"entity_{side}_equivalent_match")) for score, side in entity_sides),
            len(entity_sides),
        ),
        "matched_pairs": matched_scores,
        "unmatched_truth_joint_ids": [
            joint["id"]
            for index, joint in enumerate(truth_joints)
            if index not in typed_matched_truth
        ],
        "unmatched_prediction_indices": [
            index for index in range(pred_count) if index not in typed_matched_pred
        ],
        "body_pair_unmatched_truth_joint_ids": [
            joint["id"]
            for index, joint in enumerate(truth_joints)
            if index not in body_pair_matched_truth
        ],
        "body_pair_unmatched_prediction_indices": [
            index for index in range(pred_count) if index not in body_pair_matched_pred
        ],
    }
    result["overall_pair_score"] = _overall_pair_score(result)
    return result


def evaluate_dataset(
    subset_dir: str | Path,
    predictions_dir: str | Path | None = None,
    *,
    use_oracle: bool = False,
    prediction_factory: Callable[[dict[str, Any], Path], dict[str, Any]] | None = None,
    prediction_source: str | None = None,
    assembly_filter: Callable[[dict[str, Any]], bool] | None = None,
    assembly_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    subset_dir = Path(subset_dir)
    predictions_dir = Path(predictions_dir) if predictions_dir else None
    assembly_dirs = sorted((subset_dir / "assemblies").glob("joint_set_*"))
    if limit is not None:
        assembly_dirs = assembly_dirs[:limit]

    results = []
    for assembly_dir in assembly_dirs:
        master_path = assembly_dir / "ground_truth" / "master.json"
        if not master_path.exists():
            continue
        master = load_json(master_path)
        if assembly_ids is not None and master.get("joint_set_id") not in assembly_ids:
            continue
        if assembly_filter is not None and not assembly_filter(master):
            continue
        if use_oracle:
            prediction = make_oracle_prediction(master)
            source = "oracle"
        elif prediction_factory is not None:
            prediction = prediction_factory(master, assembly_dir)
            source = prediction_source or getattr(prediction_factory, "__name__", "prediction_factory")
        else:
            if predictions_dir is None:
                raise ValueError(
                    "predictions_dir is required unless use_oracle=True or prediction_factory is provided"
                )
            prediction_path = _prediction_path(predictions_dir, master["joint_set_id"])
            if prediction_path is None:
                prediction = {
                    "assembly_id": master.get("joint_set_id"),
                    "predicted_joints": [],
                    "_load_errors": ["prediction file missing"],
                }
                source = "missing"
            else:
                try:
                    prediction = load_json(prediction_path)
                except (OSError, json.JSONDecodeError) as exc:
                    prediction = {
                        "assembly_id": master.get("joint_set_id"),
                        "predicted_joints": [],
                        "_load_errors": [f"prediction JSON could not be loaded: {exc}"],
                    }
                source = str(prediction_path)
        result = evaluate_master_prediction(master, prediction)
        result["prediction_source"] = source
        result["lossiness_label"] = _lossiness_label(assembly_dir)
        result["ground_truth_joint_types"] = sorted(
            {joint.get("fusion_joint_type") for joint in master.get("joints", []) if joint.get("fusion_joint_type")}
        )
        results.append(result)

    return _aggregate_results(results)


def _truth_joints(master: dict[str, Any]) -> list[dict[str, Any]]:
    joints = []
    normalization_length = _master_scale(master)
    for index, joint in enumerate(master.get("joints", [])):
        joints.append(
            {
                "index": index,
                "id": joint.get("id", f"truth_{index}"),
                "body_a": joint.get("body_a"),
                "body_b": joint.get("body_b"),
                "joint_type": joint.get("fusion_joint_type"),
                "axis": joint.get("axis") or {},
                "normalization_length": normalization_length,
                "entities": {
                    "body_a": _truth_entity_record(joint.get("geometry_or_origin_one") or {}),
                    "body_b": _truth_entity_record(joint.get("geometry_or_origin_two") or {}),
                },
            }
        )
    return joints


def _prediction_joints(master: dict[str, Any], prediction: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    errors = [str(error) for error in prediction.get("_load_errors", [])]
    if prediction.get("assembly_id") not in {None, master.get("joint_set_id")}:
        errors.append(
            f"assembly_id mismatch: expected {master.get('joint_set_id')}, got {prediction.get('assembly_id')}"
        )
    body_ids = {body.get("id") for body in master.get("bodies", [])}
    raw_joints = prediction.get("predicted_joints")
    if not isinstance(raw_joints, list):
        return [], errors + ["predicted_joints must be a list"]

    joints = []
    for index, raw in enumerate(raw_joints):
        if not isinstance(raw, dict):
            errors.append(f"predicted_joints[{index}] must be an object")
            continue
        body_a = raw.get("body_a")
        body_b = raw.get("body_b")
        joint_type = raw.get("joint_type") or raw.get("fusion_joint_type")
        axis_value = raw.get("axis")
        axis = axis_value if isinstance(axis_value, dict) else {}
        axis_status = raw.get("axis_status")
        selected_entities = raw.get("selected_entities") or {}
        if body_a not in body_ids:
            errors.append(f"predicted_joints[{index}].body_a unknown: {body_a}")
        if body_b not in body_ids:
            errors.append(f"predicted_joints[{index}].body_b unknown: {body_b}")
        if joint_type not in RECOGNIZED_JOINT_TYPES:
            errors.append(f"predicted_joints[{index}].joint_type unknown: {joint_type}")
        if axis_value is None:
            if axis_status not in UNRESOLVED_AXIS_STATUSES:
                errors.append(
                    f"predicted_joints[{index}].axis missing without a recognized axis_status"
                )
        elif not _valid_axis(axis):
            errors.append(f"predicted_joints[{index}].axis invalid")
        if not isinstance(selected_entities, dict):
            errors.append(f"predicted_joints[{index}].selected_entities must be an object")
            selected_entities = {}
        for side in ("body_a", "body_b"):
            entity = selected_entities.get(side)
            if entity is not None and not _valid_entity(entity):
                errors.append(f"predicted_joints[{index}].selected_entities.{side} invalid")
        joints.append(
            {
                "index": index,
                "body_a": body_a,
                "body_b": body_b,
                "joint_type": joint_type,
                "axis": axis,
                "axis_status": axis_status,
                "selected_entities": selected_entities,
            }
        )
    return joints, errors


def _score_joint_pair(truth: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    body_pair_match = _body_pair(truth) == _body_pair(pred)
    joint_type_match = truth.get("joint_type") == pred.get("joint_type")
    dof_signature_match = DOF_SIGNATURES.get(truth.get("joint_type")) == DOF_SIGNATURES.get(
        pred.get("joint_type")
    )
    axis_truth = truth.get("axis") or {}
    axis_pred = pred.get("axis") or {}
    angular_error = None
    distance_error = None
    normalized_distance_error = None
    if _valid_axis(axis_truth) and _valid_axis(axis_pred):
        truth_direction = _normalise(_axis_direction(axis_truth))
        pred_direction = _normalise(_axis_direction(axis_pred))
        angular_error = _axis_angular_error_deg(truth_direction, pred_direction)
        distance_error = _axis_distance(axis_truth, axis_pred)
        normalized_distance_error = distance_error / max(
            float(truth.get("normalization_length") or 1.0), 1e-12
        )
    entity_scores = {}
    for side in ("body_a", "body_b"):
        truth_entity = (truth.get("entities") or {}).get(side) or {}
        predicted_entity = (pred.get("selected_entities") or {}).get(side)
        predicted, exact, equivalent = _entity_reference_match(predicted_entity, truth_entity)
        entity_scores.update(
            {
                f"entity_{side}_truth_available": bool(truth_entity.get("primary")),
                f"entity_{side}_predicted": predicted,
                f"entity_{side}_exact_match": exact,
                f"entity_{side}_equivalent_match": equivalent,
            }
        )
    score = {
        "truth_index": truth["index"],
        "truth_joint_id": truth["id"],
        "prediction_index": pred["index"],
        "truth_joint_type": truth.get("joint_type"),
        "predicted_joint_type": pred.get("joint_type"),
        "body_pair_match": body_pair_match,
        "joint_type_match": joint_type_match,
        "dof_signature_match": dof_signature_match,
        "axis_angular_error_deg": angular_error,
        "axis_distance_error": distance_error,
        "axis_distance_error_normalized": normalized_distance_error,
        "axis_predicted": _valid_axis(axis_pred),
        **entity_scores,
    }
    score["assignment_score"] = _assignment_score(score)
    return score


def _assignment_score(score: dict[str, Any]) -> float:
    value = 0.0
    if score["body_pair_match"]:
        value += 100.0
    if score["joint_type_match"]:
        value += 20.0
    if score["dof_signature_match"]:
        value += 4.0
    for side in ("body_a", "body_b"):
        if score.get(f"entity_{side}_equivalent_match"):
            value += 2.0
    angular = score.get("axis_angular_error_deg")
    distance = score.get("axis_distance_error")
    if angular is not None:
        value += max(0.0, 10.0 - angular / 9.0)
    if distance is not None:
        value += max(0.0, 5.0 - distance)
    return value


def _best_assignments(pair_scores: list[dict[str, Any]], truth_count: int, pred_count: int) -> list[int]:
    if truth_count == 0 or pred_count == 0:
        return []
    score_by_pair = {
        (score["truth_index"], score["prediction_index"]): index
        for index, score in enumerate(pair_scores)
    }
    smaller = min(truth_count, pred_count)
    if truth_count <= 8 and pred_count <= 8:
        small_is_truth = truth_count <= pred_count
        small_count = truth_count if small_is_truth else pred_count
        large_count = pred_count if small_is_truth else truth_count

        @lru_cache(maxsize=None)
        def solve(small_index: int, used_large: int) -> tuple[float, tuple[int, ...]]:
            if small_index == small_count:
                return 0.0, ()
            best_score = -1.0
            best_indices: tuple[int, ...] = ()
            for large_index in range(large_count):
                bit = 1 << large_index
                if used_large & bit:
                    continue
                truth_index, pred_index = (
                    (small_index, large_index)
                    if small_is_truth
                    else (large_index, small_index)
                )
                pair_index = score_by_pair[(truth_index, pred_index)]
                remaining_score, remaining_indices = solve(small_index + 1, used_large | bit)
                total = pair_scores[pair_index]["assignment_score"] + remaining_score
                if total > best_score:
                    best_score = total
                    best_indices = (pair_index,) + remaining_indices
            return best_score, best_indices

        return list(solve(0, 0)[1])

    # Greedy fallback for unusually dense multi-joint pairs. This keeps runtime predictable.
    ordered = sorted(range(len(pair_scores)), key=lambda index: pair_scores[index]["assignment_score"], reverse=True)
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    selected = []
    for index in ordered:
        score = pair_scores[index]
        if score["truth_index"] in used_truth or score["prediction_index"] in used_pred:
            continue
        selected.append(index)
        used_truth.add(score["truth_index"])
        used_pred.add(score["prediction_index"])
        if len(selected) == smaller:
            break
    return selected


def _body_pair(joint: dict[str, Any]) -> tuple[str, str]:
    body_a = joint.get("body_a")
    body_b = joint.get("body_b")
    return tuple(sorted((str(body_a) if body_a is not None else "", str(body_b) if body_b is not None else "")))


def _master_scale(master: dict[str, Any]) -> float:
    diagonals = []
    for body in master.get("bodies", []):
        box = (body.get("geometry_quality") or {}).get("bounding_box") or {}
        extents = box.get("extents")
        if isinstance(extents, list) and len(extents) == 3:
            try:
                diagonal = _norm([float(value) for value in extents])
            except (TypeError, ValueError):
                continue
            if math.isfinite(diagonal) and diagonal > 1e-12:
                diagonals.append(diagonal)
    return max(diagonals, default=1.0)


def _truth_entity_record(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary": geometry.get("entity_one"),
        "equivalents": geometry.get("entity_one_equivalents") or [],
    }


def _valid_entity(entity: Any) -> bool:
    if not isinstance(entity, dict):
        return False
    return entity.get("type") in {"BRepFace", "BRepEdge"} and isinstance(entity.get("index"), int)


def _entity_key(entity: dict[str, Any] | None) -> tuple[Any, Any] | None:
    if not _valid_entity(entity):
        return None
    return entity.get("type"), entity.get("index")


def _entity_reference_match(
    predicted: dict[str, Any] | None, truth: dict[str, Any]
) -> tuple[bool, bool, bool]:
    predicted_key = _entity_key(predicted)
    primary_key = _entity_key(truth.get("primary"))
    if predicted_key is None:
        return False, False, False
    exact = primary_key is not None and predicted_key == primary_key
    equivalent_keys = {
        key for key in (_entity_key(entity) for entity in truth.get("equivalents", [])) if key is not None
    }
    return True, exact, exact or predicted_key in equivalent_keys


def _valid_axis(axis: dict[str, Any]) -> bool:
    try:
        origin = _axis_origin(axis)
        direction = _axis_direction(axis)
    except Exception:
        return False
    return len(origin) == 3 and len(direction) == 3 and _norm(direction) > 1e-12


def _axis_origin(axis: dict[str, Any]) -> list[float]:
    return [float(value) for value in axis.get("origin", [])]


def _axis_direction(axis: dict[str, Any]) -> list[float]:
    return [float(value) for value in axis.get("direction", [])]


def _axis_angular_error_deg(a: list[float], b: list[float]) -> float:
    dot = min(1.0, max(-1.0, abs(_dot(_normalise(a), _normalise(b)))))
    return math.degrees(math.acos(dot))


def _axis_distance(axis_a: dict[str, Any], axis_b: dict[str, Any]) -> float:
    origin_a = _axis_origin(axis_a)
    origin_b = _axis_origin(axis_b)
    direction_a = _normalise(_axis_direction(axis_a))
    direction_b = _normalise(_axis_direction(axis_b))
    cross = _cross(direction_a, direction_b)
    cross_norm = _norm(cross)
    delta = _sub(origin_b, origin_a)
    if cross_norm < 1e-9:
        return _norm(_cross(delta, direction_a))
    return abs(_dot(delta, cross)) / cross_norm


def _type_prf(
    truth_joints: list[dict[str, Any]],
    pred_joints: list[dict[str, Any]],
    matched_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    per_type = {}
    joint_types = sorted(
        {joint.get("joint_type") for joint in truth_joints + pred_joints if joint.get("joint_type")}
    )
    for joint_type in joint_types:
        truth_count = sum(1 for joint in truth_joints if joint.get("joint_type") == joint_type)
        pred_count = sum(1 for joint in pred_joints if joint.get("joint_type") == joint_type)
        true_positive = sum(
            1
            for score in matched_scores
            if score["body_pair_match"]
            and score["truth_joint_type"] == joint_type
            and score["predicted_joint_type"] == joint_type
        )
        precision = _safe_div(true_positive, pred_count)
        recall = _safe_div(true_positive, truth_count)
        per_type[joint_type] = {
            "truth_count": truth_count,
            "predicted_count": pred_count,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    return per_type


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [result for result in results if result["l0_valid"]]
    official = _official_aggregate(results)
    summary = {
        "assembly_count": len(results),
        "l0_valid_count": len(valid),
        "l0_valid_rate": _safe_div(len(valid), len(results)),
        "joint_count_exact_rate": _mean([float(result["joint_count_exact"]) for result in results]),
        "joint_precision_mean": _mean([result["joint_instance_precision"] for result in results]),
        "joint_recall_mean": _mean([result["joint_instance_recall"] for result in results]),
        "joint_f1_mean": _mean([result["joint_instance_f1"] for result in results]),
        "body_pair_precision_mean": _mean([result["body_pair_precision"] for result in results]),
        "body_pair_recall_mean": _mean([result["body_pair_recall"] for result in results]),
        "body_pair_f1_mean": _mean([result["body_pair_f1"] for result in results]),
        "body_pair_multiplicity_precision_mean": _mean(
            [result["body_pair_multiplicity_precision"] for result in results]
        ),
        "body_pair_multiplicity_recall_mean": _mean(
            [result["body_pair_multiplicity_recall"] for result in results]
        ),
        "body_pair_multiplicity_f1_mean": _mean(
            [result["body_pair_multiplicity_f1"] for result in results]
        ),
        "joint_type_accuracy_on_matched_mean": _mean(
            [result["joint_type_accuracy_on_matched"] for result in results]
        ),
        "axis_mean_angular_error_deg": _mean(
            [
                result["axis_mean_angular_error_deg"]
                for result in results
                if result["axis_mean_angular_error_deg"] is not None
            ]
        ),
        "axis_mean_distance_error": _mean(
            [
                result["axis_mean_distance_error"]
                for result in results
                if result["axis_mean_distance_error"] is not None
            ]
        ),
        "overall_pair_score_mean": _mean([result["overall_pair_score"] for result in results]),
        **official["summary"],
    }
    return {
        "schema": "cadengbench_a_pair_evaluation_report_v2",
        "metric_specification": "cadengbench_a_pair_metrics_v1_2",
        "summary": summary,
        "joint_type_metrics": official["joint_type_metrics"],
        "by_lossiness_label": _aggregate_by_key(results, "lossiness_label"),
        "by_ground_truth_joint_type": _aggregate_by_membership(results, "ground_truth_joint_types"),
        "results": results,
    }


def _official_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    truth_total = sum(result["truth_joint_count"] for result in results)
    predicted_total = sum(result["predicted_joint_count"] for result in results)
    valid_body_pair_matches = [
        score
        for result in results
        if result["l0_valid"]
        for score in result["matched_pairs"]
        if score["body_pair_match"]
    ]
    body_pair_matched_total = len(valid_body_pair_matches)
    typed_matched_total = sum(
        bool(score["joint_type_match"]) for score in valid_body_pair_matches
    )
    dof_correct = sum(bool(score["dof_signature_match"]) for score in valid_body_pair_matches)
    primary_axis_required = [
        score
        for score in valid_body_pair_matches
        if score["joint_type_match"]
        and score["truth_joint_type"] in AXIS_DEFINED_JOINT_TYPES
    ]
    primary_axes = [
        score
        for score in primary_axis_required
        if score.get("axis_angular_error_deg") is not None
    ]
    entity_sides = [
        (score, side)
        for score in valid_body_pair_matches
        for side in ("body_a", "body_b")
        if score.get(f"entity_{side}_truth_available")
    ]
    body_pair_precision = _safe_div(body_pair_matched_total, predicted_total)
    body_pair_recall = _safe_div(body_pair_matched_total, truth_total)
    joint_precision = _safe_div(typed_matched_total, predicted_total)
    joint_recall = _safe_div(typed_matched_total, truth_total)
    type_metrics = _aggregate_type_metrics(results)
    macro_f1 = _mean(
        [data["f1"] for data in type_metrics.values() if data["truth_count"] or data["predicted_count"]]
    )
    return {
        "summary": {
            "truth_joint_count_total": truth_total,
            "predicted_joint_count_total": predicted_total,
            "matched_joint_count_total": typed_matched_total,
            "body_pair_matched_joint_count_total": body_pair_matched_total,
            "joint_precision_micro": joint_precision,
            "joint_recall_micro": joint_recall,
            "joint_f1_micro": _f1(joint_precision, joint_recall),
            "body_pair_multiplicity_precision_micro": body_pair_precision,
            "body_pair_multiplicity_recall_micro": body_pair_recall,
            "body_pair_multiplicity_f1_micro": _f1(
                body_pair_precision, body_pair_recall
            ),
            "joint_type_accuracy_on_matched_micro": _safe_div(
                typed_matched_total, body_pair_matched_total
            ),
            "joint_type_macro_f1": macro_f1,
            "dof_compatibility_accuracy_on_matched_micro": _safe_div(
                dof_correct, body_pair_matched_total
            ),
            "primary_axis_required_count": len(primary_axis_required),
            "primary_axis_eligible_count": len(primary_axes),
            "primary_axis_prediction_count": len(primary_axes),
            "primary_axis_prediction_rate": _safe_div(
                len(primary_axes), len(primary_axis_required)
            ),
            "primary_axis_mean_angular_error_deg": _mean(
                [score["axis_angular_error_deg"] for score in primary_axes]
            ),
            "primary_axis_median_angular_error_deg": _median(
                [score["axis_angular_error_deg"] for score in primary_axes]
            ),
            "primary_axis_mean_distance_error": _mean(
                [score["axis_distance_error"] for score in primary_axes]
            ),
            "primary_axis_median_distance_error": _median(
                [score["axis_distance_error"] for score in primary_axes]
            ),
            "primary_axis_mean_normalized_distance_error": _mean(
                [score["axis_distance_error_normalized"] for score in primary_axes]
            ),
            "primary_axis_median_normalized_distance_error": _median(
                [score["axis_distance_error_normalized"] for score in primary_axes]
            ),
            "entity_reference_side_count": len(entity_sides),
            "entity_reference_prediction_rate": _safe_div(
                sum(bool(score.get(f"entity_{side}_predicted")) for score, side in entity_sides),
                len(entity_sides),
            ),
            "entity_reference_exact_accuracy": _safe_div(
                sum(bool(score.get(f"entity_{side}_exact_match")) for score, side in entity_sides),
                len(entity_sides),
            ),
            "entity_reference_equivalence_accuracy": _safe_div(
                sum(
                    bool(score.get(f"entity_{side}_equivalent_match"))
                    for score, side in entity_sides
                ),
                len(entity_sides),
            ),
        },
        "joint_type_metrics": type_metrics,
    }


def _aggregate_type_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        joint_type: {"truth_count": 0, "predicted_count": 0, "true_positive": 0}
        for joint_type in sorted(RECOGNIZED_JOINT_TYPES)
    }
    for result in results:
        for joint_type, data in result["joint_type_precision_recall_f1"].items():
            totals.setdefault(
                joint_type, {"truth_count": 0, "predicted_count": 0, "true_positive": 0}
            )
            totals[joint_type]["truth_count"] += data["truth_count"]
            totals[joint_type]["predicted_count"] += data["predicted_count"]
            if result["l0_valid"]:
                totals[joint_type]["true_positive"] += data["true_positive"]
    for data in totals.values():
        data["precision"] = _safe_div(data["true_positive"], data["predicted_count"])
        data["recall"] = _safe_div(data["true_positive"], data["truth_count"])
        data["f1"] = _f1(data["precision"], data["recall"])
    return totals


def _aggregate_by_key(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get(key) or "unknown"), []).append(result)
    return {
        group: {
            "assembly_count": len(items),
            "joint_instance_f1_mean": _mean([item["joint_instance_f1"] for item in items]),
            "body_pair_f1_mean": _mean([item["body_pair_f1"] for item in items]),
            "joint_type_accuracy_on_matched_mean": _mean(
                [item["joint_type_accuracy_on_matched"] for item in items]
            ),
            "overall_pair_score_mean": _mean([item["overall_pair_score"] for item in items]),
        }
        for group, items in sorted(grouped.items())
    }


def _aggregate_by_membership(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        for value in result.get(key, []):
            grouped.setdefault(str(value), []).append(result)
    return {
        group: {
            "assembly_count": len(items),
            "joint_f1_mean": _mean([item["joint_instance_f1"] for item in items]),
            "body_pair_multiplicity_f1_mean": _mean(
                [item["body_pair_multiplicity_f1"] for item in items]
            ),
            "joint_type_accuracy_on_matched_mean": _mean(
                [item["joint_type_accuracy_on_matched"] for item in items]
            ),
            "axis_mean_angular_error_deg": _mean(
                [
                    item["axis_mean_angular_error_deg"]
                    for item in items
                    if item["axis_mean_angular_error_deg"] is not None
                ]
            ),
            "overall_pair_score_mean": _mean([item["overall_pair_score"] for item in items]),
        }
        for group, items in sorted(grouped.items())
    }


def _overall_pair_score(result: dict[str, Any]) -> float:
    if not result.get("l0_valid"):
        return 0.0
    components = [
        result["body_pair_f1"],
        result["joint_type_accuracy_on_matched"],
        float(result["joint_count_exact"]),
    ]
    angular = result.get("axis_mean_angular_error_deg")
    if angular is not None:
        components.append(max(0.0, 1.0 - angular / 180.0))
    distance = result.get("axis_mean_distance_error")
    if distance is not None:
        components.append(1.0 / (1.0 + distance))
    return _mean(components) or 0.0


def _prediction_path(predictions_dir: Path, assembly_id: str) -> Path | None:
    candidates = [
        predictions_dir / f"{assembly_id}.json",
        predictions_dir / assembly_id / "prediction.json",
        predictions_dir / assembly_id / "predictions.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _lossiness_label(assembly_dir: Path) -> str:
    path = assembly_dir / "reports" / "urdf_lossiness.json"
    if not path.exists():
        return "unknown"
    try:
        data = load_json(path)
    except Exception:
        return "unknown"
    return (
        data.get("lossiness_label")
        or data.get("urdf_export_status", {}).get("lossiness_label")
        or data.get("urdf", {}).get("export_status", {}).get("lossiness_label")
        or data.get("export_status", {}).get("lossiness_label")
        or "unknown"
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _normalise(vec: list[float]) -> list[float]:
    length = _norm(vec)
    if length <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [value / length for value in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))
