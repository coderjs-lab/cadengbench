"""Provider-neutral A2 request construction, scoring, and aggregation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pipeline.cadengbench_a_dev20 import read_json
from pipeline.cadengbench_a_pose_evaluator import evaluate_a2_kinematic_prediction
from pipeline.cadengbench_a_pose_simulation import run_authorized_pose_simulation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_hypothesis(a1_prediction: dict[str, Any], selected_rank: int) -> dict[str, Any]:
    hypotheses = a1_prediction.get("ranked_hypotheses") or []
    selected = next((row for row in hypotheses if row.get("rank") == selected_rank), None)
    if selected is None:
        raise ValueError(f"A1 prediction has no hypothesis at rank {selected_rank}")
    required = {"joint_type", "body_a_candidate_id", "body_b_candidate_id"}
    if not required.issubset(selected):
        raise ValueError("selected A1 hypothesis is incomplete")
    return selected


def build_pose_stage_package(
    item_dir: str | Path,
    prompt_dir: str | Path,
    a1_prediction: dict[str, Any],
    *,
    selected_rank: int = 1,
) -> dict[str, Any]:
    """Build a provider-neutral A2 kinematics request without reading labels."""
    item_dir = Path(item_dir)
    prompt_dir = Path(prompt_dir)
    task = read_json(item_dir / "task.json")
    selected = _selected_hypothesis(a1_prediction, selected_rank)
    system_prompt = (prompt_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (prompt_dir / "user_template.txt").read_text(encoding="utf-8")
    replacements = {
        "{{assembly_id}}": task["assembly_id"],
        "{{selected_rank}}": str(selected_rank),
        "{{joint_type}}": selected["joint_type"],
        "{{body_a_candidate_id}}": selected["body_a_candidate_id"],
        "{{body_b_candidate_id}}": selected["body_b_candidate_id"],
    }
    user_prompt = template
    for token, value in replacements.items():
        user_prompt = user_prompt.replace(token, value)
    public_geometry = item_dir / "public_geometry.json"
    candidates = item_dir / "candidates.json"
    user_prompt += (
        "\n\nPUBLIC GEOMETRY JSON:\n"
        + public_geometry.read_text(encoding="utf-8")
        + "\n\nCANDIDATE REGISTRY JSON:\n"
        + candidates.read_text(encoding="utf-8")
    )
    images = []
    for role in ("body_a", "body_b"):
        for relative in task["assets"]["views"][role]:
            path = item_dir / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append(
                {
                    "body_role": role,
                    "relative_path": relative,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "detail": "high",
                }
            )
    return {
        "schema": "cadengbench_a_a2_kinematic_request_package_v1",
        "assembly_id": task["assembly_id"],
        "stage": "A2_kinematic_verification",
        "selected_rank": selected_rank,
        "selected_hypothesis": {
            key: selected[key]
            for key in ("joint_type", "body_a_candidate_id", "body_b_candidate_id")
        },
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "output_schema": read_json(prompt_dir / "output_schema.json"),
        "images": images,
        "evidence": {
            "public_geometry_sha256": _sha256(public_geometry),
            "candidate_registry_sha256": _sha256(candidates),
        },
        "ground_truth_accessed": False,
    }


def evaluate_pose_stage(
    master: dict[str, Any],
    candidates: dict[str, Any],
    public_geometry: dict[str, Any],
    a1_prediction: dict[str, Any],
    pose_prediction: dict[str, Any],
    *,
    simulation_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a predicted frame and optionally execute authorized physics."""
    evaluation = evaluate_a2_kinematic_prediction(
        master, candidates, public_geometry, a1_prediction, pose_prediction
    )
    simulation = None
    if simulation_output_dir is not None:
        simulation = run_authorized_pose_simulation(
            master, evaluation, simulation_output_dir
        )
    return {
        "schema": "cadengbench_a_a2_kinematic_stage_result_v1",
        "assembly_id": evaluation["assembly_id"],
        "pose_evaluation": evaluation,
        "simulation": simulation,
        "headline_a1_score_modified": False,
    }


def aggregate_pose_stage_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate conditional A2 pose and simulation outcomes."""
    statuses: dict[str, int] = {}
    eligible = 0
    pose_passes = 0
    authorized = 0
    physics_passes = 0
    physics_executions = 0
    for record in records:
        evaluation = record["pose_evaluation"]
        status = evaluation["status"]
        statuses[status] = statuses.get(status, 0) + 1
        if evaluation["typed_known_positive_source_joint_ids"]:
            eligible += 1
        pose_passes += int(evaluation["pose_pass"])
        authorized += int(evaluation["kinematic_execution_authorized"])
        simulation = record.get("simulation")
        if simulation and simulation.get("physics_executed"):
            physics_executions += 1
            physics_passes += int(simulation.get("status") == "passed")
    item_count = len(records)
    return {
        "schema": "cadengbench_a_a2_kinematic_aggregate_report_v1",
        "scope": "conditional_secondary_track_separate_from_a1_headline",
        "item_count": item_count,
        "status_counts": dict(sorted(statuses.items())),
        "typed_a1_eligible_count": eligible,
        "typed_a1_eligibility_rate": eligible / item_count if item_count else 0.0,
        "conditional_pose_pass_count": pose_passes,
        "conditional_pose_pass_rate": pose_passes / eligible if eligible else None,
        "execution_authorized_count": authorized,
        "end_to_end_authorization_rate": authorized / item_count if item_count else 0.0,
        "physics_execution_count": physics_executions,
        "physics_pass_count": physics_passes,
        "physics_pass_rate": physics_passes / physics_executions if physics_executions else None,
        "headline_a1_score_modified": False,
    }
