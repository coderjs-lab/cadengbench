#!/usr/bin/env python3
"""Run CADEngBench-A v2 from public evidence through A1 and A2.

The command is intentionally provider-neutral and artifact-preserving. Every item
stores prompt text, ordered image hashes, raw provider responses, parsed JSON,
evaluator reports, usage, latency, and simulation output.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.cadengbench_a_dev20 import read_json, write_json
from pipeline.cadengbench_a_hypothesis_evaluator import evaluate_hypothesis_prediction
from pipeline.cadengbench_a_pose_stage import build_pose_stage_package, evaluate_pose_stage
from pipeline.cadengbench_a_provider_runtime import (
    MultimodalRequest,
    parse_model_output,
    send_multimodal_request,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _item_dir(dataset_root: Path, split: str, assembly_id: str) -> Path:
    candidates = (
        dataset_root / "public" / split / assembly_id,
        dataset_root / split / assembly_id,
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _label_path(labels_root: Path, split: str, assembly_id: str) -> Path:
    candidates = (
        labels_root / split / assembly_id / "master.json",
        labels_root / "sealed_ground_truth" / split / assembly_id / "master.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _validate_a0(item_dir: Path, label_path: Path) -> dict[str, Any]:
    required = ("task.json", "candidates.json", "public_geometry.json")
    missing = [name for name in required if not (item_dir / name).is_file()]
    public_label_files = [
        path.relative_to(item_dir).as_posix()
        for path in item_dir.rglob("*")
        if path.is_file() and ("ground_truth" in path.name or path.name == "master.json")
    ]
    task = read_json(item_dir / "task.json") if not missing else {}
    missing_views = [
        relative
        for paths in (task.get("assets", {}).get("views", {}) or {}).values()
        for relative in paths
        if not (item_dir / relative).is_file()
    ]
    errors = []
    if missing:
        errors.append(f"missing public files: {missing}")
    if public_label_files:
        errors.append(f"label-like files found in public evidence: {public_label_files}")
    if missing_views:
        errors.append(f"missing views: {missing_views}")
    if not label_path.is_file():
        errors.append(f"missing evaluator label: {label_path}")
    return {
        "schema": "cadengbench_a_a0_integrity_result_v2",
        "assembly_id": task.get("assembly_id") or item_dir.name,
        "pass": not errors,
        "errors": errors,
        "public_label_files": public_label_files,
        "public_file_count": sum(path.is_file() for path in item_dir.rglob("*")),
    }


def _render_a1_request(item_dir: Path, prompt_dir: Path) -> MultimodalRequest:
    task = read_json(item_dir / "task.json")
    bodies = {body["body_role"]: body["body_id"] for body in task["bodies"]}
    template = (prompt_dir / "user_template.txt").read_text(encoding="utf-8")
    user = (
        template.replace("{{assembly_id}}", task["assembly_id"])
        .replace("{{body_a_id}}", bodies["body_a"])
        .replace("{{body_b_id}}", bodies["body_b"])
    )
    user += (
        "\n\nPUBLIC GEOMETRY JSON:\n"
        + (item_dir / "public_geometry.json").read_text(encoding="utf-8")
        + "\n\nCANDIDATE REGISTRY JSON:\n"
        + (item_dir / "candidates.json").read_text(encoding="utf-8")
    )
    images = tuple(
        item_dir / relative
        for role in ("body_a", "body_b")
        for relative in task["assets"]["views"][role]
    )
    return MultimodalRequest(
        system_prompt=(prompt_dir / "system.txt").read_text(encoding="utf-8"),
        user_prompt=user,
        images=images,
        output_schema=read_json(prompt_dir / "output_schema.json"),
        max_output_tokens=2500,
    )


def _render_a2_request(
    item_dir: Path,
    prompt_dir: Path,
    prediction: dict[str, Any],
) -> MultimodalRequest:
    package = build_pose_stage_package(item_dir, prompt_dir, prediction)
    return MultimodalRequest(
        system_prompt=package["system_prompt"],
        user_prompt=package["user_prompt"],
        images=tuple(item_dir / row["relative_path"] for row in package["images"]),
        output_schema=package["output_schema"],
        max_output_tokens=2500,
    )


def _request_record(
    request: MultimodalRequest,
    *,
    provider: str,
    model: str,
    stage: str,
) -> dict[str, Any]:
    return {
        "schema": "cadengbench_a_request_record_v2",
        "stage": stage,
        "provider": provider,
        "model": model,
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "output_schema": request.output_schema,
        "max_output_tokens": request.max_output_tokens,
        "reasoning_effort": request.reasoning_effort,
        "reasoning_control": (
            "explicit"
            if provider in {"openai", "gemini"}
            else "provider_default_not_overridden"
        ),
        "images": [
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in request.images
        ],
    }


def _call_and_archive(
    provider: str,
    model: str,
    request: MultimodalRequest,
    output_dir: Path,
    *,
    base_url: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    response = send_multimodal_request(
        provider, model, request, base_url=base_url
    )
    latency = time.perf_counter() - started
    write_json(output_dir / "provider_response.json", response.payload)
    (output_dir / "raw_response.txt").write_text(response.text, encoding="utf-8")
    prediction, parser = parse_model_output(response.text, request.output_schema)
    parser.update(
        {
            "latency_seconds": latency,
            "response_id": response.response_id,
            "usage": response.usage,
        }
    )
    write_json(output_dir / "parser_report.json", parser)
    if prediction is not None:
        write_json(output_dir / "prediction.json", prediction)
    return prediction, parser


def _assembly_ids(dataset_root: Path, split: str) -> list[str]:
    base = dataset_root / "public" / split
    if not base.is_dir():
        base = dataset_root / split
    return sorted(path.name for path in base.glob("joint_set_*") if path.is_dir())


def _verify_frozen_contract(dataset_root: Path) -> dict[str, Any]:
    """Fail if active prompts or a full Core-150 manifest changed after freeze."""

    contract = read_json(
        REPO_ROOT
        / "dataset"
        / "cadengbench_a"
        / "benchmark_contract_v2"
        / "contract_manifest.json"
    )
    prompt_roots = {
        "A1": REPO_ROOT / "prompts" / "cadengbench_a" / "pair_track_b_v3",
        "A2": REPO_ROOT / "prompts" / "cadengbench_a" / "a2_kinematics_v1",
    }
    errors = []
    for layer, records in contract["prompts"].items():
        for name, expected in records.items():
            path = prompt_roots[layer] / name
            if not path.is_file() or _sha256(path) != expected:
                errors.append(f"{layer} prompt hash mismatch: {path}")
    package_manifest = dataset_root / "package_manifest.json"
    if package_manifest.is_file():
        expected = contract["dataset_manifest"]["sha256"]
        if _sha256(package_manifest) != expected:
            errors.append("Core-150 package manifest hash does not match v2 contract")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "benchmark_contract": contract["benchmark_contract"],
        "contract_manifest_sha256": _sha256(
            REPO_ROOT
            / "dataset"
            / "cadengbench_a"
            / "benchmark_contract_v2"
            / "contract_manifest.json"
        ),
    }


def main() -> int:
    """Parse CLI arguments and execute an artifact-preserving benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--provider", choices=("openai", "anthropic", "gemini", "openrouter", "mistral", "compatible"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high"),
        default="minimal",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--assembly-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ids = args.assembly_id or _assembly_ids(args.dataset_root, args.split)
    if args.limit is not None:
        ids = ids[: args.limit]
    if not ids:
        parser.error("no benchmark items selected")
    contract_identity = _verify_frozen_contract(args.dataset_root)
    if args.output_dir.exists() and not args.resume:
        parser.error("output directory exists; use --resume or choose another directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a1_prompt = REPO_ROOT / "prompts" / "cadengbench_a" / "pair_track_b_v3"
    a2_prompt = REPO_ROOT / "prompts" / "cadengbench_a" / "a2_kinematics_v1"
    manifest: dict[str, Any] = {
        "schema": "cadengbench_a_end_to_end_run_v2",
        "benchmark_layers": ["A0", "A1", "A2"],
        **contract_identity,
        "provider": args.provider,
        "model": args.model,
        "split": args.split,
        "dry_run": args.dry_run,
        "started_at": _utc_now(),
        "assembly_ids": ids,
        "items": [],
    }
    write_json(args.output_dir / "run_manifest.json", manifest)

    for assembly_id in ids:
        item_out = args.output_dir / "items" / assembly_id
        item_out.mkdir(parents=True, exist_ok=True)
        if args.resume and (item_out / "result.json").is_file():
            manifest["items"].append(read_json(item_out / "result.json"))
            continue
        item_dir = _item_dir(args.dataset_root, args.split, assembly_id)
        label_path = _label_path(args.labels_root, args.split, assembly_id)
        a0 = _validate_a0(item_dir, label_path)
        write_json(item_out / "a0_integrity.json", a0)
        if not a0["pass"]:
            result = {"assembly_id": assembly_id, "status": "a0_failed"}
            write_json(item_out / "result.json", result)
            manifest["items"].append(result)
            continue

        a1_dir = item_out / "a1"
        a1_dir.mkdir(exist_ok=True)
        a1_request = _render_a1_request(item_dir, a1_prompt)
        a1_request = replace(
            a1_request,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        )
        write_json(
            a1_dir / "request.json",
            _request_record(
                a1_request, provider=args.provider, model=args.model, stage="A1"
            ),
        )
        if args.dry_run:
            result = {"assembly_id": assembly_id, "status": "dry_run_packaged"}
            write_json(item_out / "result.json", result)
            manifest["items"].append(result)
            continue

        try:
            a1_prediction, a1_parser = _call_and_archive(
                args.provider,
                args.model,
                a1_request,
                a1_dir,
                base_url=args.base_url,
            )
        except Exception as exc:
            write_json(
                a1_dir / "provider_error.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            result = {"assembly_id": assembly_id, "status": "a1_provider_failed"}
            write_json(item_out / "result.json", result)
            manifest["items"].append(result)
            write_json(args.output_dir / "run_manifest.json", manifest)
            continue
        master = read_json(label_path)
        candidates = read_json(item_dir / "candidates.json")
        a1_evaluation = evaluate_hypothesis_prediction(
            master, candidates, a1_prediction or {}
        )
        if a1_parser["schema_errors"]:
            a1_evaluation["l0_valid"] = False
        write_json(a1_dir / "evaluation.json", a1_evaluation)

        a2_status = "not_requested_invalid_a1"
        a2_evaluation = None
        a2_parser = None
        if a1_prediction and a1_evaluation["l0_valid"] and a1_prediction.get("ranked_hypotheses"):
            a2_dir = item_out / "a2"
            a2_dir.mkdir(exist_ok=True)
            a2_request = _render_a2_request(item_dir, a2_prompt, a1_prediction)
            a2_request = replace(
                a2_request,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
            write_json(
                a2_dir / "request.json",
                _request_record(
                    a2_request, provider=args.provider, model=args.model, stage="A2"
                ),
            )
            try:
                a2_prediction, a2_parser = _call_and_archive(
                    args.provider,
                    args.model,
                    a2_request,
                    a2_dir,
                    base_url=args.base_url,
                )
                public_geometry = read_json(item_dir / "public_geometry.json")
                a2_evaluation = evaluate_pose_stage(
                    master,
                    candidates,
                    public_geometry,
                    a1_prediction,
                    a2_prediction or {},
                    simulation_output_dir=a2_dir / "simulation",
                )
                if a2_parser["schema_errors"]:
                    a2_evaluation["pose_evaluation"]["l0_valid"] = False
                write_json(a2_dir / "evaluation.json", a2_evaluation)
                a2_status = a2_evaluation["pose_evaluation"]["status"]
            except Exception as exc:
                write_json(
                    a2_dir / "provider_or_evaluator_error.json",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
                a2_status = "provider_or_evaluator_failed"

        result = {
            "assembly_id": assembly_id,
            "status": "completed",
            "a0_pass": True,
            "a1_valid": a1_evaluation["l0_valid"],
            "a1_top1_typed_entity_hit": a1_evaluation["top1_typed_entity_hit"],
            "a1_top3_typed_entity_hit": a1_evaluation["top3_typed_entity_hit"],
            "a2_status": a2_status,
            "a2_pose_pass": (
                a2_evaluation["pose_evaluation"]["pose_pass"]
                if a2_evaluation
                else False
            ),
            "a2_physics_status": (
                (a2_evaluation.get("simulation") or {}).get("status")
                if a2_evaluation
                else None
            ),
            "usage": {
                "a1": a1_parser["usage"],
                "a2": a2_parser["usage"] if a2_parser else None,
            },
        }
        write_json(item_out / "result.json", result)
        manifest["items"].append(result)
        write_json(args.output_dir / "run_manifest.json", manifest)

    manifest["completed_at"] = _utc_now()
    manifest["counts"] = {
        status: sum(item["status"] == status for item in manifest["items"])
        for status in sorted({item["status"] for item in manifest["items"]})
    }
    manifest["usage"] = {
        key: sum(
            int(((item.get("usage") or {}).get(stage) or {}).get(key, 0))
            for item in manifest["items"]
            for stage in ("a1", "a2")
        )
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(args.output_dir / "run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
