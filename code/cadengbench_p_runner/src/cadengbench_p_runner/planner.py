"""Freeze an immutable request roster before candidate inference."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .io import read_json, semantic_hash, sha256_file, sha256_json, write_json
from .materializer import INPUT_PROFILES, materialize_request
from .model_registry import ModelSpec, load_model_spec

RUN_MANIFEST_SCHEMA = "cadengbench-p.candidate-run-manifest.v1"
RUN_MANIFEST_V2_SCHEMA = "cadengbench-p.candidate-run-manifest.v2"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _runner_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runner_build_hash() -> str:
    root = _runner_root()
    records = []
    directory = root / "src"
    for path in sorted(
        value
        for value in directory.rglob("*")
        if value.is_file() and value.suffix in {".py", ".json"}
    ):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return sha256_json(records)


def dependency_lock_hash() -> str:
    lock_path = _runner_root() / "uv.lock"
    if not lock_path.is_file():
        raise ValueError(
            "runner uv.lock is absent; freeze dependencies before planning"
        )
    return sha256_file(lock_path)


def _select_items(items_root: Path, item_ids: Sequence[str] | None) -> list[Path]:
    if not items_root.is_dir():
        raise ValueError(f"items root is not a directory: {items_root}")
    available = {path.name: path for path in items_root.iterdir() if path.is_dir()}
    if item_ids is None:
        selected = sorted(available)
    else:
        selected = list(item_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("item selection contains duplicates")
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"selected items are missing: {missing}")
    if not selected:
        raise ValueError("run roster is empty")
    return [available[item_id] for item_id in selected]


def _assert_empty_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"run output already contains files: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def plan_run(
    *,
    items_root: Path,
    model_spec_path: Path,
    output_root: Path,
    run_id: str,
    input_profile: str,
    task_kind: str = "zero_to_cad",
    sample_count: int = 1,
    max_attempts: int = 3,
    request_timeout_seconds: int = 180,
    max_retry_elapsed_seconds: int = 900,
    item_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize and hash every request without calling a model endpoint."""

    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be a short filesystem-safe identifier")
    if input_profile not in INPUT_PROFILES:
        raise ValueError(f"unsupported input profile: {input_profile}")
    if task_kind not in {"zero_to_cad", "functional_edit"}:
        raise ValueError(f"unsupported task kind: {task_kind}")
    sample_count = _positive_int(sample_count, "sample_count")
    max_attempts = _positive_int(max_attempts, "max_attempts")
    request_timeout_seconds = _positive_int(
        request_timeout_seconds, "request_timeout_seconds"
    )
    max_retry_elapsed_seconds = _positive_int(
        max_retry_elapsed_seconds, "max_retry_elapsed_seconds"
    )
    model = load_model_spec(model_spec_path)
    if input_profile != "text" and not model.image_capable:
        raise ValueError("selected input profile requires an image-capable model")
    if not model.text_capable:
        raise ValueError("CADEngBench-P requests require text capability")
    _assert_empty_output(output_root)

    dependency_digest = dependency_lock_hash()
    item_roots = _select_items(items_root, item_ids)
    records: list[dict[str, Any]] = []
    release_ids: set[str] = set()
    for item_root in item_roots:
        request = materialize_request(
            item_root,
            run_root=output_root,
            input_profile=input_profile,
            task_kind=task_kind,
        )
        request_path = output_root / "requests" / f"{item_root.name}.json"
        write_json(request_path, request)
        release_ids.add(str(request["dataset"]["release_id"]))
        records.append(
            {
                "item_id": item_root.name,
                "request_path": request_path.relative_to(output_root).as_posix(),
                "request_hash": request["request_hash"],
            }
        )
    if len(release_ids) != 1:
        raise ValueError(f"run roster spans multiple dataset releases: {release_ids}")

    manifest: dict[str, Any] = {
        "schema_version": (
            RUN_MANIFEST_SCHEMA
            if task_kind == "zero_to_cad"
            else RUN_MANIFEST_V2_SCHEMA
        ),
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "release_id": next(iter(release_ids)),
            "item_count": len(records),
            "item_roster_hash": sha256_json([record["item_id"] for record in records]),
        },
        "input_profile": input_profile,
        "sample_count": sample_count,
        "model": model.as_public_dict(),
        "execution": {
            "max_attempts": max_attempts,
            "request_timeout_seconds": request_timeout_seconds,
            "max_retry_elapsed_seconds": max_retry_elapsed_seconds,
            "max_concurrency": model.max_concurrency,
            "requests_per_minute": model.requests_per_minute,
            "retry_policy": "classified_exponential_backoff_with_jitter.v1",
        },
        "requests": records,
        "runner": {
            "version": __version__,
            "build_hash": runner_build_hash(),
            "python": sys.version.split()[0],
            "dependency_lock_sha256": dependency_digest,
        },
    }
    if task_kind != "zero_to_cad":
        manifest["task_kind"] = task_kind
    manifest["manifest_hash"] = semantic_hash(manifest, "manifest_hash")
    write_json(output_root / "run_manifest.json", manifest)
    return manifest


def load_run_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "run_manifest.json"
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") not in {
        RUN_MANIFEST_SCHEMA,
        RUN_MANIFEST_V2_SCHEMA,
    }:
        raise ValueError(f"{path} is not a supported candidate-run manifest")
    if semantic_hash(value, "manifest_hash") != value.get("manifest_hash"):
        raise ValueError("run manifest hash is invalid")
    required = {
        "schema_version",
        "run_id",
        "created_at",
        "dataset",
        "input_profile",
        "sample_count",
        "model",
        "execution",
        "requests",
        "runner",
        "manifest_hash",
    }
    if value["schema_version"] == RUN_MANIFEST_V2_SCHEMA:
        required.add("task_kind")
    if set(value) != required:
        raise ValueError("run manifest fields differ from the frozen contract")
    try:
        datetime.fromisoformat(str(value["created_at"]))
    except ValueError as exc:
        raise ValueError("run manifest created_at is invalid") from exc
    if not _RUN_ID.fullmatch(str(value["run_id"])):
        raise ValueError("run manifest run_id is invalid")
    if value.get("input_profile") not in INPUT_PROFILES:
        raise ValueError("run manifest input profile is invalid")
    task_kind = value.get("task_kind", "zero_to_cad")
    if task_kind not in {"zero_to_cad", "functional_edit"}:
        raise ValueError("run manifest task kind is invalid")
    sample_count = value.get("sample_count")
    _positive_int(sample_count, "sample_count")
    requests = value.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("run manifest request roster is empty")
    if len(
        {record.get("item_id") for record in requests if isinstance(record, dict)}
    ) != len(requests):
        raise ValueError("run manifest item IDs are malformed or duplicated")
    for record in requests:
        if not isinstance(record, dict) or set(record) != {
            "item_id",
            "request_path",
            "request_hash",
        }:
            raise ValueError("run manifest request record is malformed")
        if record["request_path"] != f"requests/{record['item_id']}.json":
            raise ValueError("run manifest request path is noncanonical")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["request_hash"])):
            raise ValueError("run manifest request hash is malformed")
    dataset = value.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {
        "release_id",
        "item_count",
        "item_roster_hash",
    }:
        raise ValueError("run manifest dataset record is malformed")
    item_ids = [record["item_id"] for record in requests]
    if dataset["item_count"] != len(item_ids) or dataset[
        "item_roster_hash"
    ] != sha256_json(item_ids):
        raise ValueError("run manifest dataset roster binding is invalid")
    model = value.get("model")
    if not isinstance(model, dict):
        raise ValueError("run manifest model record is absent")
    semantic_model = dict(model)
    recorded_spec_hash = semantic_model.pop("spec_hash", None)
    if sha256_json(semantic_model) != recorded_spec_hash:
        raise ValueError("frozen model spec hash is invalid")
    parsed_model = model_from_public_record(model)
    execution = value.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "max_attempts",
        "request_timeout_seconds",
        "max_retry_elapsed_seconds",
        "max_concurrency",
        "requests_per_minute",
        "retry_policy",
    }:
        raise ValueError("run manifest execution record is malformed")
    for field in (
        "max_attempts",
        "request_timeout_seconds",
        "max_retry_elapsed_seconds",
        "max_concurrency",
        "requests_per_minute",
    ):
        _positive_int(execution[field], field)
    if execution["retry_policy"] != "classified_exponential_backoff_with_jitter.v1":
        raise ValueError("run manifest retry policy is unsupported")
    if (
        execution["max_concurrency"] != parsed_model.max_concurrency
        or execution["requests_per_minute"] != parsed_model.requests_per_minute
    ):
        raise ValueError("run execution limits differ from the frozen model spec")
    runner = value.get("runner")
    if not isinstance(runner, dict) or set(runner) != {
        "version",
        "build_hash",
        "python",
        "dependency_lock_sha256",
    }:
        raise ValueError("run manifest runner record is malformed")
    for field in ("build_hash", "dependency_lock_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(runner[field])):
            raise ValueError(f"run manifest runner {field} is malformed")
    return value


def model_from_public_record(value: dict[str, Any]) -> ModelSpec:
    model = dict(value)
    model.pop("spec_hash", None)
    # Reparse the exact frozen semantic fields instead of the original registry file.
    temporary = {
        "schema_version": model["schema_version"],
        "model_label": model["model_label"],
        "provider": model["provider"],
        "protocol": model["protocol"],
        "model_id": model["model_id"],
        "base_url": model["base_url"],
        "api_key_env": model["api_key_env"],
        "api_key_required": model["api_key_required"],
        "capabilities": model["capabilities"],
        "request_defaults": model["request_defaults"],
        "rate_limits": model["rate_limits"],
    }
    from .model_registry import parse_model_spec

    return parse_model_spec(temporary)


def model_from_manifest(manifest: dict[str, Any]) -> ModelSpec:
    return model_from_public_record(manifest["model"])
