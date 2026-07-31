"""Deterministic aggregation of per-item evaluator results."""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import (
    Status,
    evaluation_result_from_dict,
    passed_value,
    reduce_required_status,
)
from .integrity import RESULT_DIGEST_NAME, verify_evidence_manifest
from .io import read_json, sha256_file, sha256_json, write_json

RUN_MANIFEST_SCHEMA = "cadengbench-p.run-manifest.v2"
RESULT_SCHEMA = "cadengbench-p.evaluation-result.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ITEM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REFERENCE_DIAGNOSTIC_METRICS = (
    "chamfer_mm",
    "chamfer_normalized_by_target_diagonal",
    "surface_precision",
    "surface_recall",
    "surface_f1",
    "hd95_mm",
    "hd95_normalized_by_target_diagonal",
    "voxel_iou",
    "voxel_dice",
    "volume_relative_error",
    "surface_area_relative_error",
    "centroid_distance_mm",
    "centroid_distance_normalized_by_target_diagonal",
    "bbox_dimension_max_error_mm",
    "bbox_dimension_max_error_normalized_by_target_diagonal",
)


def _layer_index(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    layers = result.get("layers")
    if not isinstance(layers, list):
        raise ValueError("evaluation result lacks a layers list")
    index: dict[str, Mapping[str, Any]] = {}
    for layer in layers:
        if not isinstance(layer, Mapping) or not isinstance(layer.get("layer"), str):
            raise ValueError("evaluation result contains a malformed layer")
        name = str(layer["layer"])
        if name in index:
            raise ValueError(f"duplicate layer {name}")
        index[name] = layer
    return index


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _required_check_pass_rate(layer: Mapping[str, Any]) -> float | None:
    checks = [
        check
        for check in layer.get("checks", [])
        if isinstance(check, Mapping)
        and bool(check.get("required", True))
        and check.get("status") != Status.NOT_APPLICABLE.value
    ]
    if not checks:
        return None
    return sum(check.get("status") == Status.PASSED.value for check in checks) / len(
        checks
    )


def benchmark_owned_not_scoreable(check: Mapping[str, Any]) -> bool:
    if check.get("status") != Status.NOT_SCOREABLE.value:
        return False
    check_id = str(check.get("check_id", ""))
    summary = str(check.get("summary", ""))
    return (
        check_id == "L2.feasibility_contract"
        and summary.startswith("invalid feasibility contract:")
    ) or (
        check_id.startswith("L2.solver.")
        and summary.startswith("dataset qualification failed:")
    )


def _benchmark_excluded_layer(layer: Mapping[str, Any]) -> bool:
    if layer.get("status") != Status.NOT_SCOREABLE.value:
        return False
    required_not_scoreable = [
        check
        for check in layer.get("checks", [])
        if isinstance(check, Mapping)
        and bool(check.get("required", True))
        and check.get("status") == Status.NOT_SCOREABLE.value
    ]
    return bool(required_not_scoreable) and all(
        benchmark_owned_not_scoreable(check) for check in required_not_scoreable
    )


def aggregate_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(results)
    item_ids: set[str] = set()
    status_counts: dict[str, int] = {status.value: 0 for status in Status}
    layer_counts = {
        layer: {status.value: 0 for status in Status} for layer in ("L0", "L1", "L2")
    }
    layer_required_check_rates: dict[str, list[float]] = {
        layer: [] for layer in ("L0", "L1", "L2")
    }
    source_counts: dict[str, int] = {}
    source_excluded: dict[str, int] = {}
    source_passes: dict[str, int] = {}
    protocol_counts: dict[str, int] = {}
    protocol_excluded: dict[str, int] = {}
    protocol_passes: dict[str, int] = {}
    l2_contract_counts: dict[str, int] = {}
    l2_contract_excluded: dict[str, int] = {}
    l2_contract_passes: dict[str, int] = {}
    check_counts: dict[str, dict[str, int]] = {}
    evaluator_error_items = 0
    required_not_scoreable_items = 0
    benchmark_excluded_items = 0
    layer_excluded_items = {layer: 0 for layer in ("L0", "L1", "L2")}
    diagnostic_item_count = 0
    exact_reconstruction_item_count = 0
    reference_values: dict[str, list[float]] = {
        key: [] for key in _REFERENCE_DIAGNOSTIC_METRICS
    }
    for record in records:
        item_id = record.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in item_ids:
            raise ValueError("aggregation requires unique nonempty item IDs")
        item_ids.add(item_id)
        overall = str(record.get("overall_status"))
        if overall not in status_counts:
            raise ValueError(f"invalid overall status for {item_id}: {overall}")
        status_counts[overall] += 1
        source = str(record.get("identity", {}).get("source_family", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        source_passes[source] = source_passes.get(source, 0) + (overall == "passed")
        protocol = str(record.get("protocol", "unknown"))
        protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        protocol_passes[protocol] = protocol_passes.get(protocol, 0) + (
            overall == "passed"
        )
        l2_contract_mode = str(
            record.get("identity", {}).get("l2_contract_mode", "unknown")
        )
        l2_contract_counts[l2_contract_mode] = (
            l2_contract_counts.get(l2_contract_mode, 0) + 1
        )
        layers = _layer_index(record)
        excluded_layers = {
            name
            for name, layer in layers.items()
            if _benchmark_excluded_layer(layer)
        }
        required_not_scoreable_checks = [
            check
            for layer in layers.values()
            for check in layer.get("checks", [])
            if isinstance(check, Mapping)
            and bool(check.get("required", True))
            and check.get("status") == Status.NOT_SCOREABLE.value
        ]
        benchmark_excluded = (
            overall == Status.NOT_SCOREABLE.value
            and bool(required_not_scoreable_checks)
            and all(
                benchmark_owned_not_scoreable(check)
                for check in required_not_scoreable_checks
            )
        )
        benchmark_excluded_items += int(benchmark_excluded)
        source_excluded[source] = source_excluded.get(source, 0) + int(
            benchmark_excluded
        )
        protocol_excluded[protocol] = protocol_excluded.get(protocol, 0) + int(
            benchmark_excluded
        )
        l2_contract_excluded[l2_contract_mode] = l2_contract_excluded.get(
            l2_contract_mode, 0
        ) + int("L2" in excluded_layers)
        l2_status = str((layers.get("L2") or {}).get("status", "not_reached"))
        l2_contract_passes[l2_contract_mode] = l2_contract_passes.get(
            l2_contract_mode, 0
        ) + int(l2_status == Status.PASSED.value)
        item_has_evaluator_error = overall == Status.EVALUATOR_ERROR.value
        item_has_required_not_scoreable = False
        for layer_name in ("L0", "L1", "L2"):
            layer = layers.get(layer_name)
            status = "not_reached" if layer is None else str(layer.get("status"))
            if status not in layer_counts[layer_name]:
                raise ValueError(f"invalid {layer_name} status for {item_id}: {status}")
            layer_counts[layer_name][status] += 1
            layer_excluded_items[layer_name] += int(layer_name in excluded_layers)
            item_has_evaluator_error = item_has_evaluator_error or (
                status == Status.EVALUATOR_ERROR.value
            )
            if layer is None:
                continue
            if status in {Status.PASSED.value, Status.FAILED.value}:
                required_check_rate = _required_check_pass_rate(layer)
                if required_check_rate is not None:
                    layer_required_check_rates[layer_name].append(
                        required_check_rate
                    )
            for check in layer.get("checks", []):
                if not isinstance(check, Mapping):
                    continue
                check_id = str(check.get("check_id"))
                check_status = str(check.get("status"))
                if check_status not in status_counts:
                    continue
                item_has_evaluator_error = item_has_evaluator_error or (
                    check_status == Status.EVALUATOR_ERROR.value
                )
                item_has_required_not_scoreable = item_has_required_not_scoreable or (
                    check_status == Status.NOT_SCOREABLE.value
                    and bool(check.get("required", True))
                    and not benchmark_owned_not_scoreable(check)
                )
                counts = check_counts.setdefault(
                    check_id, {status.value: 0 for status in Status}
                )
                counts[check_status] += 1
        l1 = layers.get("L1")
        l1_metrics = l1.get("metrics") if isinstance(l1, Mapping) else None
        diagnostics = (
            l1_metrics.get("reference_geometry_diagnostics")
            if isinstance(l1_metrics, Mapping)
            else None
        )
        if isinstance(diagnostics, Mapping) and diagnostics.get("status") == "computed":
            diagnostic_item_count += 1
            exact_reconstruction_item_count += int(
                diagnostics.get("score_bearing") is True
            )
            for key in _REFERENCE_DIAGNOSTIC_METRICS:
                value = diagnostics.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    reference_values[key].append(float(value))
        evaluator_error_items += int(item_has_evaluator_error)
        required_not_scoreable_items += int(item_has_required_not_scoreable)
    item_count = len(records)
    source_rates = {
        source: _ratio(source_passes[source], count - source_excluded.get(source, 0))
        for source, count in sorted(source_counts.items())
    }
    defined_source_rates = [
        value for value in source_rates.values() if value is not None
    ]
    publication_valid = (
        evaluator_error_items == 0
        and required_not_scoreable_items == 0
        and status_counts[Status.NOT_SCOREABLE.value] == benchmark_excluded_items
        and status_counts[Status.NOT_APPLICABLE.value] == 0
        and status_counts[Status.EVALUATOR_ERROR.value] == 0
    )
    scoreable_item_count = item_count - benchmark_excluded_items
    aggregate = {
        "schema_version": "cadengbench-p.aggregate-result.v3",
        "item_count": item_count,
        "overall": {
            "status_counts": status_counts,
            "raw_strict_pass_rate": _ratio(status_counts["passed"], item_count),
            "strict_pass_rate": _ratio(status_counts["passed"], scoreable_item_count),
            "scoreable_item_count": scoreable_item_count,
            "benchmark_excluded_item_count": benchmark_excluded_items,
            "evaluator_error_rate": _ratio(
                status_counts["evaluator_error"], item_count
            ),
            "source_macro_strict_pass_rate": (
                sum(defined_source_rates) / len(defined_source_rates)
                if defined_source_rates
                else None
            ),
            "publication_valid": publication_valid,
            "publication_strict_pass_rate": (
                _ratio(status_counts["passed"], scoreable_item_count)
                if publication_valid
                else None
            ),
            "evaluator_error_item_count": evaluator_error_items,
            "required_not_scoreable_item_count": required_not_scoreable_items,
        },
        "layers": {
            layer: {
                "status_counts": counts,
                "raw_strict_pass_rate": _ratio(counts["passed"], item_count),
                "strict_pass_rate": _ratio(
                    counts["passed"], item_count - layer_excluded_items[layer]
                ),
                "scoreable_item_count": item_count - layer_excluded_items[layer],
                "benchmark_excluded_item_count": layer_excluded_items[layer],
                "reach_rate": _ratio(
                    counts["passed"] + counts["failed"], item_count
                ),
                "scoreable_rate": _ratio(
                    counts["passed"] + counts["failed"], item_count
                ),
                "conditional_pass_rate": _ratio(
                    counts["passed"], counts["passed"] + counts["failed"]
                ),
                "mean_required_check_pass_rate_on_reached": (
                    statistics.fmean(layer_required_check_rates[layer])
                    if layer_required_check_rates[layer]
                    else None
                ),
            }
            for layer, counts in layer_counts.items()
        },
        "sources": {
            source: {
                "item_count": count,
                "scoreable_item_count": count - source_excluded.get(source, 0),
                "strict_pass_rate": source_rates[source],
            }
            for source, count in sorted(source_counts.items())
        },
        "protocols": {
            protocol: {
                "item_count": count,
                "scoreable_item_count": count - protocol_excluded.get(protocol, 0),
                "strict_pass_rate": _ratio(
                    protocol_passes[protocol],
                    count - protocol_excluded.get(protocol, 0),
                ),
            }
            for protocol, count in sorted(protocol_counts.items())
        },
        "l2_authority": {
            mode: {
                "item_count": count,
                "scoreable_item_count": count - l2_contract_excluded.get(mode, 0),
                "strict_pass_rate": _ratio(
                    l2_contract_passes[mode],
                    count - l2_contract_excluded.get(mode, 0),
                ),
            }
            for mode, count in sorted(l2_contract_counts.items())
        },
        "checks": dict(sorted(check_counts.items())),
        "reference_geometry_diagnostics": {
            "computed_item_count": diagnostic_item_count,
            "coverage_rate": _ratio(diagnostic_item_count, item_count),
            "exact_reconstruction_scoring_item_count": (
                exact_reconstruction_item_count
            ),
            "metric_means": {
                key: statistics.fmean(values) if values else None
                for key, values in reference_values.items()
            },
            "metric_item_counts": {
                key: len(values) for key, values in reference_values.items()
            },
            "score_bearing_for_general_tasks": False,
        },
        "llm_judge_used": False,
    }
    aggregate["aggregate_hash"] = sha256_json(aggregate)
    return aggregate


def _validate_run_manifest(value: Any) -> tuple[dict[str, Any], set[str]]:
    if not isinstance(value, Mapping):
        raise ValueError("run manifest must be an object")
    if value.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported run manifest schema")
    identity = value.get("identity")
    inventory = value.get("submission_inventory")
    if not isinstance(identity, Mapping) or not isinstance(inventory, list):
        raise ValueError("run manifest lacks immutable identity or inventory")
    if value.get("run_hash") != sha256_json(identity):
        raise ValueError("run manifest failed its identity hash validation")
    if sha256_json(inventory) != identity.get("submission_inventory_hash"):
        raise ValueError("run manifest inventory failed its hash validation")
    if value.get("item_count") != len(inventory):
        raise ValueError("run manifest item_count does not match its inventory")
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise ValueError(f"run manifest flattened identity {key!r} differs")
    if value.get("model_id") != identity.get("model_id"):
        raise ValueError("run manifest model_id differs from immutable identity")

    item_ids: set[str] = set()
    for index, record in enumerate(inventory):
        if not isinstance(record, Mapping) or record.get("index") != index:
            raise ValueError("run manifest inventory order is malformed")
        item_id = record.get("item_id")
        relative = record.get("path")
        state = record.get("state")
        digest = record.get("sha256")
        size = record.get("size_bytes")
        if (
            not isinstance(item_id, str)
            or _ITEM_ID_PATTERN.fullmatch(item_id) is None
            or item_id in item_ids
        ):
            raise ValueError("run manifest inventory has unsafe or duplicate item IDs")
        item_ids.add(item_id)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"inventory path is malformed for {item_id}")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != relative
        ):
            raise ValueError(f"inventory path is unsafe for {item_id}")
        if state == "present":
            if (
                not isinstance(digest, str)
                or _SHA256_PATTERN.fullmatch(digest) is None
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ValueError(f"present inventory record is malformed for {item_id}")
        elif state == "missing":
            if digest != "missing" or size is not None:
                raise ValueError(f"missing inventory record is malformed for {item_id}")
        else:
            raise ValueError(f"inventory state is invalid for {item_id}")
    return dict(identity), item_ids


def _overall_status(layer_statuses: set[Status], has_evaluator_error: bool) -> Status:
    if has_evaluator_error or Status.EVALUATOR_ERROR in layer_statuses:
        return Status.EVALUATOR_ERROR
    if Status.FAILED in layer_statuses:
        return Status.FAILED
    if Status.NOT_SCOREABLE in layer_statuses:
        return Status.NOT_SCOREABLE
    if Status.NOT_REACHED in layer_statuses or Status.NOT_APPLICABLE in layer_statuses:
        return Status.NOT_SCOREABLE
    return Status.PASSED


def _validate_result_record(
    value: Any,
    *,
    item_id: str,
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"result for {item_id} must be an object")
    try:
        result = evaluation_result_from_dict(dict(value))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"result for {item_id} is malformed: {exc}") from exc
    if result.schema_version != RESULT_SCHEMA or result.item_id != item_id:
        raise ValueError(f"result identity or schema differs for {item_id}")
    if result.llm_judge_used:
        raise ValueError(f"result for {item_id} illegally declares an LLM judge")
    if tuple(layer.layer for layer in result.layers) != ("L0", "L1", "L2"):
        raise ValueError(f"result for {item_id} must contain exactly L0/L1/L2")
    raw_layers = value.get("layers")
    if not isinstance(raw_layers, list) or len(raw_layers) != len(result.layers):
        raise ValueError(f"result layer roster is malformed for {item_id}")
    has_evaluator_error = False
    for raw_layer, layer in zip(raw_layers, result.layers, strict=True):
        if not isinstance(raw_layer, Mapping):
            raise ValueError(f"result contains a malformed layer for {item_id}")
        if raw_layer.get("passed") != passed_value(layer.status):
            raise ValueError(f"layer passed field is inconsistent for {item_id}")
        if reduce_required_status(layer.checks) is not layer.status:
            raise ValueError(f"layer status does not reduce from checks for {item_id}")
        raw_checks = raw_layer.get("checks")
        if not isinstance(raw_checks, list) or len(raw_checks) != len(layer.checks):
            raise ValueError(f"check roster is malformed for {item_id}")
        check_ids: set[str] = set()
        for raw_check, check in zip(raw_checks, layer.checks, strict=True):
            if not isinstance(raw_check, Mapping) or check.check_id in check_ids:
                raise ValueError(
                    f"check record is malformed or duplicated for {item_id}"
                )
            check_ids.add(check.check_id)
            if raw_check.get("passed") != passed_value(check.status):
                raise ValueError(f"check passed field is inconsistent for {item_id}")
            has_evaluator_error = (
                has_evaluator_error or check.status is Status.EVALUATOR_ERROR
            )
    expected_overall = _overall_status(
        {layer.status for layer in result.layers}, has_evaluator_error
    )
    if result.overall_status is not expected_overall:
        raise ValueError(f"overall status does not reduce from layers for {item_id}")
    if value.get("passed") != passed_value(result.overall_status):
        raise ValueError(f"overall passed field is inconsistent for {item_id}")

    for key in ("evaluator_build_hash", "environment_hash"):
        if result.identity.get(key) != run_identity.get(key):
            raise ValueError(f"result {key} differs from run identity for {item_id}")
    if result.environment.get("environment_hash") != run_identity.get(
        "environment_hash"
    ):
        raise ValueError(f"result environment differs from run identity for {item_id}")
    return dict(value)


def aggregate_run(
    run_root: Path, *, expected_item_ids: set[str] | None = None
) -> dict[str, Any]:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("aggregation requires the immutable run manifest")
    manifest = read_json(manifest_path)
    run_identity, manifest_item_ids = _validate_run_manifest(manifest)
    if expected_item_ids is not None and expected_item_ids != manifest_item_ids:
        raise ValueError("requested aggregation roster differs from the run manifest")
    expected = manifest_item_ids
    items_root = run_root / "items"
    if items_root.is_symlink():
        raise ValueError("run items root cannot be a symlink")
    if items_root.is_dir() and any(path.is_symlink() for path in items_root.iterdir()):
        raise ValueError("run item directories cannot be symlinks")
    observed_directories = (
        {path.name for path in items_root.iterdir() if path.is_dir()}
        if items_root.is_dir()
        else set()
    )
    if observed_directories != expected:
        raise ValueError(
            "item output directories differ from the frozen run roster; "
            f"missing={sorted(expected - observed_directories)}, "
            f"extra={sorted(observed_directories - expected)}"
        )
    result_paths = [
        items_root / item_id / "result.json" for item_id in sorted(expected)
    ]
    missing_results = {path.parent.name for path in result_paths if not path.is_file()}
    if missing_results:
        raise ValueError(
            f"run is incomplete; missing results for {sorted(missing_results)}"
        )
    records: list[dict[str, Any]] = []
    for path in result_paths:
        item_root = path.parent
        digest_path = item_root / RESULT_DIGEST_NAME
        if not digest_path.is_file():
            raise ValueError(f"result digest is missing for {item_root.name}")
        digest = digest_path.read_text(encoding="ascii").strip()
        if _SHA256_PATTERN.fullmatch(digest) is None or sha256_file(path) != digest:
            raise ValueError(f"result digest validation failed for {item_root.name}")
        verify_evidence_manifest(item_root)
        records.append(
            _validate_result_record(
                read_json(path),
                item_id=item_root.name,
                run_identity=run_identity,
            )
        )
    aggregate = aggregate_results(records)
    aggregate["run_hash"] = manifest["run_hash"]
    aggregate.pop("aggregate_hash", None)
    aggregate["aggregate_hash"] = sha256_json(aggregate)
    output_root = run_root / "aggregate"
    write_json(output_root / "metrics.json", aggregate)
    write_json(run_root / "aggregate.json", aggregate)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "failures.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["item_id", "source_family", "overall_status", "first_failure"],
        )
        writer.writeheader()
        for record in records:
            if record.get("overall_status") == "passed":
                continue
            first_failure = ""
            for layer in record.get("layers", []):
                for check in layer.get("checks", []):
                    if check.get("status") in {
                        "failed",
                        "evaluator_error",
                        "not_scoreable",
                    }:
                        first_failure = str(
                            check.get("failure_code") or check.get("check_id") or ""
                        )
                        break
                if first_failure:
                    break
            writer.writerow(
                {
                    "item_id": record.get("item_id"),
                    "source_family": record.get("identity", {}).get("source_family"),
                    "overall_status": record.get("overall_status"),
                    "first_failure": first_failure,
                }
            )
    return aggregate
