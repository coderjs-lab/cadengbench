"""Versioned evaluator-profile loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json, sha256_json

PROFILE_SCHEMA = "cadengbench-p.evaluator-profile.v1"


def _positive_int(section: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"profile {key} must be an integer >= {minimum}")
    return value


def _nonnegative_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"profile {key} must be a non-negative integer")
    return value


def _finite_number(
    section: dict[str, Any], key: str, *, minimum: float, maximum: float | None = None
) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"profile {key} must be numeric")
    number = float(value)
    if (
        not math.isfinite(number)
        or number < minimum
        or (maximum is not None and number > maximum)
    ):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"profile {key} must be finite and {bounds}")
    return number


def _validate_profile(value: dict[str, Any]) -> None:
    execution = value["execution"]
    geometry = value["geometry"]
    dfm = value["dfm"]
    l2 = value["l2"]

    _positive_int(execution, "timeout_seconds")
    _nonnegative_int(execution, "worker_startup_grace_seconds")
    _positive_int(execution, "max_source_bytes")
    _positive_int(execution, "max_step_bytes")
    roots = execution.get("allowed_import_roots")
    if (
        not isinstance(roots, list)
        or not roots
        or not all(isinstance(root, str) and root for root in roots)
    ):
        raise ValueError("profile allowed_import_roots must be nonempty strings")

    if "evaluation_timeout_seconds" in geometry:
        _finite_number(geometry, "evaluation_timeout_seconds", minimum=1e-9)
    _finite_number(geometry, "linear_tessellation_tolerance_mm", minimum=1e-12)
    _positive_int(geometry, "surface_sample_count")
    _finite_number(geometry, "surface_tolerance_mm", minimum=1e-12)
    _positive_int(geometry, "voxel_resolution", minimum=4)
    _finite_number(
        geometry,
        "voxel_margin_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    seed = geometry.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("profile geometry seed must be an integer")

    _positive_int(dfm, "wall_sample_count", minimum=2)
    _positive_int(dfm, "wall_neighbor_count")
    _finite_number(dfm, "opposed_normal_cosine_min", minimum=0.0, maximum=1.0)
    _finite_number(dfm, "hole_axis_tolerance_degrees", minimum=0.0, maximum=180.0)

    for key in (
        "max_variants",
        "directed_points_per_parameter",
        "morris_levels",
        "morris_trajectories",
        "z3_timeout_ms",
    ):
        _positive_int(l2, key)
    l2_seed = l2.get("seed")
    if isinstance(l2_seed, bool) or not isinstance(l2_seed, int):
        raise ValueError("profile l2 seed must be an integer")


@dataclass(frozen=True)
class EvaluatorProfile:
    data: dict[str, Any]
    digest: str

    @property
    def profile_id(self) -> str:
        return str(self.data["profile_id"])

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"profile section {name!r} is missing")
        return value


def load_profile(path: Path) -> EvaluatorProfile:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(f"{path} is not a {PROFILE_SCHEMA} profile")
    if not isinstance(value.get("profile_id"), str) or not value["profile_id"]:
        raise ValueError("profile_id must be a nonempty string")
    for section in ("execution", "geometry", "dfm", "l2"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"profile section {section!r} is required")
    _validate_profile(value)
    return EvaluatorProfile(data=value, digest=sha256_json(value))
