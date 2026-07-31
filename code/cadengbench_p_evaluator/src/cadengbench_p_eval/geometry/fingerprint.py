"""Kernel- and profile-frozen fingerprints for imported STEP artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .metrics import sample_brep_surface
from .step import StepGeometry, load_step

FINGERPRINT_VERSION = "cadengbench-p.brep-geometry-fingerprint.v2"


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _quantize(values: np.ndarray, quantum: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if not np.isfinite(source).all():
        raise ValueError("geometry fingerprint input contains non-finite coordinates")
    scaled = source / quantum
    integer_limit = float(np.iinfo(np.int64).max - 1)
    if np.any(np.abs(scaled) > integer_limit):
        raise ValueError("geometry coordinates exceed fingerprint integer range")
    return np.rint(scaled).astype(np.int64)


def _canonical_point_bytes(points: np.ndarray, quantum: float) -> bytes:
    canonical = _quantize(points, quantum)
    point_order = np.lexsort(tuple(canonical[:, index] for index in range(2, -1, -1)))
    canonical = np.ascontiguousarray(canonical[point_order], dtype="<i8")
    return canonical.tobytes(order="C")


def _quantized_scalar(value: float, quantum: float) -> int:
    if not math.isfinite(value):
        raise ValueError("geometry fingerprint input contains a non-finite scalar")
    return round(value / quantum)


def geometry_fingerprint(
    source: str | Path | StepGeometry,
    *,
    surface_sample_count: int = 512,
    seed: int = 0,
    coordinate_quantization_mm: float = 1e-7,
) -> str:
    """Return a SHA-256 digest of deterministic B-Rep geometry evidence.

    ``source`` may be a STEP path or an already imported :class:`StepGeometry`.
    The digest excludes STEP serialization metadata and uses no mesh. It remains
    intentionally bound to the frozen CAD kernel, B-Rep topology, and sampling
    profile. It detects parameter response; it is not a representation-independent
    proof that arbitrary B-Reps are equivalent.
    """

    if (
        isinstance(surface_sample_count, bool)
        or not isinstance(surface_sample_count, int)
        or surface_sample_count <= 0
    ):
        raise ValueError("surface_sample_count must be a positive integer")
    quantum = _finite_positive(
        coordinate_quantization_mm,
        "coordinate_quantization_mm",
    )
    geometry = source if isinstance(source, StepGeometry) else load_step(Path(source))
    samples = sample_brep_surface(
        geometry,
        count=surface_sample_count,
        seed=seed,
    )

    length_scale = max(geometry.bbox.diagonal, 1.0)
    area_quantum = max(quantum * length_scale, quantum**2)
    volume_quantum = max(quantum * length_scale**2, quantum**3)
    metadata = {
        "contract": FINGERPRINT_VERSION,
        "coordinate_quantization_mm_hex": quantum.hex(),
        "surface_sample_count": surface_sample_count,
        "seed": int(seed),
        "bbox_minimum": _quantize(
            np.asarray(geometry.bbox.minimum, dtype=np.float64), quantum
        ).tolist(),
        "bbox_maximum": _quantize(
            np.asarray(geometry.bbox.maximum, dtype=np.float64), quantum
        ).tolist(),
        "centroid": _quantize(
            np.asarray(geometry.centroid_mm, dtype=np.float64), quantum
        ).tolist(),
        "surface_area": _quantized_scalar(
            geometry.surface_area_mm2,
            area_quantum,
        ),
        "volume": _quantized_scalar(geometry.volume_mm3, volume_quantum),
        "topology": geometry.topology,
    }
    encoded_metadata = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    digest = hashlib.sha256()
    digest.update(encoded_metadata)
    digest.update(b"\0")
    digest.update(_canonical_point_bytes(samples.points, quantum))
    return digest.hexdigest()
