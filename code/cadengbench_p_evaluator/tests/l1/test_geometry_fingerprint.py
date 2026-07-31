import hashlib
import re
from pathlib import Path

import cadquery as cq

from cadengbench_p_eval.geometry.fingerprint import geometry_fingerprint


def _export(tmp_path: Path, name: str, shape: cq.Workplane) -> Path:
    path = tmp_path / f"{name}.step"
    cq.exporters.export(shape, str(path))
    return path


def test_geometry_fingerprint_ignores_step_header_metadata(tmp_path: Path) -> None:
    original = _export(tmp_path, "original", cq.Workplane("XY").box(10, 8, 2))
    metadata_only_copy = tmp_path / "metadata_only.step"
    rewritten, replacements = re.subn(
        r"(FILE_NAME\s*\()\s*'[^']*'",
        r"\1'metadata-only-copy'",
        original.read_text(encoding="utf-8"),
        count=1,
    )
    assert replacements == 1
    metadata_only_copy.write_text(rewritten, encoding="utf-8")
    assert (
        hashlib.sha256(original.read_bytes()).digest()
        != hashlib.sha256(metadata_only_copy.read_bytes()).digest()
    )

    assert geometry_fingerprint(
        original,
        surface_sample_count=128,
        seed=7,
    ) == geometry_fingerprint(
        metadata_only_copy,
        surface_sample_count=128,
        seed=7,
    )


def test_geometry_fingerprint_changes_with_shape(tmp_path: Path) -> None:
    shallow = _export(tmp_path, "shallow", cq.Workplane("XY").box(10, 8, 2))
    deep = _export(tmp_path, "deep", cq.Workplane("XY").box(10, 8, 3))

    assert geometry_fingerprint(
        shallow,
        surface_sample_count=128,
        seed=7,
    ) != geometry_fingerprint(
        deep,
        surface_sample_count=128,
        seed=7,
    )


def test_independent_exports_of_same_shape_have_same_fingerprint(
    tmp_path: Path,
) -> None:
    first = _export(tmp_path, "first", cq.Workplane("XY").box(10, 8, 2))
    second = _export(tmp_path, "second", cq.Workplane("XY").box(10, 8, 2))

    assert geometry_fingerprint(
        first,
        surface_sample_count=128,
        seed=7,
    ) == geometry_fingerprint(
        second,
        surface_sample_count=128,
        seed=7,
    )
