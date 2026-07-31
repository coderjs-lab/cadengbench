"""Evaluator and environment provenance records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .io import sha256_json


def environment_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "cadquery",
        "cadquery-ocp",
        "vtk",
        "numpy",
        "scipy",
        "jsonschema",
        "z3-solver",
        "gmsh",
        "httpx",
        "matplotlib",
        "pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    package_root = Path(__file__).resolve().parents[2]
    lockfile = package_root / "uv.lock"
    record = {
        "evaluator_version": __version__,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "uv_lock_sha256": (
            hashlib.sha256(lockfile.read_bytes()).hexdigest()
            if lockfile.is_file()
            else None
        ),
        "container_digest": os.environ.get("CADENGBENCH_CONTAINER_DIGEST"),
    }
    record["environment_hash"] = sha256_json(record)
    return record


def evaluator_build_hash() -> str:
    """Hash the installed evaluator Python sources in stable relative order."""

    package_root = Path(__file__).resolve().parent
    records = []
    for path in sorted(package_root.rglob("*.py")):
        records.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sha256_json(records)


def physics_evaluator_build_hash() -> str:
    """Hash only code and schema that can change CalculiX probe evidence."""

    package_root = Path(__file__).resolve().parent
    project_root = package_root.parents[1]
    schema = project_root / "schemas/l3_physics_card.v2.schema.json"
    if not schema.is_file():
        schema = package_root / "_schemas/l3_physics_card.v2.schema.json"
    paths = (
        package_root / "physics/calculix.py",
        package_root / "physics/subprocess_worker.py",
        package_root / "io.py",
        schema,
    )
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"physics identity input is missing: {path}")
        records.append(
            {
                "path": (
                    path.relative_to(package_root).as_posix()
                    if package_root in path.parents
                    else f"schemas/{path.name}"
                ),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sha256_json(records)
