from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

VALID_SOURCE = """\
import cadquery as cq
from cadquery import exporters

DEFAULT_PARAMS = {
    "overall_width": 20.0,
}

PARAMETER_MANIFEST = {
    "P-001": {
        "name": "overall_width",
        "unit": "mm",
        "role": "principal_dimension",
    },
}

def build(params=None):
    values = dict(DEFAULT_PARAMS)
    if params:
        values.update(params)
    return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)

result = build()
exporters.export(result, "model.step")
"""


TWO_SOLID_SOURCE = """\
import cadquery as cq
from cadquery import exporters

DEFAULT_PARAMS = {
    "overall_width": 20.0,
}

PARAMETER_MANIFEST = {
    "P-001": {
        "name": "overall_width",
        "unit": "mm",
        "role": "principal_dimension",
    },
}

def build(params=None):
    values = dict(DEFAULT_PARAMS)
    if params:
        values.update(params)
    first = cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0).val()
    second = (
        cq.Workplane("XY")
        .transformed(offset=(30.0, 0.0, 0.0))
        .box(5.0, 5.0, 5.0)
        .val()
    )
    return cq.Workplane("XY").newObject([first, second])

result = build()
exporters.export(result, "model.step")
"""


@pytest.fixture
def evaluator_case() -> dict[str, Any]:
    return {
        "item_id": "synthetic_l0",
        "task_contract": {
            "submission": {
                "code_submission_contract": {
                    "build_signature": "build(params=None)",
                    "interface_version": "cadengbench-p.cadquery-submission.v1",
                    "required_top_level_symbols": [
                        "DEFAULT_PARAMS",
                        "PARAMETER_MANIFEST",
                        "build",
                        "result",
                    ],
                    "step_output_path": "model.step",
                }
            },
            "solid_contract": {
                "connectivity": "one_connected_solid",
                "required_body_count": 1,
            },
        },
        "parameter_system": {
            "interface_version": "cadengbench-p.cadquery-submission.v1",
            "parameters": [
                {
                    "parameter_id": "P-001",
                    "engineering_name": "overall_width",
                    "nominal": 20.0,
                    "unit": "mm",
                    "role": "principal_dimension",
                }
            ],
            "relations": [],
        },
        "evaluator_profile": {
            "execution": {
                "timeout_seconds": 30,
                "worker_startup_grace_seconds": 45,
                "max_source_bytes": 100_000,
                "max_step_bytes": 16 * 1024 * 1024,
                "allowed_import_roots": ["cadquery", "math"],
            }
        },
    }


@pytest.fixture
def case_factory(
    evaluator_case: dict[str, Any],
) -> Callable[..., dict[str, Any]]:
    def make(*, body_count: int = 1, timeout_seconds: int = 30) -> dict[str, Any]:
        value = deepcopy(evaluator_case)
        value["task_contract"]["solid_contract"]["required_body_count"] = body_count
        value["evaluator_profile"]["execution"]["timeout_seconds"] = timeout_seconds
        return value

    return make


@pytest.fixture
def source_writer(tmp_path: Path) -> Callable[[str, str], Path]:
    def write(source: str, name: str = "submission.py") -> Path:
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        return path

    return write
