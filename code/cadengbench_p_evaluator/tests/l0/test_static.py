from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from cadengbench_p_eval.execution.static import (
    StaticValidationError,
    inspect_submission,
)

from .conftest import VALID_SOURCE


def _failure_code(
    source_writer: Callable[[str, str], Any],
    evaluator_case: dict[str, Any],
    source: str,
) -> str:
    path = source_writer(source, "invalid.py")
    with pytest.raises(StaticValidationError) as caught:
        inspect_submission(path, case=evaluator_case)
    return caught.value.failure_code


def test_accepts_exact_v1_contract(source_writer, evaluator_case) -> None:
    inspection = inspect_submission(source_writer(VALID_SOURCE), case=evaluator_case)

    assert inspection.interface_version == "cadengbench-p.cadquery-submission.v1"
    assert inspection.default_params == {"overall_width": 20.0}
    assert inspection.parameter_manifest["P-001"]["name"] == "overall_width"
    assert inspection.step_output_path == "model.step"


def test_accepts_ordinary_dictionary_get(source_writer, evaluator_case) -> None:
    source = VALID_SOURCE.replace(
        'values["overall_width"]',
        'values.get("overall_width")',
    )

    inspection = inspect_submission(source_writer(source), case=evaluator_case)

    assert inspection.default_params == {"overall_width": 20.0}


def test_accepts_ordinary_exception_handler(source_writer, evaluator_case) -> None:
    source = VALID_SOURCE.replace(
        '    return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)',
        """\
    try:
        return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)
    except Exception:
        raise
""".rstrip(),
    )

    inspection = inspect_submission(source_writer(source), case=evaluator_case)

    assert inspection.default_params == {"overall_width": 20.0}


@pytest.mark.parametrize(
    ("old", "new", "expected_code"),
    [
        (
            '"name": "overall_width"',
            '"name": "other_width"',
            "parameter_manifest_mismatch",
        ),
        (
            '"role": "principal_dimension"',
            '"role": "feature_dimension"',
            "parameter_manifest_mismatch",
        ),
        (
            '"overall_width": 20.0',
            '"overall_width": 21.0',
            "parameter_default_mismatch",
        ),
        (
            "def build(params=None):",
            "def build(params={}):",
            "invalid_submission_interface",
        ),
        (
            "result = build()",
            "result = build(DEFAULT_PARAMS)",
            "invalid_submission_interface",
        ),
        (
            'exporters.export(result, "model.step")',
            'exporters.export(result, "other.step")',
            "invalid_step_export",
        ),
    ],
)
def test_rejects_interface_mutations(
    source_writer, evaluator_case, old: str, new: str, expected_code: str
) -> None:
    assert (
        _failure_code(source_writer, evaluator_case, VALID_SOURCE.replace(old, new, 1))
        == expected_code
    )


def test_rejects_nonliteral_defaults(source_writer, evaluator_case) -> None:
    source = VALID_SOURCE.replace(
        'DEFAULT_PARAMS = {\n    "overall_width": 20.0,\n}',
        "DEFAULT_PARAMS = dict(overall_width=20.0)",
    )
    assert (
        _failure_code(source_writer, evaluator_case, source)
        == "invalid_submission_interface"
    )


def test_rejects_extra_public_default(source_writer, evaluator_case) -> None:
    source = VALID_SOURCE.replace(
        '"overall_width": 20.0,',
        '"overall_width": 20.0,\n    "unscored_width": 3.0,',
        1,
    )
    assert (
        _failure_code(source_writer, evaluator_case, source)
        == "parameter_default_mismatch"
    )


def test_rejects_forbidden_import(source_writer, evaluator_case) -> None:
    source = "import os\n" + VALID_SOURCE
    assert _failure_code(source_writer, evaluator_case, source) == "forbidden_import"


@pytest.mark.parametrize(
    "handler",
    [
        "except:",
        "except BaseException:",
        "except SystemExit:",
        "except KeyboardInterrupt:",
        "except GeneratorExit:",
        "except (Exception, BaseException):",
    ],
)
def test_rejects_deadline_swallowing_exception_handlers(
    source_writer, evaluator_case, handler: str
) -> None:
    source = VALID_SOURCE.replace(
        '    return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)',
        f"""\
    try:
        return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)
    {handler}
        raise
""".rstrip(),
    )

    assert (
        _failure_code(source_writer, evaluator_case, source)
        == "forbidden_exception_handler"
    )


def test_rejects_reference_geometry_import(source_writer, evaluator_case) -> None:
    source = VALID_SOURCE.replace(
        'return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)',
        'return cq.importers.importStep("reference.step")',
    )
    assert (
        _failure_code(source_writer, evaluator_case, source)
        == "forbidden_source_operation"
    )


def test_rejects_duplicate_literal_keys(source_writer, evaluator_case) -> None:
    source = VALID_SOURCE.replace(
        '"overall_width": 20.0,',
        '"overall_width": 20.0,\n    "overall_width": 20.0,',
        1,
    )
    assert (
        _failure_code(source_writer, evaluator_case, source)
        == "invalid_submission_interface"
    )
