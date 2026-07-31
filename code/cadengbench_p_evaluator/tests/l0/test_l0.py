from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.layers.l0 import evaluate_l0

from .conftest import TWO_SOLID_SOURCE, VALID_SOURCE

pytestmark = pytest.mark.integration


def test_l0_returns_shared_layer_contract(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    output = tmp_path / "l0"
    result = evaluate_l0(
        case=evaluator_case,
        submission_path=source_writer(VALID_SOURCE),
        output_dir=output,
    )

    assert result.layer == "L0"
    assert result.status is Status.PASSED
    assert len(result.checks) == 7
    assert all(check.status is Status.PASSED for check in result.checks)
    persisted = json.loads((output / "l0.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "passed"
    assert persisted["metrics"]["containment"]["security_boundary"] is False


def test_l0_preserves_model_failure_ownership(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    source = VALID_SOURCE.replace("result = build()", "result = build(DEFAULT_PARAMS)")
    result = evaluate_l0(
        case=evaluator_case,
        submission_path=source_writer(source),
        output_dir=tmp_path / "invalid",
    )

    assert result.status is Status.FAILED
    failed = next(check for check in result.checks if check.status is Status.FAILED)
    assert failed.check_id == "L0.submission.contract"
    assert failed.failure_code == "invalid_submission_interface"
    assert failed.evidence[-1]["failure_ownership"] == "model"


def test_l0_reports_body_count_mismatch_as_diagnostic(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    result = evaluate_l0(
        case=evaluator_case,
        submission_path=source_writer(TWO_SOLID_SOURCE),
        output_dir=tmp_path / "wrong-count",
    )

    diagnostic = next(
        check for check in result.checks
        if check.check_id == "L0.geometry.solid_count"
    )
    assert result.status is Status.PASSED
    assert diagnostic.required is False
    assert diagnostic.status is Status.FAILED
    assert diagnostic.failure_code == "solid_count_mismatch"
    assert diagnostic.expected == 1
    assert diagnostic.observed == 2


def test_l0_maps_invalid_case_to_evaluator_error(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    evaluator_case["parameter_system"]["interface_version"] = "wrong"
    result = evaluate_l0(
        case=evaluator_case,
        submission_path=source_writer(VALID_SOURCE),
        output_dir=tmp_path / "invalid-case",
    )

    assert result.status is Status.EVALUATOR_ERROR
    assert result.checks[0].status is Status.EVALUATOR_ERROR
    assert result.checks[0].evidence[-1]["failure_ownership"] == "evaluator"
