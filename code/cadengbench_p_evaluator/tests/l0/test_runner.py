from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest

from cadengbench_p_eval.execution import runner as execution_runner
from cadengbench_p_eval.execution.runner import run_submission

from .conftest import TWO_SOLID_SOURCE, VALID_SOURCE

pytestmark = pytest.mark.integration


def test_runs_valid_submission_and_independent_roundtrip(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    result = run_submission(
        source_path=source_writer(VALID_SOURCE),
        case=evaluator_case,
        output_dir=tmp_path / "run",
    )

    assert result["status"] == "passed"
    assert result["stages"]["step_reimport"]["status"] == "passed"
    assert result["stages"]["roundtrip"]["status"] == "passed"
    assert result["facts"]["in_memory"]["solid_count"] == 1
    assert result["facts"]["step_reimport"]["solid_count"] == 1
    assert (tmp_path / "run" / "candidate.step").is_file()
    assert result["containment"]["canonical"] is False
    assert result["containment"]["security_boundary"] is False


def test_parameter_override_is_executed_through_build(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    result = run_submission(
        source_path=source_writer(VALID_SOURCE),
        case=evaluator_case,
        output_dir=tmp_path / "variant",
        overrides={"overall_width": 30.0},
    )

    assert result["status"] == "passed"
    assert result["override"] == {"overall_width": 30.0}
    assert result["facts"]["in_memory"]["bbox_mm"]["xlen"] == pytest.approx(30.0)
    assert result["facts"]["step_reimport"]["bbox_mm"]["xlen"] == pytest.approx(30.0)
    assert (
        result["stages"]["step_output"]["observed"]["evaluator_exported_override"]
        is True
    )


def test_reports_model_owned_body_count_failure(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    result = run_submission(
        source_path=source_writer(TWO_SOLID_SOURCE),
        case=evaluator_case,
        output_dir=tmp_path / "wrong-count",
    )

    stage = result["stages"]["body_count"]
    assert result["status"] == "failed"
    assert stage["status"] == "failed"
    assert stage["ownership"] == "model"
    assert stage["failure_code"] == "solid_count_mismatch"
    assert result["stages"]["step_reimport"]["status"] == "passed"


def test_supports_declared_two_solid_contract(
    tmp_path: Path, source_writer, case_factory
) -> None:
    result = run_submission(
        source_path=source_writer(TWO_SOLID_SOURCE),
        case=case_factory(body_count=2),
        output_dir=tmp_path / "two-solids",
    )

    assert result["status"] == "passed"
    assert result["facts"]["step_reimport"]["solid_count"] == 2


def test_runtime_exception_is_a_model_failure(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    source = VALID_SOURCE.replace(
        'return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)',
        'raise RuntimeError("construction failed")',
    )
    result = run_submission(
        source_path=source_writer(source),
        case=evaluator_case,
        output_dir=tmp_path / "runtime-error",
    )

    stage = result["stages"]["execution"]
    assert stage["status"] == "failed"
    assert stage["ownership"] == "model"
    assert stage["failure_code"] == "submission_runtime_error"
    assert result["stages"]["step_output"]["status"] == "not_reached"


def test_timeout_preserves_log_evidence(
    tmp_path: Path, source_writer, case_factory
) -> None:
    source = VALID_SOURCE.replace(
        "result = build()", "while True:\n    pass\n\nresult = build()"
    )
    output = tmp_path / "timeout"
    result = run_submission(
        source_path=source_writer(source),
        case=case_factory(timeout_seconds=1),
        output_dir=output,
    )

    stage = result["stages"]["execution"]
    assert stage["status"] == "failed"
    assert stage["failure_code"] == "execution_timeout"
    assert (output / "execution.stdout.log").is_file()
    assert (output / "execution.stderr.log").is_file()


def test_timeout_is_not_caught_by_model_exception_handler(
    tmp_path: Path, source_writer, case_factory
) -> None:
    source = VALID_SOURCE.replace(
        "result = build()",
        """\
try:
    while True:
        pass
except Exception:
    pass

result = build()
""".rstrip(),
    )
    result = run_submission(
        source_path=source_writer(source),
        case=case_factory(timeout_seconds=1),
        output_dir=tmp_path / "catch-exception-timeout",
    )

    stage = result["stages"]["execution"]
    assert result["status"] == "failed"
    assert stage["status"] == "failed"
    assert stage["ownership"] == "model"
    assert stage["failure_code"] == "execution_timeout"


def test_override_timeout_is_not_caught_by_model_exception_handler(
    tmp_path: Path, source_writer, case_factory
) -> None:
    source = VALID_SOURCE.replace(
        '    return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)',
        """\
    if params:
        try:
            while True:
                pass
        except Exception:
            pass
    return cq.Workplane("XY").box(values["overall_width"], 10.0, 5.0)
""".rstrip(),
    )
    result = run_submission(
        source_path=source_writer(source),
        case=case_factory(timeout_seconds=1),
        output_dir=tmp_path / "catch-exception-override-timeout",
        overrides={"overall_width": 30.0},
    )

    stage = result["stages"]["execution"]
    assert result["status"] == "failed"
    assert stage["status"] == "failed"
    assert stage["ownership"] == "model"
    assert stage["failure_code"] == "execution_timeout"


def _timed_out_process(phase: str | None) -> dict[str, Any]:
    return {
        "timed_out": True,
        "returncode": None,
        "runtime_seconds": 0.01,
        "report": None,
        "phase": {"phase": phase} if phase is not None else None,
    }


def test_worker_sigxcpu_during_model_execution_is_model_failure(
    tmp_path: Path, source_writer, evaluator_case, monkeypatch
) -> None:
    def cpu_limited_worker(command, **kwargs):
        del command, kwargs
        return {
            "timed_out": False,
            "returncode": -signal.SIGXCPU,
            "runtime_seconds": 1.0,
            "report": None,
            "phase": {"phase": "source_execution"},
        }

    monkeypatch.setattr(execution_runner, "_invoke_worker", cpu_limited_worker)
    result = run_submission(
        source_path=source_writer(VALID_SOURCE),
        case=evaluator_case,
        output_dir=tmp_path / "cpu-limit",
    )

    stage = result["stages"]["execution"]
    assert result["status"] == "failed"
    assert stage["status"] == "failed"
    assert stage["ownership"] == "model"
    assert stage["failure_code"] == "execution_cpu_limit_exceeded"


def test_worker_sigsegv_during_model_execution_is_model_failure(
    tmp_path: Path, source_writer, evaluator_case, monkeypatch
) -> None:
    def crashed_worker(command, **kwargs):
        del command, kwargs
        return {
            "timed_out": False,
            "returncode": -signal.SIGSEGV,
            "runtime_seconds": 1.0,
            "report": None,
            "phase": {"phase": "source_execution"},
        }

    monkeypatch.setattr(execution_runner, "_invoke_worker", crashed_worker)
    result = run_submission(
        source_path=source_writer(VALID_SOURCE),
        case=evaluator_case,
        output_dir=tmp_path / "native-crash",
    )

    stage = result["stages"]["execution"]
    assert result["status"] == "failed"
    assert stage["status"] == "failed"
    assert stage["ownership"] == "model"
    assert stage["failure_code"] == "execution_native_process_crash"
    assert "SIGSEGV" in stage["summary"]


def test_parent_timeout_after_model_start_is_model_failure_and_counts_override_budget(
    tmp_path: Path, source_writer, case_factory, monkeypatch
) -> None:
    observed: dict[str, Any] = {}

    def fake_invoke(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return _timed_out_process("parameter_execution")

    monkeypatch.setattr(execution_runner, "_invoke_worker", fake_invoke)
    result = run_submission(
        source_path=source_writer(VALID_SOURCE),
        case=case_factory(timeout_seconds=3),
        output_dir=tmp_path / "outer-model-timeout",
        overrides={"overall_width": 30.0},
    )

    stage = result["stages"]["execution"]
    assert observed["timeout_seconds"] == 45 + 2 * 3
    assert "--phase-marker" in observed["command"]
    assert observed["phase_path"] is not None
    assert stage["status"] == "failed"
    assert stage["ownership"] == "model"
    assert stage["failure_code"] == "execution_timeout"


def test_parent_timeout_before_model_start_is_evaluator_error(
    tmp_path: Path, source_writer, case_factory, monkeypatch
) -> None:
    observed: dict[str, Any] = {}

    def fake_invoke(command, **kwargs):
        del command
        observed.update(kwargs)
        return _timed_out_process(None)

    monkeypatch.setattr(execution_runner, "_invoke_worker", fake_invoke)
    result = run_submission(
        source_path=source_writer(VALID_SOURCE),
        case=case_factory(timeout_seconds=3),
        output_dir=tmp_path / "outer-startup-timeout",
    )

    stage = result["stages"]["execution"]
    assert observed["timeout_seconds"] == 45 + 3
    assert result["status"] == "evaluator_error"
    assert stage["status"] == "evaluator_error"
    assert stage["ownership"] == "evaluator"
    assert stage["failure_code"] == "execution_worker_startup_timeout"


def test_step_reimport_timeout_is_evaluator_error(
    tmp_path: Path, source_writer, evaluator_case, monkeypatch
) -> None:
    original_invoke = execution_runner._invoke_worker

    def timeout_reimport(command, **kwargs):
        if "inspect" in command:
            return _timed_out_process(None)
        return original_invoke(command, **kwargs)

    monkeypatch.setattr(execution_runner, "_invoke_worker", timeout_reimport)
    result = run_submission(
        source_path=source_writer(VALID_SOURCE),
        case=evaluator_case,
        output_dir=tmp_path / "reimport-timeout",
    )

    stage = result["stages"]["step_reimport"]
    assert result["status"] == "evaluator_error"
    assert stage["status"] == "evaluator_error"
    assert stage["ownership"] == "evaluator"
    assert stage["failure_code"] == "step_reimport_worker_timeout"


def test_unknown_override_is_an_evaluator_error(
    tmp_path: Path, source_writer, evaluator_case
) -> None:
    result = run_submission(
        source_path=source_writer(VALID_SOURCE),
        case=evaluator_case,
        output_dir=tmp_path / "bad-override",
        overrides={"not_public": 2.0},
    )

    stage = result["stages"]["static_contract"]
    assert stage["status"] == "evaluator_error"
    assert stage["ownership"] == "evaluator"
