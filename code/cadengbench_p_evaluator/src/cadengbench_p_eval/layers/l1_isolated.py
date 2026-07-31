"""Parent-side process isolation for the existing L1 evaluator."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from copy import deepcopy
from pathlib import Path
from typing import Any

from cadengbench_p_eval.case import LoadedCase
from cadengbench_p_eval.contracts import (
    CheckResult,
    LayerResult,
    Status,
    layer_result_from_dict,
    reduce_required_status,
)
from cadengbench_p_eval.io import read_json, write_json

L1_JOB_SCHEMA = "cadengbench-p.l1-isolated-job.v1"
L1_WORKER_REPORT_SCHEMA = "cadengbench-p.l1-isolated-report.v1"
_LOG_TAIL_CHARACTERS = 16_384
_TERMINATION_GRACE_SECONDS = 1.0


class _MalformedWorkerReport(ValueError):
    """A worker report cannot be trusted as a LayerResult."""


def _serialize_case(
    case: LoadedCase | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(case, LoadedCase):
        return {
            "kind": "loaded_case",
            "path": str(case.path.resolve()),
            "data": deepcopy(case.data),
        }
    if not isinstance(case, Mapping):
        raise TypeError("case must be a LoadedCase or mapping")
    data = deepcopy(dict(case))
    source_item_root = data.get("source_item_root")
    if isinstance(source_item_root, str):
        data["source_item_root"] = str(Path(source_item_root or ".").resolve())
    return {"kind": "mapping", "data": data}


def _job_payload(
    case: LoadedCase | Mapping[str, Any],
    candidate_step: Path,
    *,
    target_step: Path | None,
    is_parameter_variant: bool,
    l0_passed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": L1_JOB_SCHEMA,
        "case": _serialize_case(case),
        "candidate_step": str(Path(candidate_step).resolve()),
        "target_step": (
            str(Path(target_step).resolve()) if target_step is not None else None
        ),
        "is_parameter_variant": is_parameter_variant,
        "l0_passed": l0_passed,
    }


def _worker_command(job_path: Path, report_path: Path) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-m",
        "cadengbench_p_eval.layers.l1_worker",
        "--job",
        str(job_path),
        "--report",
        str(report_path),
    ]


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TBB_NUM_THREADS": "1",
        }
    )
    return environment


def _terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    """Terminate the worker and every descendant in its dedicated process group."""

    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        return True
    except subprocess.TimeoutExpired:
        pass
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        return True
    except subprocess.TimeoutExpired:
        return False


def _log_tail(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return value[-_LOG_TAIL_CHARACTERS:]


def _error_result(
    code: str,
    summary: str,
    *,
    runtime_seconds: float,
    returncode: int | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    details: Mapping[str, Any] | None = None,
) -> LayerResult:
    observed: dict[str, Any] = {"error_code": code}
    if returncode is not None:
        observed["worker_returncode"] = returncode
    if details:
        observed["details"] = dict(details)
    evidence: list[dict[str, Any]] = []
    if stdout_path is not None or stderr_path is not None:
        evidence.append(
            {
                "worker_stdout_tail": (
                    _log_tail(stdout_path) if stdout_path is not None else ""
                ),
                "worker_stderr_tail": (
                    _log_tail(stderr_path) if stderr_path is not None else ""
                ),
            }
        )
    check = CheckResult(
        check_id="L1.process_isolation",
        status=Status.EVALUATOR_ERROR,
        required=True,
        summary=summary,
        observed=observed,
        evidence=evidence,
        runtime_seconds=runtime_seconds,
    )
    return LayerResult.from_checks("L1", (check,), runtime_seconds=runtime_seconds)


def _validate_result_payload(value: Any) -> LayerResult:
    if not isinstance(value, Mapping):
        raise _MalformedWorkerReport("result must be an object")
    raw = dict(value)
    try:
        result = layer_result_from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise _MalformedWorkerReport(f"invalid LayerResult: {exc}") from exc
    if result.layer != "L1":
        raise _MalformedWorkerReport("worker returned a non-L1 result")
    reduced = reduce_required_status(result.checks)
    if result.status is not reduced:
        raise _MalformedWorkerReport(
            "worker LayerResult status does not match its required checks"
        )
    if raw.get("passed") is not result.passed:
        raise _MalformedWorkerReport("worker LayerResult passed value is inconsistent")
    raw_checks = raw.get("checks")
    if not isinstance(raw_checks, list) or len(raw_checks) != len(result.checks):
        raise _MalformedWorkerReport("worker LayerResult checks are malformed")
    for raw_check, check in zip(raw_checks, result.checks, strict=True):
        if (
            not isinstance(raw_check, Mapping)
            or raw_check.get("passed") is not check.passed
        ):
            raise _MalformedWorkerReport("worker check passed value is inconsistent")
    runtime = result.runtime_seconds
    if runtime is not None and (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) < 0.0
    ):
        raise _MalformedWorkerReport("worker runtime must be finite and non-negative")
    return result


def _read_worker_result(report_path: Path) -> LayerResult:
    try:
        report = read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _MalformedWorkerReport(f"report cannot be read: {exc}") from exc
    if not isinstance(report, Mapping):
        raise _MalformedWorkerReport("report must be an object")
    if report.get("schema_version") != L1_WORKER_REPORT_SCHEMA:
        raise _MalformedWorkerReport("report schema is absent or unsupported")
    outcome = report.get("outcome")
    if outcome == "worker_error":
        error = report.get("error")
        detail = dict(error) if isinstance(error, Mapping) else {}
        raise _MalformedWorkerReport(
            f"worker reported an internal error: {detail.get('type', 'unknown')}: "
            f"{detail.get('message', '')}"
        )
    if outcome != "completed":
        raise _MalformedWorkerReport("report outcome is absent or unsupported")
    return _validate_result_payload(report.get("result"))


@contextmanager
def _execution_directory(work_dir: Path | None) -> Iterator[Path]:
    if work_dir is not None:
        path = Path(work_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="cadengbench-p-l1-") as raw:
        yield Path(raw)


def _evaluate_in_directory(
    case: LoadedCase | Mapping[str, Any],
    candidate_step: Path,
    *,
    target_step: Path | None,
    is_parameter_variant: bool,
    l0_passed: bool,
    timeout_seconds: float,
    execution_dir: Path,
) -> LayerResult:
    started = time.monotonic()
    job_path = execution_dir / "worker.job.json"
    report_path = execution_dir / "worker.report.json"
    stdout_path = execution_dir / "worker.stdout.log"
    stderr_path = execution_dir / "worker.stderr.log"
    try:
        for path in (job_path, report_path, stdout_path, stderr_path):
            path.unlink(missing_ok=True)
        write_json(
            job_path,
            _job_payload(
                case,
                candidate_step,
                target_step=target_step,
                is_parameter_variant=is_parameter_variant,
                l0_passed=l0_passed,
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        runtime = time.monotonic() - started
        return _error_result(
            "invalid_isolated_l1_request",
            f"isolated L1 request could not be serialized: {type(exc).__name__}: {exc}",
            runtime_seconds=runtime,
        )

    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                _worker_command(job_path, report_path),
                cwd=execution_dir,
                env=_worker_environment(),
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process_reaped = _terminate_process_group(process)
                runtime = time.monotonic() - started
                return _error_result(
                    "l1_worker_timeout",
                    f"isolated L1 evaluation exceeded {timeout_seconds:g} seconds",
                    runtime_seconds=runtime,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    details={"process_group_reaped": process_reaped},
                )
            except BaseException:
                _terminate_process_group(process)
                raise
    except OSError as exc:
        runtime = time.monotonic() - started
        return _error_result(
            "l1_worker_launch_failed",
            f"isolated L1 worker could not start: {type(exc).__name__}: {exc}",
            runtime_seconds=runtime,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    runtime = time.monotonic() - started
    if returncode != 0:
        details: dict[str, Any] = {}
        try:
            report = read_json(report_path)
            if isinstance(report, Mapping) and isinstance(report.get("error"), Mapping):
                details = dict(report["error"])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        return _error_result(
            "l1_worker_crash",
            f"isolated L1 worker exited with status {returncode}",
            runtime_seconds=runtime,
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            details=details,
        )
    try:
        return _read_worker_result(report_path)
    except _MalformedWorkerReport as exc:
        return _error_result(
            "malformed_l1_worker_report",
            f"isolated L1 worker produced no trustworthy report: {exc}",
            runtime_seconds=runtime,
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


def evaluate_l1_isolated(
    case: LoadedCase | Mapping[str, Any],
    candidate_step: Path,
    *,
    target_step: Path | None = None,
    is_parameter_variant: bool = False,
    l0_passed: bool = True,
    timeout_seconds: float = 120.0,
    work_dir: Path | None = None,
) -> LayerResult:
    """Run ``evaluate_l1`` in a fresh, bounded subprocess.

    The deadline includes worker startup and all geometry/DFM work. When
    ``work_dir`` is supplied, the atomic job/report and bounded logs are retained
    there for audit; otherwise they live only for the duration of this call.
    """

    if os.name != "posix":
        return _error_result(
            "unsupported_process_isolation_platform",
            "isolated L1 evaluation currently supports macOS and Linux",
            runtime_seconds=0.0,
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        return _error_result(
            "invalid_l1_timeout",
            "isolated L1 timeout must be finite and positive",
            runtime_seconds=0.0,
        )
    if not isinstance(is_parameter_variant, bool) or not isinstance(l0_passed, bool):
        return _error_result(
            "invalid_isolated_l1_request",
            "variant and L0 flags must be booleans",
            runtime_seconds=0.0,
        )
    try:
        with _execution_directory(work_dir) as execution_dir:
            return _evaluate_in_directory(
                case,
                Path(candidate_step),
                target_step=Path(target_step) if target_step is not None else None,
                is_parameter_variant=is_parameter_variant,
                l0_passed=l0_passed,
                timeout_seconds=float(timeout_seconds),
                execution_dir=execution_dir,
            )
    except OSError as exc:
        return _error_result(
            "l1_work_directory_failed",
            f"isolated L1 work directory failed: {type(exc).__name__}: {exc}",
            runtime_seconds=0.0,
        )
