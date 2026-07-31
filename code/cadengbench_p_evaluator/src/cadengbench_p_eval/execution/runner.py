"""Parent-side orchestration for one CadQuery submission execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .static import StaticInspection, StaticValidationError, inspect_submission

RUNNER_SCHEMA = "cadengbench-p.execution-result.v1"
RUNNER_VERSION = "cadengbench-p.local-runner.v1"
CONTAINMENT_NOTICE = (
    "Local isolated subprocess execution is non-canonical containment and is not "
    "a security boundary. Publication scoring requires the pinned Linux sandbox."
)
_STAGE_NAMES = (
    "static_contract",
    "execution",
    "step_output",
    "geometry_validity",
    "body_count",
    "step_reimport",
    "roundtrip",
)
_CASE_ERROR_CODES = {"invalid_evaluator_case", "unsupported_submission_interface"}
_LOG_TAIL_CHARS = 65_536
_MODEL_EXECUTION_PHASES = {"source_execution", "parameter_execution"}


def _stage(
    status: str,
    summary: str,
    *,
    ownership: str | None = None,
    failure_code: str | None = None,
    expected: Any = None,
    observed: Any = None,
    metrics: Mapping[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "ownership": ownership,
        "failure_code": failure_code,
        "expected": expected,
        "observed": observed,
        "metrics": dict(metrics or {}),
        "evidence": list(evidence or []),
    }


def _initial_stages() -> dict[str, dict[str, Any]]:
    return {
        name: _stage("not_reached", "an earlier L0 stage did not complete")
        for name in _STAGE_NAMES
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _execution_policy(case: Mapping[str, Any]) -> dict[str, Any]:
    try:
        execution = case["evaluator_profile"]["execution"]
        solid_contract = case["task_contract"]["solid_contract"]
        timeout = execution["timeout_seconds"]
        startup_grace = execution.get("worker_startup_grace_seconds", 30)
        max_step_bytes = execution["max_step_bytes"]
        roots = execution["allowed_import_roots"]
        body_count = solid_contract["required_body_count"]
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluator case lacks the L0 execution contract") from exc
    if (
        not isinstance(timeout, int)
        or timeout <= 0
        or not isinstance(startup_grace, int)
        or startup_grace < 0
        or not isinstance(max_step_bytes, int)
        or max_step_bytes <= 0
        or not isinstance(roots, list)
        or not all(isinstance(root, str) and root for root in roots)
        or not isinstance(body_count, int)
        or isinstance(body_count, bool)
        or body_count <= 0
    ):
        raise ValueError("evaluator case contains an invalid L0 execution contract")
    return {
        "timeout_seconds": timeout,
        "worker_startup_grace_seconds": startup_grace,
        "max_step_bytes": max_step_bytes,
        "allowed_import_roots": sorted(set(roots) | {"__future__"}),
        "required_body_count": body_count,
    }


def _normalize_overrides(
    overrides: Mapping[str, Any] | None, inspection: StaticInspection
) -> dict[str, float]:
    if overrides is None:
        return {}
    unknown = sorted(set(overrides) - set(inspection.default_params))
    if unknown:
        raise ValueError(f"parameter override names are not public: {unknown}")
    normalized: dict[str, float] = {}
    for name, value in overrides.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"parameter override {name!r} must be finite")
        normalized[name] = float(value)
    return dict(sorted(normalized.items()))


def _worker_environment(sandbox: Path) -> dict[str, str]:
    home = sandbox / "home"
    temporary = sandbox / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    environment = {
        "PATH": os.environ.get("PATH", str(Path(sys.executable).parent)),
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TBB_NUM_THREADS": "1",
    }
    for name in ("CONDA_PREFIX", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _log_record(path: Path, public_path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    truncated = len(text) > _LOG_TAIL_CHARS
    tail = text[-_LOG_TAIL_CHARS:]
    public_path.write_text(tail, encoding="utf-8")
    return {
        "path": public_path.name,
        "sha256": _sha256(public_path),
        "characters_retained": len(tail),
        "truncated": truncated,
    }


def _invoke_worker(
    command: list[str],
    *,
    sandbox: Path,
    report_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    phase_path: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=sandbox,
                env=_worker_environment(sandbox),
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    return {
        "timed_out": timed_out,
        "returncode": returncode,
        "runtime_seconds": time.monotonic() - started,
        "report": _read_json(report_path),
        "phase": _read_json(phase_path) if phase_path is not None else None,
    }


def _roundtrip_comparison(
    in_memory: Mapping[str, Any], reimported: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    source_volume = float(in_memory["volume_mm3"])
    imported_volume = float(reimported["volume_mm3"])
    volume_error = abs(source_volume - imported_volume)
    volume_tolerance = max(1e-6, abs(source_volume) * 1e-7)
    source_bbox = in_memory["bbox_mm"]
    imported_bbox = reimported["bbox_mm"]
    coordinates = ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")
    bbox_errors = {
        name: abs(float(source_bbox[name]) - float(imported_bbox[name]))
        for name in coordinates
    }
    scale = max(
        1.0,
        float(source_bbox["xlen"]),
        float(source_bbox["ylen"]),
        float(source_bbox["zlen"]),
    )
    bbox_tolerance = max(1e-6, scale * 1e-7)
    body_count_equal = int(in_memory["solid_count"]) == int(reimported["solid_count"])
    passed = (
        body_count_equal
        and volume_error <= volume_tolerance
        and max(bbox_errors.values(), default=0.0) <= bbox_tolerance
    )
    return passed, {
        "body_count_equal": body_count_equal,
        "volume_absolute_error_mm3": volume_error,
        "volume_tolerance_mm3": volume_tolerance,
        "bbox_absolute_errors_mm": bbox_errors,
        "bbox_tolerance_mm": bbox_tolerance,
    }


def _overall_status(stages: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = {str(stage["status"]) for stage in stages.values()}
    if "evaluator_error" in statuses:
        return "evaluator_error"
    if "failed" in statuses:
        return "failed"
    return "passed"


def _base_result(
    *, source_path: Path, overrides: Mapping[str, float]
) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA,
        "runner_version": RUNNER_VERSION,
        "status": "not_reached",
        "passed": None,
        "containment": {
            "mode": "local_python_isolated_subprocess",
            "canonical": False,
            "security_boundary": False,
            "notice": CONTAINMENT_NOTICE,
        },
        "source": {
            "path": str(source_path.resolve()),
            "available": source_path.is_file(),
        },
        "override": dict(overrides),
        "static_inspection": None,
        "stages": _initial_stages(),
        "facts": {"in_memory": None, "step_reimport": None},
        "artifacts": {},
        "runtime_seconds": 0.0,
    }


def _finish(result: dict[str, Any], output_dir: Path, started: float) -> dict[str, Any]:
    result["status"] = _overall_status(result["stages"])
    result["passed"] = (
        True
        if result["status"] == "passed"
        else False
        if result["status"] == "failed"
        else None
    )
    result["runtime_seconds"] = time.monotonic() - started
    _write_json(output_dir / "execution.json", result)
    return result


def run_submission(
    *,
    source_path: Path,
    case: Mapping[str, Any],
    output_dir: Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and execute one submission, returning auditable L0 stages.

    ``overrides`` is evaluator-owned input for L2.  It is passed only through
    the submission's validated ``build(params)`` interface.
    """

    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "candidate.step",
        "execution.json",
        "execution.stdout.log",
        "execution.stderr.log",
        "reimport.stdout.log",
        "reimport.stderr.log",
    ):
        (output_dir / name).unlink(missing_ok=True)
    result = _base_result(source_path=source_path, overrides={})
    stages = result["stages"]

    try:
        policy = _execution_policy(case)
        inspection = inspect_submission(source_path, case=case)
        normalized_overrides = _normalize_overrides(overrides, inspection)
    except StaticValidationError as exc:
        ownership = "evaluator" if exc.failure_code in _CASE_ERROR_CODES else "model"
        status = "evaluator_error" if ownership == "evaluator" else "failed"
        stages["static_contract"] = _stage(
            status,
            str(exc),
            ownership=ownership,
            failure_code=exc.failure_code,
            evidence=[{"source_path": str(source_path)}],
        )
        return _finish(result, output_dir, started)
    except (OSError, ValueError) as exc:
        stages["static_contract"] = _stage(
            "evaluator_error",
            str(exc),
            ownership="evaluator",
            failure_code="invalid_execution_request",
        )
        return _finish(result, output_dir, started)

    result["override"] = normalized_overrides
    result["static_inspection"] = inspection.as_dict()
    result["source"].update(
        {"sha256": inspection.source_sha256, "bytes": inspection.source_bytes}
    )
    stages["static_contract"] = _stage(
        "passed",
        "source satisfies the frozen CadQuery v1 interface and policy",
        observed={
            "parameter_count": len(inspection.parameter_manifest),
            "source_bytes": inspection.source_bytes,
        },
    )

    worker_path = Path(__file__).with_name("worker.py").resolve()
    timeout = int(policy["timeout_seconds"])
    startup_grace = int(policy["worker_startup_grace_seconds"])
    model_execution_count = 2 if normalized_overrides else 1
    worker_timeout = startup_grace + timeout * model_execution_count
    inspection_timeout = startup_grace + timeout
    max_step_bytes = int(policy["max_step_bytes"])
    with tempfile.TemporaryDirectory(prefix="cadengbench-p-l0-", dir=output_dir) as raw:
        sandbox = Path(raw)
        staged_source = sandbox / "submission.py"
        shutil.copyfile(source_path, staged_source)
        execute_report_path = sandbox / "execute.report.json"
        execute_phase_path = sandbox / "execute.phase.json"
        execute_stdout = sandbox / "execute.stdout.log"
        execute_stderr = sandbox / "execute.stderr.log"
        command = [
            sys.executable,
            "-I",
            str(worker_path),
            "execute",
            "--source",
            str(staged_source),
            "--report",
            str(execute_report_path),
            "--phase-marker",
            str(execute_phase_path),
            "--allowed-import-roots",
            json.dumps(policy["allowed_import_roots"]),
            "--cpu-seconds",
            str(max(1, worker_timeout)),
            "--model-wall-seconds",
            str(timeout),
            "--max-step-bytes",
            str(max_step_bytes),
        ]
        if normalized_overrides:
            command.extend(["--override-json", json.dumps(normalized_overrides)])
        process = _invoke_worker(
            command,
            sandbox=sandbox,
            report_path=execute_report_path,
            stdout_path=execute_stdout,
            stderr_path=execute_stderr,
            timeout_seconds=worker_timeout,
            phase_path=execute_phase_path,
        )
        log_evidence = {
            "stdout": _log_record(execute_stdout, output_dir / "execution.stdout.log"),
            "stderr": _log_record(execute_stderr, output_dir / "execution.stderr.log"),
            "returncode": process["returncode"],
            "runtime_seconds": process["runtime_seconds"],
            "worker_phase": process["phase"],
            "model_execution_count": model_execution_count,
            "parent_timeout_seconds": worker_timeout,
        }
        result["artifacts"].update(
            {
                "execution_stdout": log_evidence["stdout"],
                "execution_stderr": log_evidence["stderr"],
            }
        )
        if process["timed_out"]:
            phase_record = process.get("phase")
            phase = (
                str(phase_record.get("phase"))
                if isinstance(phase_record, Mapping)
                else None
            )
            model_owned = phase in _MODEL_EXECUTION_PHASES
            stages["execution"] = _stage(
                "failed" if model_owned else "evaluator_error",
                (
                    f"model {phase} exceeded its bounded execution budget"
                    if model_owned
                    else "execution worker timed out before model execution started"
                    if phase is None
                    else f"execution worker timed out during evaluator phase {phase}"
                ),
                ownership="model" if model_owned else "evaluator",
                failure_code=(
                    "execution_timeout"
                    if model_owned
                    else "execution_worker_startup_timeout"
                    if phase is None
                    else "execution_worker_timeout"
                ),
                evidence=[log_evidence],
            )
            return _finish(result, output_dir, started)

        report = process["report"]
        if report is None:
            phase_record = process.get("phase")
            phase = (
                str(phase_record.get("phase"))
                if isinstance(phase_record, Mapping)
                else None
            )
            cpu_limit_exceeded = (
                phase in _MODEL_EXECUTION_PHASES
                and process.get("returncode") == -signal.SIGXCPU
            )
            returncode = process.get("returncode")
            model_process_crashed = (
                phase in _MODEL_EXECUTION_PHASES
                and isinstance(returncode, int)
                and returncode < 0
            )
            if cpu_limit_exceeded:
                stages["execution"] = _stage(
                    "failed",
                    "model execution exceeded its bounded CPU budget",
                    ownership="model",
                    failure_code="execution_cpu_limit_exceeded",
                    evidence=[log_evidence],
                )
            elif model_process_crashed:
                signal_number = -returncode
                try:
                    signal_name = signal.Signals(signal_number).name
                except ValueError:
                    signal_name = f"signal {signal_number}"
                stages["execution"] = _stage(
                    "failed",
                    f"model execution process terminated by {signal_name}",
                    ownership="model",
                    failure_code="execution_native_process_crash",
                    evidence=[log_evidence],
                )
            else:
                stages["execution"] = _stage(
                    "evaluator_error",
                    "execution worker exited without a valid report",
                    ownership="evaluator",
                    failure_code="worker_report_missing",
                    evidence=[log_evidence],
                )
            return _finish(result, output_dir, started)
        if report.get("status") != "passed":
            ownership = str(report.get("ownership") or "evaluator")
            status = "failed" if ownership == "model" else "evaluator_error"
            failed_stage = str(report.get("failed_stage") or "source_execution")
            target = {
                "output_contract": "step_output",
                "in_memory_geometry": "geometry_validity",
            }.get(failed_stage, "execution")
            if target != "execution":
                stages["execution"] = _stage(
                    "passed", "submission source executed without an exception"
                )
            if target == "geometry_validity":
                stages["step_output"] = _stage(
                    "passed", "submission authored a nonempty model.step"
                )
            stages[target] = _stage(
                status,
                str(report.get("summary") or "worker rejected the submission"),
                ownership=ownership,
                failure_code=str(report.get("failure_code") or "worker_failure"),
                evidence=[log_evidence],
            )
            return _finish(result, output_dir, started)

        in_memory = report.get("facts")
        if not isinstance(in_memory, Mapping):
            stages["execution"] = _stage(
                "evaluator_error",
                "execution worker omitted geometry facts",
                ownership="evaluator",
                failure_code="worker_report_incomplete",
                evidence=[log_evidence],
            )
            return _finish(result, output_dir, started)
        result["facts"]["in_memory"] = dict(in_memory)
        stages["execution"] = _stage(
            "passed",
            "submission executed in a fresh subprocess",
            evidence=[log_evidence],
        )
        stages["step_output"] = _stage(
            "passed",
            "submission authored a nonempty model.step",
            observed={
                "bytes": report.get("step_bytes"),
                "evaluator_exported_override": report.get(
                    "evaluator_exported_override"
                ),
            },
        )
        stages["geometry_validity"] = _stage(
            "passed",
            "in-memory result is a valid nonempty solid set",
            observed=dict(in_memory),
        )
        required_count = int(policy["required_body_count"])
        observed_count = int(in_memory["solid_count"])
        if observed_count == required_count:
            stages["body_count"] = _stage(
                "passed",
                "solid count matches the public task contract",
                expected=required_count,
                observed=observed_count,
            )
        else:
            stages["body_count"] = _stage(
                "failed",
                "solid count differs from the public task contract",
                ownership="model",
                failure_code="solid_count_mismatch",
                expected=required_count,
                observed=observed_count,
            )

        candidate_in_sandbox = sandbox / "model.step"
        candidate = output_dir / "candidate.step"
        pending_candidate = output_dir / "candidate.step.tmp"
        shutil.copyfile(candidate_in_sandbox, pending_candidate)
        os.replace(pending_candidate, candidate)
        candidate_record = {
            "path": candidate.name,
            "sha256": _sha256(candidate),
            "bytes": candidate.stat().st_size,
        }
        result["artifacts"]["candidate_step"] = candidate_record

        inspect_report_path = sandbox / "inspect.report.json"
        inspect_stdout = sandbox / "inspect.stdout.log"
        inspect_stderr = sandbox / "inspect.stderr.log"
        inspect_command = [
            sys.executable,
            "-I",
            str(worker_path),
            "inspect",
            "--step",
            str(candidate),
            "--report",
            str(inspect_report_path),
            "--cpu-seconds",
            str(max(1, inspection_timeout)),
            "--max-step-bytes",
            str(max_step_bytes),
        ]
        inspection_process = _invoke_worker(
            inspect_command,
            sandbox=sandbox,
            report_path=inspect_report_path,
            stdout_path=inspect_stdout,
            stderr_path=inspect_stderr,
            timeout_seconds=inspection_timeout,
        )
        inspection_logs = {
            "stdout": _log_record(inspect_stdout, output_dir / "reimport.stdout.log"),
            "stderr": _log_record(inspect_stderr, output_dir / "reimport.stderr.log"),
            "returncode": inspection_process["returncode"],
            "runtime_seconds": inspection_process["runtime_seconds"],
        }
        result["artifacts"].update(
            {
                "reimport_stdout": inspection_logs["stdout"],
                "reimport_stderr": inspection_logs["stderr"],
            }
        )
        if inspection_process["timed_out"]:
            stages["step_reimport"] = _stage(
                "evaluator_error",
                (
                    "independent STEP reimport exceeded the evaluator-owned "
                    "worker budget"
                ),
                ownership="evaluator",
                failure_code="step_reimport_worker_timeout",
                evidence=[inspection_logs],
            )
            return _finish(result, output_dir, started)
        inspect_report = inspection_process["report"]
        if inspect_report is None:
            stages["step_reimport"] = _stage(
                "evaluator_error",
                "STEP inspection worker exited without a valid report",
                ownership="evaluator",
                failure_code="worker_report_missing",
                evidence=[inspection_logs],
            )
            return _finish(result, output_dir, started)
        if inspect_report.get("status") != "passed":
            ownership = str(inspect_report.get("ownership") or "evaluator")
            stages["step_reimport"] = _stage(
                "failed" if ownership == "model" else "evaluator_error",
                str(inspect_report.get("summary") or "STEP reimport failed"),
                ownership=ownership,
                failure_code=str(
                    inspect_report.get("failure_code") or "step_reimport_failed"
                ),
                evidence=[inspection_logs],
            )
            return _finish(result, output_dir, started)
        reimported = inspect_report.get("facts")
        if not isinstance(reimported, Mapping):
            stages["step_reimport"] = _stage(
                "evaluator_error",
                "STEP inspection worker omitted geometry facts",
                ownership="evaluator",
                failure_code="worker_report_incomplete",
                evidence=[inspection_logs],
            )
            return _finish(result, output_dir, started)
        result["facts"]["step_reimport"] = dict(reimported)
        stages["step_reimport"] = _stage(
            "passed",
            "candidate STEP independently reimported as valid solids",
            observed=dict(reimported),
            evidence=[inspection_logs],
        )
        roundtrip_passed, metrics = _roundtrip_comparison(in_memory, reimported)
        stages["roundtrip"] = _stage(
            "passed" if roundtrip_passed else "failed",
            "STEP round trip preserved body count, volume, and world-frame bounds"
            if roundtrip_passed
            else "STEP round trip changed a required geometry invariant",
            ownership=None if roundtrip_passed else "model",
            failure_code=None if roundtrip_passed else "step_roundtrip_mismatch",
            metrics=metrics,
        )
    return _finish(result, output_dir, started)
