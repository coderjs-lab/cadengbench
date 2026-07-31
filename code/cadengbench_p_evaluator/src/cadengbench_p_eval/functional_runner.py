"""End-to-end deterministic Protocol-E item orchestration."""

from __future__ import annotations

import shutil
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import __version__
from .case import load_case
from .contracts import CheckResult, EvaluationResult, LayerResult, Status
from .io import sha256_file, write_json
from .layers.l1_isolated import evaluate_l1_isolated
from .layers.l2_functional import evaluate_l2_functional
from .protocols.functional_edit import (
    FunctionalEditAdapter,
    FunctionalEditAdapterError,
    FunctionalSourceResult,
    solid_count_requirement,
)
from .provenance import environment_record, evaluator_build_hash
from .runner import (
    RESULT_SCHEMA,
    _assert_safe_output_root,
    _not_reached,
    _overall_status,
    _prepare_item_output,
    _resume_result,
    _write_layer,
    _write_result,
)


def _source_check(
    source: FunctionalSourceResult, check_id: str
) -> Mapping[str, Any] | None:
    return source.checks.get(check_id)


def _l0_from_source(source: FunctionalSourceResult) -> LayerResult:
    valid = _source_check(source, "core.valid_solid")
    interface_ok = bool(source.normalized_result.get("checks"))
    candidate_ok = source.candidate_step is not None
    checks = (
        CheckResult(
            "L0.protocol_e.source_adapter_interface",
            Status.PASSED if interface_ok else Status.FAILED,
            failure_code=None if interface_ok else "invalid_functional_edit_submission",
            observed={
                "source_status": source.normalized_result.get("status"),
                "source_failure": source.normalized_result.get("failure"),
            },
        ),
        CheckResult(
            "L0.protocol_e.valid_solid",
            Status.PASSED if valid and valid.get("passed") is True else Status.FAILED,
            failure_code=(
                None
                if valid and valid.get("passed") is True
                else "functional_edit_invalid_solid"
            ),
            observed=dict(valid or {}),
        ),
        CheckResult(
            "L0.protocol_e.step_artifact",
            Status.PASSED if candidate_ok else Status.FAILED,
            failure_code=None if candidate_ok else "functional_edit_step_missing",
            observed={
                "candidate_step": str(source.candidate_step)
                if source.candidate_step is not None
                else None
            },
        ),
    )
    return LayerResult.from_checks(
        "L0",
        checks,
        artifacts={
            "candidate_step": (
                {
                    "path": str(source.candidate_step),
                    "sha256": sha256_file(source.candidate_step),
                    "bytes": source.candidate_step.stat().st_size,
                }
                if source.candidate_step is not None
                else None
            ),
            "source_evaluation": source.normalized_result,
        },
    )


def _l1_from_source(
    case_data: Mapping[str, Any],
    source: FunctionalSourceResult,
    *,
    output_root: Path,
) -> LayerResult:
    assert source.candidate_step is not None
    edited_case = deepcopy(dict(case_data))
    edited_case["requirements"] = solid_count_requirement(source)
    profile = edited_case.get("evaluator_profile") or {}
    execution = profile.get("execution") or {}
    timeout = float(execution.get("timeout_seconds", 120)) + float(
        execution.get("worker_startup_grace_seconds", 60)
    )
    evaluated = evaluate_l1_isolated(
        edited_case,
        source.candidate_step,
        l0_passed=True,
        timeout_seconds=timeout,
        work_dir=output_root / "l1-worker",
    )
    solid_count = _source_check(source, "core.solid_count_preserved")
    invariant = CheckResult(
        "L1.protocol_e.declared_solid_count",
        (
            Status.PASSED
            if solid_count and solid_count.get("passed") is True
            else Status.FAILED
        ),
        failure_code=(
            None
            if solid_count and solid_count.get("passed") is True
            else "functional_edit_solid_count_changed"
        ),
        observed=dict(solid_count or {}),
    )
    return LayerResult.from_checks(
        "L1",
        (*evaluated.checks, invariant),
        metrics=evaluated.metrics,
        artifacts=evaluated.artifacts,
        runtime_seconds=evaluated.runtime_seconds,
    )


def evaluate_functional_item(
    *,
    repo_root: Path,
    case_path: Path,
    submission_path: Path,
    output_root: Path,
    force: bool = False,
) -> EvaluationResult:
    """Evaluate one BenchCAD source edit or Fusion bounded JSON patch."""

    started = time.monotonic()
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    case = load_case(case_path, verify_artifacts=True)
    if case.data.get("protocol") != "functional_edit":
        raise ValueError("Protocol-E runner requires a functional_edit case")
    submission_present = submission_path.is_file()
    _assert_safe_output_root(
        output_root,
        protected_paths=(
            (case_path, submission_path) if submission_present else (case_path,)
        ),
        protected_roots=(case.item_root,),
    )
    _prepare_item_output(output_root, force=force)

    environment = environment_record()
    submission_hash = sha256_file(submission_path) if submission_present else "missing"
    identity = {
        "case_hash": str(case.data["case_hash"]),
        "profile_hash": str(case.data["profile_hash"]),
        "raw_model_output_hash": submission_hash,
        "submission_source_hash": submission_hash,
        "evaluator_build_hash": evaluator_build_hash(),
        "environment_hash": str(environment["environment_hash"]),
        "source_family": str(case.data["source_family"]),
    }
    if not force:
        existing = _resume_result(
            output_root,
            identity,
            case=case,
            environment=environment,
        )
        if existing is not None:
            return existing
    if (output_root / "result.json").exists() and not force:
        raise ValueError(
            "functional-edit output belongs to different inputs; use --force"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "case.json", case.data)
    write_json(output_root / "environment.json", environment)
    submission_root = output_root / "submission"
    submission_root.mkdir()
    copied = submission_root / submission_path.name
    if submission_present:
        shutil.copyfile(submission_path, copied)
    write_json(
        submission_root / "manifest.json",
        {
            "raw_sha256": submission_hash,
            "source_sha256": submission_hash,
            "filename": copied.name if submission_present else None,
            "state": "present" if submission_present else "missing",
            "protocol": "functional_edit",
        },
    )

    if not submission_present:
        l0 = LayerResult.from_checks(
            "L0",
            (
                CheckResult(
                    "L0.protocol_e.submission",
                    Status.FAILED,
                    failure_code="functional_edit_submission_missing",
                    observed={"submission_present": False},
                ),
            ),
        )
        l1 = _not_reached("L1", "L0")
        l2 = _not_reached("L2", "L0")
    else:
        try:
            adapter = FunctionalEditAdapter(
                repo_root=repo_root,
                case=case,
                submission_path=copied,
                output_root=output_root / "protocol_e",
            )
            source = adapter.evaluate_source_submission()
        except Exception as exc:  # source adapter and external CAD backend boundary
            l0 = LayerResult.from_checks(
                "L0",
                (
                    CheckResult(
                        "L0.protocol_e.source_adapter",
                        Status.EVALUATOR_ERROR,
                        summary=f"{type(exc).__name__}: {exc}",
                        observed={
                            "adapter_exception": isinstance(
                                exc, FunctionalEditAdapterError
                            )
                        },
                    ),
                ),
            )
            l1 = _not_reached("L1", "L0")
            l2 = _not_reached("L2", "L0")
        else:
            l0 = _l0_from_source(source)
            if l0.status is Status.PASSED:
                l1 = _l1_from_source(case.data, source, output_root=output_root)
            else:
                l1 = _not_reached("L1", "L0")
            if l0.status is Status.PASSED and l1.status is Status.PASSED:
                l2 = evaluate_l2_functional(case.data, adapter=adapter, source=source)
            else:
                l2 = _not_reached("L2", "L1")

    for layer in (l0, l1, l2):
        _write_layer(output_root, layer)
    layers = (l0, l1, l2)
    result = EvaluationResult(
        schema_version=RESULT_SCHEMA,
        evaluator_version=__version__,
        item_id=case.item_id,
        protocol="functional_edit",
        identity=identity,
        layers=layers,
        overall_status=_overall_status(layers),
        environment=environment,
        runtime_seconds=time.monotonic() - started,
    )
    _write_result(output_root, result)
    return result
