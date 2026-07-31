"""L0: submission, execution, B-Rep, and STEP-deliverable validity."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status
from cadengbench_p_eval.execution.runner import run_submission

_CHECK_IDS = {
    "static_contract": "L0.submission.contract",
    "execution": "L0.submission.execution",
    "step_output": "L0.deliverable.model_step",
    "geometry_validity": "L0.geometry.brep_validity",
    "body_count": "L0.geometry.solid_count",
    "step_reimport": "L0.deliverable.step_reimport",
    "roundtrip": "L0.deliverable.roundtrip_invariants",
}


def _check_result(name: str, stage: Mapping[str, Any]) -> CheckResult:
    status = Status(str(stage["status"]))
    evidence = list(stage.get("evidence") or [])
    ownership = stage.get("ownership")
    runner_failure = stage.get("failure_code")
    if ownership is not None or runner_failure is not None:
        evidence.append(
            {
                "failure_ownership": ownership,
                "runner_failure_code": runner_failure,
            }
        )
    return CheckResult(
        check_id=_CHECK_IDS[name],
        status=status,
        required=name != "body_count",
        failure_code=str(runner_failure)
        if status is Status.FAILED and runner_failure
        else None,
        summary=str(stage.get("summary") or ""),
        expected=stage.get("expected"),
        observed=stage.get("observed"),
        metrics=dict(stage.get("metrics") or {}),
        evidence=evidence,
    )


def _write_layer_result(path: Path, layer: LayerResult) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(layer.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate_l0(
    *,
    case: Mapping[str, Any],
    submission_path: Path,
    output_dir: Path,
    overrides: Mapping[str, Any] | None = None,
) -> LayerResult:
    """Evaluate one CadQuery submission without consulting target geometry.

    ``overrides`` is reserved for L2 replay and still traverses every L0 gate.
    Local execution is explicitly recorded as non-canonical containment.
    """

    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        execution = run_submission(
            source_path=submission_path,
            case=case,
            output_dir=output_dir,
            overrides=overrides,
        )
        checks = tuple(
            _check_result(name, execution["stages"][name]) for name in _CHECK_IDS
        )
        layer = LayerResult.from_checks(
            "L0",
            checks,
            metrics={
                "facts": execution.get("facts"),
                "override": execution.get("override"),
                "containment": execution.get("containment"),
            },
            artifacts=dict(execution.get("artifacts") or {}),
            runtime_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        layer = LayerResult.from_checks(
            "L0",
            (
                CheckResult(
                    check_id="L0.evaluator.internal",
                    status=Status.EVALUATOR_ERROR,
                    summary=f"L0 evaluator raised {type(exc).__name__}: {exc}",
                    evidence=[
                        {
                            "failure_ownership": "evaluator",
                            "exception_type": type(exc).__name__,
                        }
                    ],
                ),
            ),
            runtime_seconds=time.monotonic() - started,
        )
    _write_layer_result(output_dir / "l0.json", layer)
    return layer


evaluate = evaluate_l0
