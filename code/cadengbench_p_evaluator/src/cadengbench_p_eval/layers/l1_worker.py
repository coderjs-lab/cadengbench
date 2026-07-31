"""Fresh-process worker for one L1 geometry and DFM evaluation."""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cadengbench_p_eval.case import LoadedCase
from cadengbench_p_eval.io import read_json, write_json
from cadengbench_p_eval.layers.l1 import evaluate_l1
from cadengbench_p_eval.layers.l1_isolated import (
    L1_JOB_SCHEMA,
    L1_WORKER_REPORT_SCHEMA,
)

_TRACEBACK_CHARACTERS = 16_384


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _deserialize_case(value: Any) -> LoadedCase | Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("case payload must be an object")
    kind = value.get("kind")
    data = value.get("data")
    if not isinstance(data, dict):
        raise ValueError("case data must be an object")
    if kind == "mapping":
        return data
    if kind == "loaded_case":
        path = value.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("loaded case payload requires an absolute case path")
        return LoadedCase(path=Path(path), data=data)
    raise ValueError("case payload kind is unsupported")


def _boolean(job: Mapping[str, Any], key: str) -> bool:
    value = job.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _run_job(job_path: Path) -> dict[str, Any]:
    job = read_json(job_path)
    if not isinstance(job, Mapping) or job.get("schema_version") != L1_JOB_SCHEMA:
        raise ValueError("job schema is absent or unsupported")
    candidate = job.get("candidate_step")
    target = job.get("target_step")
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("candidate STEP path must be a non-empty string")
    if target is not None and (not isinstance(target, str) or not target):
        raise ValueError("target STEP path must be null or a non-empty string")
    result = evaluate_l1(
        _deserialize_case(job.get("case")),
        Path(candidate),
        target_step=Path(target) if target is not None else None,
        is_parameter_variant=_boolean(job, "is_parameter_variant"),
        l0_passed=_boolean(job, "l0_passed"),
    )
    return result.as_dict()


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = _run_job(arguments.job)
    except BaseException as exc:
        report = {
            "schema_version": L1_WORKER_REPORT_SCHEMA,
            "outcome": "worker_error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-_TRACEBACK_CHARACTERS:],
            },
        }
        try:
            write_json(arguments.report, report)
        except BaseException:
            return 2
        return 1
    write_json(
        arguments.report,
        {
            "schema_version": L1_WORKER_REPORT_SCHEMA,
            "outcome": "completed",
            "result": result,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
