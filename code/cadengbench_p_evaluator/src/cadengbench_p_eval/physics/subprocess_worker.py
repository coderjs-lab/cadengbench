"""Short-lived native-isolation worker for L3 source and variant runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cadengbench_p_eval.io import read_json, write_json
from cadengbench_p_eval.physics.calculix import load_physics_card
from cadengbench_p_eval.physics.runner import _qualify_corpus_item, _run_variant

JOB_SCHEMA = "cadengbench-p.l3-subprocess-worker-job.v1"


def _path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("worker path is missing")
    return Path(value)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("schema_version") != JOB_SCHEMA:
        raise ValueError("unsupported L3 subprocess-worker job")
    operation = job.get("operation")
    if operation == "source_qualification":
        return _qualify_corpus_item(
            item_id=str(job["item_id"]),
            card_path=_path(job["card_path"]),
            reference_step=_path(job["reference_step"]),
            qualification_dir=_path(job["qualification_dir"]),
            promoted_card_path=_path(job["promoted_card_path"]),
            ccx_path=_path(job["ccx_path"]),
            timeout_seconds=int(job["timeout_seconds"]),
            render=bool(job["render"]),
            force=bool(job["force"]),
        )
    if operation == "variant":
        record = job.get("record")
        if not isinstance(record, dict):
            raise ValueError("variant worker record is missing")
        return _run_variant(
            card=load_physics_card(_path(job["card_path"])),
            record=record,
            output_dir=_path(job["output_dir"]),
            ccx_path=_path(job["ccx_path"]),
            timeout_seconds=int(job["timeout_seconds"]),
            render=bool(job["render"]),
        )
    raise ValueError(f"unsupported L3 subprocess-worker operation: {operation}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    job = read_json(args.job)
    if not isinstance(job, dict):
        raise ValueError("worker job root is not an object")
    write_json(args.result, run_job(job))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
