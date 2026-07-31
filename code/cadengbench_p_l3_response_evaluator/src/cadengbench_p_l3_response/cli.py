"""Command-line interface for the standalone response evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cadengbench_p_l3_response.provenance import evaluator_identity
from cadengbench_p_l3_response.runner import (
    bind_family_manifest,
    calibrate_item,
    evaluate_pair,
    run_family,
)


def _ccx_default() -> Path | None:
    value = os.environ.get("CADENGBENCH_CCX")
    return Path(value) if value else None


def _add_solver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ccx", type=Path, default=_ccx_default(), required=_ccx_default() is None
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cadengbench-p-l3-response")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor")
    _add_solver_arguments(doctor)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--profile", type=Path, required=True)
    calibrate.add_argument("--card", type=Path, required=True)
    calibrate.add_argument("--reference-step", type=Path, required=True)
    calibrate.add_argument("--mesh-study", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--force", action="store_true")
    _add_solver_arguments(calibrate)

    pair = commands.add_parser("evaluate-pair")
    pair.add_argument("--contract", type=Path, required=True)
    pair.add_argument("--card", type=Path, required=True)
    pair.add_argument("--reference-step", type=Path, required=True)
    pair.add_argument("--candidate-step", type=Path, required=True)
    pair.add_argument("--variant-id", required=True)
    pair.add_argument("--upstream-status", default="passed")
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--render", action="store_true")
    pair.add_argument("--force", action="store_true")
    _add_solver_arguments(pair)

    bind = commands.add_parser("bind-family")
    bind.add_argument("--contract", type=Path, required=True)
    bind.add_argument("--card", type=Path, required=True)
    bind.add_argument("--variants", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)

    family = commands.add_parser("run-family")
    family.add_argument("--manifest", type=Path, required=True)
    family.add_argument("--output", type=Path, required=True)
    family.add_argument("--workers", type=int, default=1)
    family.add_argument("--render", action="store_true")
    family.add_argument("--force", action="store_true")
    _add_solver_arguments(family)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "doctor":
            from cadengbench_p_eval.physics.calculix import calculix_version

            runtime_lock = os.environ.get("CADENGBENCH_L3_RUNTIME_LOCK")
            report = {
                "evaluator": evaluator_identity(),
                "ccx_path": str(arguments.ccx.resolve()),
                "ccx_version": calculix_version(arguments.ccx),
                "runtime_lock": runtime_lock,
                "runtime_lock_available": bool(
                    runtime_lock and Path(runtime_lock).is_file()
                ),
            }
            if not report["runtime_lock_available"]:
                raise RuntimeError("CADENGBENCH_L3_RUNTIME_LOCK is unavailable")
            result = report
        elif arguments.command == "calibrate":
            result = calibrate_item(
                profile_path=arguments.profile,
                card_path=arguments.card,
                reference_step=arguments.reference_step,
                mesh_study_path=arguments.mesh_study,
                output_dir=arguments.output,
                ccx_path=arguments.ccx,
                timeout_seconds=arguments.timeout_seconds,
                force=arguments.force,
            )
        elif arguments.command == "evaluate-pair":
            result = evaluate_pair(
                contract_path=arguments.contract,
                card_path=arguments.card,
                reference_step=arguments.reference_step,
                candidate_step=arguments.candidate_step,
                variant_id=arguments.variant_id,
                upstream_status=arguments.upstream_status,
                output_dir=arguments.output,
                ccx_path=arguments.ccx,
                timeout_seconds=arguments.timeout_seconds,
                render=arguments.render,
                force=arguments.force,
            )
        elif arguments.command == "bind-family":
            result = bind_family_manifest(
                contract_path=arguments.contract,
                card_path=arguments.card,
                variants_spec_path=arguments.variants,
                output_path=arguments.output,
            )
        else:
            result = run_family(
                manifest_path=arguments.manifest,
                output_dir=arguments.output,
                ccx_path=arguments.ccx,
                workers=arguments.workers,
                timeout_seconds=arguments.timeout_seconds,
                render=arguments.render,
                force=arguments.force,
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
