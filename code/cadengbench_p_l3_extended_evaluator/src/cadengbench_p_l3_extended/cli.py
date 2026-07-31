"""Command-line interface for the L3 extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cadengbench_p_l3_extended.annotations import compile_body_cards
from cadengbench_p_l3_extended.audit import run_audit
from cadengbench_p_l3_extended.body_response import (
    calibrate_body_response,
    evaluate_body_response_pair,
)
from cadengbench_p_l3_extended.qualification import qualify_body_cards


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cadengbench-p-l3-extended")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-coverage")
    audit.add_argument("--repo-root", type=Path, required=True)
    audit.add_argument("--profile", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    compile_cards = subparsers.add_parser("compile-body-cards")
    compile_cards.add_argument("--profile", type=Path, required=True)
    compile_cards.add_argument("--audit-root", type=Path, required=True)
    compile_cards.add_argument("--output", type=Path, required=True)
    qualify = subparsers.add_parser("qualify-body")
    qualify.add_argument("--repo-root", type=Path, required=True)
    qualify.add_argument("--profile", type=Path, required=True)
    qualify.add_argument("--audit-root", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--ccx", type=Path, required=True)
    qualify.add_argument("--workers", type=int, default=3)
    qualify.add_argument("--timeout-seconds", type=int, default=900)
    qualify.add_argument("--item-limit", type=int)
    qualify.add_argument("--render-count", type=int, default=0)
    calibrate = subparsers.add_parser("calibrate-response")
    calibrate.add_argument("--profile", type=Path, required=True)
    calibrate.add_argument("--card", type=Path, required=True)
    calibrate.add_argument("--reference-step", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--ccx", type=Path, required=True)
    calibrate.add_argument("--timeout-seconds", type=int, default=300)
    calibrate.add_argument("--force", action="store_true")
    pair = subparsers.add_parser("evaluate-response-pair")
    pair.add_argument("--contract", type=Path, required=True)
    pair.add_argument("--card", type=Path, required=True)
    pair.add_argument("--reference-step", type=Path, required=True)
    pair.add_argument("--candidate-step", type=Path, required=True)
    pair.add_argument("--variant-id", required=True)
    pair.add_argument("--upstream-status", default="passed")
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--ccx", type=Path, required=True)
    pair.add_argument("--timeout-seconds", type=int, default=300)
    pair.add_argument("--render", action="store_true")
    pair.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "audit-coverage":
        result = run_audit(
            repo_root=args.repo_root,
            profile_path=args.profile,
            output_dir=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "compile-body-cards":
        result = compile_body_cards(
            profile_path=args.profile,
            audit_root=args.audit_root,
            output_root=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "qualify-body":
        result = qualify_body_cards(
            repo_root=args.repo_root,
            profile_path=args.profile,
            audit_root=args.audit_root,
            output_root=args.output,
            ccx_path=args.ccx,
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
            item_limit=args.item_limit,
            render_count=args.render_count,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "calibrate-response":
        result = calibrate_body_response(
            profile_path=args.profile,
            card_path=args.card,
            reference_step=args.reference_step,
            output_dir=args.output,
            ccx_path=args.ccx,
            timeout_seconds=args.timeout_seconds,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate-response-pair":
        result = evaluate_body_response_pair(
            contract_path=args.contract,
            card_path=args.card,
            reference_step=args.reference_step,
            candidate_step=args.candidate_step,
            variant_id=args.variant_id,
            upstream_status=args.upstream_status,
            output_dir=args.output,
            ccx_path=args.ccx,
            timeout_seconds=args.timeout_seconds,
            render=args.render,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
