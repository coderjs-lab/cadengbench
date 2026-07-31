"""Command-line interface for planning, running, resuming, and packaging."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .benchmark import prepare_benchmark, run_benchmark
from .io import read_json
from .l3_benchmark import (
    apply_l3_pair_selection,
    build_binding_index,
    plan_l3_run,
    run_l3_plan,
    summarize_l3_run,
    validate_release,
)
from .packaging import package_run
from .planner import plan_run
from .scheduler import run_generation


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cadengbench-p-run",
        description="Provider-neutral CADEngBench-P candidate-model runner",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="report runner version and protocol support")

    plan = subparsers.add_parser("plan", help="freeze requests before inference")
    plan.add_argument("--items-root", required=True, type=_path)
    plan.add_argument("--model-spec", required=True, type=_path)
    plan.add_argument("--output", required=True, type=_path)
    plan.add_argument("--run-id", required=True)
    plan.add_argument(
        "--task-kind",
        choices=("zero_to_cad", "functional_edit"),
        default="zero_to_cad",
    )
    plan.add_argument(
        "--input-profile",
        choices=("text", "text_iso", "text_engineering_views"),
        default="text_iso",
    )
    plan.add_argument("--sample-count", type=int, default=1)
    plan.add_argument("--max-attempts", type=int, default=3)
    plan.add_argument("--request-timeout-seconds", type=int, default=180)
    plan.add_argument("--max-retry-elapsed-seconds", type=int, default=900)
    plan.add_argument(
        "--item-id",
        action="append",
        dest="item_ids",
        help="select an item; repeat to freeze an explicit ordered roster",
    )

    for name, help_text in (
        ("run", "execute pending requests"),
        ("resume", "resume an interrupted request run"),
        ("status", "verify and summarize durable run state"),
        ("package", "rebuild the sorted evaluator-facing manifest"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--run", required=True, type=_path)

    for name, help_text in (
        ("native-batch", "execute through an OpenAI-compatible Batch API"),
        ("gemini-batch", "execute through Gemini Batch with Gemini file I/O"),
        ("anthropic-batch", "execute through Anthropic Message Batches"),
        ("mistral-batch", "execute through Mistral Batch Jobs"),
    ):
        native_batch = subparsers.add_parser(name, help=help_text)
        native_batch.add_argument("--run", required=True, type=_path)
        native_batch.add_argument("--base-url", required=True)
        native_batch.add_argument("--api-key-env", required=True)
        native_batch.add_argument("--chunk-size", type=int, default=25)
        native_batch.add_argument("--poll-seconds", type=int, default=60)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="resume generation and scoring for both frozen L0-L2 protocols",
    )
    benchmark.add_argument("--evalset", required=True, type=_path)
    benchmark.add_argument("--model-spec", required=True, type=_path)
    benchmark.add_argument("--output", required=True, type=_path)
    benchmark.add_argument("--run-id", required=True)
    benchmark.add_argument("--evaluator-cli", required=True, type=_path)
    benchmark.add_argument("--repo-root", required=True, type=_path)
    benchmark.add_argument(
        "--input-profile",
        choices=("text", "text_iso", "text_engineering_views"),
        default="text_iso",
    )
    benchmark.add_argument("--workers", type=int, default=1)
    benchmark.add_argument(
        "--item-id",
        action="append",
        dest="item_ids",
        help="benchmark only this item from the frozen release; repeat for a subset",
    )

    preflight = subparsers.add_parser(
        "preflight",
        help="freeze and verify both 300-request protocols without model inference",
    )
    preflight.add_argument("--evalset", required=True, type=_path)
    preflight.add_argument("--model-spec", required=True, type=_path)
    preflight.add_argument("--output", required=True, type=_path)
    preflight.add_argument("--run-id", required=True)
    preflight.add_argument("--evaluator-cli", required=True, type=_path)
    preflight.add_argument(
        "--input-profile",
        choices=("text", "text_iso", "text_engineering_views"),
        default="text_iso",
    )
    preflight.add_argument(
        "--item-id",
        action="append",
        dest="item_ids",
        help="preflight only this item from the frozen release; repeat for a subset",
    )

    validate = subparsers.add_parser(
        "validate-release", help="validate the frozen CADEngBench-P roster and payload"
    )
    validate.add_argument("--release", required=True, type=_path)
    validate.add_argument(
        "--metadata-only",
        action="store_true",
        help="allow a metadata-only mirror without the canonical items/ payload",
    )

    l3_index = subparsers.add_parser(
        "l3-index", help="bind the frozen release to qualified L3 artifacts"
    )
    l3_index.add_argument("--release", required=True, type=_path)
    l3_index.add_argument("--qualification-summary", required=True, type=_path)
    l3_index.add_argument("--sidecar", required=True, type=_path)
    l3_index.add_argument("--contract-root", required=True, type=_path)
    l3_index.add_argument("--applicability-index", type=_path)
    l3_index.add_argument("--body-contract-root", type=_path)
    l3_index.add_argument("--output", required=True, type=_path)

    l3_plan = subparsers.add_parser(
        "l3-plan", help="compose L0-L2 evidence into an executable L3 plan"
    )
    l3_plan.add_argument("--release", required=True, type=_path)
    l3_plan.add_argument("--l02-run", required=True, type=_path)
    l3_plan.add_argument("--bindings", required=True, type=_path)
    l3_plan.add_argument("--reference-bank", required=True, type=_path)
    l3_plan.add_argument("--alignment", required=True, type=_path)
    l3_plan.add_argument("--response-profile", required=True, type=_path)
    l3_plan.add_argument("--body-response-profile", type=_path)
    l3_plan.add_argument("--pair-selection", type=_path)
    l3_plan.add_argument("--output", required=True, type=_path)

    l3_select = subparsers.add_parser(
        "l3-select-pairs", help="apply a fixed pair-selection policy to an L3 plan"
    )
    l3_select.add_argument("--plan", required=True, type=_path)
    l3_select.add_argument("--pair-selection", required=True, type=_path)
    l3_select.add_argument("--output", required=True, type=_path)

    l3_run = subparsers.add_parser(
        "l3-run", help="execute all scoreable pairs from a frozen L3 plan"
    )
    l3_run.add_argument("--plan", required=True, type=_path)
    l3_run.add_argument("--output", required=True, type=_path)
    l3_run.add_argument("--l3-cli", required=True, type=_path)
    l3_run.add_argument("--body-l3-cli", type=_path)
    l3_run.add_argument("--ccx", required=True, type=_path)
    l3_run.add_argument("--workers", type=int, default=1)
    l3_run.add_argument("--timeout-seconds", type=int, default=300)
    l3_run.add_argument("--render", action="store_true")

    l3_summary = subparsers.add_parser(
        "l3-summarize", help="write item and protocol-variant L3 result tables"
    )
    l3_summary.add_argument("--plan", required=True, type=_path)
    l3_summary.add_argument("--run", required=True, type=_path)
    l3_summary.add_argument("--output", required=True, type=_path)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "doctor":
            _print(
                {
                    "ready": True,
                    "runner_version": __version__,
                    "providers": ["openai_compatible"],
                    "protocols": ["zero_to_cad", "functional_edit"],
                    "evaluation_layers": ["L0", "L1", "L2", "L3"],
                    "l3_protocols": ["zero_to_cad"],
                    "l3_comparison_modes": ["matched_parameter_response"],
                    "l3_profiles": ["frozen_v1", "extended_v1"],
                    "l3_default_hpc_profile": "extended_v1",
                    "l3_scenario_families": [
                        "axial_tension",
                        "cantilever_transverse",
                        "single_support_body_acceleration",
                    ],
                    "generation_scoring_separated": True,
                    "llm_judge_used": False,
                }
            )
            return 0
        if arguments.command == "plan":
            manifest = plan_run(
                items_root=arguments.items_root,
                model_spec_path=arguments.model_spec,
                output_root=arguments.output,
                run_id=arguments.run_id,
                task_kind=arguments.task_kind,
                input_profile=arguments.input_profile,
                sample_count=arguments.sample_count,
                max_attempts=arguments.max_attempts,
                request_timeout_seconds=arguments.request_timeout_seconds,
                max_retry_elapsed_seconds=arguments.max_retry_elapsed_seconds,
                item_ids=arguments.item_ids,
            )
            _print(
                {
                    "run_id": manifest["run_id"],
                    "manifest_hash": manifest["manifest_hash"],
                    "item_count": manifest["dataset"]["item_count"],
                    "sample_count": manifest["sample_count"],
                    "output": str(arguments.output),
                }
            )
            return 0
        if arguments.command in {"run", "resume"}:
            summary = run_generation(arguments.run)
            _print(summary)
            return 0 if summary["run_settled"] else 2
        if arguments.command in {
            "native-batch",
            "gemini-batch",
            "anthropic-batch",
            "mistral-batch",
        }:
            from .native_batch import (
                run_anthropic_batch,
                run_gemini_batch,
                run_mistral_batch,
                run_openai_batch,
            )

            runners = {
                "native-batch": run_openai_batch,
                "gemini-batch": run_gemini_batch,
                "anthropic-batch": run_anthropic_batch,
                "mistral-batch": run_mistral_batch,
            }
            _print(
                runners[arguments.command](
                    run_root=arguments.run,
                    base_url=arguments.base_url,
                    api_key_env=arguments.api_key_env,
                    chunk_size=arguments.chunk_size,
                    poll_seconds=arguments.poll_seconds,
                )
            )
            return 0
        if arguments.command in {"status", "package"}:
            summary = package_run(arguments.run)
            _print(summary)
            return 0 if summary["run_settled"] else 2
        if arguments.command == "benchmark":
            summary = run_benchmark(
                evalset_root=arguments.evalset,
                model_spec_path=arguments.model_spec,
                output_root=arguments.output,
                run_id=arguments.run_id,
                evaluator_cli=arguments.evaluator_cli,
                repo_root=arguments.repo_root,
                input_profile=arguments.input_profile,
                workers=arguments.workers,
                item_ids=arguments.item_ids,
            )
            _print(summary)
            return 0
        if arguments.command == "preflight":
            summary = prepare_benchmark(
                evalset_root=arguments.evalset,
                model_spec_path=arguments.model_spec,
                output_root=arguments.output,
                run_id=arguments.run_id,
                evaluator_cli=arguments.evaluator_cli,
                input_profile=arguments.input_profile,
                item_ids=arguments.item_ids,
            )
            _print(summary)
            return 0
        if arguments.command == "validate-release":
            release = validate_release(
                arguments.release, require_items=not arguments.metadata_only
            )
            _print(
                {
                    "dataset_version": release["release"]["dataset_version"],
                    "item_count": len(release["item_ids"]),
                    "payload_checked": not arguments.metadata_only,
                    "valid": True,
                }
            )
            return 0
        if arguments.command == "l3-index":
            index = build_binding_index(
                release_root=arguments.release,
                qualification_summary_path=arguments.qualification_summary,
                sidecar_root=arguments.sidecar,
                contract_root=arguments.contract_root,
                output_path=arguments.output,
                applicability_index_path=arguments.applicability_index,
                body_contract_root=arguments.body_contract_root,
            )
            _print(
                {
                    "item_count": index["item_count"],
                    "solver_qualified_count": index["solver_qualified_count"],
                    "output": str(arguments.output),
                }
            )
            return 0
        if arguments.command == "l3-plan":
            plan = plan_l3_run(
                release_root=arguments.release,
                l02_run=arguments.l02_run,
                bindings_path=arguments.bindings,
                reference_bank_path=arguments.reference_bank,
                alignment_path=arguments.alignment,
                response_profile_path=arguments.response_profile,
                body_response_profile_path=arguments.body_response_profile,
                pair_selection_path=arguments.pair_selection,
                output_path=arguments.output,
            )
            _print(
                {
                    "item_count": plan["item_count"],
                    "ready_item_count": plan["ready_item_count"],
                    "output": str(arguments.output),
                }
            )
            return 0
        if arguments.command == "l3-select-pairs":
            plan = apply_l3_pair_selection(
                plan=read_json(arguments.plan),
                selection_path=arguments.pair_selection,
                output_path=arguments.output,
            )
            _print(
                {
                    "candidate_pair_count": plan["pair_selection"][
                        "candidate_pair_count"
                    ],
                    "selected_pair_count": plan["pair_selection"][
                        "selected_pair_count"
                    ],
                    "output": str(arguments.output),
                }
            )
            return 0
        if arguments.command == "l3-run":
            execution = run_l3_plan(
                plan_path=arguments.plan,
                output_root=arguments.output,
                l3_cli=arguments.l3_cli,
                body_l3_cli=arguments.body_l3_cli,
                ccx_path=arguments.ccx,
                workers=arguments.workers,
                timeout_seconds=arguments.timeout_seconds,
                render=arguments.render,
            )
            _print(
                {
                    "planned_pair_count": execution["planned_pair_count"],
                    "completed_pair_count": execution["completed_pair_count"],
                    "calibration_error_count": len(execution["calibration_errors"]),
                    "output": str(arguments.output),
                }
            )
            return (
                0
                if (
                    execution["completed_pair_count"] == execution["planned_pair_count"]
                    and not execution["calibration_errors"]
                )
                else 2
            )
        if arguments.command == "l3-summarize":
            summary = summarize_l3_run(
                plan_path=arguments.plan,
                run_root=arguments.run,
                output_root=arguments.output,
            )
            _print(
                {
                    key: summary[key]
                    for key in (
                        "item_count",
                        "l3_applicable_item_count",
                        "protocol_variant_row_count",
                        "physical_pair_count",
                        "physical_diagnostic_status",
                        "physical_diagnostic_reason",
                        "l3_scoreable_pair_count",
                        "l3_status",
                        "l3_reason",
                        "l3_scoreable_variant_row_count",
                        "solver_completion_rate",
                        "engineering_requirement_pass_rate_on_solved",
                        "response_fidelity_pass_rate_on_solved",
                        "l3_pass_rate",
                        "l3_family_pass_rate",
                        "mean_response_fidelity_on_solved",
                    )
                }
            )
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
