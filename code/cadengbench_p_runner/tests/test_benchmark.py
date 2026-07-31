from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import write_json

from cadengbench_p_runner import benchmark


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_frozen_release_versions_are_supported(version: str) -> None:
    assert benchmark._is_frozen_release(
        {
            "schema_version": f"cadengbench-p.l0-l2-release.{version}",
            "item_count": 300,
        }
    )


def test_unknown_release_version_is_rejected() -> None:
    assert not benchmark._is_frozen_release(
        {
            "schema_version": "cadengbench-p.l0-l2-release.v4",
            "item_count": 300,
        }
    )


def test_complete_benchmark_exposes_zero_to_cad_l3_handoff(
    monkeypatch, tmp_path: Path
) -> None:
    evalset = tmp_path / "evalset"
    write_json(
        evalset / "release_manifest.json",
        {
            "schema_version": "cadengbench-p.l0-l2-release.v1",
            "item_count": 300,
            "release_hash": "release-test",
        },
    )
    model_spec = tmp_path / "model.json"
    write_json(model_spec, {"model_id": "mock"})
    evaluator_cli = tmp_path / "cadengbench-p-eval"
    evaluator_cli.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(
        benchmark,
        "prepare_benchmark",
        lambda **values: {
            "benchmark_plan_hash": "plan-test",
            "evalset_release_hash": "release-test",
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_plan_or_load",
        lambda **values: {
            "manifest_hash": f"manifest-{values['protocol']}",
            "model": {"model_id": "mock"},
        },
    )
    monkeypatch.setattr(
        benchmark,
        "run_generation",
        lambda run_root: {
            "expected_jobs": 300,
            "succeeded_jobs": 299,
            "terminal_failed_jobs": 1,
            "truncated_jobs": 1,
            "run_settled": True,
            "attempt_count": 301,
            "attempt_status_counts": {"extraction_failure": 1, "succeeded": 300},
            "recorded_latency_seconds": 1234.5,
            "usage_totals": {"completion_tokens": 1000.0},
        },
    )
    monkeypatch.setattr(
        benchmark,
        "package_run",
        lambda run_root: {"artifact_manifest": {"sha256": "artifact-test"}},
    )

    def fake_command(
        command: list[str],
        log_path: Path,
        *,
        completed_summary_path: Path | None = None,
    ) -> None:
        del completed_summary_path
        if command[1] == "aggregate":
            scoring_root = Path(command[command.index("--run") + 1])
            write_json(scoring_root / "aggregate.json", {"status": "complete"})

    monkeypatch.setattr(benchmark, "_run_command", fake_command)
    output = tmp_path / "run"
    summary = benchmark.run_benchmark(
        evalset_root=evalset,
        model_spec_path=model_spec,
        output_root=output,
        run_id="mock-run",
        evaluator_cli=evaluator_cli,
        repo_root=tmp_path,
        input_profile="text_iso",
        workers=2,
    )
    assert set(summary["protocols"]) == {"zero_to_cad", "functional_edit"}
    assert summary["benchmark_plan_hash"] == "plan-test"
    assert summary["protocols"]["zero_to_cad"]["generation"] == {
        "expected_jobs": 300,
        "succeeded_jobs": 299,
        "terminal_failed_jobs": 1,
        "truncated_jobs": 1,
        "run_settled": True,
        "attempt_count": 301,
        "attempt_status_counts": {"extraction_failure": 1, "succeeded": 300},
        "recorded_latency_seconds": 1234.5,
        "usage_totals": {"completion_tokens": 1000.0},
    }
    assert summary["l3_handoff"] == {
        "protocol": "zero_to_cad",
        "l02_run": "zero_to_cad/scoring",
        "status": "ready_for_conditional_l3",
    }
    assert (output / "benchmark_summary.json").is_file()


def test_complete_model_outcome_does_not_abort_benchmark(
    monkeypatch, tmp_path: Path
) -> None:
    summary_path = tmp_path / "scoring" / "summary.json"
    write_json(
        summary_path,
        {
            "execution_complete": True,
            "item_evaluator_error_count": 0,
            "publication_valid": False,
            "result_roster_exact": True,
            "valid_complete_run": True,
        },
    )
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=2,
            stdout="",
            stderr="publication validity gate failed",
        ),
    )

    benchmark._run_command(
        ["cadengbench-p-eval", "evaluate"],
        tmp_path / "evaluate.log.json",
        completed_summary_path=summary_path,
    )
