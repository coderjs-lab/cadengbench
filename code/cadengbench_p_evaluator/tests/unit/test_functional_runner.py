from pathlib import Path

import cadengbench_p_eval.functional_runner as functional_runner
from cadengbench_p_eval.case import LoadedCase
from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.io import sha256_json


def test_missing_functional_submission_is_a_model_failure(
    tmp_path: Path, monkeypatch
) -> None:
    profile = {
        "schema_version": "cadengbench-p.evaluator-profile.v1",
        "profile_id": "unit-profile-v1",
    }
    case = LoadedCase(
        path=tmp_path / "case.json",
        data={
            "item_id": "p_benchcad_missing",
            "case_hash": "case-hash",
            "source_family": "benchcad",
            "source_item_root": str(tmp_path / "source-item"),
            "profile_hash": sha256_json(profile),
            "evaluator_profile": profile,
            "protocol": "functional_edit",
        },
    )
    monkeypatch.setattr(
        functional_runner, "load_case", lambda *_args, **_kwargs: case
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output_root = tmp_path / "result"

    result = functional_runner.evaluate_functional_item(
        repo_root=repo_root,
        case_path=case.path,
        submission_path=tmp_path / "missing/submission.py",
        output_root=output_root,
    )

    assert result.overall_status is Status.FAILED
    assert [layer.status for layer in result.layers] == [
        Status.FAILED,
        Status.NOT_REACHED,
        Status.NOT_REACHED,
    ]
    assert (
        result.layers[0].checks[0].failure_code
        == "functional_edit_submission_missing"
    )
    assert result.identity["raw_model_output_hash"] == "missing"
    assert (output_root / "result.json").is_file()


def test_missing_runtime_dependency_is_an_evaluator_error(
    tmp_path: Path, monkeypatch
) -> None:
    profile = {
        "schema_version": "cadengbench-p.evaluator-profile.v1",
        "profile_id": "unit-profile-v1",
    }
    case = LoadedCase(
        path=tmp_path / "case.json",
        data={
            "item_id": "p_fusion360_dependency",
            "case_hash": "case-hash",
            "source_family": "fusion360_reconstruction",
            "source_item_root": str(tmp_path / "source-item"),
            "profile_hash": sha256_json(profile),
            "evaluator_profile": profile,
            "protocol": "functional_edit",
        },
    )
    monkeypatch.setattr(
        functional_runner, "load_case", lambda *_args, **_kwargs: case
    )

    class MissingDependencyAdapter:
        def __init__(self, **_kwargs) -> None:
            raise FileNotFoundError("required evaluator schema is missing")

    monkeypatch.setattr(
        functional_runner, "FunctionalEditAdapter", MissingDependencyAdapter
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    submission = tmp_path / "fusion_patch.json"
    submission.write_text("{}\n", encoding="utf-8")

    result = functional_runner.evaluate_functional_item(
        repo_root=repo_root,
        case_path=case.path,
        submission_path=submission,
        output_root=tmp_path / "result",
    )

    assert result.overall_status is Status.EVALUATOR_ERROR
    assert result.layers[0].status is Status.EVALUATOR_ERROR
    assert result.layers[0].checks[0].failure_code is None
