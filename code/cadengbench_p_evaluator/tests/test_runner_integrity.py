from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import cadengbench_p_eval.runner as runner
from cadengbench_p_eval import __version__
from cadengbench_p_eval.case import LoadedCase
from cadengbench_p_eval.contracts import (
    CheckResult,
    EvaluationResult,
    LayerResult,
    Status,
)
from cadengbench_p_eval.io import sha256_file, sha256_json, write_json, write_text


def _case(
    path: Path,
    *,
    item_id: str = "p_benchcad_integrity",
    case_hash: str = "case-hash",
    source_family: str = "benchcad",
    protocol: str = "zero_to_cad",
    profile: dict[str, Any] | None = None,
) -> LoadedCase:
    evaluator_profile = profile or {
        "schema_version": "cadengbench-p.evaluator-profile.v1",
        "profile_id": "integrity-profile-v1",
    }
    return LoadedCase(
        path=path,
        data={
            "item_id": item_id,
            "case_hash": case_hash,
            "source_family": source_family,
            "profile_hash": sha256_json(evaluator_profile),
            "evaluator_profile": evaluator_profile,
            "protocol": protocol,
        },
    )


def _write_evalset(evalset: Path, case: LoadedCase) -> Path:
    case_path = evalset / "cases" / f"{case.item_id}.case.json"
    write_json(case_path, case.data)
    record = {
        "item_id": case.item_id,
        "protocol": case.data["protocol"],
        "case_path": case_path.relative_to(evalset).as_posix(),
        "case_hash": case.data["case_hash"],
        "source_family": case.data["source_family"],
    }
    manifest = {
        "schema_version": "cadengbench-p.evalset-manifest.v1",
        "profile_id": case.data["evaluator_profile"]["profile_id"],
        "profile_hash": case.data["profile_hash"],
        "protocol": case.data["protocol"],
        "item_count": 1,
        "items": [record],
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    write_json(evalset / "manifest.json", manifest)
    return case_path


def _check(check_id: str, status: Status, *, required: bool = True) -> CheckResult:
    return CheckResult(
        check_id,
        status,
        required=required,
        failure_code="model_failure" if status is Status.FAILED else None,
    )


def _batch_result(
    item_id: str,
    status: Status = Status.PASSED,
    *,
    protocol: str = "zero_to_cad",
) -> dict[str, Any]:
    if status is Status.PASSED:
        layers = tuple(
            LayerResult.from_checks(name, (_check(f"{name}.check", status),))
            for name in ("L0", "L1", "L2")
        )
    elif status is Status.EVALUATOR_ERROR:
        layers = tuple(
            LayerResult.from_checks(
                name,
                (
                    _check(f"{name}.required", Status.PASSED),
                    _check(f"{name}.optional", status, required=False),
                ),
            )
            for name in ("L0", "L1", "L2")
        )
    elif status is Status.NOT_SCOREABLE:
        layers = (
            LayerResult.from_checks("L0", (_check("L0.check", Status.PASSED),)),
            LayerResult.from_checks("L1", (_check("L1.check", status),)),
            runner._not_reached("L2", "L1"),
        )
    else:
        raise AssertionError(f"unsupported test status: {status}")
    return EvaluationResult(
        schema_version=runner.RESULT_SCHEMA,
        evaluator_version=__version__,
        item_id=item_id,
        protocol=protocol,
        identity={"source_family": "benchcad"},
        layers=layers,
        overall_status=runner._overall_status(layers),
        environment={},
        runtime_seconds=0.0,
    ).as_dict()


def _install_fake_batch(
    monkeypatch: pytest.MonkeyPatch,
    case: LoadedCase,
    result: dict[str, Any],
    *,
    extra_result: bool = False,
) -> None:
    monkeypatch.setattr(runner, "load_case", lambda *_args, **_kwargs: case)

    def fake_batch(
        _case_path: str,
        _submission_path: str,
        output_root: str,
        _force: bool,
    ) -> dict[str, Any]:
        item_output = Path(output_root)
        write_json(item_output / "result.json", result)
        if extra_result:
            write_json(
                item_output.parents[0] / "unexpected" / "result.json",
                _batch_result("unexpected"),
            )
        return result

    monkeypatch.setattr(runner, "_batch_job", fake_batch)


def _fake_evalset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: Status = Status.PASSED,
    extra_result: bool = False,
) -> tuple[Path, Path, Path, LoadedCase]:
    evalset = tmp_path / "evalset"
    submissions = tmp_path / "submissions"
    output = tmp_path / "output"
    case_path = evalset / "cases/p_benchcad_integrity.case.json"
    case = _case(case_path)
    _write_evalset(evalset, case)
    submission = submissions / case.item_id / "submission.py"
    submission.parent.mkdir(parents=True)
    submission.write_text("result = None\n", encoding="utf-8")
    _install_fake_batch(
        monkeypatch,
        case,
        _batch_result(case.item_id, status),
        extra_result=extra_result,
    )
    return evalset, submissions, output, case


def _resume_bundle(
    tmp_path: Path,
) -> tuple[
    Path,
    LoadedCase,
    dict[str, str],
    dict[str, Any],
    EvaluationResult,
]:
    output = tmp_path / "item-output"
    case = _case(tmp_path / "case.json")
    environment = {"environment_hash": "environment-hash", "runtime": "test"}
    submission_root = output / "submission"
    source = submission_root / "submission.py"
    raw = submission_root / "raw_output.py"
    source.parent.mkdir(parents=True)
    source.write_text("result = None\n", encoding="utf-8")
    raw.write_text("result = None\n", encoding="utf-8")
    identity = {
        "case_hash": case.data["case_hash"],
        "profile_hash": case.data["profile_hash"],
        "raw_model_output_hash": sha256_file(raw),
        "submission_source_hash": sha256_file(source),
        "evaluator_build_hash": "evaluator-build-hash",
        "environment_hash": environment["environment_hash"],
        "source_family": case.data["source_family"],
    }
    write_json(
        submission_root / "manifest.json",
        {
            "raw_sha256": identity["raw_model_output_hash"],
            "source_sha256": identity["submission_source_hash"],
            "extraction": "test",
        },
    )
    candidate = output / "l0/candidate.step"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"ISO-10303-21; test candidate")
    l0 = LayerResult.from_checks(
        "L0",
        (_check("L0.check", Status.PASSED),),
        artifacts={
            "candidate_step": {
                "path": "candidate.step",
                "sha256": sha256_file(candidate),
                "bytes": candidate.stat().st_size,
            }
        },
    )
    l1 = LayerResult.from_checks("L1", (_check("L1.check", Status.PASSED),))
    l2 = LayerResult.from_checks("L2", (_check("L2.check", Status.PASSED),))
    result = EvaluationResult(
        schema_version=runner.RESULT_SCHEMA,
        evaluator_version=__version__,
        item_id=case.item_id,
        protocol=case.data["protocol"],
        identity=identity,
        layers=(l0, l1, l2),
        overall_status=Status.PASSED,
        environment=environment,
        runtime_seconds=0.1,
    )
    write_json(output / "case.json", case.data)
    write_json(output / "environment.json", environment)
    write_text(output / "l1-worker/stderr.txt", "worker evidence\n")
    for layer in result.layers:
        runner._write_layer(output, layer)
    runner._write_result(output, result)
    return output, case, identity, environment, result


def test_swapped_case_is_rejected_before_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evalset = tmp_path / "evalset"
    expected = _case(evalset / "cases/p_benchcad_integrity.case.json")
    _write_evalset(evalset, expected)
    swapped = _case(
        expected.path,
        item_id="p_benchcad_swapped",
        case_hash=expected.data["case_hash"],
    )
    monkeypatch.setattr(runner, "load_case", lambda *_args, **_kwargs: swapped)
    submissions = tmp_path / "submissions"
    submissions.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "must-survive.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="resolves to case"):
        runner.evaluate_evalset(
            evalset_root=evalset,
            submissions_root=submissions,
            output_root=output,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("case_hash", "case_hash"),
        ("source_family", "source_family"),
        ("profile_hash", "profile_hash"),
        ("record_profile_hash", "record has a mismatched profile_hash"),
        ("profile_id", "profile_id"),
        ("profile_content", "failed its hash binding"),
    ],
)
def test_evalset_record_requires_all_case_bindings(
    tmp_path: Path, mutation: str, message: str
) -> None:
    case = _case(tmp_path / "case.json")
    record = {
        "item_id": case.item_id,
        "case_hash": case.data["case_hash"],
        "source_family": case.data["source_family"],
    }
    manifest = {
        "profile_id": case.data["evaluator_profile"]["profile_id"],
        "profile_hash": case.data["profile_hash"],
    }
    if mutation == "case_hash":
        record["case_hash"] = "different"
    elif mutation == "source_family":
        record["source_family"] = "fusion360_reconstruction"
    elif mutation == "profile_hash":
        manifest["profile_hash"] = "different"
    elif mutation == "record_profile_hash":
        record["profile_hash"] = "different"
    elif mutation == "profile_id":
        manifest["profile_id"] = "different"
    elif mutation == "profile_content":
        case.data["evaluator_profile"]["unexpected"] = True

    with pytest.raises(ValueError, match=message):
        runner._validate_evalset_case_binding(record, case, manifest)


def test_run_identity_rejects_submission_inventory_and_model_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evalset, submissions, output, case = _fake_evalset(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "aggregate_run",
        lambda *_args, **_kwargs: {"item_count": 1},
    )

    first = runner.evaluate_evalset(
        evalset_root=evalset,
        submissions_root=submissions,
        output_root=output,
        model_id="model-a",
    )
    assert first["valid_complete_run"] is True
    run_manifest = runner.read_json(output / "manifest.json")
    assert run_manifest["identity"]["model_id"] == "model-a"
    assert len(run_manifest["submission_inventory"]) == 1

    submission = submissions / case.item_id / "submission.py"
    original = submission.read_text(encoding="utf-8")
    submission.write_text("result = 'changed'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="submission inventory differs"):
        runner.evaluate_evalset(
            evalset_root=evalset,
            submissions_root=submissions,
            output_root=output,
            resume=True,
            model_id="model-a",
        )

    submission.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="model_id differs"):
        runner.evaluate_evalset(
            evalset_root=evalset,
            submissions_root=submissions,
            output_root=output,
            resume=True,
            model_id="model-b",
        )


def test_functional_evalset_requires_repo_root_and_dispatches_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evalset = tmp_path / "evalset"
    submissions = tmp_path / "submissions"
    output = tmp_path / "output"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    case = _case(
        evalset / "cases/p_fusion360_integrity.case.json",
        item_id="p_fusion360_integrity",
        source_family="fusion360_reconstruction",
        protocol="functional_edit",
    )
    _write_evalset(evalset, case)
    submission = submissions / case.item_id / "fusion_patch.json"
    submission.parent.mkdir(parents=True)
    submission.write_text('{"edits": []}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "load_case", lambda *_args, **_kwargs: case)
    result = _batch_result(case.item_id, protocol="functional_edit")
    calls: list[tuple[str, str]] = []

    def fake_batch(
        _case_path: str,
        submission_path: str,
        output_root: str,
        _force: bool,
        received_repo_root: str,
    ) -> dict[str, Any]:
        calls.append((submission_path, received_repo_root))
        write_json(Path(output_root) / "result.json", result)
        return result

    monkeypatch.setattr(runner, "_batch_job", fake_batch)
    monkeypatch.setattr(
        runner,
        "aggregate_run",
        lambda *_args, **_kwargs: {"item_count": 1},
    )

    with pytest.raises(ValueError, match="require --repo-root"):
        runner.evaluate_evalset(
            evalset_root=evalset,
            submissions_root=submissions,
            output_root=output,
        )

    summary = runner.evaluate_evalset(
        evalset_root=evalset,
        submissions_root=submissions,
        output_root=output,
        model_id="functional-model",
        repo_root=repo_root,
    )

    assert summary["valid_complete_run"] is True
    assert calls == [(str(submission), str(repo_root.resolve()))]


def test_missing_submission_is_hash_bound_in_order(tmp_path: Path) -> None:
    submissions = tmp_path / "submissions"
    submissions.mkdir()
    _path, missing = runner._submission_inventory_record(
        submissions, "p_benchcad_missing", 0
    )

    assert missing == {
        "index": 0,
        "item_id": "p_benchcad_missing",
        "path": "p_benchcad_missing/submission.py",
        "state": "missing",
        "sha256": "missing",
        "size_bytes": None,
    }


def test_resume_validates_result_digest(tmp_path: Path) -> None:
    output, case, identity, environment, _result = _resume_bundle(tmp_path)
    result_path = output / "result.json"
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="digest validation"):
        runner._resume_result(
            output,
            identity,
            case=case,
            environment=environment,
        )


def test_resume_recomputes_statuses_even_with_new_digest(tmp_path: Path) -> None:
    output, case, identity, environment, _result = _resume_bundle(tmp_path)
    result_path = output / "result.json"
    value = runner.read_json(result_path)
    value["layers"][0]["status"] = Status.FAILED.value
    value["layers"][0]["passed"] = False
    value["overall_status"] = Status.FAILED.value
    value["passed"] = False
    write_json(result_path, value)
    write_text(output / runner.RESULT_DIGEST_NAME, f"{sha256_file(result_path)}\n")

    with pytest.raises(ValueError, match="does not reduce from its checks"):
        runner._resume_result(
            output,
            identity,
            case=case,
            environment=environment,
        )


def test_resume_verifies_candidate_artifact_hash(tmp_path: Path) -> None:
    output, case, identity, environment, result = _resume_bundle(tmp_path)
    assert (
        runner._resume_result(
            output,
            identity,
            case=case,
            environment=environment,
        ).as_dict()
        == result.as_dict()
    )
    (output / "l0/candidate.step").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="candidate STEP artifact failed"):
        runner._resume_result(
            output,
            identity,
            case=case,
            environment=environment,
        )


def test_resume_verifies_submission_source_hash(tmp_path: Path) -> None:
    output, case, identity, environment, _result = _resume_bundle(tmp_path)
    (output / "submission/submission.py").write_text(
        "result = 'tampered'\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source artifact failed"):
        runner._resume_result(
            output,
            identity,
            case=case,
            environment=environment,
        )


def test_resume_verifies_unreferenced_worker_evidence(tmp_path: Path) -> None:
    output, case, identity, environment, _result = _resume_bundle(tmp_path)
    (output / "l1-worker/stderr.txt").write_text(
        "tampered worker evidence\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="evidence bundle"):
        runner._resume_result(
            output,
            identity,
            case=case,
            environment=environment,
        )


def test_interrupted_item_output_is_recomputed_from_a_clean_boundary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "interrupted"
    write_json(output / "result.json", {"partial": True})
    write_text(output / "variant/partial.log", "partial\n")

    runner._prepare_item_output(output, force=False)

    assert not output.exists()


def test_force_cannot_replace_case_input_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    case_path = source_root / "case.json"
    submission = tmp_path / "submission.py"
    source_root.mkdir()
    case_path.write_text("{}\n", encoding="utf-8")
    submission.write_text("result = None\n", encoding="utf-8")
    sentinel = source_root / "must-survive.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    case = _case(case_path)
    monkeypatch.setattr(runner, "load_case", lambda *_args, **_kwargs: case)

    with pytest.raises(ValueError, match="protected path"):
        runner.evaluate_item(
            case_path=case_path,
            submission_path=submission,
            output_root=source_root,
            force=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_evaluator_error_result_invalidates_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evalset, submissions, output, _case_value = _fake_evalset(
        tmp_path,
        monkeypatch,
        status=Status.EVALUATOR_ERROR,
    )
    monkeypatch.setattr(
        runner,
        "aggregate_run",
        lambda *_args, **_kwargs: pytest.fail("invalid run must not aggregate"),
    )

    summary = runner.evaluate_evalset(
        evalset_root=evalset,
        submissions_root=submissions,
        output_root=output,
    )

    assert summary["execution_complete"] is True
    assert summary["result_roster_exact"] is True
    assert summary["item_evaluator_error_count"] == 1
    assert summary["valid_complete_run"] is False
    assert summary["publication_valid"] is False


def test_required_not_scoreable_is_diagnostic_but_not_publication_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evalset, submissions, output, _case_value = _fake_evalset(
        tmp_path,
        monkeypatch,
        status=Status.NOT_SCOREABLE,
    )
    monkeypatch.setattr(
        runner,
        "aggregate_run",
        lambda *_args, **_kwargs: {"diagnostic": True},
    )

    summary = runner.evaluate_evalset(
        evalset_root=evalset,
        submissions_root=submissions,
        output_root=output,
    )

    assert summary["execution_complete"] is True
    assert summary["valid_complete_run"] is True
    assert summary["publication_valid"] is False
    assert summary["required_not_scoreable_item_count"] == 1
    assert summary["non_scoreable_outcome_item_count"] == 1
    assert summary["aggregate"] == {"diagnostic": True}


def test_extra_result_breaks_exact_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evalset, submissions, output, _case_value = _fake_evalset(
        tmp_path,
        monkeypatch,
        extra_result=True,
    )
    monkeypatch.setattr(
        runner,
        "aggregate_run",
        lambda *_args, **_kwargs: pytest.fail("inexact roster must not aggregate"),
    )

    summary = runner.evaluate_evalset(
        evalset_root=evalset,
        submissions_root=submissions,
        output_root=output,
    )

    assert summary["execution_complete"] is True
    assert summary["result_roster_exact"] is False
    assert summary["extra_result_item_ids"] == ["unexpected"]
    assert summary["valid_complete_run"] is False
    assert summary["publication_valid"] is False


def test_result_status_tampering_fixture_is_independent(tmp_path: Path) -> None:
    """Guard against accidental shared mutation in the integrity fixtures."""

    original = _batch_result("p_benchcad_integrity")
    changed = deepcopy(original)
    changed["layers"][0]["status"] = Status.FAILED.value

    assert original["layers"][0]["status"] == Status.PASSED.value
