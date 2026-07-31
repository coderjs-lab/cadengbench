from __future__ import annotations

import json
from pathlib import Path

from cadengbench_p_runner.planner import load_run_manifest, plan_run
from cadengbench_p_runner.schema import validate_canonical_request


def test_plan_freezes_manifest_and_requests(
    synthetic_items: Path, model_spec_writer, tmp_path: Path
) -> None:
    model = model_spec_writer(tmp_path / "model.json", "http://127.0.0.1:9")
    run = tmp_path / "run"
    manifest = plan_run(
        items_root=synthetic_items,
        model_spec_path=model,
        output_root=run,
        run_id="test-run",
        input_profile="text_iso",
    )
    assert load_run_manifest(run)["manifest_hash"] == manifest["manifest_hash"]
    request = json.loads((run / manifest["requests"][0]["request_path"]).read_text())
    validate_canonical_request(request)
    assert manifest["model"]["api_key_env"] is None
    assert '"api_key":' not in json.dumps(manifest)


def test_functional_edit_plan_freezes_source_native_request(
    synthetic_items: Path, model_spec_writer, tmp_path: Path
) -> None:
    model = model_spec_writer(tmp_path / "model.json", "http://127.0.0.1:9")
    run = tmp_path / "functional-run"
    manifest = plan_run(
        items_root=synthetic_items,
        model_spec_path=model,
        output_root=run,
        run_id="functional-test-run",
        input_profile="text_iso",
        task_kind="functional_edit",
    )
    loaded = load_run_manifest(run)
    assert loaded["schema_version"] == "cadengbench-p.candidate-run-manifest.v2"
    assert loaded["task_kind"] == "functional_edit"
    request = json.loads((run / manifest["requests"][0]["request_path"]).read_text())
    validate_canonical_request(request)
    assert request["task_kind"] == "functional_edit"
    assert request["response_contract"]["interface_version"] == (
        "cadengbench-p.benchcad-source-edit.v1"
    )
    user_text = request["messages"][1]["parts"][0]["text"]
    assert "BEGIN EDITABLE SOURCE" in user_text
    assert "Increase the block height" in user_text
    assert "Minimum wall thickness: 1.0 mm" in user_text


def test_fusion_functional_edit_uses_complete_compact_parameter_index(
    synthetic_fusion_items: Path, model_spec_writer, tmp_path: Path
) -> None:
    model = model_spec_writer(tmp_path / "model.json", "http://127.0.0.1:9")
    run = tmp_path / "fusion-functional-run"
    manifest = plan_run(
        items_root=synthetic_fusion_items,
        model_spec_path=model,
        output_root=run,
        run_id="fusion-functional-test-run",
        input_profile="text_iso",
        task_kind="functional_edit",
    )
    request = json.loads((run / manifest["requests"][0]["request_path"]).read_text())
    validate_canonical_request(request)
    user_text = request["messages"][1]["parts"][0]["text"]
    assert "BEGIN EDITABLE SOURCE" not in user_text
    assert "BEGIN EDITABLE PARAMETER INDEX" in user_text
    assert "/entities/extrude-1/extent_one/distance/value" in user_text
    assert (
        "/entities/sketch-1/profiles/profile-1/loops/0/"
        "profile_curves/0/radius"
    ) in user_text
    assert '"editable_parameter_count": 2' in user_text
    assert request["response_contract"]["interface_version"] == (
        "cadengbench-p.fusion360-patch-submission.v1"
    )


def test_canary_request_contract_materializes_all_30(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    items = repo / "data/interim/cadengbench_p_r25_combined_hpc_canary_30/dataset/items"
    if not items.is_dir():
        return
    from conftest import model_spec, write_json

    model = tmp_path / "model.json"
    write_json(model, model_spec("http://127.0.0.1:9"))
    manifest = plan_run(
        items_root=items,
        model_spec_path=model,
        output_root=tmp_path / "canary-run",
        run_id="canary-contract-test",
        input_profile="text_iso",
    )
    assert manifest["dataset"]["item_count"] == 30
    assert len({record["request_hash"] for record in manifest["requests"]}) == 30
