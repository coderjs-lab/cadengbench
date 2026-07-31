from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cadengbench_p_runner.io import semantic_hash, sha256_file, write_json
from cadengbench_p_runner.materializer import materialize_request
from cadengbench_p_runner.model_registry import parse_model_spec
from cadengbench_p_runner.schema import (
    validate_candidate_submission,
    validate_canonical_request,
)


def test_canonical_request_is_hash_bound(synthetic_items: Path, tmp_path: Path) -> None:
    item = next(synthetic_items.iterdir())
    request = materialize_request(
        item, run_root=tmp_path / "run", input_profile="text_iso"
    )
    validate_canonical_request(request)
    assert len(request["messages"][1]["parts"]) == 3
    request["messages"][1]["parts"][0]["text"] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        validate_canonical_request(request)


def test_consumer_ready_item_materializes_a_hash_bound_request_source(
    synthetic_items: Path, tmp_path: Path
) -> None:
    item = next(synthetic_items.iterdir())
    shutil.rmtree(item / "zero_to_cad")
    system_prompt = tmp_path / "configs/prompts/zero_to_cad/model.system.txt"
    system_prompt.parent.mkdir(parents=True)
    system_prompt.write_text("Return one CadQuery Python program.\n", encoding="utf-8")
    write_json(
        item / "benchmark/consumer_ready_item_v2.json",
        {
            "schema_version": "cadengbench-p.consumer-ready-item.v2",
            "item_id": item.name,
            "model_facing": {
                "zero_to_cad": {
                    "system_prompt": "configs/prompts/zero_to_cad/model.system.txt",
                    "user_prompt": "Build one editable parametric block.",
                    "visual_assets": [],
                }
            },
        },
    )
    write_json(
        item / "evaluation/qualification.v2.json",
        {
            "schema_version": "cadengbench-p.evaluator-qualification.v3",
            "item_id": item.name,
            "requirements": [
                {
                    "requirement_id": "R-SOLID",
                    "category": "solid_contract",
                    "value_and_tolerance": {"body_count": 1},
                },
                {
                    "requirement_id": "R-BBOX",
                    "category": "global_envelope",
                    "value_and_tolerance": {
                        axis: {"value": 10.0, "tolerance": 0.05}
                        for axis in ("x", "y", "z")
                    },
                },
            ],
            "parameter_system": {
                "parameters": [
                    {
                        "parameter_id": "P-001",
                        "engineering_name": "overall_length",
                        "nominal": 10.0,
                        "unit": "mm",
                        "role": "principal_dimension",
                    }
                ]
            },
            "dfm": {
                "families": [
                    {
                        "family": "minimum_wall_thickness",
                        "scoreable": True,
                        "thresholds": {"minimum_mm": 1.0},
                    }
                ]
            },
        },
    )
    run_root = tmp_path / "run"

    request = materialize_request(item, run_root=run_root, input_profile="text_iso")

    source_path = (
        run_root / "request_sources" / item.name / "zero_to_cad_design_source.json"
    )
    assert request["dataset"]["design_spec_sha256"] == sha256_file(source_path)
    assert request["messages"][0]["parts"][0]["text"] == system_prompt.read_text()
    user_prompt = request["messages"][1]["parts"][0]["text"]
    assert user_prompt.startswith("Build one editable parametric block.\n")
    assert '"overall_length": 10.0' in user_prompt
    assert '"P-001": {' in user_prompt
    assert "overall envelope X=10.0+/-0.05 mm" in user_prompt
    assert "Minimum wall thickness: 1.0 mm" in user_prompt
    validate_canonical_request(request)


def test_candidate_contract_rejects_semantic_tampering() -> None:
    candidate = {
        "schema_version": "cadengbench-p.candidate-submission.v1",
        "submission_id": "run:p_benchcad_000001:0",
        "run_id": "run",
        "item_id": "p_benchcad_000001",
        "sample_index": 0,
        "request_hash": "1" * 64,
        "run_manifest_hash": "2" * 64,
        "model": {
            "provider": "openai_compatible",
            "requested_model_id": "model",
            "returned_model_id": "model-revision",
            "provider_request_id": "request-1",
        },
        "successful_attempt": {
            "attempt_index": 1,
            "ledger_record_hash": "3" * 64,
        },
        "raw_response": {"path": "raw.json", "sha256": "4" * 64, "size_bytes": 1},
        "extraction": {
            "policy": "single_python_artifact_no_repair.v1",
            "method": "raw_python",
            "status": "succeeded",
        },
        "artifact": {
            "path": "artifacts/p_benchcad_000001/submission.py",
            "sha256": "5" * 64,
            "size_bytes": 1,
            "media_type": "text/x-python",
            "interface_version": "cadengbench-p.cadquery-submission.v1",
        },
    }
    candidate["candidate_hash"] = semantic_hash(candidate, "candidate_hash")
    validate_candidate_submission(candidate)
    candidate["model"]["requested_model_id"] = "other"
    with pytest.raises(ValueError, match="hash"):
        validate_candidate_submission(candidate)


def test_hosted_reasoning_model_defaults_are_supported() -> None:
    spec = {
        "schema_version": "cadengbench-p.model-spec.v1",
        "model_label": "hosted-reasoning-model",
        "provider": "openai_compatible",
        "protocol": "openai_chat",
        "model_id": "gpt-5.4-mini-2026-03-17",
        "base_url": "https://api.openai.com",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_required": True,
        "provider_routing": {
            "order": ["PinnedProvider"],
            "only": ["PinnedProvider"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "capabilities": {"text": True, "images": True},
        "request_defaults": {
            "max_completion_tokens": 16384,
            "reasoning_effort": "low",
            "presence_penalty": 1.5,
            "top_k": 20,
        },
        "rate_limits": {"max_concurrency": 2, "requests_per_minute": 30},
    }

    parsed = parse_model_spec(spec)

    assert parsed.request_defaults == {
        "max_completion_tokens": 16384,
        "reasoning_effort": "low",
        "presence_penalty": 1.5,
        "top_k": 20,
    }
    assert parsed.provider_routing == {
        "order": ["PinnedProvider"],
        "only": ["PinnedProvider"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_model_spec_digest_uses_normalized_public_values() -> None:
    spec = {
        "schema_version": "cadengbench-p.model-spec.v1",
        "model_label": "integer-defaults-model",
        "provider": "openai_compatible",
        "protocol": "openai_chat",
        "model_id": "model",
        "base_url": "http://127.0.0.1:8000/v1/",
        "api_key_env": None,
        "api_key_required": False,
        "capabilities": {"text": True, "images": True},
        "request_defaults": {"max_tokens": 8192, "temperature": 0, "top_p": 1},
        "rate_limits": {"max_concurrency": 1, "requests_per_minute": 20},
    }

    parsed = parse_model_spec(spec)
    public = parsed.as_public_dict()

    assert public["base_url"] == "http://127.0.0.1:8000/v1"
    assert public["request_defaults"]["top_p"] == 1.0
    assert public["spec_hash"] == semantic_hash(public, "spec_hash")


def test_model_defaults_reject_conflicting_token_limits() -> None:
    spec = {
        "schema_version": "cadengbench-p.model-spec.v1",
        "model_label": "invalid-hosted-model",
        "provider": "openai_compatible",
        "protocol": "openai_chat",
        "model_id": "gpt-5.4-mini-2026-03-17",
        "base_url": "https://api.openai.com",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_required": True,
        "capabilities": {"text": True, "images": True},
        "request_defaults": {
            "max_tokens": 1024,
            "max_completion_tokens": 1024,
        },
        "rate_limits": {"max_concurrency": 2, "requests_per_minute": 30},
    }

    with pytest.raises(ValueError, match="cannot set both"):
        parse_model_spec(spec)
