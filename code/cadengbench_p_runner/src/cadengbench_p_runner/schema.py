"""Load and validate the two frozen public runner contracts."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from .io import semantic_hash, sha256_text

CANONICAL_REQUEST_SCHEMA = "canonical_model_request_v1.schema.json"
CANDIDATE_SUBMISSION_SCHEMA = "candidate_submission_v1.schema.json"
CANONICAL_REQUEST_V2_SCHEMA = "canonical_model_request_v2.schema.json"
CANDIDATE_SUBMISSION_V2_SCHEMA = "candidate_submission_v2.schema.json"


def load_schema(name: str) -> dict[str, Any]:
    resource = files("cadengbench_p_runner.schemas").joinpath(name)
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_schema(value: Any, name: str) -> None:
    Draft202012Validator(load_schema(name)).validate(value)


def validate_canonical_request(value: dict[str, Any]) -> None:
    schema_version = value.get("schema_version")
    schema_name = {
        "cadengbench-p.canonical-model-request.v1": CANONICAL_REQUEST_SCHEMA,
        "cadengbench-p.canonical-model-request.v2": CANONICAL_REQUEST_V2_SCHEMA,
    }.get(schema_version)
    if schema_name is None:
        raise ValueError(f"unsupported canonical request schema: {schema_version!r}")
    validate_schema(value, schema_name)
    if semantic_hash(value, "request_hash") != value["request_hash"]:
        raise ValueError("canonical request hash is invalid")
    expected_id = (
        f"{value['item_id']}:{value['input_profile']}:v1"
        if schema_version == "cadengbench-p.canonical-model-request.v1"
        else (f"{value['item_id']}:{value['task_kind']}:{value['input_profile']}:v2")
    )
    if value["request_id"] != expected_id:
        raise ValueError("canonical request ID is not derived from item and profile")
    for message in value["messages"]:
        for part in message["parts"]:
            if part["type"] == "text" and sha256_text(part["text"]) != part["sha256"]:
                raise ValueError("canonical request text-part hash is invalid")
    user_parts = value["messages"][1]["parts"]
    if user_parts[0]["type"] != "text" or any(
        part["type"] != "image" for part in user_parts[1:]
    ):
        raise ValueError("user request must contain text followed by ordered images")
    expected_roles = {
        "text": [],
        "text_iso": ["isometric_front_right_top", "isometric_rear_left_top"],
        "text_engineering_views": ["engineering_four_view_visible_hidden"],
    }[value["input_profile"]]
    observed_roles = [part["view_role"] for part in user_parts[1:]]
    if observed_roles != expected_roles:
        raise ValueError("canonical request visual roles differ from its input profile")


def validate_candidate_submission(value: dict[str, Any]) -> None:
    schema_version = value.get("schema_version")
    schema_name = {
        "cadengbench-p.candidate-submission.v1": CANDIDATE_SUBMISSION_SCHEMA,
        "cadengbench-p.candidate-submission.v2": CANDIDATE_SUBMISSION_V2_SCHEMA,
    }.get(schema_version)
    if schema_name is None:
        raise ValueError(f"unsupported candidate submission schema: {schema_version!r}")
    validate_schema(value, schema_name)
    if semantic_hash(value, "candidate_hash") != value["candidate_hash"]:
        raise ValueError("candidate submission hash is invalid")
    expected = f"{value['run_id']}:{value['item_id']}:{value['sample_index']}"
    if value["submission_id"] != expected:
        raise ValueError("candidate submission ID is not derived from its run job")
