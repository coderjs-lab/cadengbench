"""JSON-schema loading and identity validation."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from cadengbench_p_l3_response.io import read_json, sha256_json

SCHEMAS = {
    "profile": "response_profile.v1.schema.json",
    "contract": "response_contract.v1.schema.json",
    "pair_result": "response_pair_result.v1.schema.json",
    "family_manifest": "response_family_manifest.v1.schema.json",
}


class ResponseSchemaError(ValueError):
    """A response-evaluator artifact violates its frozen contract."""


def _schema_document(name: str) -> dict[str, Any]:
    filename = SCHEMAS[name]
    source_path = Path(__file__).resolve().parents[2] / "schemas" / filename
    if source_path.is_file():
        return read_json(source_path)
    resource = resources.files("cadengbench_p_l3_response").joinpath(
        "_schemas", filename
    )
    value = resource.read_text(encoding="utf-8")
    import json

    document = json.loads(value)
    if not isinstance(document, dict):
        raise ResponseSchemaError(f"schema is not an object: {filename}")
    return document


def validate_document(name: str, document: dict[str, Any]) -> None:
    validator = Draft202012Validator(_schema_document(name))
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if errors:
        rendered = "; ".join(
            f"/{'/'.join(str(value) for value in error.path)}: {error.message}"
            for error in errors[:8]
        )
        raise ResponseSchemaError(f"invalid {name}: {rendered}")


def load_profile(path: Path) -> dict[str, Any]:
    profile = read_json(path)
    validate_document("profile", profile)
    return profile


def load_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    validate_document("contract", contract)
    body = dict(contract)
    expected = body.pop("contract_sha256")
    if sha256_json(body) != expected:
        raise ResponseSchemaError("response contract self-hash changed")
    return contract


def load_family_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    validate_document("family_manifest", manifest)
    body = dict(manifest)
    expected = body.pop("manifest_sha256")
    if sha256_json(body) != expected:
        raise ResponseSchemaError("response family manifest self-hash changed")
    return manifest
