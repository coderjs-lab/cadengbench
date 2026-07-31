"""Deterministic provider response to benchmark artifact extraction."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

EXTRACTION_POLICY = "single_python_artifact_no_repair.v1"
JSON_EXTRACTION_POLICY = "single_json_artifact_no_repair.v1"
_PYTHON_FENCE = re.compile(
    r"```(?:python|py)\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL
)
_JSON_FENCE = re.compile(r"```json\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)
_CADQUERY_IMPORT = re.compile(r"^(?:import\s+cadquery\b|from\s+cadquery\b)")


class ExtractionError(ValueError):
    """A provider response does not contain one unambiguous Python artifact."""


@dataclass(frozen=True)
class ExtractionResult:
    source: str
    method: str
    policy: str = EXTRACTION_POLICY


def response_content(body: Any) -> str:
    """Return the unique provider text without interpreting or repairing it."""

    if not isinstance(body, dict):
        raise ExtractionError("provider body is not an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ExtractionError("response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = [
            part.get("text")
            for part in content
            if isinstance(part, dict)
            and part.get("type") in {"text", "output_text"}
            and isinstance(part.get("text"), str)
        ]
        if len(text) == 1:
            return text[0]
    raise ExtractionError("choice message has no unique text content")


def response_finish_reason(body: Any) -> str | None:
    """Return the provider's single-choice termination reason when available."""

    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    for key in ("finish_reason", "finishReason", "stop_reason"):
        value = choice.get(key)
        if value is not None:
            return str(value)
    return None


def extract_python(body: Any, *, max_source_bytes: int = 1_000_000) -> ExtractionResult:
    raw = response_content(body).strip()
    fences = list(_PYTHON_FENCE.finditer(raw))
    raw_source = raw + ("\n" if raw and not raw.endswith("\n") else "")
    if not raw_source.strip():
        raise ExtractionError("extracted Python artifact is empty")
    if len(raw_source.encode("utf-8")) > max_source_bytes:
        raise ExtractionError("extracted Python artifact exceeds the size limit")

    if not fences:
        try:
            ast.parse(raw_source)
        except SyntaxError:
            pass
        else:
            return ExtractionResult(
                source=raw_source,
                method="raw_python",
                policy=EXTRACTION_POLICY,
            )

    candidates: list[tuple[str, str]] = []
    for fence in fences:
        candidate = fence.group("code").strip() + "\n"
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        if candidate not in {source for source, _method in candidates}:
            candidates.append((candidate, "single_valid_python_fence"))

    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if not _CADQUERY_IMPORT.match(line.strip()):
            continue
        candidate = "\n".join(lines[index:]).strip() + "\n"
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        if candidate not in {source for source, _method in candidates}:
            candidates.append((candidate, "unique_cadquery_suffix"))

    if len(candidates) != 1:
        raise ExtractionError(
            "response does not contain one unambiguous valid Python artifact"
        )
    source, method = candidates[0]
    try:
        ast.parse(source)
    except SyntaxError as exc:  # guarded by candidate validation above
        raise ExtractionError(f"extracted artifact is not valid Python: {exc}") from exc
    return ExtractionResult(source=source, method=method, policy=EXTRACTION_POLICY)


def extract_json(body: Any, *, max_source_bytes: int = 1_000_000) -> ExtractionResult:
    raw = response_content(body).strip()
    fences = list(_JSON_FENCE.finditer(raw))
    if len(fences) > 1:
        raise ExtractionError("response contains multiple JSON code fences")
    if fences:
        source = fences[0].group("code").strip()
        method = "single_json_fence"
    else:
        source = raw
        method = "raw_json"
    if not source:
        raise ExtractionError("extracted JSON artifact is empty")
    if len(source.encode("utf-8")) > max_source_bytes:
        raise ExtractionError("extracted JSON artifact exceeds the size limit")
    try:
        value = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"extracted artifact is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtractionError("functional-edit JSON submission must be an object")
    canonical = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    )
    return ExtractionResult(
        source=canonical,
        method=method,
        policy=JSON_EXTRACTION_POLICY,
    )


def extract_artifact(body: Any, response_contract: dict[str, Any]) -> ExtractionResult:
    policy = response_contract.get("extraction_policy")
    if policy == EXTRACTION_POLICY:
        return extract_python(body)
    if policy == JSON_EXTRACTION_POLICY:
        return extract_json(body)
    raise ExtractionError(f"unsupported response extraction policy: {policy!r}")
