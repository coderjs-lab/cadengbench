"""Multimodal provider adapters for reproducible CADEngBench-A execution.

The benchmark runner passes the same provider-neutral request to every adapter.
Adapters may differ in wire format, but they may not alter prompts, images, output
schemas, or evaluation policy.
"""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from models.env import load_env_file


@dataclass(frozen=True)
class MultimodalRequest:
    """Provider-neutral model request with ordered image evidence."""

    system_prompt: str
    user_prompt: str
    images: tuple[Path, ...]
    output_schema: dict[str, Any]
    max_output_tokens: int
    reasoning_effort: str = "minimal"


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response plus the lossless provider payload."""

    text: str
    payload: dict[str, Any]
    response_id: str | None
    usage: dict[str, int]


def _api_key(name: str) -> str:
    load_env_file()
    value = os.environ.get(name)
    if not value or value.startswith("your_"):
        raise RuntimeError(f"Missing provider credential: set {name} in .env")
    return value


def _mime(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "image/png"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _data_url(path: Path) -> str:
    return f"data:{_mime(path)};base64,{_b64(path)}"


def _schema_instruction(request: MultimodalRequest) -> str:
    return (
        f"{request.user_prompt}\n\nOUTPUT JSON SCHEMA:\n"
        f"{json.dumps(request.output_schema, separators=(',', ':'), sort_keys=True)}"
    )


def _structured_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove metadata and normalize unions for OpenAI Structured Outputs."""

    translated = copy.deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "oneOf" in value and "anyOf" not in value:
                value["anyOf"] = value.pop("oneOf")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(translated)
    translated.pop("$schema", None)
    translated.pop("$id", None)
    return translated


def _normalized_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or payload.get("usage_metadata") or {}
    input_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    output_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("candidates_token_count")
        or 0
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(
            usage.get("total_tokens")
            or usage.get("total_token_count")
            or input_tokens + output_tokens
        ),
    }


def parse_model_output(
    raw_text: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Parse strict JSON with a recorded, bounded first-object recovery."""

    strict_error = None
    prediction = None
    recovery = None
    try:
        prediction = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        strict_error = str(exc)
        candidate = raw_text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            candidate = "\n".join(lines[1:-1]) if len(lines) >= 3 else candidate
            recovery = "removed_code_fence"
        start = candidate.find("{")
        if start >= 0:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(candidate)):
                char = candidate[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            prediction = json.loads(candidate[start : index + 1])
                            recovery = recovery or "extracted_first_json_object"
                        except json.JSONDecodeError:
                            prediction = None
                        break
    schema_errors = []
    if prediction is not None:
        schema_errors = sorted(
            error.message
            for error in Draft202012Validator(schema).iter_errors(prediction)
        )
    return prediction, {
        "schema": "cadengbench_a_parser_report_v2",
        "strict_json_parse": strict_error is None,
        "strict_first_pass_valid": strict_error is None and not schema_errors,
        "recovery": recovery,
        "recovered_valid": prediction is not None and not schema_errors,
        "json_parse_error": strict_error,
        "schema_errors": schema_errors,
    }


def _send_openai(model: str, request: MultimodalRequest) -> ProviderResponse:
    from openai import OpenAI

    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": request.user_prompt}
    ]
    content.extend(
        {"type": "input_image", "image_url": _data_url(path), "detail": "high"}
        for path in request.images
    )
    response = OpenAI(api_key=_api_key("OPENAI_API_KEY")).responses.create(
        model=model,
        instructions=request.system_prompt,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": str(request.output_schema.get("$id") or "cadengbench_a_output"),
                "schema": _structured_output_schema(request.output_schema),
                "strict": True,
            }
        },
        reasoning={"effort": request.reasoning_effort},
        max_output_tokens=request.max_output_tokens,
    )
    payload = response.model_dump(mode="json")
    return ProviderResponse(
        text=getattr(response, "output_text", "") or "",
        payload=payload,
        response_id=getattr(response, "id", None),
        usage=_normalized_usage(payload),
    )


def _send_anthropic(model: str, request: MultimodalRequest) -> ProviderResponse:
    import anthropic

    content: list[dict[str, Any]] = [
        {"type": "text", "text": _schema_instruction(request)}
    ]
    content.extend(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _mime(path),
                "data": _b64(path),
            },
        }
        for path in request.images
    )
    response = anthropic.Anthropic(
        api_key=_api_key("ANTHROPIC_API_KEY")
    ).messages.create(
        model=model,
        system=request.system_prompt,
        messages=[{"role": "user", "content": content}],
        max_tokens=request.max_output_tokens,
        temperature=0,
    )
    payload = response.model_dump(mode="json")
    text = "\n".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    return ProviderResponse(
        text=text,
        payload=payload,
        response_id=getattr(response, "id", None),
        usage=_normalized_usage(payload),
    )


def _send_gemini(model: str, request: MultimodalRequest) -> ProviderResponse:
    from google import genai
    from google.genai import types

    contents: list[Any] = [_schema_instruction(request)]
    contents.extend(
        types.Part.from_bytes(data=path.read_bytes(), mime_type=_mime(path))
        for path in request.images
    )
    response = genai.Client(api_key=_api_key("GEMINI_API_KEY")).models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            response_mime_type="application/json",
            response_json_schema=request.output_schema,
            max_output_tokens=request.max_output_tokens,
            temperature=0,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_level=(
                    "minimal"
                    if request.reasoning_effort == "none"
                    else request.reasoning_effort
                ),
            ),
        ),
    )
    payload = response.model_dump(mode="json")
    return ProviderResponse(
        text=getattr(response, "text", "") or "",
        payload=payload,
        response_id=getattr(response, "response_id", None),
        usage=_normalized_usage(payload),
    )


def _send_openai_compatible(
    provider: str,
    model: str,
    request: MultimodalRequest,
    *,
    base_url: str | None,
) -> ProviderResponse:
    from openai import OpenAI

    defaults = {
        "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
    }
    if provider not in defaults and not base_url:
        raise ValueError("OpenAI-compatible providers require --base-url")
    key_name, default_url = defaults.get(provider, ("OPENAI_COMPATIBLE_API_KEY", ""))
    content: list[dict[str, Any]] = [
        {"type": "text", "text": _schema_instruction(request)}
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": _data_url(path)}}
        for path in request.images
    )
    response = OpenAI(
        api_key=_api_key(key_name), base_url=base_url or default_url
    ).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=request.max_output_tokens,
        temperature=0,
    )
    payload = response.model_dump(mode="json")
    return ProviderResponse(
        text=response.choices[0].message.content or "",
        payload=payload,
        response_id=getattr(response, "id", None),
        usage=_normalized_usage(payload),
    )


def send_multimodal_request(
    provider: str,
    model: str,
    request: MultimodalRequest,
    *,
    base_url: str | None = None,
) -> ProviderResponse:
    """Send one frozen request through a supported provider adapter."""

    if provider == "openai":
        return _send_openai(model, request)
    if provider == "anthropic":
        return _send_anthropic(model, request)
    if provider == "gemini":
        return _send_gemini(model, request)
    return _send_openai_compatible(
        provider, model, request, base_url=base_url
    )
