"""OpenAI-compatible chat adapter for hosted endpoints and local vLLM."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from cadengbench_p_runner.io import (
    safe_relative_path,
    sha256_file,
    sha256_json,
)
from cadengbench_p_runner.model_registry import ModelSpec
from cadengbench_p_runner.schema import validate_canonical_request

from .base import ProviderResult

_RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}
_RETAINED_HEADERS = {
    "retry-after",
    "x-request-id",
    "request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
}


def _endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _headers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    normalized = {str(key).lower(): str(item) for key, item in value.items()}
    return {key: normalized[key] for key in sorted(_RETAINED_HEADERS & set(normalized))}


def _retry_after(headers: dict[str, str]) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0.0 else None


def _parse_body(payload: bytes) -> Any:
    text = payload.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"unparsed_text": text}


def _translated_request_defaults(model: ModelSpec) -> dict[str, Any]:
    defaults = dict(model.request_defaults)
    effort = defaults.pop("reasoning_effort", None)
    if urlparse(model.base_url).hostname == "openrouter.ai" and effort is not None:
        defaults["reasoning"] = (
            {"enabled": False} if effort == "none" else {"effort": effort}
        )
    elif effort is not None:
        defaults["reasoning_effort"] = effort
    return defaults


class OpenAICompatibleAdapter:
    """Translate only transport syntax; benchmark semantics remain untouched."""

    def __init__(self, model: ModelSpec, *, environment: dict[str, str] | None = None):
        self.model = model
        environment = os.environ if environment is None else environment
        self.api_key = (
            environment.get(model.api_key_env)
            if model.api_key_env is not None
            else None
        )
        if model.api_key_required and not self.api_key:
            raise ValueError(
                f"required API key environment variable is absent: {model.api_key_env}"
            )

    def _message_parts(
        self, request: dict[str, Any], *, run_root: Path
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request["messages"]:
            content: list[dict[str, Any]] = []
            for part in message["parts"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                    continue
                path = safe_relative_path(run_root, part["asset_path"])
                if not path.is_file() or sha256_file(path) != part["sha256"]:
                    raise ValueError(
                        f"request image failed its hash binding: {part['asset_path']}"
                    )
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{part['media_type']};base64,{encoded}"
                        },
                    }
                )
            messages.append({"role": message["role"], "content": content})
        return messages

    def provider_request(
        self, request: dict[str, Any], *, run_root: Path
    ) -> dict[str, Any]:
        validate_canonical_request(request)
        payload = {
            "model": self.model.model_id,
            "messages": self._message_parts(request, run_root=run_root),
            **_translated_request_defaults(self.model),
        }
        if self.model.provider_routing:
            payload["provider"] = self.model.provider_routing
        return payload

    def generate(
        self,
        request: dict[str, Any],
        *,
        run_root: Path,
        timeout_seconds: int,
    ) -> ProviderResult:
        payload = self.provider_request(request, run_root=run_root)
        request_hash = sha256_json(payload)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = Request(
            _endpoint(self.model.base_url),
            data=json.dumps(payload, ensure_ascii=True, allow_nan=False).encode(
                "utf-8"
            ),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=timeout_seconds) as response:
                status = int(response.status)
                response_headers = _headers(response.headers)
                body = _parse_body(response.read())
        except HTTPError as exc:
            response_headers = _headers(exc.headers)
            body = _parse_body(exc.read())
            message = (
                str((body.get("error") or {}).get("message"))
                if isinstance(body, dict) and isinstance(body.get("error"), dict)
                else f"HTTP {exc.code}"
            )
            return ProviderResult(
                outcome="error",
                retryable=exc.code in _RETRYABLE_HTTP,
                http_status=int(exc.code),
                error_class="http_error",
                error_message=message,
                provider_request_id=response_headers.get("x-request-id")
                or response_headers.get("request-id"),
                returned_model_id=None,
                body=body,
                response_headers=response_headers,
                provider_request_hash=request_hash,
                retry_after_seconds=_retry_after(response_headers),
            )
        except TimeoutError as exc:
            return ProviderResult(
                outcome="error",
                retryable=True,
                http_status=None,
                error_class="timeout",
                error_message=str(exc),
                provider_request_id=None,
                returned_model_id=None,
                provider_request_hash=request_hash,
            )
        except URLError as exc:
            return ProviderResult(
                outcome="error",
                retryable=True,
                http_status=None,
                error_class="connection_error",
                error_message=str(exc.reason),
                provider_request_id=None,
                returned_model_id=None,
                provider_request_hash=request_hash,
            )

        if not isinstance(body, dict):
            return ProviderResult(
                outcome="error",
                retryable=False,
                http_status=status,
                error_class="invalid_provider_response",
                error_message="provider response is not a JSON object",
                provider_request_id=response_headers.get("x-request-id")
                or response_headers.get("request-id"),
                returned_model_id=None,
                body=body,
                response_headers=response_headers,
                provider_request_hash=request_hash,
            )
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        returned_model = body.get("model")
        provider_request_id = body.get("id")
        return ProviderResult(
            outcome="response",
            retryable=False,
            http_status=status,
            error_class=None,
            error_message=None,
            provider_request_id=str(provider_request_id)
            if provider_request_id is not None
            else response_headers.get("x-request-id")
            or response_headers.get("request-id"),
            returned_model_id=str(returned_model)
            if returned_model is not None
            else None,
            usage=usage,
            body=body,
            response_headers=response_headers,
            provider_request_hash=request_hash,
        )
