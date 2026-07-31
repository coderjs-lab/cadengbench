"""Strict model registry entries with no secret values."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io import read_json, sha256_json

MODEL_SPEC_SCHEMA = "cadengbench-p.model-spec.v1"
SUPPORTED_REQUEST_DEFAULTS = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "reasoning_effort",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "top_k",
}

SUPPORTED_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class ModelSpec:
    model_label: str
    provider: str
    protocol: str
    model_id: str
    base_url: str
    api_key_env: str | None
    api_key_required: bool
    provider_routing: dict[str, Any]
    text_capable: bool
    image_capable: bool
    request_defaults: dict[str, Any]
    max_concurrency: int
    requests_per_minute: int
    digest: str

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SPEC_SCHEMA,
            "model_label": self.model_label,
            "provider": self.provider,
            "protocol": self.protocol,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_required": self.api_key_required,
            "provider_routing": self.provider_routing,
            "capabilities": {
                "text": self.text_capable,
                "images": self.image_capable,
            },
            "request_defaults": self.request_defaults,
            "rate_limits": {
                "max_concurrency": self.max_concurrency,
                "requests_per_minute": self.requests_per_minute,
            },
            "spec_hash": self.digest,
        }


def parse_model_spec(value: Any) -> ModelSpec:
    if not isinstance(value, dict) or value.get("schema_version") != MODEL_SPEC_SCHEMA:
        raise ValueError(f"model spec must use {MODEL_SPEC_SCHEMA}")
    required = {
        "schema_version",
        "model_label",
        "provider",
        "protocol",
        "model_id",
        "base_url",
        "api_key_env",
        "api_key_required",
        "capabilities",
        "request_defaults",
        "rate_limits",
    }
    optional = {"provider_routing"}
    extras = set(value) - required - optional
    missing = required - set(value)
    if extras or missing:
        raise ValueError(
            "model spec fields differ; "
            f"missing={sorted(missing)}, extra={sorted(extras)}"
        )
    strings = [value.get(key) for key in ("model_label", "model_id", "base_url")]
    if not all(isinstance(item, str) and item for item in strings):
        raise ValueError("model label, ID, and base URL are required")
    if value.get("provider") != "openai_compatible":
        raise ValueError("only the openai_compatible provider is implemented")
    if value.get("protocol") != "openai_chat":
        raise ValueError("only the openai_chat protocol is implemented")
    api_key_env = value.get("api_key_env")
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not api_key_env
    ):
        raise ValueError("api_key_env must be null or a nonempty environment name")
    if isinstance(api_key_env, str) and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", api_key_env
    ):
        raise ValueError("api_key_env is not a valid environment variable name")
    if not isinstance(value.get("api_key_required"), bool):
        raise ValueError("api_key_required must be boolean")
    routing = value.get("provider_routing", {})
    routing_allowed = {
        "order",
        "only",
        "allow_fallbacks",
        "require_parameters",
        "quantizations",
    }
    if not isinstance(routing, dict) or set(routing) - routing_allowed:
        raise ValueError("provider_routing has unsupported fields")
    if routing:
        order = routing.get("order")
        if order is not None and (
            not isinstance(order, list)
            or not order
            or len(order) != len(set(order))
            or any(not isinstance(item, str) or not item for item in order)
        ):
            raise ValueError("provider_routing.order must be unique nonempty strings")
        only = routing.get("only")
        if only is not None and (
            not isinstance(only, list)
            or not only
            or len(only) != len(set(only))
            or any(not isinstance(item, str) or not item for item in only)
        ):
            raise ValueError("provider_routing.only must be unique nonempty strings")
        allow_fallbacks = routing.get("allow_fallbacks")
        if allow_fallbacks is not None and not isinstance(allow_fallbacks, bool):
            raise ValueError("provider_routing.allow_fallbacks must be boolean")
        require_parameters = routing.get("require_parameters")
        if require_parameters is not None and not isinstance(require_parameters, bool):
            raise ValueError("provider_routing.require_parameters must be boolean")
        quantizations = routing.get("quantizations")
        supported_quantizations = {
            "int4",
            "int8",
            "fp4",
            "fp6",
            "fp8",
            "fp16",
            "bf16",
            "fp32",
            "unknown",
        }
        if quantizations is not None and (
            not isinstance(quantizations, list)
            or not quantizations
            or len(quantizations) != len(set(quantizations))
            or any(item not in supported_quantizations for item in quantizations)
        ):
            raise ValueError("provider_routing.quantizations is invalid")
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) != {"text", "images"}:
        raise ValueError("capabilities must contain exactly text and images")
    if not all(isinstance(capabilities[key], bool) for key in capabilities):
        raise ValueError("capabilities must be booleans")
    defaults = value.get("request_defaults")
    if not isinstance(defaults, dict):
        raise ValueError("request_defaults must be an object")
    unsupported = set(defaults) - SUPPORTED_REQUEST_DEFAULTS
    if unsupported:
        raise ValueError(f"unsupported request defaults: {sorted(unsupported)}")
    if "max_tokens" in defaults and "max_completion_tokens" in defaults:
        raise ValueError(
            "request defaults cannot set both max_tokens and max_completion_tokens"
        )
    normalized_defaults: dict[str, Any] = {}
    for key, raw in defaults.items():
        if key in {"max_tokens", "max_completion_tokens"}:
            normalized_defaults[key] = _positive_int(raw, key)
        elif key == "reasoning_effort":
            if not isinstance(raw, str) or raw not in SUPPORTED_REASONING_EFFORTS:
                raise ValueError(
                    "reasoning_effort must be one of "
                    f"{sorted(SUPPORTED_REASONING_EFFORTS)}"
                )
            normalized_defaults[key] = raw
        elif key == "seed":
            normalized_defaults[key] = _nonnegative_int(raw, key)
        elif key == "top_k":
            normalized_defaults[key] = _positive_int(raw, key)
        else:
            normalized_defaults[key] = _finite_number(raw, key)
    if not 0.0 <= normalized_defaults.get("top_p", 1.0) <= 1.0:
        raise ValueError("top_p must be between 0 and 1")
    if not 0.0 <= normalized_defaults.get("temperature", 0.0) <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    for key in ("frequency_penalty", "presence_penalty"):
        if key in normalized_defaults and not -2.0 <= normalized_defaults[key] <= 2.0:
            raise ValueError(f"{key} must be between -2 and 2")
    parsed_url = urlparse(str(value["base_url"]))
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    rate_limits = value.get("rate_limits")
    if not isinstance(rate_limits, dict) or set(rate_limits) != {
        "max_concurrency",
        "requests_per_minute",
    }:
        raise ValueError("rate_limits fields are incomplete")
    semantic = {
        "schema_version": MODEL_SPEC_SCHEMA,
        "model_label": str(value["model_label"]),
        "provider": "openai_compatible",
        "protocol": "openai_chat",
        "model_id": str(value["model_id"]),
        "base_url": str(value["base_url"]).rstrip("/"),
        "api_key_env": api_key_env,
        "api_key_required": bool(value["api_key_required"]),
        "provider_routing": routing,
        "capabilities": {
            "text": bool(capabilities["text"]),
            "images": bool(capabilities["images"]),
        },
        "request_defaults": normalized_defaults,
        "rate_limits": {
            "max_concurrency": _positive_int(
                rate_limits["max_concurrency"], "max_concurrency"
            ),
            "requests_per_minute": _positive_int(
                rate_limits["requests_per_minute"], "requests_per_minute"
            ),
        },
    }
    digest = sha256_json(semantic)
    return ModelSpec(
        model_label=str(value["model_label"]),
        provider="openai_compatible",
        protocol="openai_chat",
        model_id=str(value["model_id"]),
        base_url=str(value["base_url"]).rstrip("/"),
        api_key_env=api_key_env,
        api_key_required=bool(value["api_key_required"]),
        provider_routing=routing,
        text_capable=bool(capabilities["text"]),
        image_capable=bool(capabilities["images"]),
        request_defaults=normalized_defaults,
        max_concurrency=_positive_int(
            rate_limits["max_concurrency"], "max_concurrency"
        ),
        requests_per_minute=_positive_int(
            rate_limits["requests_per_minute"], "requests_per_minute"
        ),
        digest=digest,
    )


def load_model_spec(path: Path) -> ModelSpec:
    return parse_model_spec(read_json(path))
