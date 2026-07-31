from __future__ import annotations

from cadengbench_p_runner.model_registry import parse_model_spec
from cadengbench_p_runner.providers.openai_compatible import (
    _translated_request_defaults,
)


def _model(base_url: str) -> object:
    return parse_model_spec(
        {
            "schema_version": "cadengbench-p.model-spec.v1",
            "model_label": "test",
            "provider": "openai_compatible",
            "protocol": "openai_chat",
            "model_id": "test/model",
            "base_url": base_url,
            "api_key_env": None,
            "api_key_required": False,
            "capabilities": {"text": True, "images": True},
            "request_defaults": {
                "temperature": 0,
                "max_tokens": 1024,
                "reasoning_effort": "none",
            },
            "rate_limits": {
                "max_concurrency": 1,
                "requests_per_minute": 20,
            },
        }
    )


def test_openrouter_none_disables_native_reasoning() -> None:
    defaults = _translated_request_defaults(_model("https://openrouter.ai/api/v1"))
    assert defaults["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in defaults


def test_non_openrouter_retains_reasoning_effort() -> None:
    defaults = _translated_request_defaults(_model("https://example.com/v1"))
    assert defaults["reasoning_effort"] == "none"
    assert "reasoning" not in defaults
