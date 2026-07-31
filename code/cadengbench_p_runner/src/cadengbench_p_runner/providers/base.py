"""Narrow provider result contract used by the scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderResult:
    outcome: str
    retryable: bool
    http_status: int | None
    error_class: str | None
    error_message: str | None
    provider_request_id: str | None
    returned_model_id: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    body: Any = None
    response_headers: dict[str, str] = field(default_factory=dict)
    provider_request_hash: str | None = None
    retry_after_seconds: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome == "response"

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "retryable": self.retryable,
            "http_status": self.http_status,
            "error_class": self.error_class,
            "error_message": self.error_message,
            "provider_request_id": self.provider_request_id,
            "returned_model_id": self.returned_model_id,
            "usage": self.usage,
            "body": self.body,
            "response_headers": self.response_headers,
            "provider_request_hash": self.provider_request_hash,
            "retry_after_seconds": self.retry_after_seconds,
        }
