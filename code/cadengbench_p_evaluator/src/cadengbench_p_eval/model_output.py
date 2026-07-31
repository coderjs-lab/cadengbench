"""Strict, repair-free extraction of model-generated submission artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .io import sha256_file, write_text

_FENCE = re.compile(
    r"```(?P<language>python|py)?[ \t]*\r?\n(?P<code>.*?)\r?\n```",
    flags=re.DOTALL | re.IGNORECASE,
)


class ModelOutputError(ValueError):
    """A model response cannot be mapped to exactly one permitted artifact."""


@dataclass(frozen=True)
class MaterializedSubmission:
    source_path: Path
    raw_sha256: str
    source_sha256: str
    extraction: str


def _extract_source(path: Path, *, max_bytes: int) -> tuple[str, str]:
    if not path.is_file():
        raise ModelOutputError(f"model output is missing: {path}")
    if path.stat().st_size > max_bytes:
        raise ModelOutputError(f"model output exceeds {max_bytes} bytes")
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ModelOutputError("model output must be UTF-8") from exc
    if path.suffix.lower() == ".py":
        if "```" in raw:
            raise ModelOutputError("a .py submission cannot contain Markdown fences")
        return raw, "direct_python_file"
    matches = list(_FENCE.finditer(raw))
    if len(matches) != 1:
        raise ModelOutputError(
            "a text response must contain exactly one Python code fence"
        )
    outside = raw[: matches[0].start()] + raw[matches[0].end() :]
    if outside.strip():
        raise ModelOutputError("prose outside the Python code fence is not permitted")
    source = matches[0].group("code")
    if not source.strip():
        raise ModelOutputError("the Python code fence is empty")
    return source + ("" if source.endswith("\n") else "\n"), "single_python_fence"


def materialize_submission(
    model_output_path: Path,
    *,
    output_root: Path,
    max_bytes: int,
) -> MaterializedSubmission:
    """Materialize one exact source artifact without semantic repair."""

    source, extraction = _extract_source(model_output_path, max_bytes=max_bytes)
    source_path = output_root / "submission.py"
    write_text(source_path, source)
    return MaterializedSubmission(
        source_path=source_path,
        raw_sha256=sha256_file(model_output_path),
        source_sha256=sha256_file(source_path),
        extraction=extraction,
    )
