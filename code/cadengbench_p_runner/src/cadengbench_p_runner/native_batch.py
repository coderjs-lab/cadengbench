"""Resumable native batch transports for the hosted P-campaign models."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .io import read_json, sha256_json, write_json
from .planner import model_from_manifest
from .providers.base import ProviderResult
from .providers.openai_compatible import OpenAICompatibleAdapter
from .scheduler import RAW_ENVELOPE_SCHEMA

TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}


def _json_request(
    url: str,
    *,
    api_key: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 300,
    extra_headers: dict[str, str] | None = None,
    bearer_auth: bool = True,
) -> dict[str, Any]:
    body = (
        json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        if payload is not None
        else None
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if bearer_auth:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra_headers or {})
    request = Request(
        url,
        data=body,
        method=method,
        headers=headers,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"batch API HTTP {exc.code}: {detail}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("batch API returned a non-object response")
    return value


def _upload(
    url: str, *, api_key: str, path: Path, timeout: int = 300
) -> dict[str, Any]:
    boundary = f"----cadengbench-{uuid.uuid4().hex}"
    content = path.read_bytes()
    chunks = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n',
        f"--{boundary}\r\n".encode(),
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{path.name}"\r\n'
        ).encode(),
        b"Content-Type: application/jsonl\r\n\r\n",
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    request = Request(
        url,
        data=b"".join(chunks),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"batch upload HTTP {exc.code}: {detail}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise RuntimeError("batch upload did not return a file ID")
    return value


def _download(
    url: str,
    *,
    api_key: str,
    timeout: int = 300,
    extra_headers: dict[str, str] | None = None,
    bearer_auth: bool = True,
) -> bytes:
    headers = {"Accept": "*/*"}
    if bearer_auth:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra_headers or {})
    request = Request(
        url,
        headers=headers,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"batch download HTTP {exc.code}: {detail}") from exc


def _custom_id(item_id: str, sample_index: int) -> str:
    return f"{item_id}--{sample_index:03d}"


def _requests(
    run_root: Path, adapter: OpenAICompatibleAdapter
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    manifest = read_json(run_root / "run_manifest.json")
    records = []
    for record in manifest["requests"]:
        request = read_json(run_root / record["request_path"])
        for sample_index in range(int(manifest["sample_count"])):
            custom_id = _custom_id(record["item_id"], sample_index)
            payload = adapter.provider_request(request, run_root=run_root)
            records.append((custom_id, record, payload))
    return records


def _write_chunks(
    run_root: Path,
    records: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    chunk_size: int,
) -> list[Path]:
    batch_root = run_root / "native_batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for offset in range(0, len(records), chunk_size):
        path = batch_root / f"input_{offset // chunk_size:04d}.jsonl"
        lines = [
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": payload,
                },
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
            for custom_id, _record, payload in records[offset : offset + chunk_size]
        ]
        expected = "\n".join(lines) + "\n"
        if path.is_file() and path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"existing native batch input differs: {path}")
        path.write_text(expected, encoding="utf-8")
        paths.append(path)
    return paths


def _write_mistral_chunks(
    run_root: Path,
    records: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    chunk_size: int,
) -> list[Path]:
    batch_root = run_root / "native_batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for offset in range(0, len(records), chunk_size):
        path = batch_root / f"input_{offset // chunk_size:04d}.jsonl"
        lines = [
            json.dumps(
                {"custom_id": custom_id, "body": payload},
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
            for custom_id, _record, payload in records[offset : offset + chunk_size]
        ]
        expected = "\n".join(lines) + "\n"
        if path.is_file() and path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"existing native batch input differs: {path}")
        path.write_text(expected, encoding="utf-8")
        paths.append(path)
    return paths


def _anthropic_params(payload: dict[str, Any]) -> dict[str, Any]:
    system: list[dict[str, str]] = []
    messages: list[dict[str, Any]] = []
    for message in payload["messages"]:
        blocks = []
        for part in message["content"]:
            if part["type"] == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                url = part["image_url"]["url"]
                prefix, data = url.split(",", 1)
                media_type = prefix.removeprefix("data:").removesuffix(";base64")
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            else:
                raise ValueError(f"unsupported Anthropic content part: {part['type']}")
        if message["role"] == "system":
            system.extend(blocks)
        else:
            messages.append({"role": message["role"], "content": blocks})
    params: dict[str, Any] = {
        "model": payload["model"],
        "messages": messages,
        "max_tokens": payload["max_tokens"],
    }
    if system:
        params["system"] = system
    for field in ("temperature", "top_p"):
        if payload.get(field) is not None:
            params[field] = payload[field]
    return params


def _anthropic_result_line(
    line: dict[str, Any], request_payload: dict[str, Any]
) -> ProviderResult:
    result = line.get("result")
    if isinstance(result, dict) and result.get("type") == "succeeded":
        message = result.get("message")
        if not isinstance(message, dict):
            raise ValueError("Anthropic succeeded result has no message")
        text = "".join(
            str(block.get("text", ""))
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        raw_usage = dict(message.get("usage") or {})
        body = {
            "id": message.get("id"),
            "model": message.get("model"),
            "choices": [
                {
                    "finish_reason": message.get("stop_reason"),
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {
                **raw_usage,
                "prompt_tokens": raw_usage.get("input_tokens", 0),
                "completion_tokens": raw_usage.get("output_tokens", 0),
                "total_tokens": raw_usage.get("input_tokens", 0)
                + raw_usage.get("output_tokens", 0),
            },
        }
        return ProviderResult(
            outcome="response",
            retryable=False,
            http_status=200,
            error_class=None,
            error_message=None,
            provider_request_id=str(message.get("id") or ""),
            returned_model_id=str(message.get("model"))
            if message.get("model") is not None
            else None,
            usage=body["usage"],
            body=body,
            provider_request_hash=sha256_json(request_payload),
        )
    return ProviderResult(
        outcome="error",
        retryable=False,
        http_status=None,
        error_class=f"anthropic_batch_{(result or {}).get('type', 'error')}",
        error_message=str((result or {}).get("error") or result or line),
        provider_request_id=None,
        returned_model_id=None,
        provider_request_hash=sha256_json(request_payload),
    )


def _result_from_line(
    line: dict[str, Any], request_payload: dict[str, Any]
) -> ProviderResult:
    response = line.get("response")
    if isinstance(response, dict) and isinstance(response.get("body"), dict):
        body = response["body"]
        status = int(response.get("status_code", 200))
        return ProviderResult(
            outcome="response" if 200 <= status < 300 else "error",
            retryable=False,
            http_status=status,
            error_class=None if 200 <= status < 300 else "batch_http_error",
            error_message=None if 200 <= status < 300 else str(body),
            provider_request_id=str(body.get("id") or line.get("id") or ""),
            returned_model_id=str(body.get("model"))
            if body.get("model") is not None
            else None,
            usage=dict(body.get("usage") or {}),
            body=body,
            provider_request_hash=sha256_json(request_payload),
        )
    return ProviderResult(
        outcome="error",
        retryable=False,
        http_status=None,
        error_class="batch_request_error",
        error_message=str(line.get("error") or line),
        provider_request_id=None,
        returned_model_id=None,
        provider_request_hash=sha256_json(request_payload),
    )


def _import_results(
    run_root: Path,
    records: list[tuple[str, dict[str, Any], dict[str, Any]]],
    output_paths: list[Path],
    result_parser: Any = _result_from_line,
) -> None:
    manifest = read_json(run_root / "run_manifest.json")
    by_id = {custom_id: (record, payload) for custom_id, record, payload in records}
    seen: set[str] = set()
    now = datetime.now(UTC).isoformat()
    for output_path in output_paths:
        for raw in output_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            line = json.loads(raw)
            custom_id = line.get("custom_id")
            if custom_id not in by_id or custom_id in seen:
                raise ValueError(f"unexpected native batch custom_id: {custom_id}")
            seen.add(custom_id)
            record, payload = by_id[custom_id]
            sample_index = int(custom_id.rsplit("--", 1)[1])
            result = result_parser(line, payload)
            envelope = {
                "schema_version": RAW_ENVELOPE_SCHEMA,
                "run_id": manifest["run_id"],
                "item_id": record["item_id"],
                "sample_index": sample_index,
                "attempt_index": 1,
                "request_hash": record["request_hash"],
                "started_at": now,
                "ended_at": now,
                "latency_seconds": 0.0,
                "provider_result": result.as_dict(),
            }
            path = (
                run_root
                / "raw_responses"
                / record["item_id"]
                / f"sample_{sample_index:03d}"
                / "attempt_0001.json"
            )
            if path.is_file():
                existing = read_json(path)
                if (
                    existing.get("run_id") != manifest["run_id"]
                    or existing.get("item_id") != record["item_id"]
                    or existing.get("sample_index") != sample_index
                    or existing.get("request_hash") != record["request_hash"]
                    or existing.get("provider_result") != result.as_dict()
                ):
                    raise ValueError(f"existing batch envelope differs: {path}")
                continue
            write_json(path, envelope)
    missing = set(by_id) - seen
    if missing:
        raise RuntimeError(
            f"native batch results are incomplete: {len(missing)} missing"
        )


def run_openai_batch(
    *,
    run_root: Path,
    base_url: str,
    api_key_env: str,
    chunk_size: int = 25,
    poll_seconds: int = 60,
) -> dict[str, Any]:
    """Submit, resume, download, and import all requests in one planned run."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"required API key is absent: {api_key_env}")
    manifest = read_json(run_root / "run_manifest.json")
    model = model_from_manifest(manifest)
    adapter = OpenAICompatibleAdapter(model)
    records = _requests(run_root, adapter)
    chunks = _write_chunks(run_root, records, chunk_size=chunk_size)
    root_url = base_url.rstrip("/")
    state_path = run_root / "native_batch/state.json"
    state = read_json(state_path) if state_path.is_file() else {"jobs": []}
    jobs = {job["input"]: job for job in state["jobs"]}

    for chunk in chunks:
        relative = chunk.relative_to(run_root).as_posix()
        if relative in jobs:
            continue
        uploaded = _upload(f"{root_url}/files", api_key=api_key, path=chunk)
        batch = _json_request(
            f"{root_url}/batches",
            api_key=api_key,
            method="POST",
            payload={
                "input_file_id": uploaded["id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"run_id": manifest["run_id"], "input": relative},
            },
        )
        job = {"input": relative, "file_id": uploaded["id"], "batch_id": batch["id"]}
        state["jobs"].append(job)
        jobs[relative] = job
        write_json(state_path, state)

    output_paths = []
    for job in state["jobs"]:
        while True:
            batch = _json_request(
                f"{root_url}/batches/{job['batch_id']}", api_key=api_key
            )
            status = str(batch.get("status"))
            job["status"] = status
            write_json(state_path, state)
            if status in TERMINAL_BATCH_STATES:
                break
            time.sleep(poll_seconds)
        if status != "completed":
            raise RuntimeError(f"native batch {job['batch_id']} ended as {status}")
        job_outputs = []
        for kind, field in (
            ("output", "output_file_id"),
            ("error", "error_file_id"),
        ):
            file_id = batch.get(field)
            if not isinstance(file_id, str):
                continue
            output = (
                run_root
                / "native_batch"
                / f"{kind}_{job['batch_id']}.jsonl"
            )
            if not output.is_file():
                output.write_bytes(
                    _download(
                        f"{root_url}/files/{file_id}/content",
                        api_key=api_key,
                    )
                )
            output_paths.append(output)
            job_outputs.append(output)
        if not job_outputs:
            raise RuntimeError(
                f"native batch {job['batch_id']} completed without result files"
            )
    _import_results(run_root, records, output_paths)
    summary = {
        "schema_version": "cadengbench-p.native-batch.v1",
        "run_id": manifest["run_id"],
        "request_count": len(records),
        "batch_count": len(state["jobs"]),
        "complete": True,
    }
    write_json(run_root / "native_batch/summary.json", summary)
    return summary


def run_gemini_batch(
    *,
    run_root: Path,
    base_url: str,
    api_key_env: str,
    chunk_size: int = 25,
    poll_seconds: int = 60,
) -> dict[str, Any]:
    """Execute Gemini's OpenAI-compatible Batch API with Gemini file I/O."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"required API key is absent: {api_key_env}")

    from google import genai

    manifest = read_json(run_root / "run_manifest.json")
    model = model_from_manifest(manifest)
    adapter = OpenAICompatibleAdapter(model)
    records = _requests(run_root, adapter)
    chunks = _write_chunks(run_root, records, chunk_size=chunk_size)
    root_url = base_url.rstrip("/")
    state_path = run_root / "native_batch/state.json"
    state = read_json(state_path) if state_path.is_file() else {"jobs": []}
    jobs = {job["input"]: job for job in state["jobs"]}
    client = genai.Client(api_key=api_key)

    try:
        for chunk in chunks:
            relative = chunk.relative_to(run_root).as_posix()
            if relative in jobs:
                continue
            uploaded = client.files.upload(
                file=chunk,
                config={
                    "display_name": chunk.name,
                    "mime_type": "application/jsonl",
                },
            )
            if not isinstance(uploaded.name, str):
                raise RuntimeError("Gemini file upload did not return a file name")
            batch = _json_request(
                f"{root_url}/batches",
                api_key=api_key,
                method="POST",
                payload={
                    "input_file_id": uploaded.name,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "metadata": {"run_id": manifest["run_id"], "input": relative},
                },
            )
            job = {
                "input": relative,
                "file_id": uploaded.name,
                "batch_id": batch["id"],
            }
            state["jobs"].append(job)
            jobs[relative] = job
            write_json(state_path, state)

        outputs = []
        for job in state["jobs"]:
            while True:
                batch = _json_request(
                    f"{root_url}/batches/{job['batch_id']}",
                    api_key=api_key,
                )
                status = str(batch.get("status"))
                job["status"] = status
                write_json(state_path, state)
                if status in TERMINAL_BATCH_STATES:
                    break
                time.sleep(poll_seconds)
            if status != "completed":
                raise RuntimeError(
                    f"Gemini batch {job['batch_id']} ended as {status}"
                )
            job_outputs = []
            for kind, field in (
                ("output", "output_file_id"),
                ("error", "error_file_id"),
            ):
                file_id = batch.get(field)
                if not isinstance(file_id, str):
                    continue
                output = (
                    run_root
                    / "native_batch"
                    / f"{kind}_{job['batch_id']}.jsonl"
                )
                if not output.is_file():
                    output.write_bytes(client.files.download(file=file_id))
                outputs.append(output)
                job_outputs.append(output)
            if not job_outputs:
                raise RuntimeError(
                    f"Gemini batch {job['batch_id']} completed without result files"
                )
    finally:
        client.close()

    _import_results(run_root, records, outputs)
    summary = {
        "schema_version": "cadengbench-p.native-batch.v1",
        "provider": "gemini",
        "run_id": manifest["run_id"],
        "request_count": len(records),
        "batch_count": len(state["jobs"]),
        "complete": True,
    }
    write_json(run_root / "native_batch/summary.json", summary)
    return summary


def run_anthropic_batch(
    *,
    run_root: Path,
    base_url: str,
    api_key_env: str,
    chunk_size: int = 25,
    poll_seconds: int = 60,
) -> dict[str, Any]:
    """Execute one planned run through Anthropic Message Batches."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"required API key is absent: {api_key_env}")
    manifest = read_json(run_root / "run_manifest.json")
    model = model_from_manifest(manifest)
    adapter = OpenAICompatibleAdapter(model)
    records = _requests(run_root, adapter)
    root_url = base_url.rstrip("/")
    batch_root = run_root / "native_batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    state_path = batch_root / "state.json"
    state = read_json(state_path) if state_path.is_file() else {"jobs": []}
    jobs = {job["offset"]: job for job in state["jobs"]}
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    for offset in range(0, len(records), chunk_size):
        key = offset // chunk_size
        if key in jobs:
            continue
        requests = [
            {"custom_id": custom_id, "params": _anthropic_params(payload)}
            for custom_id, _record, payload in records[offset : offset + chunk_size]
        ]
        batch = _json_request(
            f"{root_url}/v1/messages/batches",
            api_key=api_key,
            method="POST",
            payload={"requests": requests},
            extra_headers=headers,
            bearer_auth=False,
        )
        job = {"offset": key, "batch_id": batch["id"]}
        state["jobs"].append(job)
        jobs[key] = job
        write_json(state_path, state)

    outputs = []
    for job in state["jobs"]:
        while True:
            batch = _json_request(
                f"{root_url}/v1/messages/batches/{job['batch_id']}",
                api_key=api_key,
                extra_headers=headers,
                bearer_auth=False,
            )
            status = str(batch.get("processing_status"))
            job["status"] = status
            write_json(state_path, state)
            if status == "ended":
                break
            time.sleep(poll_seconds)
        output = batch_root / f"output_{job['batch_id']}.jsonl"
        if not output.is_file():
            output.write_bytes(
                _download(
                    f"{root_url}/v1/messages/batches/{job['batch_id']}/results",
                    api_key=api_key,
                    extra_headers=headers,
                    bearer_auth=False,
                )
            )
        outputs.append(output)
    _import_results(
        run_root,
        records,
        outputs,
        result_parser=_anthropic_result_line,
    )
    summary = {
        "schema_version": "cadengbench-p.native-batch.v1",
        "provider": "anthropic",
        "run_id": manifest["run_id"],
        "request_count": len(records),
        "batch_count": len(state["jobs"]),
        "complete": True,
    }
    write_json(batch_root / "summary.json", summary)
    return summary


def run_mistral_batch(
    *,
    run_root: Path,
    base_url: str,
    api_key_env: str,
    chunk_size: int = 25,
    poll_seconds: int = 60,
) -> dict[str, Any]:
    """Execute one planned run through Mistral Batch Jobs."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"required API key is absent: {api_key_env}")
    manifest = read_json(run_root / "run_manifest.json")
    model = model_from_manifest(manifest)
    adapter = OpenAICompatibleAdapter(model)
    records = _requests(run_root, adapter)
    chunks = _write_mistral_chunks(run_root, records, chunk_size=chunk_size)
    root_url = base_url.rstrip("/")
    state_path = run_root / "native_batch/state.json"
    state = read_json(state_path) if state_path.is_file() else {"jobs": []}
    jobs = {job["input"]: job for job in state["jobs"]}

    for chunk in chunks:
        relative = chunk.relative_to(run_root).as_posix()
        if relative in jobs:
            continue
        uploaded = _upload(f"{root_url}/files", api_key=api_key, path=chunk)
        batch = _json_request(
            f"{root_url}/batch/jobs",
            api_key=api_key,
            method="POST",
            payload={
                "input_files": [uploaded["id"]],
                "model": model.model_id,
                "endpoint": "/v1/chat/completions",
                "metadata": {"run_id": manifest["run_id"], "input": relative},
            },
        )
        job = {"input": relative, "file_id": uploaded["id"], "batch_id": batch["id"]}
        state["jobs"].append(job)
        jobs[relative] = job
        write_json(state_path, state)

    outputs = []
    for job in state["jobs"]:
        while True:
            batch = _json_request(
                f"{root_url}/batch/jobs/{job['batch_id']}", api_key=api_key
            )
            status = str(batch.get("status"))
            job["status"] = status
            write_json(state_path, state)
            if status not in {"QUEUED", "RUNNING"}:
                break
            time.sleep(poll_seconds)
        if status != "SUCCESS":
            raise RuntimeError(f"Mistral batch {job['batch_id']} ended as {status}")
        job_outputs = []
        for kind, field in (("output", "output_file"), ("error", "error_file")):
            file_id = batch.get(field)
            if not isinstance(file_id, str):
                continue
            output = (
                run_root
                / "native_batch"
                / f"{kind}_{job['batch_id']}.jsonl"
            )
            if not output.is_file():
                output.write_bytes(
                    _download(
                        f"{root_url}/files/{file_id}/content",
                        api_key=api_key,
                    )
                )
            outputs.append(output)
            job_outputs.append(output)
        if not job_outputs:
            raise RuntimeError(
                f"Mistral batch {job['batch_id']} completed without result files"
            )
    _import_results(run_root, records, outputs)
    summary = {
        "schema_version": "cadengbench-p.native-batch.v1",
        "provider": "mistral",
        "run_id": manifest["run_id"],
        "request_count": len(records),
        "batch_count": len(state["jobs"]),
        "complete": True,
    }
    write_json(run_root / "native_batch/summary.json", summary)
    return summary
