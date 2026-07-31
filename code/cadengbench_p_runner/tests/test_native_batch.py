import json
from pathlib import Path
from types import SimpleNamespace

import cadengbench_p_runner.native_batch as native_batch
from cadengbench_p_runner.native_batch import (
    _anthropic_params,
    _anthropic_result_line,
    _result_from_line,
    _write_chunks,
    _write_mistral_chunks,
)


def test_native_batch_chunks_are_deterministic(tmp_path: Path) -> None:
    records = [
        (
            f"item-{index}--000",
            {"item_id": f"item-{index}", "request_hash": str(index)},
            {"model": "mock", "messages": [{"role": "user", "content": "x"}]},
        )
        for index in range(3)
    ]

    paths = _write_chunks(tmp_path, records, chunk_size=2)

    assert [path.name for path in paths] == ["input_0000.jsonl", "input_0001.jsonl"]
    assert len(paths[0].read_text(encoding="utf-8").splitlines()) == 2
    assert _write_chunks(tmp_path, records, chunk_size=2) == paths


def test_native_batch_result_preserves_openai_body() -> None:
    body = {
        "id": "response-1",
        "model": "mock-model",
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }

    result = _result_from_line(
        {
            "custom_id": "item--000",
            "response": {"status_code": 200, "body": body},
        },
        {"model": "mock-model"},
    )

    assert result.succeeded
    assert result.body == body
    assert result.returned_model_id == "mock-model"
    assert result.usage["completion_tokens"] == 2


def test_native_batch_error_is_terminal() -> None:
    result = _result_from_line(
        {"custom_id": "item--000", "error": {"message": "rejected"}},
        {"model": "mock-model"},
    )

    assert not result.succeeded
    assert not result.retryable
    assert result.error_class == "batch_request_error"


def test_mistral_batch_uses_custom_id_and_body(tmp_path: Path) -> None:
    records = [
        (
            "item-1--000",
            {"item_id": "item-1", "request_hash": "hash"},
            {"model": "mistral-small", "messages": [{"role": "user", "content": "x"}]},
        )
    ]

    [path] = _write_mistral_chunks(tmp_path, records, chunk_size=25)
    line = json.loads(path.read_text(encoding="utf-8"))

    assert set(line) == {"custom_id", "body"}
    assert line["custom_id"] == "item-1--000"
    assert line["body"]["model"] == "mistral-small"


def test_anthropic_params_convert_system_text_and_image() -> None:
    params = _anthropic_params(
        {
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Return CadQuery."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Build this part."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                },
            ],
        }
    )

    assert params["system"] == [{"type": "text", "text": "Return CadQuery."}]
    assert params["messages"][0]["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "AAAA",
        },
    }


def test_anthropic_batch_result_becomes_openai_compatible_body() -> None:
    result = _anthropic_result_line(
        {
            "custom_id": "item--000",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_1",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "answer"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            },
        },
        {"model": "claude-sonnet-4-5"},
    )

    assert result.succeeded
    assert result.body["choices"][0]["message"]["content"] == "answer"
    assert result.usage["prompt_tokens"] == 10
    assert result.usage["completion_tokens"] == 3
    assert result.usage["total_tokens"] == 13


def test_gemini_batch_uses_native_file_io(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run_manifest.json").write_text(
        json.dumps({"run_id": "gemini-test"}),
        encoding="utf-8",
    )
    records = [
        (
            "item-1--000",
            {"item_id": "item-1", "request_hash": "hash"},
            {"model": "gemini-test", "messages": []},
        )
    ]
    calls = []
    imported = []

    class FakeFiles:
        def upload(self, *, file, config):
            calls.append(("upload", Path(file).name, config["mime_type"]))
            return SimpleNamespace(name="files/input-1")

        def download(self, *, file):
            calls.append(("download", file))
            return b'{"custom_id":"item-1--000","response":{"body":{}}}\n'

    class FakeClient:
        def __init__(self, *, api_key):
            calls.append(("client", api_key))
            self.files = FakeFiles()

        def close(self):
            calls.append(("close",))

    def fake_json_request(url, **kwargs):
        calls.append(("request", url, kwargs.get("method", "GET")))
        if kwargs.get("method") == "POST":
            assert kwargs["payload"]["input_file_id"] == "files/input-1"
            return {"id": "batch-1"}
        return {
            "id": "batch-1",
            "status": "completed",
            "output_file_id": "files/output-1",
        }

    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setattr(native_batch, "model_from_manifest", lambda _value: object())
    monkeypatch.setattr(
        native_batch,
        "OpenAICompatibleAdapter",
        lambda _value: object(),
    )
    monkeypatch.setattr(native_batch, "_requests", lambda *_args: records)
    monkeypatch.setattr(native_batch, "_json_request", fake_json_request)
    monkeypatch.setattr(
        native_batch,
        "_import_results",
        lambda _root, _records, paths: imported.extend(paths),
    )
    monkeypatch.setattr("google.genai.Client", FakeClient)

    summary = native_batch.run_gemini_batch(
        run_root=run_root,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        poll_seconds=0,
    )

    assert summary["provider"] == "gemini"
    assert summary["request_count"] == 1
    assert calls[0] == ("client", "secret")
    assert ("upload", "input_0000.jsonl", "application/jsonl") in calls
    assert ("download", "files/output-1") in calls
    assert [path.name for path in imported] == ["output_batch-1.jsonl"]
