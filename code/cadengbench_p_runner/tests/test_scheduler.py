from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cadengbench_p_runner.io import write_json
from cadengbench_p_runner.ledger import AttemptLedger
from cadengbench_p_runner.packaging import load_candidate_submission
from cadengbench_p_runner.planner import plan_run
from cadengbench_p_runner.scheduler import run_generation


class MockHandler(BaseHTTPRequestHandler):
    calls = 0
    fail_first = True
    response_content = (
        "```python\nimport cadquery as cq\nresult = cq.Workplane('XY')\n```"
    )

    def do_POST(self) -> None:
        type(self).calls += 1
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        assert request["model"] == "mock-cad-model-v1"
        if type(self).fail_first and type(self).calls == 1:
            payload = {"error": {"message": "rate limited"}}
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
            return
        payload = {
            "id": f"mock-{type(self).calls}",
            "model": "mock-cad-model-revision",
            "choices": [{"message": {"content": type(self).response_content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture
def mock_server():
    MockHandler.calls = 0
    MockHandler.fail_first = True
    MockHandler.response_content = (
        "```python\nimport cadquery as cq\nresult = cq.Workplane('XY')\n```"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.integration
def test_retry_checkpoint_and_resume_without_duplicate_call(
    synthetic_items: Path, model_spec_writer, tmp_path: Path, mock_server: str
) -> None:
    model = model_spec_writer(tmp_path / "model.json", mock_server)
    run = tmp_path / "run"
    plan_run(
        items_root=synthetic_items,
        model_spec_path=model,
        output_root=run,
        run_id="retry-resume-test",
        input_profile="text_iso",
        max_attempts=2,
    )
    summary = run_generation(run)
    assert summary["valid_complete_run"] is True
    assert MockHandler.calls == 2
    records = AttemptLedger(run / "attempts.jsonl").records()
    assert [record["status"] for record in records] == [
        "retryable_failure",
        "succeeded",
    ]
    assert records[0]["retry_decision"] == "retry"
    candidate_path = run / "submissions/p_benchcad_000001/candidate_submission.json"
    candidate = load_candidate_submission(candidate_path, run_root=run)
    assert candidate["successful_attempt"]["attempt_index"] == 2
    assert (run / "artifacts/p_benchcad_000001/submission.py").is_file()

    resumed = run_generation(run)
    assert resumed["valid_complete_run"] is True
    assert MockHandler.calls == 2
    assert len(AttemptLedger(run / "attempts.jsonl").records()) == 2


@pytest.mark.integration
def test_recovers_unledgered_success_envelope_without_api_call(
    synthetic_items: Path, model_spec_writer, tmp_path: Path, mock_server: str
) -> None:
    MockHandler.fail_first = False
    model = model_spec_writer(tmp_path / "model.json", mock_server)
    run = tmp_path / "run"
    manifest = plan_run(
        items_root=synthetic_items,
        model_spec_path=model,
        output_root=run,
        run_id="orphan-envelope-test",
        input_profile="text",
    )
    request_record = manifest["requests"][0]
    raw = run / "raw_responses/p_benchcad_000001/sample_000/attempt_0001.json"
    write_json(
        raw,
        {
            "schema_version": "cadengbench-p.raw-provider-envelope.v1",
            "run_id": manifest["run_id"],
            "item_id": "p_benchcad_000001",
            "sample_index": 0,
            "attempt_index": 1,
            "request_hash": request_record["request_hash"],
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:01+00:00",
            "latency_seconds": 1.0,
            "provider_result": {
                "outcome": "response",
                "retryable": False,
                "http_status": 200,
                "error_class": None,
                "error_message": None,
                "provider_request_id": "saved-request",
                "returned_model_id": "mock-cad-model-revision",
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
                "body": {
                    "id": "saved-request",
                    "model": "mock-cad-model-revision",
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "import cadquery as cq\n"
                                    "result = cq.Workplane('XY')\n"
                                )
                            }
                        }
                    ],
                },
                "response_headers": {},
                "provider_request_hash": "9" * 64,
                "retry_after_seconds": None,
            },
        },
    )
    summary = run_generation(run)
    assert summary["valid_complete_run"] is True
    assert MockHandler.calls == 0
    records = AttemptLedger(run / "attempts.jsonl").records()
    assert len(records) == 1
    assert records[0]["status"] == "succeeded"


@pytest.mark.integration
def test_functional_edit_generation_packages_v2_candidate(
    synthetic_items: Path, model_spec_writer, tmp_path: Path, mock_server: str
) -> None:
    MockHandler.fail_first = False
    MockHandler.response_content = (
        "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 12)\n"
    )
    model = model_spec_writer(tmp_path / "model.json", mock_server)
    run = tmp_path / "functional-run"
    plan_run(
        items_root=synthetic_items,
        model_spec_path=model,
        output_root=run,
        run_id="functional-generation-test",
        input_profile="text_iso",
        task_kind="functional_edit",
    )
    summary = run_generation(run)
    assert summary["valid_complete_run"] is True
    candidate = load_candidate_submission(
        run / "submissions/p_benchcad_000001/candidate_submission.json",
        run_root=run,
    )
    assert candidate["schema_version"] == "cadengbench-p.candidate-submission.v2"
    assert candidate["task_kind"] == "functional_edit"
    assert candidate["artifact"]["interface_version"] == (
        "cadengbench-p.benchcad-source-edit.v1"
    )


@pytest.mark.integration
def test_invalid_model_output_is_settled_and_preserved_for_l0(
    synthetic_items: Path, model_spec_writer, tmp_path: Path, mock_server: str
) -> None:
    MockHandler.fail_first = False
    MockHandler.response_content = "I could not construct the requested CAD artifact."
    model = model_spec_writer(tmp_path / "model.json", mock_server)
    run = tmp_path / "invalid-output-run"
    plan_run(
        items_root=synthetic_items,
        model_spec_path=model,
        output_root=run,
        run_id="invalid-output-test",
        input_profile="text",
    )

    summary = run_generation(run)

    assert summary["run_settled"] is True
    assert summary["valid_complete_run"] is False
    assert summary["terminal_failed_jobs"] == 1
    assert summary["pending_jobs"] == 0
    assert (
        run / "artifacts/p_benchcad_000001/response.txt"
    ).read_text(encoding="utf-8") == MockHandler.response_content


@pytest.mark.integration
def test_output_truncation_is_not_packaged_as_model_cad_failure(
    synthetic_items: Path, model_spec_writer, tmp_path: Path, mock_server: str
) -> None:
    MockHandler.fail_first = False
    model = model_spec_writer(tmp_path / "model.json", mock_server)
    run = tmp_path / "truncated-output-run"
    plan_run(
        items_root=synthetic_items,
        model_spec_path=model,
        output_root=run,
        run_id="truncated-output-test",
        input_profile="text",
    )
    original = MockHandler.do_POST

    def truncated_response(handler: MockHandler) -> None:
        type(handler).calls += 1
        length = int(handler.headers["Content-Length"])
        json.loads(handler.rfile.read(length))
        payload = {
            "id": "mock-truncated",
            "model": "mock-cad-model-revision",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": (
                            "import cadquery as cq\n"
                            "result = cq.Workplane('XY')\n"
                        )
                    },
                }
            ],
        }
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps(payload).encode())

    MockHandler.do_POST = truncated_response
    try:
        summary = run_generation(run)
    finally:
        MockHandler.do_POST = original

    assert summary["run_settled"] is True
    assert summary["truncated_jobs"] == 1
    assert summary["valid_complete_run"] is False
    records = AttemptLedger(run / "attempts.jsonl").records()
    assert records[0]["error_class"] == "output_truncated"
    assert not (run / "artifacts/p_benchcad_000001/submission.py").exists()
