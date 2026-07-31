from __future__ import annotations

import pytest

from cadengbench_p_runner.extraction import (
    ExtractionError,
    extract_artifact,
    extract_json,
    extract_python,
    response_finish_reason,
)


def response(content: object) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def test_extracts_one_python_fence_without_semantic_repair() -> None:
    result = extract_python(response("Here is the file:\n```python\nx = 1\n```"))
    assert result.method == "single_valid_python_fence"
    assert result.source == "x = 1\n"


def test_rejects_ambiguous_or_invalid_output() -> None:
    with pytest.raises(ExtractionError, match="unambiguous"):
        extract_python(response("```python\nx=1\n```\n```python\ny=2\n```"))
    with pytest.raises(ExtractionError, match="valid Python"):
        extract_python(response("this is not python prose"))


def test_extracts_one_valid_cadquery_suffix_after_prose() -> None:
    result = extract_python(
        response(
            "I used a centered workplane.\n"
            "import cadquery as cq\n"
            "result = cq.Workplane('XY').box(1, 2, 3)\n"
        )
    )

    assert result.method == "unique_cadquery_suffix"
    assert result.source.startswith("import cadquery as cq\n")


def test_does_not_repair_invalid_cadquery_suffix() -> None:
    with pytest.raises(ExtractionError, match="unambiguous"):
        extract_python(
            response(
                "Here is the model.\n"
                "import cadquery as cq\n"
                "result = cq.Workplane('XY').box(\n"
            )
        )


def test_selects_only_valid_complete_program_among_illustrative_fences() -> None:
    result = extract_python(
        response(
            "Change this:\n```python\n.hole(2.3)\n```\n"
            "to this:\n```python\n.hole(2.53)\n```\n"
            "```python\nimport cadquery as cq\n"
            "result = cq.Workplane('XY').box(1, 2, 3)\n```\n"
        )
    )

    assert result.method == "single_valid_python_fence"
    assert result.source.startswith("import cadquery as cq\n")


def test_recovers_valid_suffix_after_invalid_example_fence() -> None:
    result = extract_python(
        response(
            "The relevant operation is:\n```python\n.circle(1.25)\n```\n"
            "import cadquery as cq\n"
            "result = cq.Workplane('XY').circle(1.375).extrude(3)\n"
        )
    )

    assert result.method == "unique_cadquery_suffix"


def test_extracts_and_canonicalizes_one_json_submission() -> None:
    result = extract_json(
        response('```json\n{"edits": [], "schema_version": "x"}\n```')
    )
    assert result.method == "single_json_fence"
    assert result.policy == "single_json_artifact_no_repair.v1"
    assert result.source == '{\n  "edits": [],\n  "schema_version": "x"\n}\n'


def test_dispatches_extraction_from_frozen_response_contract() -> None:
    result = extract_artifact(
        response('{"schema_version": "x"}'),
        {"extraction_policy": "single_json_artifact_no_repair.v1"},
    )
    assert result.method == "raw_json"


@pytest.mark.parametrize("field", ["finish_reason", "finishReason", "stop_reason"])
def test_reads_provider_finish_reason_variants(field: str) -> None:
    body = response("partial")
    body["choices"][0][field] = "length"  # type: ignore[index]

    assert response_finish_reason(body) == "length"
