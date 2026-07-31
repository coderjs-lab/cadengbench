from pathlib import Path

import pytest

from cadengbench_p_eval.model_output import ModelOutputError, materialize_submission


def test_direct_python_is_preserved(tmp_path: Path) -> None:
    response = tmp_path / "answer.py"
    response.write_text("result = 1\n", encoding="utf-8")
    submission = materialize_submission(
        response, output_root=tmp_path / "out", max_bytes=1000
    )
    assert submission.extraction == "direct_python_file"
    assert submission.source_path.read_text(encoding="utf-8") == "result = 1\n"


def test_exactly_one_code_only_fence_is_accepted(tmp_path: Path) -> None:
    response = tmp_path / "answer.txt"
    response.write_text("```python\nresult = 1\n```\n", encoding="utf-8")
    submission = materialize_submission(
        response, output_root=tmp_path / "out", max_bytes=1000
    )
    assert submission.extraction == "single_python_fence"


@pytest.mark.parametrize(
    "value",
    [
        "Here is the code:\n```python\nresult = 1\n```\n",
        "```python\na = 1\n```\n```python\nb = 2\n```\n",
        "no code",
    ],
)
def test_ambiguous_or_prose_response_is_rejected(tmp_path: Path, value: str) -> None:
    response = tmp_path / "answer.txt"
    response.write_text(value, encoding="utf-8")
    with pytest.raises(ModelOutputError):
        materialize_submission(response, output_root=tmp_path / "out", max_bytes=1000)
