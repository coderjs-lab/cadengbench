from __future__ import annotations

import json

from cadengbench_p_runner.cli import main


def test_doctor_reports_provider_free_evaluator_boundary(capsys) -> None:
    assert main(["doctor"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["providers"] == ["openai_compatible"]
    assert value["generation_scoring_separated"] is True
    assert value["llm_judge_used"] is False
