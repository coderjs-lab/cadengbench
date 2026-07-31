import json
from pathlib import Path

from cadengbench_p_eval.cli import main


def test_doctor_reports_no_llm_runtime(capsys) -> None:
    assert main(["doctor"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["ready"] is True
    assert "vllm" not in value["packages"]


def test_prepare_case_and_validate(tmp_path: Path, capsys) -> None:
    repo = Path(__file__).resolve().parents[3]
    item = (
        repo
        / "data/interim/cadengbench_p_r25_combined_hpc_canary_30"
        / "dataset/items/p_fusion360_000966"
    )
    if not item.is_dir():
        return
    profile = Path(__file__).resolve().parents[2] / "configs/profiles/dev.json"
    case = tmp_path / "case.json"
    assert (
        main(
            [
                "prepare-case",
                "--item-root",
                str(item),
                "--profile",
                str(profile),
                "--output",
                str(case),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["validate-case", "--case", str(case)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
