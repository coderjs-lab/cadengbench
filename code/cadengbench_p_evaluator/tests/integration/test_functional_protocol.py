from __future__ import annotations

import os
from pathlib import Path

import pytest

from cadengbench_p_eval.case import compile_functional_case
from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.functional_runner import evaluate_functional_item
from cadengbench_p_eval.profile import load_profile

REPO_ROOT = Path(__file__).resolve().parents[3]
CANARY = (
    REPO_ROOT / "data/interim/cadengbench_p_r25_combined_hpc_canary_30/dataset/items"
)
PROFILE = REPO_ROOT / "cadengbench_p_evaluator/configs/profiles/dev_slice_10.v2.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("CADENGBENCH_RUN_REAL_PROTOCOL_E") != "1",
    reason="set CADENGBENCH_RUN_REAL_PROTOCOL_E=1 for the seven-minute CAD suite",
)


@pytest.mark.parametrize(
    ("item_id", "submission_relative"),
    (
        (
            "p_benchcad_000578",
            "domain/parametric_edits/v2/replay/probe_01_8p272/model.cadquery.py",
        ),
        (
            "p_fusion360_001531",
            "quality/evaluator_certification/functional_edit/generations/"
            "d774567036454000b56caf64e1c96404/submissions/golden_patch.json",
        ),
    ),
)
def test_real_protocol_e_golden_edit_passes_all_layers(
    tmp_path: Path,
    item_id: str,
    submission_relative: str,
) -> None:
    item_root = CANARY / item_id
    case_path = tmp_path / f"{item_id}.case.json"
    compile_functional_case(
        item_root,
        profile=load_profile(PROFILE),
        output_path=case_path,
    )

    result = evaluate_functional_item(
        repo_root=REPO_ROOT,
        case_path=case_path,
        submission_path=item_root / submission_relative,
        output_root=tmp_path / f"{item_id}.result",
    )

    assert result.overall_status is Status.PASSED
    assert [layer.status for layer in result.layers] == [
        Status.PASSED,
        Status.PASSED,
        Status.PASSED,
    ]
    l1 = result.layers[1]
    assert not any(
        token in check.check_id.lower()
        for check in l1.checks
        for token in ("chamfer", "voxel", "surface_f1", "similarity")
    )
    l2 = result.layers[2]
    assert l2.metrics["replay_robustness"] == 1.0
    assert l2.metrics["parameter_integrity_score"] == 1.0
    assert l2.metrics["llm_judge_used"] is False

    resumed = evaluate_functional_item(
        repo_root=REPO_ROOT,
        case_path=case_path,
        submission_path=item_root / submission_relative,
        output_root=tmp_path / f"{item_id}.result",
    )
    assert resumed.as_dict() == result.as_dict()


def test_real_protocol_e_no_op_is_rejected_before_l2(tmp_path: Path) -> None:
    item_id = "p_fusion360_001531"
    item_root = CANARY / item_id
    case_path = tmp_path / f"{item_id}.case.json"
    compile_functional_case(
        item_root,
        profile=load_profile(PROFILE),
        output_path=case_path,
    )
    no_op = (
        item_root / "quality/evaluator_certification/functional_edit/generations/"
        "d774567036454000b56caf64e1c96404/submissions/no_op_patch.json"
    )

    result = evaluate_functional_item(
        repo_root=REPO_ROOT,
        case_path=case_path,
        submission_path=no_op,
        output_root=tmp_path / "no-op-result",
    )

    assert result.overall_status is Status.FAILED
    assert result.layers[0].status is Status.PASSED
    assert result.layers[1].status is Status.PASSED
    assert result.layers[2].status is Status.FAILED
    assert any(
        check.status is Status.FAILED
        and check.check_id
        in {
            "L2.functional.target.target_parameter_assignment",
            "L2.functional.effect.geometry_mediated_engineering_effect",
        }
        for check in result.layers[2].checks
    )
