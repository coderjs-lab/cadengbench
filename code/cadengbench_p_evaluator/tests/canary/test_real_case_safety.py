from pathlib import Path

import cadquery as cq
import pytest

from cadengbench_p_eval.case import compile_case
from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.layers.l1 import evaluate_l1
from cadengbench_p_eval.profile import load_profile

REPO_ROOT = Path(__file__).resolve().parents[3]
ITEM_ROOT = (
    REPO_ROOT
    / "data/interim/cadengbench_p_r25_combined_hpc_canary_30/dataset/items"
    / "p_benchcad_000215"
)
PROFILE = Path(__file__).resolve().parents[2] / "configs/profiles/dev.json"


@pytest.mark.canary
def test_explicitly_unconstrained_item_accepts_requirement_equivalent_geometry(
    tmp_path: Path,
) -> None:
    if not ITEM_ROOT.is_dir():
        pytest.skip("repository-local canary is unavailable")
    case = compile_case(ITEM_ROOT, profile=load_profile(PROFILE))
    envelope = next(
        requirement["value_and_tolerance"]
        for requirement in case["requirements"]
        if requirement["requirement_id"] == "R-002"
    )
    impostor = tmp_path / "envelope_impostor.step"
    cq.exporters.export(
        cq.Workplane("XY").box(
            envelope["x"]["value"],
            envelope["y"]["value"],
            envelope["z"]["value"],
        ),
        str(impostor),
    )

    result = evaluate_l1(case, impostor)

    assert result.status is Status.PASSED
    assert result.metrics["requirement_coverage"] == {
        "requirement_count": 2,
        "required_requirement_count": 2,
        "categories": ["dimension", "topology"],
        "reference_similarity_used": False,
    }
    freedoms = case["task_contract"]["allowed_design_freedom"]
    assert (
        "The mounting pattern is not constrained by the supplied facts"
        in freedoms["acceptable_equivalences"]
    )
    assert all("reference" not in check.check_id for check in result.checks)
