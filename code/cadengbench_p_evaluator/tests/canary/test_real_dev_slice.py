from pathlib import Path

import pytest

from cadengbench_p_eval.case import compile_case, load_case
from cadengbench_p_eval.execution.static import inspect_submission
from cadengbench_p_eval.io import read_json
from cadengbench_p_eval.profile import load_profile

REPO_ROOT = Path(__file__).resolve().parents[3]
SLICE_ROOT = REPO_ROOT / "data/eval/cadengbench_p_dev_slice_10_v1"
ITEMS_ROOT = SLICE_ROOT / "items"
PROFILE = Path(__file__).resolve().parents[2] / "configs/profiles/dev_slice_10.v1.json"


@pytest.mark.canary
def test_real_development_slice_has_scoreable_contracts(tmp_path: Path) -> None:
    if not ITEMS_ROOT.is_dir():
        pytest.skip("repository-local real development slice is unavailable")
    plan = read_json(SLICE_ROOT / "slice_plan.json")
    assert isinstance(plan, dict)
    records = plan["items"]
    assert isinstance(records, list)
    assert len(records) == 10
    assert len({record["item_id"] for record in records}) == 10

    profile = load_profile(PROFILE)
    source_counts: dict[str, int] = {}
    dfm_scoreable = 0
    for record in records:
        item_id = record["item_id"]
        item_root = ITEMS_ROOT / item_id
        output = tmp_path / f"{item_id}.case.json"
        case = compile_case(item_root, profile=profile, output_path=output)
        loaded = load_case(output)
        source = SLICE_ROOT / "submissions/positive" / f"{item_id}.py"
        inspection = inspect_submission(source, case=case)

        assert loaded.item_id == item_id
        assert inspection.interface_version == ("cadengbench-p.cadquery-submission.v1")
        assert case["task_contract"]["solid_contract"]["required_body_count"] >= 1
        assert case["parameter_system"]["complete_domains"] is True
        assert case["parameter_system"]["complete_measurement_bindings"] is True
        assert case["parameter_system"]["qualified_parameter_count"] == 1
        assert len(case["parameter_system"]["solver_obligations"]) == 1
        source_family = case["source_family"]
        source_counts[source_family] = source_counts.get(source_family, 0) + 1
        dfm_scoreable += case["dfm"]["scoreable_family_count"]

    assert source_counts == {"benchcad": 6, "fusion360_reconstruction": 4}
    assert dfm_scoreable == 2
