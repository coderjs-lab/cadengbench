from copy import deepcopy
from pathlib import Path

import pytest

from cadengbench_p_eval.io import read_json, write_json
from cadengbench_p_eval.profile import load_profile

PROFILE = Path(__file__).resolve().parents[2] / "configs/profiles/dev.json"


def test_repository_dev_profile_is_valid() -> None:
    profile = load_profile(PROFILE)
    assert profile.profile_id == "cadengbench-p.dev.requirements-l1-complete-l2.v1"


def test_invalid_profile_fails_before_case_execution(tmp_path: Path) -> None:
    value = deepcopy(read_json(PROFILE))
    value["geometry"]["surface_sample_count"] = 0
    path = tmp_path / "invalid-profile.json"
    write_json(path, value)

    with pytest.raises(ValueError, match="surface_sample_count"):
        load_profile(path)
