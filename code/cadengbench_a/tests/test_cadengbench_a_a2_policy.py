from __future__ import annotations

import pytest

from pipeline.cadengbench_a_a2_policy import build_a2_policy_manifest


def _direct_or_compound(total: int) -> dict:
    return {
        "pose_audit": {"alternative_count": total},
        "smoke": {"status_counts": {"passed": total}},
    }


def test_policy_manifest_accounts_for_supported_and_capability_gap_cases() -> None:
    ball = {
        "alternative_count": 2,
        "execution_status_counts": {"passed": 1, "unsupported_source_limits": 1},
        "records": [
            {
                "assembly_id": "joint_set_1",
                "joint_id": "joint_0",
                "result": {
                    "status": "unsupported_source_limits",
                    "reason": "unsupported",
                    "enabled_limit_groups": {"pitch": {}},
                },
            }
        ],
    }

    manifest = build_a2_policy_manifest(_direct_or_compound(3), _direct_or_compound(4), ball)

    assert manifest["accounting"]["all_alternatives"] == 9
    assert manifest["accounting"]["faithfully_executed_passes"] == 8
    assert manifest["accounting"]["simulator_capability_gaps"] == 1
    assert manifest["collision_policy"]["contact_count_affects_kinematic_verdict"] is False


def test_policy_manifest_rejects_unaccounted_failures() -> None:
    direct = _direct_or_compound(3)
    direct["smoke"]["status_counts"] = {"passed": 2, "failed": 1}
    ball = {"alternative_count": 0, "execution_status_counts": {}, "records": []}

    with pytest.raises(ValueError, match="accounting is inconsistent"):
        build_a2_policy_manifest(direct, _direct_or_compound(0), ball)
