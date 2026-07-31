"""Construction-accounting policy for supported A2 simulator mappings."""

from __future__ import annotations

from typing import Any


def build_a2_policy_manifest(
    direct: dict[str, Any],
    compound: dict[str, Any],
    ball: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate the frozen A2 construction-accounting policy."""
    direct_total = int(direct["pose_audit"]["alternative_count"])
    compound_total = int(compound["pose_audit"]["alternative_count"])
    ball_total = int(ball["alternative_count"])
    direct_passed = int(direct["smoke"]["status_counts"].get("passed", 0))
    compound_passed = int(compound["smoke"]["status_counts"].get("passed", 0))
    ball_passed = int(ball["execution_status_counts"].get("passed", 0))
    capability_gaps = [
        {
            "assembly_id": record["assembly_id"],
            "joint_id": record["joint_id"],
            "status": record["result"]["status"],
            "reason": record["result"]["reason"],
            "enabled_limit_groups": record["result"]["enabled_limit_groups"],
        }
        for record in ball.get("records", [])
        if (record.get("result") or {}).get("status") == "unsupported_source_limits"
    ]
    total = direct_total + compound_total + ball_total
    supported = direct_passed + compound_passed + ball_passed
    gap_count = len(capability_gaps)
    unexpected = {
        "direct": {key: value for key, value in direct["smoke"]["status_counts"].items() if key != "passed"},
        "compound": {key: value for key, value in compound["smoke"]["status_counts"].items() if key != "passed"},
        "ball": {
            key: value
            for key, value in ball["execution_status_counts"].items()
            if key not in {"passed", "unsupported_source_limits"}
        },
    }
    accounting_valid = (
        all(not values for values in unexpected.values())
        and direct_passed == direct_total
        and compound_passed == compound_total
        and supported + gap_count == total
    )
    if not accounting_valid:
        raise ValueError(f"A2 construction accounting is inconsistent: {unexpected}")

    return {
        "schema": "cadengbench_a_a2_policy_manifest_v1",
        "scope": "construction_policy_not_model_evaluation",
        "accounting": {
            "all_alternatives": total,
            "faithfully_executed_passes": supported,
            "simulator_capability_gaps": gap_count,
            "supported_coverage": supported / total if total else 0.0,
            "accounting_valid": True,
            "by_representation": {
                "direct": {"total": direct_total, "passed": direct_passed},
                "compound": {"total": compound_total, "passed": compound_passed},
                "native_spherical": {
                    "total": ball_total,
                    "passed": ball_passed,
                    "capability_gaps": gap_count,
                },
            },
        },
        "scoring_policy": {
            "supported_kinematic_denominator": supported,
            "capability_gaps_excluded_from_kinematic_pass_denominator": True,
            "capability_gaps_reported_against_full_coverage_denominator": total,
            "capability_gap_is_not_model_or_source_failure": True,
            "model_benchmarking_deferred_until_pipeline_freeze": True,
        },
        "collision_policy": {
            "role": "diagnostic_only",
            "contact_count_affects_kinematic_verdict": False,
            "reason": "valid mating interfaces may start in contact and no allowable-contact mask is available",
        },
        "capability_gap_records": capability_gaps,
    }
