"""Map an L3 perturbation-family result into the public layer contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status


def l3_layer_from_family_result(value: Mapping[str, Any]) -> LayerResult:
    checks = []
    for record in value.get("variants") or []:
        variant_id = str(record["variant_id"])
        final_status = str(record["final_status"])
        if final_status == "probe_pass":
            status = Status.PASSED
            failure_code = None
        elif final_status == "skipped_upstream":
            status = Status.NOT_REACHED
            failure_code = None
        elif final_status == "requirement_fail":
            status = Status.FAILED
            failure_code = "physics_requirement_failed"
        elif record.get("failure_owner") == "evaluator":
            status = Status.EVALUATOR_ERROR
            failure_code = None
        else:
            status = Status.FAILED
            failure_code = "physics_pipeline_failed"
        checks.append(
            CheckResult(
                check_id=f"L3.variant.{variant_id}",
                status=status,
                failure_code=failure_code,
                observed={
                    "final_status": final_status,
                    "failure_stage": record.get("failure_stage"),
                    "failure_owner": record.get("failure_owner"),
                },
                runtime_seconds=record.get("runtime_seconds"),
            )
        )
    return LayerResult.from_checks(
        "L3",
        checks,
        metrics={
            key: value.get(key)
            for key in (
                "variant_count",
                "l3_reached_count",
                "probe_pass_count",
                "evaluator_error_count",
                "conditional_l3_robustness",
                "end_to_end_family_robustness",
            )
        },
        artifacts={"family_result_schema": value.get("schema_version")},
        runtime_seconds=value.get("runtime_seconds"),
    )
