import pytest

from cadengbench_p_eval.contracts import (
    CheckResult,
    LayerResult,
    Status,
    check_result_from_dict,
    reduce_required_status,
)


def test_status_has_nullable_pass_semantics() -> None:
    assert CheckResult("ok", Status.PASSED).passed is True
    assert CheckResult("bad", Status.FAILED, failure_code="bad").passed is False
    assert CheckResult("na", Status.NOT_APPLICABLE).passed is None


def test_optional_diagnostic_does_not_block_layer() -> None:
    checks = [
        CheckResult("hard", Status.PASSED),
        CheckResult("diagnostic", Status.NOT_SCOREABLE, required=False),
    ]
    assert reduce_required_status(checks) is Status.PASSED
    assert LayerResult.from_checks("L1", checks).passed is True


def test_required_missing_authority_prevents_pass() -> None:
    checks = [CheckResult("authority", Status.NOT_SCOREABLE)]
    assert reduce_required_status(checks) is Status.NOT_SCOREABLE


def test_deserializer_does_not_coerce_string_required_flag() -> None:
    with pytest.raises(ValueError, match="required must be a boolean"):
        check_result_from_dict(
            {
                "check_id": "bad-required",
                "status": "passed",
                "required": "false",
            }
        )
