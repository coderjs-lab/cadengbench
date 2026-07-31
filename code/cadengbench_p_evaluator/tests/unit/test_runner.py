import json
from pathlib import Path

import pytest

from cadengbench_p_eval.case import LoadedCase
from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status
from cadengbench_p_eval.runner import (
    _l1_timeout,
    _missing_l1_analysis_status,
    _overall_status,
    _variant_case,
    evaluate_evalset,
)


def test_optional_evaluator_error_invalidates_item_outcome() -> None:
    layers = (
        LayerResult.from_checks(
            "L0",
            (
                CheckResult("L0.required", Status.PASSED),
                CheckResult(
                    "L0.optional_diagnostic",
                    Status.EVALUATOR_ERROR,
                    required=False,
                ),
            ),
        ),
        LayerResult.from_checks("L1", (CheckResult("L1.required", Status.PASSED),)),
        LayerResult.from_checks("L2", (CheckResult("L2.required", Status.PASSED),)),
    )

    assert all(layer.status is Status.PASSED for layer in layers)
    assert _overall_status(layers) is Status.EVALUATOR_ERROR


@pytest.mark.parametrize(
    ("l1_status", "expected"),
    [
        (Status.FAILED, Status.FAILED),
        (Status.NOT_SCOREABLE, Status.FAILED),
        (Status.EVALUATOR_ERROR, Status.EVALUATOR_ERROR),
    ],
)
def test_missing_l1_analysis_preserves_failure_ownership(
    l1_status: Status, expected: Status
) -> None:
    layer = LayerResult("L1", l1_status)

    assert _missing_l1_analysis_status(layer) is expected


def test_l1_timeout_allows_bounded_budget_for_complex_multibody_case() -> None:
    base = {
        "evaluator_profile": {
            "execution": {
                "timeout_seconds": 60,
                "worker_startup_grace_seconds": 60,
            },
            "geometry": {"evaluation_timeout_seconds": 600},
        },
        "requirements": [
            {
                "category": "solid_contract",
                "value_and_tolerance": {"body_count": 38},
            }
        ],
    }

    assert _l1_timeout(base) == 1200.0
    base["requirements"][0]["value_and_tolerance"]["body_count"] = 8
    assert _l1_timeout(base) == 600.0


def test_variant_case_rebinds_envelope_and_opening_requirements() -> None:
    case = LoadedCase(
        path=Path("/tmp/synthetic.case.json"),
        data={
            "requirements": [
                {
                    "requirement_id": "R-002",
                    "value_and_tolerance": {
                        "x": {"value": 10.0, "tolerance": 0.05},
                        "y": {"value": 4.0, "tolerance": 0.05},
                        "z": {"value": 2.0, "tolerance": 0.05},
                    },
                },
                {
                    "requirement_id": "R-003",
                    "value_and_tolerance": {"diameter": 3.0, "depth": 2.0},
                },
            ],
            "parameter_system": {
                "parameters": [
                    {
                        "parameter_id": "P-001",
                        "measurement_binding": {
                            "requirement_id": "R-002",
                            "requirement_value_path": [
                                "value_and_tolerance",
                                "x",
                                "value",
                            ],
                        },
                    },
                    {
                        "parameter_id": "P-002",
                        "measurement_binding": {
                            "requirement_id": "R-003",
                            "requirement_value_path": [
                                "value_and_tolerance",
                                "diameter",
                            ],
                        },
                    },
                ]
            },
        },
    )

    variant, unbound = _variant_case(case, {"P-001": 12.0, "P-002": 3.5})

    assert unbound == []
    assert variant["requirements"][0]["value_and_tolerance"]["x"]["value"] == 12.0
    assert variant["requirements"][1]["value_and_tolerance"]["diameter"] == 3.5
    assert case.data["requirements"][0]["value_and_tolerance"]["x"]["value"] == 10.0


def test_variant_case_refuses_unbound_public_parameter() -> None:
    case = LoadedCase(
        path=Path("/tmp/synthetic.case.json"),
        data={
            "requirements": [],
            "parameter_system": {
                "parameters": [{"parameter_id": "P-001", "measurement_binding": None}]
            },
        },
    )

    _variant, unbound = _variant_case(case, {"P-001": 12.0})

    assert unbound == ["P-001"]


def test_variant_case_uses_invariants_for_geometry_response_only_parameter() -> None:
    case = LoadedCase(
        path=Path("/tmp/synthetic.case.json"),
        data={
            "requirements": [
                {
                    "requirement_id": "R-SOLID",
                    "category": "solid_contract",
                    "value_and_tolerance": {"body_count": 1},
                },
                {
                    "requirement_id": "R-BBOX",
                    "category": "global_envelope",
                    "value_and_tolerance": {"x": {"value": 10.0}},
                },
            ],
            "dfm": {
                "scoreable_family_count": 1,
                "families": [
                    {
                        "family": "minimum_wall_thickness",
                        "required": True,
                        "scoreable": True,
                    }
                ],
            },
            "parameter_system": {
                "parameters": [
                    {
                        "parameter_id": "P-001",
                        "measurement_binding": None,
                        "variant_requirement_contract": {
                            "kind": "geometry_response_only",
                            "authority": "verified_source_native_replay",
                            "evidence_ref": "oracle.json",
                        },
                    }
                ]
            },
        },
    )

    variant, unbound = _variant_case(case, {"P-001": 12.0})

    assert unbound == []
    assert [value["requirement_id"] for value in variant["requirements"]] == [
        "R-SOLID"
    ]
    assert variant["dfm"]["scoreable_family_count"] == 0
    assert variant["dfm"]["families"][0]["required"] is False
    assert variant["variant_evaluation_scope"]["mode"] == "geometry_response_only"


def test_invalid_evalset_does_not_replace_existing_output(tmp_path: Path) -> None:
    evalset = tmp_path / "evalset"
    submissions = tmp_path / "submissions"
    output = tmp_path / "output"
    evalset.mkdir()
    submissions.mkdir()
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (evalset / "manifest.json").write_text(
        json.dumps(
            {"items": [{"item_id": "../escape", "case_path": "cases/escape.json"}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe unique IDs"):
        evaluate_evalset(
            evalset_root=evalset,
            submissions_root=submissions,
            output_root=output,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
