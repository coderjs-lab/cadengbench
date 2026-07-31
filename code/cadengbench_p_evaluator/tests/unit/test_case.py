from pathlib import Path

import pytest

import cadengbench_p_eval.case as case_module
from cadengbench_p_eval.case import (
    CaseError,
    LoadedCase,
    _apply_orientation_free_contract,
    _resolve_reference,
    _validate_measurement_bindings,
    compile_functional_case,
)
from cadengbench_p_eval.io import sha256_file


def test_loaded_case_rejects_artifact_path_escape(tmp_path: Path) -> None:
    item_root = tmp_path / "item"
    item_root.mkdir()
    outside = tmp_path / "outside.step"
    outside.write_text("outside", encoding="utf-8")
    case = LoadedCase(
        path=tmp_path / "case.json",
        data={
            "item_id": "p_benchcad_escape",
            "source_item_root": str(item_root),
            "artifacts": {
                "target_step": {
                    "item_relative_path": "../outside.step",
                    "sha256": sha256_file(outside),
                }
            },
        },
    )

    with pytest.raises(CaseError, match="escapes its item root"):
        case.artifact_path("target_step")


def test_orientation_free_contract_merges_indistinguishable_opening_groups() -> None:
    task = {"reference_frame": "source_canonical_frame"}
    requirements = [
        {
            "requirement_id": "R-HOLE-001",
            "check_ids": ["hidden.check.cylindrical_opening_group.v1"],
            "value_and_tolerance": {
                "quantity": 1,
                "diameter": 8.0,
                "depth": 2.0,
                "axis_direction": [1.0, 0.0, 0.0],
            },
        },
        {
            "requirement_id": "R-HOLE-002",
            "check_ids": ["hidden.check.cylindrical_opening_group.v1"],
            "value_and_tolerance": {
                "quantity": 2,
                "diameter": 8.0,
                "depth": 5.0,
                "axis_direction": [0.0, 1.0, 0.0],
            },
        },
    ]

    _apply_orientation_free_contract(task, requirements, {"parameters": []})

    assert len(requirements) == 1
    assert requirements[0]["value_and_tolerance"]["quantity"] == 3
    assert requirements[0]["source_requirement_ids"] == [
        "R-HOLE-001",
        "R-HOLE-002",
    ]
    assert requirements[0]["value_and_tolerance"]["axis_assignment"] == "free"
    assert requirements[0]["value_and_tolerance"]["depth_scoring"] == "not_disclosed"


def test_reference_resolution_prefers_release_local_item(tmp_path: Path) -> None:
    production = tmp_path / "production/dataset/items/p_benchcad_test/source"
    production.mkdir(parents=True)
    absolute = production / "model.step"
    absolute.write_text("production", encoding="utf-8")
    item_root = tmp_path / "release/items/p_benchcad_test"
    local = item_root / "source/model.step"
    local.parent.mkdir(parents=True)
    local.write_text("release", encoding="utf-8")

    resolved = _resolve_reference(
        item_root,
        [str(absolute)],
        suffixes=(".step",),
    )

    assert resolved == local.absolute()


def test_measurement_binding_accepts_bounded_source_rounding() -> None:
    requirements = [
        {
            "requirement_id": "R-BBOX",
            "value_and_tolerance": {"x": {"value": 7.2800002}},
        }
    ]
    parameter_system = {
        "parameters": [
            {
                "parameter_id": "P-001",
                "nominal": 7.28,
                "measurement_binding": {
                    "measurement_key": "P-001",
                    "requirement_id": "R-BBOX",
                    "requirement_value_path": ["value_and_tolerance", "x", "value"],
                },
            }
        ]
    }

    _validate_measurement_bindings(parameter_system, requirements)


def test_functional_case_records_relocatable_item_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item_root = tmp_path / "release/items/p_benchcad_test"
    acceptance_path = item_root / "acceptance.json"
    acceptance_path.parent.mkdir(parents=True)
    acceptance_path.write_text(
        """
{
  "schema_version": "cadengbench-p.functional-edit-acceptance.v2",
  "status": "verified_compiled",
  "item_id": "p_benchcad_test",
  "source_key": "benchcad",
  "contract_id": "test-contract",
  "parameter_contract": {
    "parameter_id": "P-001",
    "parameter_key": "length",
    "canonical_unit": "mm",
    "base_value": 10.0,
    "target_value": 12.0,
    "source_pointer": "/parameters/length",
    "domain": {"lower": 8.0, "upper": 14.0}
  }
}
""".strip(),
        encoding="utf-8",
    )
    base_case = {
        "item_id": "p_benchcad_test",
        "source_family": "benchcad",
        "source_item_root": str(item_root),
        "requirements": [],
        "artifacts": {
            "functional_edit_acceptance_contract": {
                "item_relative_path": "acceptance.json"
            }
        },
    }
    monkeypatch.setattr(
        case_module,
        "compile_case",
        lambda *_args, **_kwargs: dict(base_case),
    )
    output = (
        tmp_path
        / "release/functional_edit/cases/p_benchcad_test.case.json"
    )

    compiled = compile_functional_case(
        item_root,
        profile=object(),  # type: ignore[arg-type]
        output_path=output,
    )

    assert compiled["source_item_root"] == "../../items/p_benchcad_test"
