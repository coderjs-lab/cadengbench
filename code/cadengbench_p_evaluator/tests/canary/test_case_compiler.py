import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from cadengbench_p_eval.case import compile_case, load_case
from cadengbench_p_eval.parametrics.relations import parse_relation
from cadengbench_p_eval.profile import load_profile

REPO_ROOT = Path(__file__).resolve().parents[3]
ITEMS_ROOT = (
    REPO_ROOT / "data/interim/cadengbench_p_r25_combined_hpc_canary_30/dataset/items"
)
PROFILE = Path(__file__).resolve().parents[2] / "configs/profiles/dev.json"
CASE_SCHEMA = (
    Path(__file__).resolve().parents[2] / "schemas/evaluator_case.v1.schema.json"
)


@pytest.mark.canary
def test_all_canary_items_compile_with_explicit_coverage(tmp_path: Path) -> None:
    if not ITEMS_ROOT.is_dir():
        pytest.skip("repository-local canary is unavailable")
    profile = load_profile(PROFILE)
    item_roots = sorted(path for path in ITEMS_ROOT.iterdir() if path.is_dir())
    assert len(item_roots) == 30
    source_counts: dict[str, int] = {}
    public_parameters = 0
    measurement_bound_parameters = 0
    qualified_parameters = 0
    scoreable_dfm = 0
    source_relation_count = 0
    typed_relation_count = 0
    delegated_parameterization_count = 0
    functional_protocol_count = 0
    for item_root in item_roots:
        output = tmp_path / f"{item_root.name}.json"
        case = compile_case(item_root, profile=profile, output_path=output)
        loaded = load_case(output)
        assert loaded.item_id == item_root.name
        source_counts[case["source_family"]] = (
            source_counts.get(case["source_family"], 0) + 1
        )
        public_parameters += len(case["parameter_system"]["parameters"])
        measurement_bound_parameters += case["parameter_system"][
            "measurement_bound_parameter_count"
        ]
        Draft202012Validator(
            json.loads(CASE_SCHEMA.read_text(encoding="utf-8"))
        ).validate(case)
        qualified_parameters += case["parameter_system"]["qualified_parameter_count"]
        scoreable_dfm += case["dfm"]["scoreable_family_count"]
        task_contract = case["task_contract"]
        assert task_contract["task_objective"] == "engineering_specification"
        assert task_contract["fidelity_profile"] == "requirements_only"
        assert (
            task_contract["l1_scoring_authority"]
            == "typed_engineering_requirements_and_applicable_dfm"
        )
        assert (
            task_contract["reference_geometry_role"]
            == "provenance_and_non_scoring_diagnostics"
        )
        assert task_contract["reference_frame"] == "global_orientation_free"
        assert "source_reference_frame" in task_contract
        assert "hard_gates" not in case["evaluator_profile"]["geometry"]
        for requirement in case["requirements"]:
            if "hidden.check.bbox_dimensions.v1" in requirement["check_ids"]:
                assert (
                    requirement["value_and_tolerance"]["axis_assignment"]
                    == "free_permutation"
                )
            if "hidden.check.cylindrical_opening_group.v1" in requirement["check_ids"]:
                contract = requirement["value_and_tolerance"]
                assert contract["axis_assignment"] == "free"
                assert contract["depth_scoring"] == "not_disclosed"
                assert contract["position_scoring"] == "not_disclosed"
        for parameter in case["parameter_system"]["parameters"]:
            binding = parameter.get("measurement_binding")
            if binding and binding["requirement_id"] == "R-BBOX":
                assert binding["measurement_kind"] == "bounding_box_unordered_dimension"
        source_relation_count += len(case["parameter_system"]["source_relations"])
        typed_relation_count += len(case["parameter_system"]["relations"])
        for relation in case["parameter_system"]["relations"]:
            parse_relation(relation)
        delegated_parameterization_count += sum(
            "hidden.check.cylindrical_opening_parameterization.v1"
            in requirement["check_ids"]
            for requirement in case["requirements"]
        )
        functional_protocol_count += int(
            case["functional_edit"]["separate_protocol_available"]
        )
    assert source_counts == {"benchcad": 15, "fusion360_reconstruction": 15}
    assert public_parameters == 136
    assert measurement_bound_parameters == 136
    assert qualified_parameters == 0
    assert scoreable_dfm == 0
    assert source_relation_count == 11
    assert typed_relation_count == 22
    assert delegated_parameterization_count == 11
    assert functional_protocol_count == 30
