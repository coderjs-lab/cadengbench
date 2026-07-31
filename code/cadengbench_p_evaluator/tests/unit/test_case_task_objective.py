import pytest

from cadengbench_p_eval.case import CaseError, _configure_nominal_task_contract


def test_general_task_is_requirement_scored_and_reference_diagnostic() -> None:
    contract: dict = {"fidelity_profile": "exact_reference"}

    _configure_nominal_task_contract(
        contract,
        source_fidelity_profile="exact_reference",
    )

    assert contract["task_objective"] == "engineering_specification"
    assert contract["fidelity_profile"] == "requirements_only"
    assert (
        contract["reference_geometry_role"] == "provenance_and_non_scoring_diagnostics"
    )


def test_exact_reconstruction_requires_thresholds() -> None:
    contract = {
        "task_objective": "exact_reconstruction",
        "fidelity_profile": "exact_reference",
    }

    with pytest.raises(CaseError, match="requires reference_similarity_thresholds"):
        _configure_nominal_task_contract(
            contract,
            source_fidelity_profile="exact_reference",
        )


def test_exact_reconstruction_preserves_explicit_threshold_contract() -> None:
    contract = {
        "task_objective": "exact_reconstruction",
        "reference_similarity_thresholds": {
            "chamfer_normalized_max": 0.01,
            "surface_f1_min": 0.95,
            "voxel_iou_min": 0.9,
        },
    }

    _configure_nominal_task_contract(
        contract,
        source_fidelity_profile="requirements_only",
    )

    assert contract["fidelity_profile"] == "exact_reference"
    assert "reference_similarity" in contract["l1_scoring_authority"]
    assert contract["reference_geometry_role"] == (
        "score_bearing_exact_reconstruction_target"
    )
