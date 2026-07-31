from pathlib import Path

import cadquery as cq
import pytest

from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.dfm.contracts import (
    MINIMUM_HOLE_GEOMETRY,
    MINIMUM_WALL_THICKNESS,
    resolve_dfm_gate,
)
from cadengbench_p_eval.dfm.holes import evaluate_minimum_hole_geometry
from cadengbench_p_eval.dfm.wall import evaluate_minimum_wall_thickness
from cadengbench_p_eval.geometry.features import (
    CylindricalFeatureInventory,
    extract_cylindrical_features,
)
from cadengbench_p_eval.geometry.requirements import (
    _orientation_free_opening_inventories,
    evaluate_requirements,
)
from cadengbench_p_eval.geometry.step import StepGeometryError, load_step
from cadengbench_p_eval.layers import l1 as l1_module
from cadengbench_p_eval.layers.l1 import evaluate_l1

GEOMETRY_PROFILE = {
    "linear_tessellation_tolerance_mm": 0.2,
    "surface_sample_count": 128,
    "surface_tolerance_mm": 0.2,
    "voxel_resolution": 6,
    "voxel_margin_fraction": 0.05,
    "seed": 7,
    "hard_gates": {"voxel_iou_min": None, "surface_f1_min": None},
}
DFM_PROFILE = {
    "wall_sample_count": 64,
    "wall_neighbor_count": 16,
    "opposed_normal_cosine_min": 0.9,
    "hole_axis_tolerance_degrees": 1.0,
}


def _export(tmp_path: Path, name: str, shape: cq.Workplane) -> Path:
    path = tmp_path / f"{name}.step"
    cq.exporters.export(shape, str(path))
    return path


def _scoreable_family(family: str, thresholds: dict[str, float]) -> dict:
    return {
        "family": family,
        "applicability": "applicable",
        "required": True,
        "thresholds": thresholds,
        "threshold_evidence_ref": "benchmark:test-profile:v1",
        "extractor_certified": True,
        "scoreable": True,
    }


def test_requirement_registry_detects_hole_mutation_and_unknown_check(
    tmp_path: Path,
) -> None:
    path = _export(
        tmp_path,
        "plate",
        cq.Workplane("XY").box(20, 20, 4).faces(">Z").workplane().hole(4),
    )
    geometry = load_step(path)
    requirements = [
        {
            "requirement_id": "R-001",
            "criticality": "required",
            "value_and_tolerance": {"body_count": 1},
            "check_ids": ["hidden.check.solid_count.v1"],
        },
        {
            "requirement_id": "R-002",
            "criticality": "required",
            "value_and_tolerance": {
                "quantity": 1,
                "diameter": 4.0,
                "depth": 4.0,
                "axis_direction": [0, 0, 1],
                "centers": [[0, 0, 0]],
                "diameter_tolerance": 0.01,
                "position_tolerance": 0.01,
            },
            "check_ids": ["hidden.check.cylindrical_opening_group.v1"],
        },
    ]
    assert all(
        result.status is Status.PASSED
        for result in evaluate_requirements(requirements, geometry)
    )
    requirements[1]["value_and_tolerance"]["diameter"] = 5.0
    assert evaluate_requirements(requirements, geometry)[1].status is Status.FAILED
    requirements[1]["check_ids"] = ["hidden.check.unknown.v1"]
    assert (
        evaluate_requirements(requirements, geometry)[1].status is Status.NOT_SCOREABLE
    )


def test_bbox_requirement_can_be_global_orientation_invariant(tmp_path: Path) -> None:
    geometry = load_step(
        _export(tmp_path, "rotated_key", cq.Workplane("XY").box(30.03, 3.22, 3.22))
    )
    requirement = {
        "requirement_id": "R-BBOX",
        "criticality": "required",
        "value_and_tolerance": {
            "x": {"value": 3.22, "tolerance": 0.05},
            "y": {"value": 30.03, "tolerance": 0.05},
            "z": {"value": 3.22, "tolerance": 0.05},
            "axis_assignment": "free_permutation",
        },
        "check_ids": ["hidden.check.bbox_dimensions.v1"],
    }

    invariant = evaluate_requirements([requirement], geometry)[0]
    assert invariant.status is Status.PASSED
    assert invariant.observed["axis_assignment"] == "free_permutation"

    requirement["value_and_tolerance"]["axis_assignment"] = "fixed"
    assert evaluate_requirements([requirement], geometry)[0].status is Status.FAILED


def test_hole_requirement_scores_only_disclosed_orientation_free_dimensions(
    tmp_path: Path,
) -> None:
    geometry = load_step(
        _export(
            tmp_path,
            "rotated_through_hole",
            cq.Workplane("YZ").box(20, 20, 8).faces(">X").workplane().hole(4),
        )
    )
    requirement = {
        "requirement_id": "R-HOLE",
        "criticality": "required",
        "value_and_tolerance": {
            "quantity": 1,
            "diameter": 4.0,
            "diameter_tolerance": 0.01,
            "depth": 2.0,
            "depth_tolerance": 0.01,
            "axis_direction": [0, 0, 1],
            "axis_assignment": "free",
            "depth_scoring": "not_disclosed",
            "centers": [[100, 100, 100]],
            "position_scoring": "not_disclosed",
        },
        "check_ids": ["hidden.check.cylindrical_opening_group.v1"],
    }

    result = evaluate_requirements([requirement], geometry)[0]

    assert result.status is Status.PASSED
    assert result.expected["axis_direction"] is None
    assert result.expected["depth_mm"] is None
    assert result.expected["position_scoring"] == "not_disclosed"


def test_overlapping_hole_tolerances_assign_each_face_once() -> None:
    class Hole:
        def __init__(self, diameter_mm: float) -> None:
            self.diameter_mm = diameter_mm
            self.is_hole = True

    requirements = [
        {
            "requirement_id": "R-HOLE-001",
            "check_ids": ["hidden.check.cylindrical_opening_group.v1"],
            "value_and_tolerance": {
                "quantity": 1,
                "diameter": 9.0,
                "diameter_tolerance": 0.1,
                "axis_assignment": "free",
                "depth_scoring": "not_disclosed",
                "position_scoring": "not_disclosed",
            },
        },
        {
            "requirement_id": "R-HOLE-002",
            "check_ids": ["hidden.check.cylindrical_opening_group.v1"],
            "value_and_tolerance": {
                "quantity": 1,
                "diameter": 9.1,
                "diameter_tolerance": 0.1,
                "axis_assignment": "free",
                "depth_scoring": "not_disclosed",
                "position_scoring": "not_disclosed",
            },
        },
    ]
    inventory = CylindricalFeatureInventory(
        assessments=(Hole(9.0), Hole(9.1))
    )

    assigned = _orientation_free_opening_inventories(requirements, inventory)

    assert [hole.diameter_mm for hole in assigned["R-HOLE-001"].holes] == [9.0]
    assert [hole.diameter_mm for hole in assigned["R-HOLE-002"].holes] == [9.1]


def test_parametricity_requirement_is_explicitly_delegated_to_l2(
    tmp_path: Path,
) -> None:
    geometry = load_step(_export(tmp_path, "box", cq.Workplane("XY").box(4, 3, 2)))
    result = evaluate_requirements(
        [
            {
                "requirement_id": "R-PARAM",
                "criticality": "required",
                "value_and_tolerance": {"parameter_ids": ["P-001"]},
                "check_ids": ["hidden.check.cylindrical_opening_parameterization.v1"],
            }
        ],
        geometry,
    )[0]

    assert result.status is Status.NOT_APPLICABLE
    assert result.required is False
    assert result.evidence[0]["delegated_layer"] == "L2"


def test_dfm_thresholds_pass_fail_and_require_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plate = load_step(
        _export(
            tmp_path,
            "hole_plate",
            cq.Workplane("XY").box(20, 20, 4).faces(">Z").workplane().hole(4),
        )
    )
    inventory = extract_cylindrical_features(plate)
    hole_contract = {
        "families": [
            _scoreable_family(
                MINIMUM_HOLE_GEOMETRY,
                {"minimum_diameter_mm": 3.0, "maximum_depth_to_diameter": 2.0},
            )
        ]
    }
    gate = resolve_dfm_gate(hole_contract, MINIMUM_HOLE_GEOMETRY)
    assert evaluate_minimum_hole_geometry(gate, inventory).status is Status.PASSED
    hole_contract["families"][0]["thresholds"]["minimum_diameter_mm"] = 5.0
    gate = resolve_dfm_gate(hole_contract, MINIMUM_HOLE_GEOMETRY)
    assert evaluate_minimum_hole_geometry(gate, inventory).status is Status.FAILED
    hole_contract["families"][0]["threshold_evidence_ref"] = None
    gate = resolve_dfm_gate(hole_contract, MINIMUM_HOLE_GEOMETRY)
    assert (
        evaluate_minimum_hole_geometry(gate, inventory).status is Status.NOT_SCOREABLE
    )

    wall_contract = {
        "families": [_scoreable_family(MINIMUM_WALL_THICKNESS, {"minimum_mm": 1.5})]
    }
    wall_gate = resolve_dfm_gate(wall_contract, MINIMUM_WALL_THICKNESS)

    def reject_tessellation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("L1 DFM must not tessellate")

    monkeypatch.setattr(cq.Shape, "tessellate", reject_tessellation)
    result = evaluate_minimum_wall_thickness(
        wall_gate,
        plate,
        sample_count=64,
        neighbor_count=16,
        opposed_normal_cosine_min=0.9,
        tessellation_tolerance_mm=0.2,
        seed=7,
    )
    assert result.status is Status.PASSED
    wall_contract["families"][0]["thresholds"]["minimum_mm"] = 4.5
    wall_gate = resolve_dfm_gate(wall_contract, MINIMUM_WALL_THICKNESS)
    result = evaluate_minimum_wall_thickness(
        wall_gate,
        plate,
        sample_count=64,
        neighbor_count=16,
        opposed_normal_cosine_min=0.9,
        tessellation_tolerance_mm=0.2,
        seed=7,
    )
    assert result.status is Status.FAILED


def _l1_case(
    *,
    requirements: list[dict] | None = None,
    dfm: dict | None = None,
) -> dict:
    return {
        "task_contract": {
            "task_objective": "engineering_specification",
            "fidelity_profile": "requirements_only",
        },
        "requirements": requirements
        if requirements is not None
        else [
            {
                "requirement_id": "R-001",
                "criticality": "required",
                "value_and_tolerance": {"body_count": 1},
                "check_ids": ["hidden.check.solid_count.v1"],
            }
        ],
        "dfm": dfm if dfm is not None else {"families": []},
        "evaluator_profile": {
            "geometry": {
                **GEOMETRY_PROFILE,
                "hard_gates": {"voxel_iou_min": 1.0, "surface_f1_min": 1.0},
            },
            "dfm": DFM_PROFILE,
        },
    }


def test_candidate_geometry_analysis_error_is_model_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _export(tmp_path, "candidate", cq.Workplane("XY").box(10, 8, 2))

    def reject_candidate_geometry(*_args: object, **_kwargs: object) -> None:
        raise StepGeometryError("trimmed face sampling failed")

    monkeypatch.setattr(l1_module, "geometry_fingerprint", reject_candidate_geometry)
    result = evaluate_l1(_l1_case(), candidate)

    check = next(
        check for check in result.checks if check.check_id == "L1.geometry.analysis"
    )
    assert result.status is Status.FAILED
    assert check.status is Status.FAILED
    assert check.failure_code == "geometry_analysis_failed"


def test_l1_missing_target_keeps_general_task_score_independent(
    tmp_path: Path,
) -> None:
    candidate = _export(tmp_path, "candidate", cq.Workplane("XY").box(10, 8, 2))
    case = _l1_case()

    result = evaluate_l1(
        case,
        candidate,
        target_step=tmp_path / "missing-target.step",
    )

    assert result.status is Status.PASSED
    assert len(result.metrics["geometry_fingerprint"]) == 64
    assert result.metrics["parameter_measurements"]["bbox_x_mm"] == 10.0
    assert result.metrics["parameter_measurements"]["solid_count"] == 1.0
    diagnostics = result.metrics["reference_geometry_diagnostics"]
    assert diagnostics["status"] == "skipped"
    assert diagnostics["reason"] == "non_scoring_reference_comparison_disabled"
    check = next(
        check
        for check in result.checks
        if check.check_id == "L1.reference_geometry.diagnostics"
    )
    assert check.status is Status.NOT_SCOREABLE
    assert check.required is False


def test_l1_skips_expensive_wall_check_after_required_geometry_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _export(tmp_path, "oversize", cq.Workplane("XY").box(20, 8, 2))
    case = _l1_case(
        requirements=[
            {
                "requirement_id": "R-BBOX",
                "criticality": "required",
                "value_and_tolerance": {
                    "x": {"value": 10.0, "tolerance": 0.1},
                    "y": {"value": 8.0, "tolerance": 0.1},
                    "z": {"value": 2.0, "tolerance": 0.1},
                    "axis_assignment": "fixed",
                },
                "check_ids": ["hidden.check.bbox_dimensions.v1"],
            }
        ],
        dfm={
            "families": [
                _scoreable_family(
                    MINIMUM_WALL_THICKNESS, {"minimum_mm": 1.0}
                )
            ]
        },
    )

    def reject_wall_check(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("expensive wall extraction should have been skipped")

    monkeypatch.setattr(
        "cadengbench_p_eval.dfm.evaluator.evaluate_minimum_wall_thickness",
        reject_wall_check,
    )
    result = evaluate_l1(case, candidate)

    assert result.status is Status.FAILED
    wall = next(
        check
        for check in result.checks
        if check.check_id == "L1.DFM.minimum_wall_thickness"
    )
    assert wall.status is Status.NOT_REACHED
    assert wall.required is True


def test_l1_does_not_measure_non_scoreable_wall_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _export(tmp_path, "candidate", cq.Workplane("XY").box(10, 8, 2))
    case = _l1_case(
        dfm={
            "families": [
                {
                    "family": MINIMUM_WALL_THICKNESS,
                    "required": False,
                    "scoreable": False,
                    "extractor_certified": True,
                    "thresholds": {"minimum_mm": 1.0},
                    "threshold_evidence_ref": {"authority": "test"},
                }
            ]
        }
    )

    def reject_wall_check(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("non-scoreable wall extraction should not run")

    monkeypatch.setattr(
        "cadengbench_p_eval.dfm.evaluator.evaluate_minimum_wall_thickness",
        reject_wall_check,
    )
    result = evaluate_l1(case, candidate)

    assert result.status is Status.PASSED
    wall = next(
        check
        for check in result.checks
        if check.check_id == "L1.DFM.minimum_wall_thickness"
    )
    assert wall.status is Status.NOT_SCOREABLE
    assert wall.required is False


def test_l1_skips_non_scoring_reference_metrics_for_complex_multibody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _export(
        tmp_path,
        "nine_body",
        cq.Workplane("XY")
        .pushPoints([(float(index) * 2.0, 0.0) for index in range(9)])
        .box(1, 1, 1, combine=False),
    )
    case = _l1_case(
        requirements=[
            {
                "requirement_id": "R-SOLID",
                "category": "solid_contract",
                "criticality": "required",
                "value_and_tolerance": {"body_count": 9},
                "check_ids": ["hidden.check.solid_count.v1"],
            }
        ]
    )

    def reject_reference_metrics(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("non-scoring multibody comparison should not run")

    monkeypatch.setattr(
        "cadengbench_p_eval.layers.l1.compare_geometry",
        reject_reference_metrics,
    )
    result = evaluate_l1(case, candidate, target_step=candidate)

    assert result.status is Status.PASSED
    diagnostics = result.metrics["reference_geometry_diagnostics"]
    assert diagnostics["status"] == "skipped"
    assert diagnostics["reason"] == "complex_multibody_non_scoring_diagnostic"


def test_l1_parameter_variant_does_not_compare_to_nominal_target(
    tmp_path: Path,
) -> None:
    candidate = _export(tmp_path, "variant", cq.Workplane("XY").box(10, 8, 2))

    result = evaluate_l1(
        _l1_case(),
        candidate,
        target_step=candidate,
        is_parameter_variant=True,
    )

    assert result.status is Status.PASSED
    diagnostics = result.metrics["reference_geometry_diagnostics"]
    assert diagnostics["status"] == "not_applicable"
    assert diagnostics["reason"] == "parameter_variant"
    assert all(
        check.check_id != "L1.reference_geometry.diagnostics" for check in result.checks
    )


def test_l1_typed_requirement_still_gates(tmp_path: Path) -> None:
    candidate = _export(tmp_path, "one_solid", cq.Workplane("XY").box(10, 8, 2))
    case = {
        **_l1_case(),
        "requirements": [
            {
                "requirement_id": "R-001",
                "criticality": "required",
                "value_and_tolerance": {"body_count": 2},
                "check_ids": ["hidden.check.solid_count.v1"],
            }
        ],
    }

    result = evaluate_l1(case, candidate)

    assert result.status is Status.PASSED
    requirement = next(
        check for check in result.checks if check.check_id.startswith("L1.requirement.")
    )
    assert requirement.status is Status.FAILED
    assert requirement.required is False
    assert requirement.failure_code == "solid_count_mismatch"


def test_l1_empty_requirement_ledger_is_not_scoreable(tmp_path: Path) -> None:
    candidate = _export(tmp_path, "candidate", cq.Workplane("XY").box(10, 8, 2))

    result = evaluate_l1(_l1_case(requirements=[]), candidate)

    assert result.status is Status.NOT_SCOREABLE
    contract = next(
        check for check in result.checks if check.check_id == "L1.requirement_contract"
    )
    assert contract.status is Status.NOT_SCOREABLE
    assert result.metrics["requirement_coverage"]["reference_similarity_used"] is False


def test_l1_dfm_still_gates(tmp_path: Path) -> None:
    candidate = _export(
        tmp_path,
        "hole_plate",
        cq.Workplane("XY").box(20, 20, 4).faces(">Z").workplane().hole(4),
    )
    dfm = {
        "families": [
            _scoreable_family(
                MINIMUM_HOLE_GEOMETRY,
                {"minimum_diameter_mm": 5.0, "maximum_depth_to_diameter": 2.0},
            )
        ]
    }

    result = evaluate_l1(_l1_case(dfm=dfm), candidate)

    assert result.status is Status.FAILED
    hole_check = next(
        check
        for check in result.checks
        if check.check_id == "L1.DFM.minimum_hole_geometry"
    )
    assert hole_check.status is Status.FAILED
