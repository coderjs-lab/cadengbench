from types import SimpleNamespace

import cadengbench_p_eval.dfm.wall as wall_module
from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.dfm.contracts import (
    MINIMUM_HOLE_GEOMETRY,
    resolve_dfm_gate,
)
from cadengbench_p_eval.dfm.holes import evaluate_minimum_hole_geometry
from cadengbench_p_eval.geometry.features import CylindricalFeatureInventory


def _applicable_hole_gate(*, expected_count: int | None):
    family = {
        "family": MINIMUM_HOLE_GEOMETRY,
        "applicability": "applicable",
        "required": True,
        "thresholds": {
            "minimum_diameter_mm": 3.0,
            "maximum_depth_to_diameter": 2.0,
        },
        "threshold_evidence_ref": "benchmark:test-profile:v1",
        "extractor_certified": True,
        "scoreable": True,
    }
    if expected_count is not None:
        family["expected_minimum_feature_count"] = expected_count
    return resolve_dfm_gate({"families": [family]}, MINIMUM_HOLE_GEOMETRY)


def test_applicable_hole_gate_fails_when_required_holes_are_missing() -> None:
    result = evaluate_minimum_hole_geometry(
        _applicable_hole_gate(expected_count=2),
        CylindricalFeatureInventory(assessments=()),
    )

    assert result.status is Status.FAILED
    assert result.failure_code == "required_hole_features_missing"
    assert result.expected["minimum_required_hole_count"] == 2
    assert result.observed["hole_count"] == 0


def test_applicable_hole_gate_without_expected_count_is_not_scoreable() -> None:
    result = evaluate_minimum_hole_geometry(
        _applicable_hole_gate(expected_count=None),
        CylindricalFeatureInventory(assessments=()),
    )

    assert result.status is Status.NOT_SCOREABLE
    assert result.status is not Status.NOT_APPLICABLE


def test_wall_estimate_selects_thinner_sampled_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        wall_module,
        "_planar_candidates",
        lambda _geometry: [{"value_mm": 5.0, "method": "analytic_test_candidate"}],
    )
    monkeypatch.setattr(wall_module, "_cylindrical_candidates", lambda _geometry: [])
    monkeypatch.setattr(
        wall_module,
        "_sampled_candidate",
        lambda _geometry, **_kwargs: {
            "value_mm": 2.0,
            "method": "sampled_test_candidate",
        },
    )
    geometry = SimpleNamespace(bbox=SimpleNamespace(lengths=(10.0, 8.0, 6.0)))

    result = wall_module.estimate_wall_thickness(
        geometry,
        sample_count=64,
        neighbor_count=16,
        opposed_normal_cosine_min=0.9,
        tessellation_tolerance_mm=0.2,
        seed=7,
    )

    assert result["minimum_wall_thickness_mm"] == 2.0
    assert result["selected_method"] == "sampled_test_candidate"
    assert result["qualified_candidate_count"] == 2
