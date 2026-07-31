from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.io import write_json
from cadengbench_p_eval.layers.l3 import l3_layer_from_family_result
from cadengbench_p_eval.physics.annotation import (
    L3AnnotationError,
    _guidance_schema,
    _semantic_validation,
    _validate_candidate,
)
from cadengbench_p_eval.physics.calculix import (
    MeshData,
    PhysicsCardError,
    _resolve_selector,
    validate_physics_card,
    write_calculix_deck,
)
from cadengbench_p_eval.physics.runner import (
    _qualify_corpus_item,
    _run_isolated_worker,
    build_family_manifest_from_evaluator_run,
    qualify_annotation_corpus,
)
from cadengbench_p_eval.provenance import (
    evaluator_build_hash,
    physics_evaluator_build_hash,
)


def _bundle() -> dict:
    regions = []
    for side in ("min", "max"):
        regions.append(
            {
                "region_id": f"axis:x:{side}",
                "axis": "x",
                "side": side,
            }
        )
    return {
        "inputs": {
            "geometry": {
                "deterministic_eligibility": {"eligible": True},
                "eligible_opposed_region_pairs": [
                    {
                        "minimum_region": regions[0],
                        "maximum_region": regions[1],
                    }
                ],
            }
        }
    }


def test_physics_build_identity_is_scoped_from_general_evaluator() -> None:
    physics_hash = physics_evaluator_build_hash()

    assert len(physics_hash) == 64
    assert physics_hash != evaluator_build_hash()


def _candidate() -> dict:
    return {
        "output_status": "complete",
        "applicability": "proposed",
        "scenario_family": "cantilever_transverse",
        "fixed_region_id": "axis:x:min",
        "loaded_region_id": "axis:x:max",
        "load_direction": {"axis": "z", "sign": -1},
    }


def _card() -> dict:
    selector = {
        "kind": "axis_extreme_planar_surfaces_v2",
        "region_id": "axis:x:min",
        "axis": "x",
        "side": "min",
        "normal_alignment_min": 0.98,
        "plane_tolerance_relative": 1e-6,
        "minimum_count": 1,
        "maximum_count": 64,
        "minimum_area_fraction": 0.002,
    }
    loaded = {**selector, "region_id": "axis:x:max", "side": "max"}
    return {
        "schema_version": "cadengbench-p.l3-physics-card.v2",
        "card_id": "p_benchcad_000001:probe:v1",
        "item_id": "p_benchcad_000001",
        "status": "geometry_qualified",
        "source_intent_claimed": False,
        "scenario_family": "cantilever_transverse",
        "geometry": {"units": "mm", "expected_solid_count": 1},
        "selectors": {"fixed": selector, "loaded": loaded},
        "material": {
            "profile_id": "probe",
            "model": "linear_elastic_isotropic",
            "youngs_modulus_mpa": 70000.0,
            "poisson_ratio": 0.33,
        },
        "load": {
            "kind": "surface_traction_probe",
            "traction_vector_mpa": [0.0, 0.0, -0.1],
            "distribution": "boundary_triangle_consistent_nodal_v1",
        },
        "mesh": {
            "element_type": "C3D10",
            "algorithm_3d": 10,
            "high_order_optimization": "none",
            "target_long_axis_divisions": 24,
            "target_short_axis_divisions": 5,
            "minimum_sicn": 0.02,
            "maximum_elements": 250000,
        },
        "analysis": {
            "type": "linear_static",
            "small_deformation": True,
            "max_equilibrium_relative_error": 1e-5,
            "max_normalized_displacement": 0.02,
            "stress_exclusion_mesh_lengths": 2.0,
            "regularized_stress_percentiles": [95.0, 99.0],
        },
        "checks": [
            "mesh_quality",
            "selector_stability",
            "solver_completion",
            "force_equilibrium",
            "finite_response",
            "small_deformation",
            "regularized_von_mises",
            "normalized_compliance",
        ],
        "provenance": {
            "annotation_candidate_sha256": "a" * 64,
            "context_sha256": "b" * 64,
            "profile_sha256": "c" * 64,
            "annotator_model": "model",
            "annotator_revision": "revision",
        },
    }


def test_candidate_must_use_opposed_available_regions() -> None:
    _validate_candidate(_candidate(), _bundle())
    invalid = _candidate()
    invalid["loaded_region_id"] = "axis:y:max"
    with pytest.raises(L3AnnotationError, match="unavailable region"):
        _validate_candidate(invalid, _bundle())


def test_axial_probe_must_point_outward() -> None:
    candidate = _candidate()
    candidate["scenario_family"] = "axial_tension"
    candidate["load_direction"] = {"axis": "x", "sign": -1}
    with pytest.raises(L3AnnotationError, match="points into"):
        _validate_candidate(candidate, _bundle())


def test_invalid_model_proposal_is_a_recorded_semantic_rejection() -> None:
    candidate = _candidate()
    candidate["scenario_family"] = "axial_tension"
    candidate["load_direction"] = {"axis": "x", "sign": -1}

    assert _semantic_validation(candidate, _bundle()) == {
        "status": "rejected",
        "reason": "axial tension points into the solid",
    }


def test_card_rejects_longitudinal_cantilever_load() -> None:
    card = _card()
    validate_physics_card(card)
    card["load"]["traction_vector_mpa"] = [0.1, 0.0, 0.0]
    with pytest.raises(PhysicsCardError, match="transverse"):
        validate_physics_card(card)


def test_selector_resolves_tilted_coplanar_extreme_regions() -> None:
    class FakeOcc:
        def __init__(self) -> None:
            self.areas = {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
            self.centers = {
                1: (0.5, 5.0, 10.0),
                2: (1.5, 15.0, 10.0),
                3: (20.5, 5.0, 10.0),
                4: (80.5, 5.0, 10.0),
            }

        def getMass(self, dimension: int, tag: int) -> float:
            assert dimension == 2
            return self.areas[tag]

        def getCenterOfMass(
            self, dimension: int, tag: int
        ) -> tuple[float, float, float]:
            assert dimension == 2
            return self.centers[tag]

    class FakeModel:
        def __init__(self) -> None:
            self.occ = FakeOcc()

        def getEntities(self, dimension: int) -> list[tuple[int, int]]:
            assert dimension == 2
            return [(2, tag) for tag in range(1, 5)]

        def getType(self, dimension: int, tag: int) -> str:
            assert dimension == 2
            return "Plane"

        def getBoundingBox(
            self, dimension: int, tag: int
        ) -> tuple[float, ...]:
            assert dimension == 2
            center = self.occ.centers[tag]
            return (
                center[0] - 0.5,
                center[1] - 5.0,
                0.0,
                center[0] + 0.5,
                center[1] + 5.0,
                20.0,
            )

        def getParametrizationBounds(
            self, dimension: int, tag: int
        ) -> tuple[list[float], list[float]]:
            assert dimension == 2
            return [0.0, 0.0], [1.0, 1.0]

        def getNormal(self, tag: int, parameters: list[float]) -> list[float]:
            assert parameters == [0.5, 0.5]
            return [1.0, -0.1, 0.0]

    class FakeGmsh:
        def __init__(self) -> None:
            self.model = FakeModel()

    selector = _card()["selectors"]["fixed"]
    selected, evidence = _resolve_selector(
        FakeGmsh(), selector, (0.0, 0.0, 0.0, 100.0, 20.0, 20.0)
    )

    assert selected == [1, 2]
    assert all(
        row["selection_method"]
        == "extreme_axis_intercept_of_aligned_planar_surfaces_v1"
        for row in evidence
    )


def test_card_identifier_cannot_inject_deck_directives() -> None:
    card = _card()
    card["card_id"] = "valid-id\n*INCLUDE, INPUT=untrusted.inp"
    with pytest.raises(PhysicsCardError, match="invalid physics card"):
        validate_physics_card(card)


def test_solver_qualified_card_requires_embedded_evidence() -> None:
    card = _card()
    card["status"] = "solver_qualified"
    with pytest.raises(PhysicsCardError, match="missing qualification"):
        validate_physics_card(card)


def test_guidance_schema_removes_only_unsupported_uniqueness() -> None:
    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string"},
            }
        },
    }
    projected, removed = _guidance_schema(schema)
    assert "uniqueItems" not in projected["properties"]["values"]
    assert removed == ["#/properties/values/uniqueItems"]
    assert schema["properties"]["values"]["uniqueItems"] is True


def test_deck_uses_compiled_nodal_forces(tmp_path: Path) -> None:
    mesh = MeshData(
        nodes=(
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 0.0, 1.0, 0.0),
            (4, 0.0, 0.0, 1.0),
        ),
        elements=((1, 1, 2, 3, 4),),
        element_type="C3D4",
        fixed_nodes=(1, 3, 4),
        loaded_nodes=(2,),
        nodal_forces={2: (0.0, 0.0, -2.5)},
        applied_force=(0.0, 0.0, -2.5),
        report={},
    )
    card = _card()
    path = tmp_path / "analysis.inp"
    write_calculix_deck(card, mesh, path)
    text = path.read_text(encoding="ascii")
    assert "*ELEMENT, TYPE=C3D4, ELSET=EALL" in text
    assert "2, 3, -2.5" in text
    assert "*NODE PRINT, NSET=FIXED, TOTALS=YES" in text


def test_l3_layer_preserves_evaluator_error_ownership() -> None:
    layer = l3_layer_from_family_result(
        {
            "schema_version": "cadengbench-p.l3-family-result.v1",
            "variant_count": 1,
            "l3_reached_count": 1,
            "probe_pass_count": 0,
            "evaluator_error_count": 1,
            "variants": [
                {
                    "variant_id": "nominal",
                    "final_status": "failed",
                    "failure_stage": "internal",
                    "failure_owner": "evaluator",
                }
            ],
        }
    )
    assert layer.status is Status.EVALUATOR_ERROR
    assert layer.checks[0].status is Status.EVALUATOR_ERROR


def test_corpus_resume_replaces_only_incomplete_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = tmp_path / "card.json"
    step = tmp_path / "source.step"
    ccx = tmp_path / "ccx"
    for path in (card, step, ccx):
        path.write_text("placeholder", encoding="ascii")
    output = tmp_path / "source_qualification"
    output.mkdir()
    (output / "partial.mesh").write_text("partial", encoding="ascii")
    observed: dict[str, bool] = {}

    def fake_qualify_physics_card(**kwargs):
        observed["force"] = bool(kwargs["force"])
        qualification_dir = kwargs["output_dir"]
        qualification_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            qualification_dir / "result.json",
            {
                "final_status": "failed",
                "failure": {"stage": "mesh", "owner": "model"},
            },
        )
        return {"promoted": False, "result_status": "failed"}

    monkeypatch.setattr(
        "cadengbench_p_eval.physics.runner.qualify_physics_card",
        fake_qualify_physics_card,
    )
    result = _qualify_corpus_item(
        item_id="p_benchcad_000001",
        card_path=card,
        reference_step=step,
        qualification_dir=output,
        promoted_card_path=tmp_path / "promoted.json",
        ccx_path=ccx,
        timeout_seconds=1,
        render=False,
        force=False,
    )
    assert observed["force"] is True
    assert result["recovered_incomplete"] is True
    assert result["failure_stage"] == "mesh"


def test_family_manifest_reads_real_l2_replay_record_shape(tmp_path: Path) -> None:
    card = _card()
    card["status"] = "solver_qualified"
    card["qualification"] = {
        "schema_version": "cadengbench-p.l3-card-qualification.v1",
        "item_id": card["item_id"],
        "card_sha256": "d" * 64,
        "reference_step_sha256": "e" * 64,
        "result_sha256": "f" * 64,
        "evaluator_build_hash": "1" * 64,
        "runtime_lock_sha256": "2" * 64,
        "result_status": "probe_pass",
        "promoted": True,
        "source_probe_metrics": {
            "normalized_max_displacement": 0.001,
            "regularized_von_mises_p95_mpa": 0.1,
            "regularized_von_mises_p99_mpa": 0.2,
            "stress_concentration_p95": 1.1,
            "normalized_compliance": 0.01,
            "force_equilibrium_relative_error": 1e-8,
        },
    }
    card_path = tmp_path / "card.json"
    write_json(card_path, card)

    evaluator_item = tmp_path / "evaluator-item"
    write_json(evaluator_item / "l0.json", {"status": "passed"})
    write_json(evaluator_item / "l1.json", {"status": "passed"})
    nominal = evaluator_item / "l0/candidate.step"
    nominal.parent.mkdir(parents=True, exist_ok=True)
    nominal.write_text("nominal", encoding="ascii")
    variant_id = "directed-0000-P-001"
    variant_step = evaluator_item / "l2/variants" / variant_id / "candidate.step"
    variant_step.parent.mkdir(parents=True, exist_ok=True)
    variant_step.write_text("variant", encoding="ascii")
    write_json(
        evaluator_item / "l2.json",
        {
            "artifacts": {
                "replay": {
                    "variants": [
                        {
                            "status": "passed",
                            "variant": {"variant_id": variant_id},
                        }
                    ]
                }
            }
        },
    )

    manifest = build_family_manifest_from_evaluator_run(
        card_path=card_path,
        evaluator_item_root=evaluator_item,
        output_path=tmp_path / "family.json",
    )

    assert [value["variant_id"] for value in manifest["variants"]] == [
        "nominal",
        variant_id,
    ]
    assert manifest["variants"][1]["upstream_status"] == "passed"
    assert manifest["variants"][1]["step_sha256"] is not None


def test_corpus_qualification_requires_current_validation_binding(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "sidecar"
    output_root.mkdir()
    item_ids = [f"p_benchcad_{index:06d}" for index in range(300)]
    (output_root / "requests.jsonl").write_text(
        "".join(f'{{"item_id":"{item_id}"}}\n' for item_id in item_ids),
        encoding="ascii",
    )
    stale_root = output_root / "items" / item_ids[0]
    write_json(stale_root / "physics_card.json", {"stale": True})
    write_json(
        stale_root / "annotation_validation.json",
        {
            "disposition": "rejected",
            "reason": "annotator_abstained",
            "card_sha256": None,
        },
    )
    mismatch_root = output_root / "items" / item_ids[1]
    write_json(mismatch_root / "physics_card.json", {"current": False})
    write_json(
        mismatch_root / "annotation_validation.json",
        {
            "disposition": "geometry_qualified",
            "reason": "candidate_matches_deterministic_region_contract",
            "card_sha256": "0" * 64,
        },
    )
    profile_path = tmp_path / "profile.json"
    write_json(
        profile_path,
        {
            "schema_version": "cadengbench-p.l3-probe-profile.v1",
            "dataset": {"items_root": "dataset/items"},
        },
    )

    summary = qualify_annotation_corpus(
        repo_root=tmp_path,
        profile_path=profile_path,
        output_root=output_root,
        ccx_path=tmp_path / "ccx",
    )
    records = {value["item_id"]: value for value in summary["records"]}

    assert records[item_ids[0]]["status"] == "not_compiled"
    assert records[item_ids[0]]["stale_card_ignored"] is True
    assert records[item_ids[1]]["status"] == "evaluator_error"
    assert records[item_ids[1]]["failure_stage"] == "qualification_input_binding"


def test_native_worker_crash_is_bounded_to_one_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=-11, stdout="", stderr="native crash"
        )

    monkeypatch.setattr("cadengbench_p_eval.physics.runner.subprocess.run", fake_run)
    result, report = _run_isolated_worker(
        specification={"operation": "source_qualification"},
        runtime_root=tmp_path,
        key="p_benchcad_000001",
        timeout_seconds=1,
    )

    assert result is None
    assert report["status"] == "failed"
    assert report["return_code"] == -11
    assert report["timed_out"] is False
    assert len(list(tmp_path.glob("p_benchcad_000001/*/worker_report.json"))) == 1
