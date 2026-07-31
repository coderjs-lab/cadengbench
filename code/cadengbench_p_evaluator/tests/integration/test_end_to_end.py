from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cadquery as cq
import pytest
from cadquery import exporters
from jsonschema import Draft202012Validator

from cadengbench_p_eval.case import compile_case
from cadengbench_p_eval.contracts import Status
from cadengbench_p_eval.io import write_json
from cadengbench_p_eval.profile import load_profile
from cadengbench_p_eval.runner import evaluate_item

pytestmark = [pytest.mark.integration, pytest.mark.mutation]


def _profile() -> dict[str, Any]:
    return {
        "schema_version": "cadengbench-p.evaluator-profile.v1",
        "profile_id": "cadengbench-p.synthetic-test.v1",
        "execution": {
            "timeout_seconds": 30,
            "worker_startup_grace_seconds": 45,
            "max_source_bytes": 100_000,
            "max_step_bytes": 16 * 1024 * 1024,
            "allowed_import_roots": ["cadquery", "math"],
        },
        "geometry": {
            "linear_tessellation_tolerance_mm": 0.25,
            "surface_sample_count": 128,
            "surface_tolerance_mm": 0.25,
            "voxel_resolution": 8,
            "voxel_margin_fraction": 0.05,
            "seed": 20260721,
        },
        "dfm": {
            "wall_sample_count": 64,
            "wall_neighbor_count": 8,
            "opposed_normal_cosine_min": 0.9,
            "hole_axis_tolerance_degrees": 1.0,
        },
        "l2": {
            "max_variants": 4,
            "directed_points_per_parameter": 2,
            "morris_levels": 4,
            "morris_trajectories": 1,
            "z3_timeout_ms": 1000,
            "seed": 20260721,
        },
    }


def _design_spec(item_id: str) -> dict[str, Any]:
    return {
        "schema_version": "cadengbench-p.zero-to-cad-design-spec.v2",
        "item_id": item_id,
        "task_kind": "zero_to_cad",
        "model_environment_contract": {
            "code_submission_contract": {
                "build_signature": "build(params=None)",
                "interface_version": "cadengbench-p.cadquery-submission.v1",
                "required_top_level_symbols": [
                    "DEFAULT_PARAMS",
                    "PARAMETER_MANIFEST",
                    "build",
                    "result",
                ],
                "step_output_path": "model.step",
            }
        },
        "canonical_public_spec": {
            "fidelity_profile": "requirements_only",
            "solid_contract": {
                "connectivity": "one_connected_solid",
                "required_body_count": 1,
            },
            "reference_frame": {"units": "mm"},
            "parametric_submission_contract": {
                "interface_version": "cadengbench-p.cadquery-submission.v1"
            },
            "public_parameters": [
                {
                    "parameter_id": "P-001",
                    "engineering_name": "overall_length",
                    "nominal": 10.0,
                    "unit": "mm",
                    "parameter_role": "principal_dimension",
                    "public_relation_refs": ["R-002"],
                    "variant_requirement_contract": {
                        "kind": "direct_measurement_only",
                        "authority": "synthetic_test_oracle",
                        "evidence_ref": "synthetic-domain-v1",
                    },
                    "qualified_domain": {
                        "minimum": 8.0,
                        "maximum": 12.0,
                        "authority": "synthetic_test_oracle",
                        "evidence_ref": "synthetic-domain-v1",
                    },
                }
            ],
            "public_constraint_relations": [],
            "solver_obligations": [
                {
                    "obligation_id": "length_domain_upper_bound",
                    "assumptions": [],
                    "conclusion": {
                        "type": "comparison",
                        "op": "le",
                        "left": {
                            "type": "variable",
                            "source": "parameter",
                            "key": "P-001",
                        },
                        "right": {"type": "constant", "value": 12.0},
                    },
                    "authority": "synthetic_test_oracle",
                    "evidence_ref": "synthetic-domain-v1",
                    "required": True,
                }
            ],
        },
        "public_requirement_ledger": [
            {
                "requirement_id": "R-001",
                "category": "topology",
                "criticality": "required",
                "value_and_tolerance": {"body_count": 1},
            },
            {
                "requirement_id": "R-002",
                "category": "dimension",
                "criticality": "required",
                "value_and_tolerance": {
                    "x": {"value": 10.0, "tolerance": 0.01, "unit": "mm"},
                    "y": {"value": 4.0, "tolerance": 0.01, "unit": "mm"},
                    "z": {"value": 2.0, "tolerance": 0.01, "unit": "mm"},
                },
            },
        ],
        "hidden_acceptance": {
            "reference_geometry_refs": ["source/generated_from_cadquery.step"],
            "reference_history_refs": [],
            "requirement_to_check_refs": {
                "R-001": ["hidden.check.solid_count.v1"],
                "R-002": ["hidden.check.bbox_dimensions.v1"],
            },
        },
    }


def _submission(*, width: float = 4.0) -> str:
    return f"""\
import cadquery as cq
from cadquery import exporters

DEFAULT_PARAMS = {{
    "overall_length": 10.0,
}}

PARAMETER_MANIFEST = {{
    "P-001": {{
        "name": "overall_length",
        "unit": "mm",
        "role": "principal_dimension",
    }},
}}

def build(params=None):
    values = dict(DEFAULT_PARAMS)
    if params:
        values.update(params)
    return cq.Workplane("XY").box(values["overall_length"], {width}, 2.0)

result = build()
exporters.export(result, "model.step")
"""


def _synthetic_case(tmp_path: Path) -> Path:
    item_root = tmp_path / "p_benchcad_synthetic"
    design_path = item_root / "zero_to_cad/compiled/zero_to_cad_design_spec.json"
    source_root = item_root / "source"
    source_root.mkdir(parents=True)
    exporters.export(
        cq.Workplane("XY").box(10.0, 4.0, 2.0),
        str(source_root / "generated_from_cadquery.step"),
    )
    write_json(design_path, _design_spec(item_root.name))
    profile_path = tmp_path / "profile.json"
    write_json(profile_path, _profile())
    case_path = tmp_path / "case.json"
    compile_case(
        item_root,
        profile=load_profile(profile_path),
        output_path=case_path,
    )
    return case_path


def test_end_to_end_positive_mutant_and_resume(tmp_path: Path) -> None:
    case_path = _synthetic_case(tmp_path)
    submission = tmp_path / "submission.py"
    submission.write_text(_submission(), encoding="utf-8")
    output = tmp_path / "passing-result"

    passed = evaluate_item(
        case_path=case_path,
        submission_path=submission,
        output_root=output,
    )

    assert passed.overall_status is Status.PASSED
    assert [layer.status for layer in passed.layers] == [
        Status.PASSED,
        Status.PASSED,
        Status.PASSED,
    ]
    l2 = passed.layers[2]
    assert l2.metrics["variant_count"] == 3
    schedule_metadata = l2.artifacts["replay"]["schedule"]["metadata"]
    assert schedule_metadata["input_variant_count"] == 4
    assert schedule_metadata["deduplicated_variant_count"] == 3
    assert schedule_metadata["duplicate_assignment_count"] == 1
    result_path = output / "result.json"
    first_mtime = result_path.stat().st_mtime_ns
    resumed = evaluate_item(
        case_path=case_path,
        submission_path=submission,
        output_root=output,
    )
    assert resumed.as_dict() == passed.as_dict()
    assert result_path.stat().st_mtime_ns == first_mtime

    schema_path = (
        Path(__file__).resolve().parents[2] / "schemas/evaluation_result.v1.schema.json"
    )
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
        passed.as_dict()
    )

    mutant = tmp_path / "mutant.py"
    mutant.write_text(_submission(width=5.0), encoding="utf-8")
    failed = evaluate_item(
        case_path=case_path,
        submission_path=mutant,
        output_root=tmp_path / "failing-result",
    )

    assert failed.overall_status is Status.FAILED
    assert failed.layers[0].status is Status.PASSED
    assert failed.layers[1].status is Status.FAILED
    assert failed.layers[2].status is Status.FAILED
    nominal_l1 = next(
        check
        for check in failed.layers[2].checks
        if check.check_id == "L2.context.nominal_l1"
    )
    assert nominal_l1.status is Status.FAILED
    assert nominal_l1.required is False
