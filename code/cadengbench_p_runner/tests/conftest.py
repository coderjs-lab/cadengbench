from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def synthetic_items(tmp_path: Path) -> Path:
    root = tmp_path / "items"
    item = root / "p_benchcad_000001"
    compiled = item / "zero_to_cad/compiled"
    compiled.mkdir(parents=True)
    write_json(
        item / "item.json",
        {"dataset_version": "cadengbench-p-test-v1", "item_id": item.name},
    )
    source = item / "source/cadquery.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)\n",
        encoding="utf-8",
    )
    write_json(
        item / "benchmark/consumer_ready_item_v2.json",
        {
            "schema_version": "cadengbench-p.consumer-ready-item.v2",
            "item_id": item.name,
            "source": {
                "source_key": "benchcad",
                "native_history_artifact": "source/cadquery.py",
            },
            "model_facing": {
                "functional_edit": {
                    "user_prompt": "Increase the block height to carry the stated load."
                }
            },
        },
    )
    write_json(
        item / "completion/v2/parametric/edit_target_oracle.json",
        {
            "schema_version": "cadengbench-p.edit-target-oracle.v2",
            "item_id": item.name,
            "source_key": "benchcad",
            "status": "verified",
            "llm_judge_used": False,
            "parameter_contract": {
                "parameter_key": "height",
                "base_value": 10.0,
                "target_value": 12.0,
                "canonical_unit": "mm",
            },
        },
    )
    write_json(
        item / "evaluation/qualification.v1.json",
        {
            "schema_version": "cadengbench-p.evaluator-qualification.v1",
            "item_id": item.name,
            "requirements": [
                {
                    "requirement_id": "R-SOLID",
                    "category": "solid_contract",
                    "value_and_tolerance": {"body_count": 1},
                },
                {
                    "requirement_id": "R-BBOX",
                    "category": "global_envelope",
                    "value_and_tolerance": {
                        axis: {"value": 10.0, "tolerance": 0.05}
                        for axis in ("x", "y", "z")
                    },
                },
            ],
            "parameter_system": {
                "parameters": [
                    {
                        "parameter_id": "P-001",
                        "engineering_name": "height",
                        "nominal": 10.0,
                        "unit": "mm",
                        "role": "principal_dimension",
                    }
                ]
            },
            "dfm": {
                "families": [
                    {
                        "family": "minimum_wall_thickness",
                        "scoreable": True,
                        "thresholds": {"minimum_mm": 1.0},
                    },
                    {
                        "family": "minimum_hole_geometry",
                        "scoreable": False,
                        "thresholds": {
                            "minimum_diameter_mm": 2.0,
                            "maximum_depth_to_diameter": 8.0,
                        },
                    },
                ]
            },
        },
    )
    write_json(
        compiled / "zero_to_cad_design_spec.json",
        {
            "schema_version": "cadengbench-p.zero-to-cad-design-spec.v2",
            "item_id": item.name,
            "task_kind": "zero_to_cad",
        },
    )
    (compiled / "model.system.txt").write_text(
        "Return Python only.\n", encoding="utf-8"
    )
    for name in (
        "zero_to_cad.text.v2.user.txt",
        "zero_to_cad.text_iso.v2.user.txt",
        "zero_to_cad.text_engineering_views.v2.user.txt",
    ):
        (compiled / name).write_text("Build one parametric block.\n", encoding="utf-8")
    for relative in (
        "visuals/v2/core/iso_front_right_top.shaded.png",
        "visuals/v2/core/iso_rear_left_top.shaded.png",
        "visuals/v2/engineering_4view_visible_hidden.png",
    ):
        path = item / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_1X1)
    return root


@pytest.fixture
def synthetic_fusion_items(tmp_path: Path) -> Path:
    root = tmp_path / "fusion-items"
    item = root / "p_fusion360_000001"
    write_json(
        item / "item.json",
        {"dataset_version": "cadengbench-p-test-v1", "item_id": item.name},
    )
    write_json(
        item / "source/fusion.json",
        {
            "metadata": {"component_name": "mounting plate"},
            "timeline": [
                {"index": 1, "entity": "sketch-1"},
                {"index": 2, "entity": "extrude-1"},
            ],
            "entities": {
                "sketch-1": {
                    "name": "Mounting holes",
                    "type": "Sketch",
                    "profiles": {
                        "profile-1": {
                            "loops": [
                                {
                                    "is_outer": True,
                                    "profile_curves": [
                                        {
                                            "type": "Circle3D",
                                            "center_point": {
                                                "x": 5.0,
                                                "y": 5.0,
                                                "z": 0.0,
                                            },
                                            "normal": {
                                                "x": 0.0,
                                                "y": 0.0,
                                                "z": 1.0,
                                            },
                                            "radius": 0.3,
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                },
                "extrude-1": {
                    "name": "Base Extrude",
                    "type": "ExtrudeFeature",
                    "operation": "NewBodyFeatureOperation",
                    "extent_one": {"distance": {"value": 1.0}},
                },
            },
        },
    )
    write_json(
        item / "benchmark/consumer_ready_item_v2.json",
        {
            "schema_version": "cadengbench-p.consumer-ready-item.v2",
            "item_id": item.name,
            "source": {
                "source_key": "fusion360",
                "native_history_artifact": "source/fusion.json",
            },
            "model_facing": {
                "functional_edit": {
                    "user_prompt": (
                        "Increase the base extrusion to improve interface stiffness."
                    )
                }
            },
        },
    )
    write_json(
        item / "completion/v2/parametric/edit_target_oracle.json",
        {
            "schema_version": "cadengbench-p.edit-target-oracle.v2",
            "item_id": item.name,
            "source_key": "fusion360",
            "status": "verified",
            "llm_judge_used": False,
            "parameter_contract": {
                "parameter_key": "extrude_distance",
                "base_value": 10.0,
                "target_value": 12.0,
                "canonical_unit": "mm",
                "source_unit": "cm",
                "scale_to_canonical": 10.0,
                "source_pointer": (
                    "/entities/extrude-1/extent_one/distance/value"
                ),
            },
        },
    )
    write_json(
        item / "evaluation/qualification.v1.json",
        {
            "schema_version": "cadengbench-p.evaluator-qualification.v1",
            "item_id": item.name,
            "requirements": [
                {
                    "requirement_id": "R-SOLID",
                    "category": "solid_contract",
                    "value_and_tolerance": {"body_count": 1},
                }
            ],
            "parameter_system": {
                "parameters": [
                    {
                        "parameter_id": "P-001",
                        "engineering_name": "height",
                        "nominal": 10.0,
                        "unit": "mm",
                        "role": "principal_dimension",
                    }
                ]
            },
            "dfm": {
                "families": [
                    {
                        "family": "minimum_wall_thickness",
                        "scoreable": True,
                        "thresholds": {"minimum_mm": 1.0},
                    }
                ]
            },
        },
    )
    for relative in (
        "visuals/v2/core/iso_front_right_top.shaded.png",
        "visuals/v2/core/iso_rear_left_top.shaded.png",
        "visuals/v2/engineering_4view_visible_hidden.png",
    ):
        path = item / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_1X1)
    return root


def model_spec(base_url: str, *, requests_per_minute: int = 60000) -> dict[str, object]:
    return {
        "schema_version": "cadengbench-p.model-spec.v1",
        "model_label": "mock-model",
        "provider": "openai_compatible",
        "protocol": "openai_chat",
        "model_id": "mock-cad-model-v1",
        "base_url": base_url,
        "api_key_env": None,
        "api_key_required": False,
        "capabilities": {"text": True, "images": True},
        "request_defaults": {"temperature": 0.0, "max_tokens": 1024},
        "rate_limits": {
            "max_concurrency": 2,
            "requests_per_minute": requests_per_minute,
        },
    }


@pytest.fixture
def model_spec_writer():
    def write(path: Path, base_url: str) -> Path:
        write_json(path, model_spec(base_url))
        return path

    return write
