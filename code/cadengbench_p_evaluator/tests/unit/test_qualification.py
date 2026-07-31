import json
from pathlib import Path

from cadengbench_p_eval.qualification import (
    ENGINEERING_ABSOLUTE_TOLERANCE_MM,
    ENGINEERING_RELATIVE_TOLERANCE,
    _requirements,
    _source_neutral_binding,
)


def _write_binding(item_root: Path, selected: dict) -> None:
    source = item_root / "source"
    source.mkdir(parents=True)
    (source / "edit_binding_qualification.json").write_text(
        json.dumps({"selected": selected}), encoding="utf-8"
    )


def _geometry(*, x: float, y: float, z: float, holes: list[dict]) -> dict:
    return {
        "body_count": 1,
        "bbox": {
            "xmin": 0.0,
            "xmax": x,
            "ymin": 0.0,
            "ymax": y,
            "zmin": 0.0,
            "zmax": z,
            "xlen": x,
            "ylen": y,
            "zlen": z,
        },
        "hole_inventory": {"hole_count": len(holes), "holes": holes},
    }


def test_source_neutral_binding_resolves_unique_hole(tmp_path: Path) -> None:
    _write_binding(
        tmp_path,
        {
            "binding": {"call": "hole", "argument_index": 0},
            "probe_reports": [],
        },
    )
    geometry = _geometry(
        x=20.0,
        y=10.0,
        z=5.0,
        holes=[
            {
                "diameter": 4.0,
                "depth": 5.0,
                "axis_direction": [0.0, 0.0, 1.0],
            }
        ],
    )

    name, refs, binding = _source_neutral_binding(
        item_root=tmp_path,
        evaluator_inputs={
            "functional_edit": {"effect_proxy": {"metric_kind": "hole_diameter"}}
        },
        geometry=geometry,
        requirements=_requirements(geometry),
        nominal=4.0,
        fallback_name="hole_arg_0",
    )

    assert name == "opening_group_001_diameter"
    assert refs == ["R-HOLE-001"]
    assert binding is not None
    assert binding["measurement_kind"] == "cylindrical_opening_diameter"


def test_engineering_tolerances_have_bounded_absolute_and_relative_floors() -> None:
    requirements = _requirements(
        _geometry(
            x=5.0,
            y=50.0,
            z=500.0,
            holes=[
                {
                    "diameter": 4.0,
                    "depth": 20.0,
                    "axis_direction": [0.0, 0.0, 1.0],
                }
            ],
        )
    )
    envelope = requirements[1]["value_and_tolerance"]
    assert envelope["x"]["tolerance"] == ENGINEERING_ABSOLUTE_TOLERANCE_MM
    assert envelope["y"]["tolerance"] == 0.5
    assert envelope["z"]["tolerance"] == 5.0
    opening = requirements[2]["value_and_tolerance"]
    assert opening["diameter_tolerance"] == ENGINEERING_ABSOLUTE_TOLERANCE_MM
    assert opening["depth_tolerance"] == 0.2
    assert ENGINEERING_RELATIVE_TOLERANCE == 0.01


def test_source_neutral_binding_requires_two_probe_bbox_agreements(
    tmp_path: Path,
) -> None:
    _write_binding(
        tmp_path,
        {
            "binding": {"edit_family": "extrusion_distance"},
            "probe_reports": [
                {
                    "canonical_value_mm": 8.0,
                    "step_roundtrip": {
                        "geometry": {
                            "bbox": {
                                "xmin": 0.0,
                                "xmax": 20.0,
                                "ymin": 0.0,
                                "ymax": 10.0,
                                "zmin": 0.0,
                                "zmax": 8.0,
                            }
                        }
                    },
                },
                {
                    "canonical_value_mm": 12.0,
                    "step_roundtrip": {
                        "geometry": {
                            "bbox": {
                                "xmin": 0.0,
                                "xmax": 20.0,
                                "ymin": 0.0,
                                "ymax": 10.0,
                                "zmin": 0.0,
                                "zmax": 12.0,
                            }
                        }
                    },
                },
            ],
        },
    )
    geometry = _geometry(x=20.0, y=10.0, z=10.0, holes=[])

    name, refs, binding = _source_neutral_binding(
        item_root=tmp_path,
        evaluator_inputs={
            "functional_edit": {
                "effect_proxy": {"metric_kind": "target_parameter_value"}
            }
        },
        geometry=geometry,
        requirements=_requirements(geometry),
        nominal=10.0,
        fallback_name="extrude_distance",
    )

    assert name == "overall_height"
    assert refs == ["R-BBOX"]
    assert binding is not None
    assert binding["axis"] == "z"
