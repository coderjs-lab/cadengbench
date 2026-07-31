from pathlib import Path

from cadengbench_p_eval.geometry.features import CylindricalFace
from cadengbench_p_eval.geometry.measurements import parameter_measurements
from cadengbench_p_eval.geometry.step import BoundingBox, StepGeometry


def _geometry() -> StepGeometry:
    return StepGeometry(
        path=Path("synthetic.step"),
        shape=None,  # type: ignore[arg-type]
        solids=(),
        bbox=BoundingBox((0.0, 0.0, 0.0), (10.0, 12.0, 14.0)),
        volume_mm3=100.0,
        surface_area_mm2=120.0,
        centroid_mm=(5.0, 6.0, 7.0),
        topology={"solid_count": 1},
    )


def _hole(face_id: str, diameter: float = 4.0) -> CylindricalFace:
    return CylindricalFace(
        face_id=face_id,
        diameter_mm=diameter,
        depth_mm=8.0,
        depth_to_diameter=8.0 / diameter,
        axis_direction=(0.0, 0.0, 1.0),
        axis_location_mm=(0.0, 0.0, 0.0),
        center_mm=(0.0, 0.0, 4.0),
        internal=True,
        near_full_cylinder=True,
        circular_edge_count=2,
        is_hole=True,
    )


def _case(quantity: int = 1) -> dict[str, object]:
    return {
        "requirements": [
            {
                "requirement_id": "R-HOLE-001",
                "value_and_tolerance": {
                    "quantity": quantity,
                    "diameter": 4.0,
                    "depth": 8.0,
                    "axis_direction": [0.0, 0.0, 1.0],
                    "centers": [],
                },
            }
        ],
        "parameter_system": {
            "parameters": [
                {
                    "parameter_id": "P-001",
                    "measurement_binding": {
                        "measurement_key": "P-001",
                        "measurement_kind": "cylindrical_opening_diameter",
                        "requirement_id": "R-HOLE-001",
                    },
                }
            ]
        },
        "evaluator_profile": {"dfm": {"hole_axis_tolerance_degrees": 1.0}},
    }


def test_centerless_hole_group_is_measured_when_cardinality_is_unambiguous() -> None:
    values = parameter_measurements(_case(), _geometry(), holes=(_hole("f1"),))

    assert values["P-001"] == 4.0


def test_centerless_hole_group_fails_closed_when_cardinality_is_ambiguous() -> None:
    values = parameter_measurements(
        _case(), _geometry(), holes=(_hole("f1"), _hole("f2", diameter=6.0))
    )

    assert "P-001" not in values


def test_orientation_free_hole_group_uses_expected_diameter() -> None:
    case = _case()
    contract = case["requirements"][0]["value_and_tolerance"]  # type: ignore[index]
    contract["axis_assignment"] = "free"  # type: ignore[index]
    values = parameter_measurements(
        case,
        _geometry(),
        holes=(_hole("larger", diameter=6.0), _hole("target", diameter=4.0)),
    )

    assert values["P-001"] == 4.0


def test_unordered_bbox_binding_uses_parameter_matched_requirement_value() -> None:
    case = {
        "requirements": [
            {
                "requirement_id": "R-BBOX",
                "value_and_tolerance": {
                    "x": {"value": 10.0},
                    "y": {"value": 14.0},
                    "z": {"value": 12.0},
                    "axis_assignment": "free_permutation",
                },
            }
        ],
        "parameter_system": {
            "parameters": [
                {
                    "parameter_id": "P-001",
                    "measurement_binding": {
                        "measurement_key": "P-001",
                        "measurement_kind": "bounding_box_unordered_dimension",
                        "requirement_id": "R-BBOX",
                        "requirement_value_path": [
                            "value_and_tolerance",
                            "y",
                            "value",
                        ],
                        "source_axis": "y",
                    },
                }
            ]
        },
        "evaluator_profile": {"dfm": {}},
    }

    values = parameter_measurements(case, _geometry(), holes=())

    assert values["P-001"] == 14.0
