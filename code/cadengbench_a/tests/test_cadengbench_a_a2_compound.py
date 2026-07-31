from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from pipeline.cadengbench_a_a2_compound import write_compound_fixture_urdf


@pytest.mark.parametrize(
    ("joint_type", "directions", "expected_types"),
    [
        (
            "CylindricalJointType",
            {"rotation_axis": [0, 0, 1], "slide_direction": [0, 0, 1]},
            ["revolute", "prismatic"],
        ),
        (
            "PlanarJointType",
            {
                "normal_direction": [0, 0, 1],
                "primary_slide_direction": [1, 0, 0],
                "secondary_slide_direction": [0, 1, 0],
            },
            ["prismatic", "prismatic", "revolute"],
        ),
        (
            "PinSlotJointType",
            {"rotation_axis": [0, 0, 1], "slide_direction": [1, 0, 0]},
            ["prismatic", "revolute"],
        ),
    ],
)
def test_compound_fixture_emits_expected_chain(tmp_path, joint_type, directions, expected_types) -> None:
    output = tmp_path / "fixture.urdf"
    metadata = write_compound_fixture_urdf(joint_type, directions, output)
    root = ET.parse(output).getroot()

    assert [joint.attrib["type"] for joint in root.findall("joint")] == expected_types
    assert len(metadata["joint_specs"]) == len(expected_types)


def test_pin_slot_rejects_nonorthogonal_axes(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be orthogonal"):
        write_compound_fixture_urdf(
            "PinSlotJointType",
            {"rotation_axis": [0, 0, 1], "slide_direction": [0, 0, 1]},
            tmp_path / "fixture.urdf",
        )
