from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from pipeline.cadengbench_a_a2_pybullet import write_direct_fixture_urdf, write_selected_cad_urdf


@pytest.mark.parametrize(
    ("fusion_type", "urdf_type"),
    [
        ("RigidJointType", "fixed"),
        ("RevoluteJointType", "revolute"),
        ("SliderJointType", "prismatic"),
    ],
)
def test_direct_fixture_has_one_faithful_joint(tmp_path, fusion_type: str, urdf_type: str) -> None:
    output = tmp_path / "fixture.urdf"
    metadata = write_direct_fixture_urdf(fusion_type, [1.0, 2.0, 3.0], output)
    root = ET.parse(output).getroot()
    joints = root.findall("joint")

    assert len(joints) == 1
    assert joints[0].attrib["type"] == urdf_type
    assert metadata["urdf_joint_type"] == urdf_type
    if urdf_type != "fixed":
        axis = [float(value) for value in joints[0].find("axis").attrib["xyz"].split()]
        assert sum(value * value for value in axis) == pytest.approx(1.0)


def test_richer_joint_is_rejected_by_direct_fixture_writer(tmp_path) -> None:
    with pytest.raises(ValueError, match="no direct PyBullet fixture"):
        write_direct_fixture_urdf("BallJointType", [0.0, 0.0, 1.0], tmp_path / "fixture.urdf")


def test_selected_cad_writer_applies_occurrence_placements(tmp_path) -> None:
    obj_a = tmp_path / "a.obj"
    obj_b = tmp_path / "b.obj"
    obj_a.write_text("o a\n", encoding="utf-8")
    obj_b.write_text("o b\n", encoding="utf-8")
    transform_a = [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    transform_b = [[1, 0, 0, 12], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    master = {
        "joint_set_id": "joint_set_test",
        "bodies": [
            {"id": "a", "source_files": {"obj": str(obj_a)}},
            {
                "id": "b",
                "source_files": {"obj": str(obj_b)},
                "physical_properties": {"mass": 1.0},
                "geometry_quality": {"bounding_box": {"extents": [1.0, 1.0, 1.0]}},
            },
        ],
        "joints": [
            {
                "id": "joint_0",
                "fusion_joint_type": "SliderJointType",
                "body_a": "a",
                "body_b": "b",
                "axis": {"origin": [11.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
                "motion_semantics": {
                    "resolved_directions": {"slide_direction": [1.0, 0.0, 0.0]},
                    "limit_groups": {},
                },
                "geometry_or_origin_one": {"transform_matrix": transform_a},
                "geometry_or_origin_two": {"transform_matrix": transform_b},
            }
        ],
    }
    output = tmp_path / "selected.urdf"
    metadata = write_selected_cad_urdf(master, "joint_0", output)
    root = ET.parse(output).getroot()
    parent_mesh_origin = root.findall("link")[0].find("visual/origin").attrib["xyz"]
    child_mesh_origin = root.findall("link")[1].find("visual/origin").attrib["xyz"]

    assert parent_mesh_origin == "0.1 0 0"
    assert child_mesh_origin == "0.01 0 0"
    assert metadata["target_coordinate"] == pytest.approx(0.01)
