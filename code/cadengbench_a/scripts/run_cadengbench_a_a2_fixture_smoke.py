#!/usr/bin/env python3
"""Run deterministic PyBullet fixtures for all seven A2 joint families."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.cadengbench_a_a2_ball import run_ball_fixture
from pipeline.cadengbench_a_a2_compound import (
    run_compound_fixture,
    write_compound_fixture_urdf,
)
from pipeline.cadengbench_a_a2_pybullet import (
    run_direct_fixture,
    write_direct_fixture_urdf,
)


def main() -> int:
    """Execute the seven-family adapter smoke and write one JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "cadengbench_a_a2_fixture_smoke",
    )
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for joint_type in ("RigidJointType", "RevoluteJointType", "SliderJointType"):
        path = args.output_dir / f"{joint_type}.urdf"
        write_direct_fixture_urdf(joint_type, [0, 0, 1], path)
        row = run_direct_fixture(path, joint_type, [0, 0, 1], steps=args.steps)
        row["fixture_joint_type"] = joint_type
        rows.append(row)

    compounds = {
        "CylindricalJointType": {
            "rotation_axis": [0, 0, 1],
            "slide_direction": [0, 0, 1],
        },
        "PlanarJointType": {
            "normal_direction": [0, 0, 1],
            "primary_slide_direction": [1, 0, 0],
            "secondary_slide_direction": [0, 1, 0],
        },
        "PinSlotJointType": {
            "rotation_axis": [0, 0, 1],
            "slide_direction": [1, 0, 0],
        },
    }
    for joint_type, directions in compounds.items():
        path = args.output_dir / f"{joint_type}.urdf"
        write_compound_fixture_urdf(joint_type, directions, path)
        row = run_compound_fixture(
            path, joint_type, directions, steps=args.steps
        )
        row["fixture_joint_type"] = joint_type
        rows.append(row)

    ball = run_ball_fixture(
        {
            "pitch_direction": [1, 0, 0],
            "yaw_direction": [0, 1, 0],
            "roll_direction": [0, 0, 1],
        },
        steps=args.steps,
    )
    ball["fixture_joint_type"] = "BallJointType"
    rows.append(ball)

    report = {
        "schema": "cadengbench_a_a2_seven_family_smoke_v1",
        "all_passed": all(row.get("status") == "passed" for row in rows),
        "results": rows,
    }
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
