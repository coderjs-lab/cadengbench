#!/usr/bin/env python3
"""Audit documentation coverage for the active CADEngBench-A Python surface."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "pipeline" / "cadengbench_a_provider_runtime.py",
    ROOT / "pipeline" / "cadengbench_a_hypothesis_evaluator.py",
    ROOT / "pipeline" / "cadengbench_a_pose_evaluator.py",
    ROOT / "pipeline" / "cadengbench_a_pose_simulation.py",
    ROOT / "pipeline" / "cadengbench_a_pose_stage.py",
    ROOT / "pipeline" / "cadengbench_a_a2_ball.py",
    ROOT / "pipeline" / "cadengbench_a_a2_compound.py",
    ROOT / "pipeline" / "cadengbench_a_a2_policy.py",
    ROOT / "pipeline" / "cadengbench_a_a2_pybullet.py",
    ROOT / "scripts" / "run_cadengbench_a_benchmark.py",
    ROOT / "scripts" / "build_cadengbench_a_submission_sample.py",
    ROOT / "scripts" / "verify_cadengbench_a_release.py",
    ROOT / "scripts" / "run_cadengbench_a_a2_fixture_smoke.py",
]


def main() -> int:
    missing = []
    for path in FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ast.get_docstring(tree) is None:
            missing.append({"file": str(path.relative_to(ROOT)), "symbol": "<module>"})
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and not node.name.startswith("_")
                and ast.get_docstring(node) is None
            ):
                missing.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "symbol": node.name,
                        "line": node.lineno,
                    }
                )
    report = {
        "schema": "cadengbench_a_comment_audit_v1",
        "files_checked": len(FILES),
        "pass": not missing,
        "missing_public_docstrings": missing,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
