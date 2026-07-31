"""Build identity for the standalone extension."""

from __future__ import annotations

import hashlib
from pathlib import Path

from cadengbench_p_eval.io import sha256_json


def evaluator_build_hash() -> str:
    root = Path(__file__).resolve().parent
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*.py"))
    ]
    return sha256_json(records)
