from pathlib import Path

import pytest

from cadengbench_p_l3_response.io import (
    verify_evidence_commit,
    write_evidence_commit,
    write_json,
)
from cadengbench_p_l3_response.provenance import _identity_files


def test_evidence_commit_detects_mutation(tmp_path: Path):
    result = tmp_path / "result.json"
    write_json(result, {"status": "passed"})
    write_json(tmp_path / "evidence.json", {"value": 1})
    write_evidence_commit(tmp_path, result)
    verify_evidence_commit(tmp_path)
    write_json(tmp_path / "evidence.json", {"value": 2})
    with pytest.raises(ValueError, match="evidence changed"):
        verify_evidence_commit(tmp_path)


def test_build_identity_includes_all_response_schemas():
    logical_paths = [logical_path for logical_path, _ in _identity_files()]
    assert len(logical_paths) == len(set(logical_paths))
    assert {
        "_schemas/response_contract.v1.schema.json",
        "_schemas/response_family_manifest.v1.schema.json",
        "_schemas/response_pair_result.v1.schema.json",
        "_schemas/response_profile.v1.schema.json",
    } <= set(logical_paths)
