from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest


def qualified_domain(minimum: float, maximum: float) -> dict[str, Any]:
    return {
        "kind": "continuous",
        "minimum": minimum,
        "maximum": maximum,
        "minimum_inclusive": True,
        "maximum_inclusive": True,
        "authority": "synthetic_test_oracle",
        "evidence_ref": "tests/l2/fixtures/domain.json",
    }


@pytest.fixture
def parameter_system() -> dict[str, Any]:
    return {
        "interface_version": "cadengbench-p.cadquery-submission.v1",
        "parameters": [
            {
                "parameter_id": "P-001",
                "engineering_name": "overall_length",
                "nominal": 10.0,
                "unit": "mm",
                "role": "principal_dimension",
                "domain": qualified_domain(5.0, 15.0),
                "variant_requirement_contract": {
                    "kind": "direct_measurement_only",
                    "authority": "synthetic_test_oracle",
                    "evidence_ref": "tests/l2/fixtures/domain.json",
                },
            },
            {
                "parameter_id": "P-002",
                "engineering_name": "overall_width",
                "nominal": 4.0,
                "unit": "mm",
                "role": "principal_dimension",
                "domain": qualified_domain(2.0, 8.0),
                "variant_requirement_contract": {
                    "kind": "direct_measurement_only",
                    "authority": "synthetic_test_oracle",
                    "evidence_ref": "tests/l2/fixtures/domain.json",
                },
            },
        ],
        "relations": [],
        "qualified_parameter_count": 2,
        "complete_domains": True,
    }


@pytest.fixture
def submission_contract() -> dict[str, Any]:
    return {
        "interface_version": "cadengbench-p.cadquery-submission.v1",
        "default_params": {"length": 10.0, "width": 4.0},
        "parameter_manifest": {
            "P-001": {
                "name": "length",
                "unit": "mm",
                "role": "principal_dimension",
            },
            "P-002": {
                "name": "width",
                "unit": "mm",
                "role": "principal_dimension",
            },
        },
    }


@pytest.fixture
def l2_case(parameter_system: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": "synthetic_l2_item",
        "parameter_system": deepcopy(parameter_system),
        "functional_edit": {"available": False},
        "evaluator_profile": {
            "l2": {
                "max_variants": 12,
                "directed_points_per_parameter": 2,
                "morris_levels": 4,
                "morris_trajectories": 2,
                "seed": 1729,
            }
        },
    }
