"""Schema validation for extension cards."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema


class ContractError(ValueError):
    """An extension contract is incomplete or inconsistent."""


def _schema() -> dict[str, Any]:
    source = (
        Path(__file__).resolve().parents[2]
        / "schemas/body_acceleration_card.v1.schema.json"
    )
    if source.is_file():
        return json.loads(source.read_text(encoding="utf-8"))
    packaged = resources.files("cadengbench_p_l3_extended").joinpath(
        "_schemas/body_acceleration_card.v1.schema.json"
    )
    return json.loads(packaged.read_text(encoding="utf-8"))


def validate_body_card(card: Mapping[str, Any]) -> None:
    try:
        jsonschema.validate(dict(card), _schema())
    except jsonschema.ValidationError as exc:
        raise ContractError(f"invalid body-acceleration card: {exc.message}") from exc
    vector = [float(value) for value in card["load"]["acceleration_vector_mm_s2"]]
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ContractError("body acceleration must be finite and nonzero")
    axis = {"x": 0, "y": 1, "z": 2}[str(card["selector"]["axis"])]
    nonzero = [index for index, value in enumerate(vector) if abs(value) > 1e-12]
    if nonzero != [axis]:
        raise ContractError("body acceleration must follow the support span axis")
    sign = 1.0 if card["selector"]["side"] == "min" else -1.0
    if vector[axis] * sign <= 0:
        raise ContractError("body acceleration must point away from the support")
    if card["status"] == "solver_qualified" and "qualification" not in card:
        raise ContractError("solver-qualified card lacks qualification")
    if card["status"] != "solver_qualified" and "qualification" in card:
        raise ContractError("unqualified card cannot carry qualification")
