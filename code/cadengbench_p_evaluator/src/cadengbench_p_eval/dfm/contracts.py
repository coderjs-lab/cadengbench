"""Evidence and applicability gates shared by deterministic DFM checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cadengbench_p_eval.contracts import Status

MINIMUM_WALL_THICKNESS = "minimum_wall_thickness"
MINIMUM_HOLE_GEOMETRY = "minimum_hole_geometry"

REQUIRED_THRESHOLDS = {
    MINIMUM_WALL_THICKNESS: ("minimum_mm",),
    MINIMUM_HOLE_GEOMETRY: (
        "minimum_diameter_mm",
        "maximum_depth_to_diameter",
    ),
}


@dataclass(frozen=True)
class DFMGate:
    family: str
    status: Status
    required: bool
    thresholds: dict[str, float]
    evidence_ref: Any
    extractor_certified: bool
    reason: str
    raw_contract: dict[str, Any]

    @property
    def scoreable(self) -> bool:
        return self.status is Status.PASSED

    def evidence(self) -> list[dict[str, Any]]:
        if self.evidence_ref is None:
            return []
        if isinstance(self.evidence_ref, Mapping):
            return [dict(self.evidence_ref)]
        return [{"threshold_evidence_ref": self.evidence_ref}]


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0.0 else None


def _explicitly_not_applicable(value: Any) -> bool:
    if value is False:
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "not_applicable",
        "inapplicable",
        "excluded",
        "false",
    }


def resolve_dfm_gate(dfm: Mapping[str, Any], family: str) -> DFMGate:
    """Resolve whether one DFM family may emit a pass/fail verdict."""

    if family not in REQUIRED_THRESHOLDS:
        raise ValueError(f"unsupported DFM family: {family}")
    raw_families = dfm.get("families")
    families = raw_families if isinstance(raw_families, list) else []
    record = next(
        (
            value
            for value in families
            if isinstance(value, Mapping) and value.get("family") == family
        ),
        None,
    )
    if record is None:
        return DFMGate(
            family=family,
            status=Status.NOT_APPLICABLE,
            required=False,
            thresholds={},
            evidence_ref=None,
            extractor_certified=False,
            reason="the item declares no contract for this DFM family",
            raw_contract={},
        )
    required = bool(record.get("required", False))
    raw = dict(record)
    if _explicitly_not_applicable(record.get("applicability")):
        return DFMGate(
            family=family,
            status=Status.NOT_APPLICABLE,
            required=required,
            thresholds={},
            evidence_ref=record.get("threshold_evidence_ref"),
            extractor_certified=record.get("extractor_certified") is True,
            reason="the frozen item contract marks this family not applicable",
            raw_contract=raw,
        )
    raw_thresholds = record.get("thresholds")
    threshold_mapping = raw_thresholds if isinstance(raw_thresholds, Mapping) else {}
    thresholds: dict[str, float] = {}
    missing: list[str] = []
    for key in REQUIRED_THRESHOLDS[family]:
        value = _finite_positive(threshold_mapping.get(key))
        if value is None:
            missing.append(key)
        else:
            thresholds[key] = value
    evidence_ref = record.get("threshold_evidence_ref")
    extractor_certified = record.get("extractor_certified") is True
    reasons: list[str] = []
    if record.get("scoreable") is not True:
        reasons.append("the case does not promote this family as scoreable")
    if not extractor_certified:
        reasons.append("the deterministic extractor is not certified")
    if evidence_ref is None or evidence_ref == "":
        reasons.append("threshold evidence is absent")
    if missing:
        reasons.append("missing finite positive thresholds: " + ", ".join(missing))
    if reasons:
        return DFMGate(
            family=family,
            status=Status.NOT_SCOREABLE,
            required=required,
            thresholds=thresholds,
            evidence_ref=evidence_ref,
            extractor_certified=extractor_certified,
            reason="; ".join(reasons),
            raw_contract=raw,
        )
    return DFMGate(
        family=family,
        status=Status.PASSED,
        required=required,
        thresholds=thresholds,
        evidence_ref=evidence_ref,
        extractor_certified=True,
        reason="applicability, extractor, thresholds, and evidence are frozen",
        raw_contract=raw,
    )
