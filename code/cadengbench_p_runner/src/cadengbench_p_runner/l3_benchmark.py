"""Compose frozen L0-L2 evidence with the standalone L3 response evaluator."""

from __future__ import annotations

import copy
import csv
import json
import statistics
import subprocess
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json

BINDING_SCHEMA = "cadengbench-p.l3-binding-index.v2"
PLAN_SCHEMA = "cadengbench-p.l3-benchmark-plan.v5"
SUMMARY_SCHEMA = "cadengbench-p.l3-benchmark-summary.v5"
PAIR_SELECTION_SCHEMA = "cadengbench-p.l3-pair-selection.v1"
PROTOCOL_E_EXCLUSION = (
    "L3 v1 has no certified source-native post-edit reference producer"
)


def _json_key(assignments: Mapping[str, Any]) -> str:
    return json.dumps(
        {str(key): float(value) for key, value in sorted(assignments.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


def _assignment_distance(
    assignments: Mapping[str, Any], nominal: Mapping[str, Any]
) -> float:
    if set(assignments) != set(nominal):
        raise ValueError("L3 variant and nominal assignments use different parameters")
    deltas = []
    for parameter_id in sorted(nominal):
        value = float(assignments[parameter_id])
        nominal_value = float(nominal[parameter_id])
        scale = max(abs(value), abs(nominal_value), 1e-12)
        deltas.append(abs(value - nominal_value) / scale)
    return max(deltas, default=0.0)


def apply_l3_pair_selection(
    *,
    plan: Mapping[str, Any],
    selection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    policy = read_json(selection_path)
    if policy.get("schema_version") != PAIR_SELECTION_SCHEMA:
        raise ValueError(f"pair selection must use {PAIR_SELECTION_SCHEMA}")
    policy_id = str(policy.get("policy_id") or "")
    if not policy_id:
        raise ValueError("pair selection policy_id must be nonempty")
    if policy.get("distance_metric") != "max_symmetric_relative_parameter_delta":
        raise ValueError("unsupported L3 pair-selection distance metric")
    full_family_item_ids = policy.get("full_family_item_ids")
    if (
        not isinstance(full_family_item_ids, list)
        or len(full_family_item_ids) != len(set(full_family_item_ids))
        or any(
            not isinstance(item_id, str) or not item_id
            for item_id in full_family_item_ids
        )
    ):
        raise ValueError("full_family_item_ids must be unique nonempty strings")

    selected_plan = copy.deepcopy(dict(plan))
    release = validate_release(
        Path(selected_plan["release_root"]), require_items=False
    )
    unknown = set(full_family_item_ids) - set(release["item_ids"])
    if unknown:
        raise ValueError(f"pair selection contains unknown item IDs: {sorted(unknown)}")

    candidate_pair_count = 0
    selected_pair_count = 0
    selected_role_counts: Counter[str] = Counter()
    ready_ablation_items = 0
    full_family_set = set(full_family_item_ids)
    for item in selected_plan["items"]:
        variants = item.get("variants") or []
        candidate_pair_count += len(variants)
        if not variants:
            item["unselected_variant_count"] = 0
            continue

        nominal_variants = [
            variant
            for variant in variants
            if variant["variant_id"] == "nominal"
            or any(
                membership.get("schedule_kind") == "nominal"
                for membership in variant["memberships"]
            )
        ]
        if len(nominal_variants) != 1:
            raise ValueError(
                f"L3 item must have exactly one nominal variant: {item['item_id']}"
            )
        nominal = nominal_variants[0]
        nominal_assignments = nominal["assignments"]
        ranked_non_nominal = []
        for variant in variants:
            if variant is nominal:
                continue
            distance = _assignment_distance(
                variant["assignments"], nominal_assignments
            )
            if distance > 1e-9:
                ranked_non_nominal.append((distance, variant["variant_id"], variant))
        ranked_non_nominal.sort(key=lambda value: (value[0], value[1]))

        roles: dict[str, set[str]] = {nominal["variant_id"]: {"nominal"}}
        if item["item_id"] in full_family_set:
            ready_ablation_items += 1
            selected_variants = list(variants)
            for variant in selected_variants:
                roles.setdefault(variant["variant_id"], set()).add(
                    "full_family_ablation"
                )
        else:
            selected_ids = {nominal["variant_id"]}
            if ranked_non_nominal:
                moderate = ranked_non_nominal[(len(ranked_non_nominal) - 1) // 2][2]
                boundary = ranked_non_nominal[-1][2]
                selected_ids.update((moderate["variant_id"], boundary["variant_id"]))
                roles.setdefault(moderate["variant_id"], set()).add("moderate")
                roles.setdefault(boundary["variant_id"], set()).add("boundary")
            selected_variants = [
                variant for variant in variants if variant["variant_id"] in selected_ids
            ]

        distance_by_id = {
            variant["variant_id"]: _assignment_distance(
                variant["assignments"], nominal_assignments
            )
            for variant in selected_variants
        }
        for variant in selected_variants:
            variant["pair_selection_roles"] = sorted(roles[variant["variant_id"]])
            variant["pair_selection_distance"] = distance_by_id[variant["variant_id"]]
            selected_role_counts.update(variant["pair_selection_roles"])
        item["unselected_variant_count"] = len(variants) - len(selected_variants)
        item["variants"] = selected_variants
        selected_pair_count += len(selected_variants)

    selected_plan["schema_version"] = PLAN_SCHEMA
    selected_plan["pair_selection"] = {
        "schema_version": PAIR_SELECTION_SCHEMA,
        "policy_id": policy_id,
        "policy_path": str(selection_path.resolve()),
        "distance_metric": policy["distance_metric"],
        "candidate_pair_count": candidate_pair_count,
        "selected_pair_count": selected_pair_count,
        "selected_role_counts": dict(sorted(selected_role_counts.items())),
        "full_family_item_count": len(full_family_item_ids),
        "ready_full_family_item_count": ready_ablation_items,
    }
    write_json(output_path, selected_plan)
    return selected_plan


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            records.append(value)
    return records


def validate_release(
    release_root: Path, *, require_items: bool = True
) -> dict[str, Any]:
    release = read_json(release_root / "RELEASE.json")
    if release.get("benchmark") != "cadengbench-p" or release.get("status") != "frozen":
        raise ValueError("release is not the frozen CADEngBench-P release")
    item_ids = [
        line.strip()
        for line in (release_root / "item_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    selected = _read_jsonl(release_root / "selected_items.jsonl")
    selected_ids = [str(record.get("item_id")) for record in selected]
    expected = int(release["item_count"])
    if len(item_ids) != expected or len(set(item_ids)) != expected:
        raise ValueError("release item ID index is incomplete or duplicated")
    if set(selected_ids) != set(item_ids) or len(selected_ids) != expected:
        raise ValueError("release selection roster differs from item_ids.txt")
    items_root = release_root / "items"
    if require_items:
        if not items_root.is_dir():
            raise ValueError(f"canonical release payload is missing: {items_root}")
        observed = sorted(path.name for path in items_root.iterdir() if path.is_dir())
        if observed != sorted(item_ids):
            raise ValueError(
                "canonical release items/ payload differs from the frozen roster"
            )
        missing_manifests = [
            item_id
            for item_id in item_ids
            if not (items_root / item_id / "item.json").is_file()
        ]
        if missing_manifests:
            raise ValueError(f"release items lack item.json: {missing_manifests[:10]}")
    return {
        "release": release,
        "item_ids": item_ids,
        "selected": {str(record["item_id"]): record for record in selected},
        "items_root": items_root,
    }


def _artifact_candidates(root: Path) -> dict[str, dict[str, list[Path]]]:
    candidates: dict[str, dict[str, list[Path]]] = {}
    if not root.is_dir():
        return candidates
    for path in sorted(root.rglob("*.json")):
        try:
            value = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(value, Mapping):
            continue
        item_id = value.get("item_id")
        schema = str(value.get("schema_version") or "")
        if not isinstance(item_id, str):
            continue
        kind = None
        if schema == "cadengbench-p.l3-physics-card.v2" and (
            (value.get("qualification") or {}).get("promoted") is True
        ):
            kind = "card"
        elif (
            schema == "cadengbench-p.l3-mesh-sensitivity-study.v1"
            and value.get("all_levels_passed") is True
        ):
            kind = "mesh_study"
        elif schema == "cadengbench-p.l3-response-contract.v1":
            kind = "contract"
        if kind is not None:
            candidates.setdefault(item_id, {}).setdefault(kind, []).append(path)
    return candidates


def _one_artifact(
    candidates: Mapping[str, list[Path]], kind: str, *, required: bool
) -> Path | None:
    paths = list(candidates.get(kind) or [])
    if not paths:
        if required:
            raise ValueError(f"solver-qualified item has no {kind} artifact")
        return None
    by_hash: dict[str, list[Path]] = {}
    for path in paths:
        by_hash.setdefault(sha256_file(path), []).append(path)
    if len(by_hash) != 1:
        raise ValueError(f"multiple nonidentical {kind} artifacts were discovered")
    return sorted(paths, key=lambda path: (len(path.parts), str(path)))[0]


def _nominal_reference(item_root: Path, source_key: str) -> Path:
    relative = (
        "source/generated_from_cadquery.step"
        if source_key == "benchcad"
        else "source/fusion360_occ_reconstruction.step"
    )
    path = item_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"nominal L3 reference is missing: {path}")
    return path


def _repository_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / "PROJECT_BRIEF.md").is_file():
            return candidate
    raise ValueError(f"cannot resolve repository root from {path}")


def _resolved_artifact_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _build_extended_binding_index(
    *,
    release: Mapping[str, Any],
    applicability_index_path: Path,
    sidecar_root: Path,
    contract_root: Path,
    body_contract_root: Path,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(applicability_index_path)
    by_id = {str(row["item_id"]): row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(release["item_ids"]):
        raise ValueError("extended L3 applicability index differs from the release")
    repo_root = _repository_root(applicability_index_path)
    canonical_sidecars = sidecar_root / "items"
    frozen_artifacts = _artifact_candidates(
        canonical_sidecars if canonical_sidecars.is_dir() else sidecar_root
    )
    items: list[dict[str, Any]] = []
    for item_id in release["item_ids"]:
        selected = release["selected"][item_id]
        source_key = str(selected["source_key"])
        row = by_id[item_id]
        if row.get("status") != "solver_qualified":
            items.append(
                {
                    "item_id": item_id,
                    "source_key": source_key,
                    "difficulty": selected["complexity"]["tier"],
                    "applicability": "not_applicable",
                    "reason": row.get("reason") or "not_qualified",
                }
            )
            continue
        card_value = row.get("physics_card_path")
        if not isinstance(card_value, str):
            raise ValueError(f"qualified extended item has no card: {item_id}")
        card = _resolved_artifact_path(repo_root, card_value)
        if not card.is_file():
            raise FileNotFoundError(f"extended L3 card is missing: {card}")
        card_document = read_json(card)
        scenario = str(card_document["scenario_family"])
        is_body = scenario == "single_support_body_acceleration"
        if is_body:
            contract = body_contract_root / "items" / item_id / "response_contract.json"
            mesh_study = None
            evaluator_mode = "body_acceleration_response_v1"
        else:
            contract = contract_root / "items" / item_id / "response_contract.json"
            mesh_study = _one_artifact(
                frozen_artifacts.get(item_id, {}), "mesh_study", required=False
            )
            evaluator_mode = "surface_force_response_v1"
        reference = _nominal_reference(release["items_root"] / item_id, source_key)
        expected_hash = (card_document.get("qualification") or {}).get(
            "reference_step_sha256"
        )
        if expected_hash != sha256_file(reference):
            raise ValueError(f"L3 card reference binding differs for {item_id}")
        items.append(
            {
                "item_id": item_id,
                "source_key": source_key,
                "difficulty": selected["complexity"]["tier"],
                "applicability": "solver_qualified",
                "reason": None,
                "scenario": scenario,
                "evaluator_mode": evaluator_mode,
                "card_path": str(card),
                "mesh_study_path": str(mesh_study.resolve()) if mesh_study else None,
                "nominal_reference_step_path": str(reference.resolve()),
                "response_contract_path": str(contract.resolve()),
                "response_contract_status": (
                    "ready" if contract.is_file() else "needs_calibration"
                ),
            }
        )
    return items


def build_binding_index(
    *,
    release_root: Path,
    qualification_summary_path: Path,
    sidecar_root: Path,
    contract_root: Path,
    output_path: Path,
    applicability_index_path: Path | None = None,
    body_contract_root: Path | None = None,
) -> dict[str, Any]:
    release = validate_release(release_root, require_items=True)
    if applicability_index_path is not None:
        if body_contract_root is None:
            raise ValueError("extended L3 indexing requires a body contract root")
        items = _build_extended_binding_index(
            release=release,
            applicability_index_path=applicability_index_path,
            sidecar_root=sidecar_root,
            contract_root=contract_root,
            body_contract_root=body_contract_root,
        )
        result = {
            "schema_version": BINDING_SCHEMA,
            "dataset_version": release["release"]["dataset_version"],
            "profile": "extended_v1",
            "item_count": len(items),
            "solver_qualified_count": sum(
                item["applicability"] == "solver_qualified" for item in items
            ),
            "family_counts": dict(
                Counter(
                    item["scenario"]
                    for item in items
                    if item["applicability"] == "solver_qualified"
                )
            ),
            "items": items,
        }
        write_json(output_path, result)
        return result
    qualification = read_json(qualification_summary_path)
    records = (
        qualification.get("records") if isinstance(qualification, Mapping) else None
    )
    if not isinstance(records, list):
        raise ValueError("L3 qualification summary has no item records")
    by_id = {str(record["item_id"]): record for record in records}
    if set(by_id) != set(release["item_ids"]):
        raise ValueError("L3 qualification roster differs from the canonical release")
    canonical_sidecars = sidecar_root / "items"
    artifacts = _artifact_candidates(
        canonical_sidecars if canonical_sidecars.is_dir() else sidecar_root
    )
    canonical_contracts = contract_root / "items"
    contract_artifacts = _artifact_candidates(
        canonical_contracts if canonical_contracts.is_dir() else contract_root
    )
    items: list[dict[str, Any]] = []
    for item_id in release["item_ids"]:
        selected = release["selected"][item_id]
        source_key = str(selected["source_key"])
        record = by_id[item_id]
        applicable = record.get("status") == "promoted"
        if not applicable:
            items.append(
                {
                    "item_id": item_id,
                    "source_key": source_key,
                    "difficulty": selected["complexity"]["tier"],
                    "applicability": "not_applicable",
                    "reason": record.get("annotation_reason")
                    or record.get("failure_stage")
                    or str(record.get("status") or "not_qualified"),
                }
            )
            continue
        available = artifacts.get(item_id, {})
        card = _one_artifact(available, "card", required=True)
        mesh_study = _one_artifact(available, "mesh_study", required=False)
        contract = _one_artifact(
            contract_artifacts.get(item_id, {}), "contract", required=False
        )
        reference = _nominal_reference(release["items_root"] / item_id, source_key)
        card_value = read_json(card)
        expected_reference_hash = (card_value.get("qualification") or {}).get(
            "reference_step_sha256"
        )
        if expected_reference_hash != sha256_file(reference):
            raise ValueError(f"L3 card reference binding differs for {item_id}")
        desired_contract = contract_root / "items" / item_id / "response_contract.json"
        items.append(
            {
                "item_id": item_id,
                "source_key": source_key,
                "difficulty": selected["complexity"]["tier"],
                "applicability": "solver_qualified",
                "reason": None,
                "scenario": card_value["scenario_family"],
                "evaluator_mode": "surface_force_response_v1",
                "card_path": str(card.resolve()),
                "mesh_study_path": str(mesh_study.resolve()) if mesh_study else None,
                "nominal_reference_step_path": str(reference.resolve()),
                "response_contract_path": str(
                    contract.resolve() if contract else desired_contract.resolve()
                ),
                "response_contract_status": "ready"
                if contract
                else "needs_calibration",
            }
        )
    result = {
        "schema_version": BINDING_SCHEMA,
        "dataset_version": release["release"]["dataset_version"],
        "profile": "frozen_v1",
        "item_count": len(items),
        "solver_qualified_count": sum(
            item["applicability"] == "solver_qualified" for item in items
        ),
        "items": items,
    }
    write_json(output_path, result)
    return result


def _layer_status(result: Mapping[str, Any], layer_name: str) -> str:
    for layer in result.get("layers") or []:
        if isinstance(layer, Mapping) and layer.get("layer") == layer_name:
            return str(layer.get("status") or "not_reached")
    return "not_reached"


def _variant_statuses(l2: Mapping[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    replay = (l2.get("artifacts") or {}).get("replay") or {}
    for record in replay.get("variants") or []:
        variant = record.get("variant") or {}
        variant_id = record.get("variant_id") or variant.get("variant_id")
        values = [
            (record.get("execution") or {}).get("status"),
            (record.get("l0") or {}).get("status"),
            (record.get("l1") or {}).get("status"),
        ]
        statuses[str(variant_id)] = (
            "passed" if all(value == "passed" for value in values) else "not_reached"
        )
    return statuses


def _variant_physical_statuses(l2: Mapping[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    replay = (l2.get("artifacts") or {}).get("replay") or {}
    for record in replay.get("variants") or []:
        variant = record.get("variant") or {}
        variant_id = record.get("variant_id") or variant.get("variant_id")
        values = [
            (record.get("execution") or {}).get("status"),
            (record.get("l0") or {}).get("status"),
        ]
        statuses[str(variant_id)] = (
            "passed" if all(value == "passed" for value in values) else "not_reached"
        )
    return statuses


def _l02_item_roster(l02_run: Path, release_item_ids: list[str]) -> list[str]:
    manifest_path = l02_run / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        inventory = manifest.get("submission_inventory")
        if not isinstance(inventory, list):
            raise ValueError("L0-L2 manifest has no submission inventory")
        item_ids = [str(record.get("item_id")) for record in inventory]
        if len(item_ids) != int(manifest.get("item_count", -1)):
            raise ValueError("L0-L2 manifest item count does not match its inventory")
    else:
        item_ids = [
            item_id
            for item_id in release_item_ids
            if (l02_run / "items" / item_id / "result.json").is_file()
        ]
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise ValueError("L0-L2 run has an empty or duplicate item roster")
    unknown = set(item_ids) - set(release_item_ids)
    if unknown:
        raise ValueError(f"L0-L2 run contains unknown item IDs: {sorted(unknown)}")
    missing = [
        item_id
        for item_id in item_ids
        if not (l02_run / "items" / item_id / "result.json").is_file()
    ]
    if missing:
        raise ValueError(f"L0-L2 run is incomplete for item IDs: {sorted(missing)}")
    selected = set(item_ids)
    return [item_id for item_id in release_item_ids if item_id in selected]


def plan_l3_run(
    *,
    release_root: Path,
    l02_run: Path,
    bindings_path: Path,
    reference_bank_path: Path,
    alignment_path: Path,
    response_profile_path: Path,
    body_response_profile_path: Path | None = None,
    pair_selection_path: Path | None = None,
    output_path: Path,
) -> dict[str, Any]:
    release = validate_release(release_root, require_items=False)
    bindings_doc = read_json(bindings_path)
    bank_doc = read_json(reference_bank_path)
    alignment_doc = read_json(alignment_path)
    bindings = {record["item_id"]: record for record in bindings_doc.get("items") or []}
    bank = {record["item_id"]: record for record in bank_doc.get("items") or []}
    alignment = {
        record["item_id"]: record for record in alignment_doc.get("items") or []
    }
    if set(bindings) != set(release["item_ids"]):
        raise ValueError("L3 binding index does not cover the frozen 300-item roster")

    l02_item_ids = _l02_item_roster(l02_run, release["item_ids"])
    items: list[dict[str, Any]] = []
    for item_id in l02_item_ids:
        binding = bindings[item_id]
        item_run = l02_run / "items" / item_id
        result = read_json(item_run / "result.json")
        l2_status = _layer_status(result, "L2")
        protocol = str(result.get("protocol") or "unknown")
        if binding["applicability"] != "solver_qualified":
            items.append(
                {
                    "item_id": item_id,
                    "protocol": protocol,
                    "applicability": "not_applicable",
                    "l2_status": l2_status,
                    "status": "not_applicable",
                    "reason": binding.get("reason"),
                    "variants": [],
                }
            )
            continue
        if protocol != "zero_to_cad":
            items.append(
                {
                    "item_id": item_id,
                    "protocol": protocol,
                    "applicability": "protocol_not_applicable",
                    "l2_status": l2_status,
                    "status": "not_applicable",
                    "reason": PROTOCOL_E_EXCLUSION,
                    "variants": [],
                }
            )
            continue
        if (
            item_id not in bank
            or item_id not in alignment
            or bank[item_id].get("status") != "ready"
            or alignment[item_id].get("status") != "ready"
        ):
            items.append(
                {
                    "item_id": item_id,
                    "protocol": protocol,
                    "applicability": "solver_qualified",
                    "l2_status": l2_status,
                    "status": "blocked",
                    "reason": "matched_reference_or_alignment_not_ready",
                    "variants": [],
                    "binding": binding,
                }
            )
            continue

        l2 = read_json(item_run / "l2.json")
        variant_status = _variant_statuses(l2)
        physical_status = _variant_physical_statuses(l2)
        references = {
            _json_key(record["assignments"]): record
            for record in bank[item_id].get("references") or []
        }
        aligned = {
            _json_key(record["assignments"]): record
            for record in alignment[item_id].get("variants") or []
        }
        variants: list[dict[str, Any]] = []
        for key, aligned_record in sorted(aligned.items()):
            reference = references.get(key)
            if reference is None:
                raise ValueError(f"no parameter-matched reference for {item_id}: {key}")
            memberships = aligned_record["memberships"]
            l02_path_status = "passed"
            physical = "passed"
            for membership in memberships:
                variant_id = membership["variant_id"]
                status = (
                    "passed"
                    if variant_id == "nominal"
                    and _layer_status(result, "L0") == "passed"
                    and _layer_status(result, "L1") == "passed"
                    else variant_status.get(variant_id, "not_reached")
                )
                if status != "passed":
                    l02_path_status = "not_reached"
                physical_variant_status = (
                    "passed"
                    if variant_id == "nominal"
                    and _layer_status(result, "L0") == "passed"
                    else physical_status.get(variant_id, "not_reached")
                )
                if physical_variant_status != "passed":
                    physical = "not_reached"
            if aligned_record.get("upstream_status") != "passed":
                physical = "not_reached"
            variants.append(
                {
                    "variant_id": aligned_record["reference_id"],
                    "assignments": aligned_record["assignments"],
                    "memberships": memberships,
                    "l02_path_status": l02_path_status,
                    "physical_status": physical,
                    "l3_scoreable": physical == "passed",
                    "comparison_mode": "matched_parameter_response",
                    "reference_step_path": reference["step_path"],
                    "candidate_step_path": aligned_record.get(
                        "aligned_candidate_step_path"
                    ),
                }
            )
        if not variants:
            items.append(
                {
                    "item_id": item_id,
                    "protocol": protocol,
                    "applicability": "solver_qualified",
                    "l2_status": l2_status,
                    "status": "blocked",
                    "reason": "matched_reference_family_is_empty",
                    "variants": [],
                    "binding": binding,
                }
            )
            continue
        evaluator_mode = str(
            binding.get("evaluator_mode") or "surface_force_response_v1"
        )
        selected_profile = (
            body_response_profile_path
            if evaluator_mode == "body_acceleration_response_v1"
            else response_profile_path
        )
        if selected_profile is None:
            raise ValueError(f"body-response profile is missing for {item_id}")
        items.append(
            {
                "item_id": item_id,
                "protocol": protocol,
                "applicability": "solver_qualified",
                "l2_status": l2_status,
                "status": "ready",
                "reason": None,
                "binding": binding,
                "response_profile_path": str(selected_profile.resolve()),
                "alignment": alignment[item_id].get("transform"),
                "variants": variants,
            }
        )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "dataset_version": release["release"]["dataset_version"],
        "release_root": str(release_root.resolve()),
        "l02_run": str(l02_run.resolve()),
        "bindings_path": str(bindings_path.resolve()),
        "reference_bank_path": str(reference_bank_path.resolve()),
        "alignment_path": str(alignment_path.resolve()),
        "response_profile_path": str(response_profile_path.resolve()),
        "body_response_profile_path": (
            str(body_response_profile_path.resolve())
            if body_response_profile_path is not None
            else None
        ),
        "item_count": len(items),
        "ready_item_count": sum(item["status"] == "ready" for item in items),
        "items": items,
    }
    if pair_selection_path is not None:
        return apply_l3_pair_selection(
            plan=plan,
            selection_path=pair_selection_path,
            output_path=output_path,
        )
    write_json(output_path, plan)
    return plan


def _run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode


def _archive_incomplete_pair_output(
    *, pair_output: Path, retry_archive_root: Path
) -> Path | None:
    if not pair_output.exists():
        return None
    if not pair_output.is_dir():
        raise ValueError(f"L3 pair output is not a directory: {pair_output}")
    target_root = retry_archive_root / pair_output.relative_to(pair_output.parents[3])
    target_root.parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    target = target_root.with_name(f"{target_root.name}.attempt-{attempt:04d}")
    while target.exists():
        attempt += 1
        target = target_root.with_name(f"{target_root.name}.attempt-{attempt:04d}")
    pair_output.rename(target)
    return target


def _ensure_contract(
    *,
    item: Mapping[str, Any],
    surface_l3_cli: Path,
    body_l3_cli: Path | None,
    ccx_path: Path,
    timeout_seconds: int,
) -> Path:
    binding = item["binding"]
    existing = Path(binding["response_contract_path"])
    if existing.is_file():
        return existing
    contract_output = existing.parent
    contract = contract_output / "response_contract.json"
    if contract.is_file():
        return contract
    body_mode = binding.get("evaluator_mode") == "body_acceleration_response_v1"
    if body_mode and body_l3_cli is None:
        raise RuntimeError("body-response CLI is required by the extended L3 plan")
    evaluator_cli = body_l3_cli if body_mode else surface_l3_cli
    command = [
        str(evaluator_cli),
        "calibrate-response" if body_mode else "calibrate",
        "--profile",
        str(item["response_profile_path"]),
        "--card",
        str(binding["card_path"]),
        "--reference-step",
        str(binding["nominal_reference_step_path"]),
        "--output",
        str(contract_output),
        "--ccx",
        str(ccx_path),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if binding.get("mesh_study_path"):
        command.extend(["--mesh-study", str(binding["mesh_study_path"])])
    returncode = _run_command(
        command,
        stdout_path=contract_output / "calibrate.stdout.json",
        stderr_path=contract_output / "calibrate.stderr.txt",
    )
    if returncode != 0 or not contract.is_file():
        raise RuntimeError(f"L3 response calibration failed for {item['item_id']}")
    return contract


def run_l3_plan(
    *,
    plan_path: Path,
    output_root: Path,
    l3_cli: Path,
    body_l3_cli: Path | None = None,
    ccx_path: Path,
    workers: int = 1,
    timeout_seconds: int = 300,
    render: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    plan = read_json(plan_path)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    calibration_errors: list[dict[str, str]] = []
    for item in plan["items"]:
        if item["status"] != "ready":
            continue
        if not any(
            variant["physical_status"] == "passed" for variant in item["variants"]
        ):
            continue
        try:
            contract = _ensure_contract(
                item=item,
                surface_l3_cli=l3_cli,
                body_l3_cli=body_l3_cli,
                ccx_path=ccx_path,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            calibration_errors.append({"item_id": item["item_id"], "message": str(exc)})
            continue
        for variant in item["variants"]:
            if variant["physical_status"] != "passed":
                continue
            jobs.append((item, variant, contract))

    def execute(job: tuple[dict[str, Any], dict[str, Any], Path]) -> dict[str, Any]:
        item, variant, contract = job
        pair_output = (
            output_root / "items" / item["item_id"] / "variants" / variant["variant_id"]
        )
        result_path = pair_output / "pair_result.json"
        if result_path.is_file():
            return {
                "item_id": item["item_id"],
                "variant_id": variant["variant_id"],
                "status": "resumed",
            }
        _archive_incomplete_pair_output(
            pair_output=pair_output,
            retry_archive_root=output_root / "retry_archive",
        )
        body_mode = (
            item["binding"].get("evaluator_mode")
            == "body_acceleration_response_v1"
        )
        if body_mode and body_l3_cli is None:
            return {
                "item_id": item["item_id"],
                "variant_id": variant["variant_id"],
                "status": "evaluator_error",
            }
        evaluator_cli = body_l3_cli if body_mode else l3_cli
        command = [
            str(evaluator_cli),
            "evaluate-response-pair" if body_mode else "evaluate-pair",
            "--contract",
            str(contract),
            "--card",
            str(item["binding"]["card_path"]),
            "--reference-step",
            str(variant["reference_step_path"]),
            "--candidate-step",
            str(variant["candidate_step_path"]),
            "--variant-id",
            str(variant["variant_id"]),
            "--upstream-status",
            "passed",
            "--output",
            str(pair_output),
            "--ccx",
            str(ccx_path),
            "--timeout-seconds",
            str(timeout_seconds),
        ]
        if render:
            command.append("--render")
        returncode = _run_command(
            command,
            stdout_path=pair_output / "evaluate.stdout.json",
            stderr_path=pair_output / "evaluate.stderr.txt",
        )
        return {
            "item_id": item["item_id"],
            "variant_id": variant["variant_id"],
            "status": "completed"
            if returncode == 0 and result_path.is_file()
            else "evaluator_error",
        }

    outcomes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            outcomes.append(future.result())
    result = {
        "schema_version": "cadengbench-p.l3-execution-summary.v1",
        "planned_pair_count": len(jobs),
        "completed_pair_count": sum(
            outcome["status"] in {"completed", "resumed"} for outcome in outcomes
        ),
        "calibration_errors": calibration_errors,
        "outcomes": sorted(
            outcomes, key=lambda value: (value["item_id"], value["variant_id"])
        ),
    }
    write_json(output_root / "execution_summary.json", result)
    return result


_ROW_FIELDS = (
    "item_id",
    "protocol",
    "source_key",
    "difficulty",
    "scenario_family",
    "evaluator_mode",
    "l2_status",
    "l2_variant_id",
    "schedule_kind",
    "pair_selection_roles",
    "assignments",
    "comparison_mode",
    "l3_applicability",
    "planned_l3_scoreable",
    "l3_scoreable",
    "rebuild",
    "selectors",
    "mesh",
    "solve",
    "numerical_validity",
    "response_fidelity",
    "engineering_requirements",
    "final_verdict",
    "stress_margin_mpa",
    "normalized_displacement_margin",
    "displacement_ratio_margin",
    "compliance_ratio_margin",
    "stress_ratio_margin",
    "body_stress_concentration_ratio_margin",
    "response_fidelity_score",
    "reference_response_metrics",
    "candidate_response_metrics",
    "candidate_reference_force_ratio",
    "runtime_seconds",
)

_PAIR_FIELDS = (
    "item_id",
    "protocol",
    "source_key",
    "difficulty",
    "scenario_family",
    "evaluator_mode",
    "l2_status",
    "physical_variant_id",
    "assignments",
    "schedule_memberships",
    "pair_selection_roles",
    "comparison_mode",
    "l3_applicability",
    "planned_l3_scoreable",
    "l3_scoreable",
    "rebuild",
    "selectors",
    "mesh",
    "solve",
    "numerical_validity",
    "response_fidelity",
    "engineering_requirements",
    "final_verdict",
    "stress_margin_mpa",
    "normalized_displacement_margin",
    "displacement_ratio_margin",
    "compliance_ratio_margin",
    "stress_ratio_margin",
    "body_stress_concentration_ratio_margin",
    "response_fidelity_score",
    "reference_response_metrics",
    "candidate_response_metrics",
    "candidate_reference_force_ratio",
    "runtime_seconds",
)


def summarize_l3_run(
    *, plan_path: Path, run_root: Path, output_root: Path
) -> dict[str, Any]:
    plan = read_json(plan_path)
    release = validate_release(Path(plan["release_root"]), require_items=False)
    selected = release["selected"]
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for item in plan["items"]:
        base = {
            "item_id": item["item_id"],
            "protocol": item.get("protocol"),
            "source_key": selected[item["item_id"]]["source_key"],
            "difficulty": selected[item["item_id"]]["complexity"]["tier"],
            "scenario_family": (item.get("binding") or {}).get("scenario"),
            "evaluator_mode": (item.get("binding") or {}).get("evaluator_mode"),
            "l2_status": item["l2_status"],
            "l3_applicability": item["applicability"],
        }
        verdicts: list[str] = []
        if not item["variants"]:
            item_rows.append(
                {
                    **base,
                    "status": item["status"],
                    "reason": item.get("reason"),
                    "variant_count": 0,
                    "l3_scoreable_variant_count": 0,
                    "pass_count": 0,
                }
            )
            continue
        l3_scoreable_count = 0
        pass_count = 0
        for variant in item["variants"]:
            result_path = (
                run_root
                / "items"
                / item["item_id"]
                / "variants"
                / variant["variant_id"]
                / "pair_result.json"
            )
            result = read_json(result_path) if result_path.is_file() else None
            verdict = (
                str(result["final_verdict"])
                if result is not None
                else (
                    "not_reached"
                    if variant["physical_status"] != "passed"
                    else "evaluator_error"
                )
            )
            effective_scoreable = bool(
                variant["l3_scoreable"] and verdict != "reference_not_scoreable"
            )
            if effective_scoreable:
                l3_scoreable_count += 1
            verdicts.append(verdict)
            if effective_scoreable and verdict == "pass":
                pass_count += 1
            stages = (result or {}).get("stages") or {}
            margins = (result or {}).get("margins") or {}
            reference_metrics = ((result or {}).get("reference") or {}).get("metrics")
            candidate_metrics = ((result or {}).get("candidate") or {}).get("metrics")
            force_ratio = ((result or {}).get("load_scaling") or {}).get(
                "candidate_reference_resultant_force_ratio"
            )
            pairs.append(
                {
                    **base,
                    "physical_variant_id": variant["variant_id"],
                    "assignments": json.dumps(variant["assignments"], sort_keys=True),
                    "schedule_memberships": json.dumps(
                        [
                            {
                                "variant_id": membership["variant_id"],
                                "schedule_kind": membership["schedule_kind"],
                            }
                            for membership in variant["memberships"]
                        ],
                        sort_keys=True,
                    ),
                    "pair_selection_roles": json.dumps(
                        variant.get("pair_selection_roles") or [], sort_keys=True
                    ),
                    "comparison_mode": variant["comparison_mode"],
                    "planned_l3_scoreable": variant["l3_scoreable"],
                    "l3_scoreable": effective_scoreable,
                    "rebuild": stages.get("rebuild", "not_reached"),
                    "selectors": stages.get("selectors", "not_reached"),
                    "mesh": stages.get("mesh", "not_reached"),
                    "solve": stages.get("solve", "not_reached"),
                    "numerical_validity": stages.get(
                        "numerical_validity", "not_reached"
                    ),
                    "response_fidelity": stages.get("response_fidelity", "not_reached"),
                    "engineering_requirements": stages.get(
                        "engineering_requirements", "not_reached"
                    ),
                    "final_verdict": verdict,
                    "stress_margin_mpa": margins.get("stress_mpa"),
                    "normalized_displacement_margin": margins.get(
                        "normalized_displacement"
                    ),
                    "displacement_ratio_margin": margins.get("displacement_ratio"),
                    "compliance_ratio_margin": margins.get("compliance_ratio"),
                    "stress_ratio_margin": margins.get("stress_ratio"),
                    "body_stress_concentration_ratio_margin": margins.get(
                        "body_stress_concentration_ratio"
                    ),
                    "response_fidelity_score": margins.get("response_fidelity_score"),
                    "reference_response_metrics": (
                        json.dumps(reference_metrics, sort_keys=True)
                        if reference_metrics is not None
                        else None
                    ),
                    "candidate_response_metrics": (
                        json.dumps(candidate_metrics, sort_keys=True)
                        if candidate_metrics is not None
                        else None
                    ),
                    "candidate_reference_force_ratio": force_ratio,
                    "runtime_seconds": (result or {}).get("runtime_seconds"),
                }
            )
            for membership in variant["memberships"]:
                rows.append(
                    {
                        **base,
                        "l2_variant_id": membership["variant_id"],
                        "schedule_kind": membership["schedule_kind"],
                        "pair_selection_roles": json.dumps(
                            variant.get("pair_selection_roles") or [], sort_keys=True
                        ),
                        "assignments": json.dumps(
                            variant["assignments"], sort_keys=True
                        ),
                        "comparison_mode": variant["comparison_mode"],
                        "planned_l3_scoreable": variant["l3_scoreable"],
                        "l3_scoreable": effective_scoreable,
                        "rebuild": stages.get("rebuild", "not_reached"),
                        "selectors": stages.get("selectors", "not_reached"),
                        "mesh": stages.get("mesh", "not_reached"),
                        "solve": stages.get("solve", "not_reached"),
                        "numerical_validity": stages.get(
                            "numerical_validity", "not_reached"
                        ),
                        "response_fidelity": stages.get(
                            "response_fidelity", "not_reached"
                        ),
                        "engineering_requirements": stages.get(
                            "engineering_requirements", "not_reached"
                        ),
                        "final_verdict": verdict,
                        "stress_margin_mpa": margins.get("stress_mpa"),
                        "normalized_displacement_margin": margins.get(
                            "normalized_displacement"
                        ),
                        "displacement_ratio_margin": margins.get("displacement_ratio"),
                        "compliance_ratio_margin": margins.get("compliance_ratio"),
                        "stress_ratio_margin": margins.get("stress_ratio"),
                        "body_stress_concentration_ratio_margin": margins.get(
                            "body_stress_concentration_ratio"
                        ),
                        "response_fidelity_score": margins.get(
                            "response_fidelity_score"
                        ),
                        "reference_response_metrics": (
                            json.dumps(reference_metrics, sort_keys=True)
                            if reference_metrics is not None
                            else None
                        ),
                        "candidate_response_metrics": (
                            json.dumps(candidate_metrics, sort_keys=True)
                            if candidate_metrics is not None
                            else None
                        ),
                        "candidate_reference_force_ratio": force_ratio,
                        "runtime_seconds": (result or {}).get("runtime_seconds"),
                    }
                )
        item_rows.append(
            {
                **base,
                "status": (
                    "evaluator_error"
                    if any(value == "evaluator_error" for value in verdicts)
                    else "completed"
                    if l3_scoreable_count
                    else "not_reached"
                ),
                "reason": None,
                "variant_count": len(item["variants"]),
                "l3_scoreable_variant_count": l3_scoreable_count,
                "pass_count": pass_count,
            }
        )

    applicable_items = [
        item for item in item_rows if item["l3_applicability"] == "solver_qualified"
    ]
    scoreable_pairs = [pair for pair in pairs if pair["l3_scoreable"]]
    solved_pairs = [pair for pair in scoreable_pairs if pair["solve"] == "passed"]
    fidelity_values = [
        float(pair["response_fidelity_score"])
        for pair in solved_pairs
        if pair["response_fidelity_score"] is not None
    ]
    scoreable_families = [
        item for item in item_rows if item["l3_scoreable_variant_count"] > 0
    ]
    scenario_counts = Counter(
        pair["scenario_family"] for pair in scoreable_pairs if pair["scenario_family"]
    )
    scenario_pass_counts = Counter(
        pair["scenario_family"]
        for pair in scoreable_pairs
        if pair["scenario_family"] and pair["final_verdict"] == "pass"
    )
    stage_names = (
        "rebuild",
        "selectors",
        "mesh",
        "solve",
        "numerical_validity",
        "response_fidelity",
        "engineering_requirements",
    )

    def stage_rates(population: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            stage: {
                "reached_count": sum(
                    pair[stage] != "not_reached" for pair in population
                ),
                "reach_rate": (
                    sum(pair[stage] != "not_reached" for pair in population)
                    / len(population)
                    if population
                    else None
                ),
                "pass_count": sum(pair[stage] == "passed" for pair in population),
                "pass_rate": (
                    sum(pair[stage] == "passed" for pair in population)
                    / len(population)
                    if population
                    else None
                ),
            }
            for stage in stage_names
        }

    l3_stage_progress = (
        statistics.fmean(
            sum(pair[stage] == "passed" for stage in stage_names) / len(stage_names)
            for pair in scoreable_pairs
        )
        if scoreable_pairs
        else None
    )
    physical_status = "completed" if pairs else "not_reached"
    physical_reason = None if pairs else "no_physically_ready_zero_to_cad_pair"
    l3_status = "completed" if scoreable_pairs else "not_reached"
    l3_reason = None if scoreable_pairs else "no_physically_ready_zero_to_cad_pair"
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "dataset_version": plan["dataset_version"],
        "pair_selection": plan.get("pair_selection"),
        "scored_protocol": "zero_to_cad",
        "excluded_protocols": {
            "functional_edit": PROTOCOL_E_EXCLUSION,
        },
        "item_count": len(item_rows),
        "l3_applicable_item_count": len(applicable_items),
        "l3_not_applicable_item_count": len(item_rows) - len(applicable_items),
        "protocol_variant_row_count": len(rows),
        "physical_pair_count": len(pairs),
        "physical_diagnostic_status": physical_status,
        "physical_diagnostic_reason": physical_reason,
        "l3_scoreable_pair_count": len(scoreable_pairs),
        "l3_status": l3_status,
        "l3_reason": l3_reason,
        "l3_scoreable_variant_row_count": sum(
            row["l3_scoreable"] for row in rows
        ),
        "l3_stage_rates": stage_rates(scoreable_pairs),
        "diagnostic_physical_stage_rates": stage_rates(pairs),
        "l3_stage_progress_score": l3_stage_progress,
        "verdict_counts": dict(Counter(pair["final_verdict"] for pair in pairs)),
        "protocol_variant_verdict_counts": dict(
            Counter(row["final_verdict"] for row in rows)
        ),
        "solver_completion_rate": (
            sum(pair["solve"] == "passed" for pair in scoreable_pairs)
            / len(scoreable_pairs)
            if scoreable_pairs
            else None
        ),
        "engineering_requirement_pass_rate_on_solved": (
            sum(pair["engineering_requirements"] == "passed" for pair in solved_pairs)
            / len(solved_pairs)
            if solved_pairs
            else None
        ),
        "response_fidelity_pass_rate_on_solved": (
            sum(pair["response_fidelity"] == "passed" for pair in solved_pairs)
            / len(solved_pairs)
            if solved_pairs
            else None
        ),
        "l3_pass_rate": (
            sum(pair["final_verdict"] == "pass" for pair in scoreable_pairs)
            / len(scoreable_pairs)
            if scoreable_pairs
            else None
        ),
        "l3_family_pass_rate": (
            sum(
                item["pass_count"] == item["l3_scoreable_variant_count"]
                for item in scoreable_families
            )
            / len(scoreable_families)
            if scoreable_families
            else None
        ),
        "mean_response_fidelity_on_solved": (
            sum(fidelity_values) / len(fidelity_values) if fidelity_values else None
        ),
        "l3_pair_count_by_scenario": dict(scenario_counts),
        "l3_pass_count_by_scenario": dict(scenario_pass_counts),
        "l3_pass_rate_by_scenario": {
            scenario: scenario_pass_counts[scenario] / count
            for scenario, count in sorted(scenario_counts.items())
        },
        "items": item_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "summary.json", summary)
    with (output_root / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=_ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "pairs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pairs)
    with (output_root / "items.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(item_rows[0]) if item_rows else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item_rows)
    return summary
