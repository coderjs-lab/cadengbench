from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SELECTION_BUCKETS = (
    "pinslot",
    "ball",
    "planar",
    "cylindrical",
    "slider",
    "revolute_only",
    "rigid_only",
    "mixed_joint_types",
    "high_complexity",
)
JOINT_TYPES = (
    "RigidJointType",
    "RevoluteJointType",
    "SliderJointType",
    "CylindricalJointType",
    "PlanarJointType",
    "BallJointType",
    "PinSlotJointType",
)
COMPLEXITY_TARGETS = {
    "single_joint": 8,
    "2_to_4_joints": 7,
    "5_to_10_joints": 3,
    "11_plus_joints": 2,
}
LOSSINESS_TARGETS = {
    "no_loss": 4,
    "likely_redundant_loss": 3,
    "unsupported_by_urdf": 4,
    "potentially_meaningful_loss": 4,
}
FACE_BIN_TARGETS = {"low": 5, "medium": 7, "high": 4}
FORBIDDEN_PUBLIC_KEYS = {
    "joint_count",
    "joint_types",
    "fusion_joint_type",
    "joint_axis",
    "selected_entities",
    "equivalent_entities",
    "lossiness_label",
    "urdf",
    "pybullet",
    "ground_truth",
    "master_json",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_pool(
    selection: dict[str, Any],
    split: dict[str, Any],
    lossiness: dict[str, Any],
    raw_root: str | Path,
) -> list[dict[str, Any]]:
    raw_root = Path(raw_root).expanduser()
    validation_ids = set(split["splits"]["validation"]["assembly_ids"])
    lossiness_by_id = {row["assembly_id"]: row for row in lossiness["assemblies"]}
    pool = []
    for source in selection["assemblies"]:
        if source["assembly_id"] not in validation_ids:
            continue
        face_count = 0
        for body_id in source["body_ids"]:
            graph = read_json(raw_root / f"{body_id}.json")
            face_count += int((graph.get("properties") or {}).get("face_count", 0))
        loss = lossiness_by_id[source["assembly_id"]]
        pool.append(
            {
                "assembly_id": source["assembly_id"],
                "body_ids": list(source["body_ids"]),
                "selection_bucket": source["selection_bucket"],
                "complexity_bin": source["complexity_bin"],
                "joint_count": int(source["joint_count"]),
                "joint_types": list(source["joint_types"]),
                "unique_joint_types": sorted(set(source["joint_types"])),
                "lossiness_label": loss["lossiness_label"],
                "total_face_count": face_count,
                "has_repeated_constraints": int(source["joint_count"]) > 1,
            }
        )
    if len(pool) != 151:
        raise ValueError(f"expected 151 validation records, found {len(pool)}")
    _assign_face_bins(pool)
    return sorted(pool, key=lambda row: row["assembly_id"])


def select_dev20(pool: list[dict[str, Any]], seed: int = 392) -> dict[str, Any]:
    if len({row["assembly_id"] for row in pool}) != len(pool):
        raise ValueError("validation pool assembly IDs must be unique")
    by_bucket = Counter(row["selection_bucket"] for row in pool)
    missing = [bucket for bucket in SELECTION_BUCKETS if by_bucket[bucket] < 2]
    if missing:
        raise ValueError(f"selection buckets lack two candidates: {missing}")

    selected: list[dict[str, Any]] = []
    for bucket in SELECTION_BUCKETS:
        for _ in range(2):
            candidates = [
                row
                for row in pool
                if row["selection_bucket"] == bucket and row not in selected
            ]
            selected.append(_best_candidate(candidates, selected, seed))
    while len(selected) < 20:
        selected.append(_best_candidate([row for row in pool if row not in selected], selected, seed))

    selected = _local_improvement(selected, pool, seed)
    coverage = summarize_coverage(selected)
    return {
        "schema": "cadengbench_a_dev20_selection_v1",
        "seed": seed,
        "algorithm": "weighted_deficit_greedy_with_deterministic_single_swap_improvement",
        "source_split": "validation",
        "selected_count": len(selected),
        "selected": [
            {"selection_rank": index + 1, **row} for index, row in enumerate(selected)
        ],
        "coverage": coverage,
        "targets": {
            "selection_buckets": {bucket: 2 for bucket in SELECTION_BUCKETS},
            "joint_types": {joint_type: 2 for joint_type in JOINT_TYPES},
            "complexity": COMPLEXITY_TARGETS,
            "lossiness": LOSSINESS_TARGETS,
            "repeated_constraints": 6,
            "face_bins": FACE_BIN_TARGETS,
        },
        "target_deficits": _coverage_deficits(coverage),
    }


def summarize_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "selection_buckets": dict(sorted(Counter(row["selection_bucket"] for row in rows).items())),
        "joint_types": dict(
            sorted(Counter(joint_type for row in rows for joint_type in row["unique_joint_types"]).items())
        ),
        "complexity": dict(sorted(Counter(row["complexity_bin"] for row in rows).items())),
        "lossiness": dict(sorted(Counter(row["lossiness_label"] for row in rows).items())),
        "repeated_constraints": sum(bool(row["has_repeated_constraints"]) for row in rows),
        "face_bins": dict(sorted(Counter(row["face_count_bin"] for row in rows).items())),
    }


def build_public_package(
    assembly_id: str,
    body_ids: list[str],
    raw_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    if len(body_ids) != 2:
        raise ValueError("pair package requires exactly two body IDs")
    raw_root = Path(raw_root).expanduser()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roles = ("body_a", "body_b")
    body_geometry = []
    candidate_bodies = []
    provenance_bodies = []
    for role, body_id in zip(roles, body_ids):
        graph_path = raw_root / f"{body_id}.json"
        obj_path = raw_root / f"{body_id}.obj"
        step_path = raw_root / f"{body_id}.step"
        for source_path in (graph_path, obj_path, step_path):
            if not source_path.exists():
                raise FileNotFoundError(source_path)
        graph = read_json(graph_path)
        candidates = extract_body_candidates(graph, role, body_id)
        body_geometry.append(_public_body_geometry(graph, role, body_id))
        candidate_bodies.append(
            {
                "body_role": role,
                "body_id": body_id,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )
        geometry_dir = output_dir / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        copied = {}
        for kind, source_path in (("obj", obj_path), ("step", step_path)):
            destination = geometry_dir / f"{role}.{kind}"
            shutil.copy2(source_path, destination)
            copied[kind] = {"path": str(destination.relative_to(output_dir)), "sha256": sha256(destination)}
        provenance_bodies.append(
            {
                "body_role": role,
                "body_id": body_id,
                "source_files": {
                    "graph_json": {"filename": graph_path.name, "sha256": sha256(graph_path)},
                    "obj": {"filename": obj_path.name, "sha256": sha256(obj_path)},
                    "step": {"filename": step_path.name, "sha256": sha256(step_path)},
                },
                "packaged_files": copied,
            }
        )

    task = {
        "schema": "cadengbench_a_pair_track_b_task_v1",
        "benchmark": "CADEngBench-A",
        "assembly_id": assembly_id,
        "bodies": [
            {"body_role": role, "body_id": body_id} for role, body_id in zip(roles, body_ids)
        ],
        "allowed_joint_types": list(JOINT_TYPES),
        "units": "source_cad_units",
        "coordinate_frame": "body-local source coordinates; no assembly transform is provided",
        "assets": {
            "public_geometry": "public_geometry.json",
            "candidates": "candidates.json",
            "body_a_obj": "geometry/body_a.obj",
            "body_b_obj": "geometry/body_b.obj",
            "body_a_step": "geometry/body_a.step",
            "body_b_step": "geometry/body_b.step",
        },
        "views_status": "pending",
        "output_schema": "../../prompts/pair_track_b_v1/output_schema.json",
    }
    public_geometry = {
        "schema": "cadengbench_a_public_geometry_v1",
        "assembly_id": assembly_id,
        "bodies": body_geometry,
    }
    candidate_registry = {
        "schema": "cadengbench_a_candidate_registry_v1",
        "assembly_id": assembly_id,
        "derivation": "all BRep face and edge nodes in source body-graph order; no joint labels consulted",
        "bodies": candidate_bodies,
    }
    provenance = {
        "schema": "cadengbench_a_public_package_provenance_v1",
        "assembly_id": assembly_id,
        "authority": {"body_graph": "S0_SOURCE", "candidates": "D1_DERIVED"},
        "bodies": provenance_bodies,
    }
    audit = {
        "schema": "cadengbench_a_public_package_audit_v1",
        "assembly_id": assembly_id,
        "automated": {
            "candidate_ids_unique": _candidate_ids_unique(candidate_bodies),
            "candidate_values_finite": _all_finite(candidate_registry),
            "forbidden_public_keys_absent": not find_forbidden_public_keys(
                {"task": task, "public_geometry": public_geometry, "candidates": candidate_registry}
            ),
            "absolute_paths_absent": not _contains_absolute_path(
                {"task": task, "public_geometry": public_geometry, "candidates": candidate_registry, "provenance": provenance}
            ),
        },
        "views": "pending",
        "blind_human_review": "pending",
        "ground_truth_coverage_review": "pending",
        "final_status": "pending",
    }
    write_json(output_dir / "task.json", task)
    write_json(output_dir / "public_geometry.json", public_geometry)
    write_json(output_dir / "candidates.json", candidate_registry)
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "audit.json", audit)
    if not all(audit["automated"].values()):
        raise ValueError(f"public package integrity failure for {assembly_id}: {audit['automated']}")
    return audit


def render_public_views(item_dir: str | Path, atlas_page_size: int = 20) -> dict[str, Any]:
    item_dir = Path(item_dir)
    task = read_json(item_dir / "task.json")
    public_geometry = read_json(item_dir / "public_geometry.json")
    registry = read_json(item_dir / "candidates.json")
    provenance = read_json(item_dir / "provenance.json")
    audit = read_json(item_dir / "audit.json")
    views_dir = item_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    views_dir.mkdir(parents=True)

    view_configs = {
        "iso": {"elevation_deg": 25.0, "azimuth_deg": -45.0, "projection": "orthographic"},
        "front": {"elevation_deg": 0.0, "azimuth_deg": -90.0, "projection": "orthographic"},
        "side": {"elevation_deg": 0.0, "azimuth_deg": 0.0, "projection": "orthographic"},
        "top": {"elevation_deg": 90.0, "azimuth_deg": -90.0, "projection": "orthographic"},
    }
    task_views: dict[str, list[str]] = {}
    view_hashes = {}
    for body in registry["bodies"]:
        role = body["body_role"]
        vertices, faces, face_entity_indices = _read_obj_mesh_with_face_groups(
            item_dir / "geometry" / f"{role}.obj"
        )
        color = "#55c7c2" if role == "body_a" else "#e4a255"
        role_views = []
        for view_name, config in view_configs.items():
            relative = Path("views") / f"{role}_{view_name}.png"
            _render_mesh_view(
                vertices,
                faces,
                item_dir / relative,
                color=color,
                elevation=config["elevation_deg"],
                azimuth=config["azimuth_deg"],
                face_entity_indices=face_entity_indices,
            )
            role_views.append(str(relative))
            view_hashes[str(relative)] = sha256(item_dir / relative)

        eligible = [candidate for candidate in body["candidates"] if candidate["prompt_eligible"]]
        atlas_paths = []
        for page_index, start in enumerate(range(0, len(eligible), atlas_page_size), start=1):
            page_candidates = eligible[start : start + atlas_page_size]
            relative = Path("views") / f"{role}_entity_atlas_{page_index:02d}.png"
            for candidate in page_candidates:
                candidate["atlas_page"] = page_index
            _render_mesh_view(
                vertices,
                faces,
                item_dir / relative,
                color=color,
                elevation=view_configs["iso"]["elevation_deg"],
                azimuth=view_configs["iso"]["azimuth_deg"],
                candidates=page_candidates,
                face_entity_indices=face_entity_indices,
                title=f"{role} candidates {start + 1}-{start + len(page_candidates)}",
            )
            atlas_paths.append(str(relative))
            view_hashes[str(relative)] = sha256(item_dir / relative)
        body["prompt_visible_candidate_count"] = len(eligible)
        body["atlas_page_count"] = len(atlas_paths)
        task_views[role] = role_views + atlas_paths

    task["assets"]["views"] = task_views
    task["views_status"] = "generated"
    task["atlas_rendering"] = {
        "version": "candidate_face_highlight_v2",
        "face_candidates": "matching legend color highlights the OBJ face group",
        "edge_candidates": "X marker shows the candidate key point; OBJ edge groups are unavailable",
    }
    public_geometry["camera_views"] = view_configs
    provenance["generated_views"] = view_hashes
    provenance["view_renderer_version"] = "candidate_face_highlight_v2"
    audit["automated"]["all_referenced_views_exist"] = all(
        (item_dir / relative).exists() for paths in task_views.values() for relative in paths
    )
    audit["views"] = "generated_pending_human_review"
    write_json(item_dir / "task.json", task)
    write_json(item_dir / "public_geometry.json", public_geometry)
    write_json(item_dir / "candidates.json", registry)
    write_json(item_dir / "provenance.json", provenance)
    write_json(item_dir / "audit.json", audit)
    return {
        "assembly_id": task["assembly_id"],
        "view_count": len(view_hashes),
        "body_view_counts": {role: len(paths) for role, paths in task_views.items()},
    }


def extract_body_candidates(
    graph: dict[str, Any], body_role: str, body_id: str
) -> list[dict[str, Any]]:
    nodes = graph.get("nodes") or []
    faces = [node for node in nodes if node.get("surface_type")]
    edges = [node for node in nodes if node.get("curve_type")]
    node_to_candidate = {}
    candidates = []
    for entity_type, family, label in (("BRepFace", faces, "face"), ("BRepEdge", edges, "edge")):
        for entity_index, node in enumerate(family):
            candidate_id = f"{body_role}_{label}_{entity_index:04d}"
            node_to_candidate[node["id"]] = candidate_id
            candidates.append(
                _candidate_record(node, candidate_id, body_role, body_id, entity_type, entity_index)
            )
    adjacency: dict[str, set[str]] = {candidate["candidate_id"]: set() for candidate in candidates}
    for link in graph.get("links") or []:
        source = node_to_candidate.get(link.get("source"))
        target = node_to_candidate.get(link.get("target"))
        if source and target and source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)
    for candidate in candidates:
        candidate["adjacent_candidate_ids"] = sorted(adjacency[candidate["candidate_id"]])
    return candidates


def find_forbidden_public_keys(value: Any) -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_PUBLIC_KEYS:
                found.append(key)
            found.extend(find_forbidden_public_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_forbidden_public_keys(child))
    return found


def audit_candidate_coverage(
    master: dict[str, Any], candidate_registry: dict[str, Any]
) -> dict[str, Any]:
    lookup = {
        (candidate["body_id"], candidate["entity_type"], candidate["entity_index"]): candidate
        for body in candidate_registry["bodies"]
        for candidate in body["candidates"]
    }
    side_results = []
    joint_results = []
    for joint in master.get("joints", []):
        joint_sides = []
        for side, geometry_key in (("body_a", "geometry_or_origin_one"), ("body_b", "geometry_or_origin_two")):
            geometry = joint.get(geometry_key) or {}
            primary = geometry.get("entity_one") or {}
            equivalents = geometry.get("entity_one_equivalents") or []
            primary_candidate = _lookup_entity_candidate(lookup, primary)
            equivalent_candidates = [
                candidate
                for entity in equivalents
                if (candidate := _lookup_entity_candidate(lookup, {**entity, "body": primary.get("body")}))
            ]
            selected = primary_candidate or (equivalent_candidates[0] if equivalent_candidates else None)
            expected_type = primary.get("surface_type") or primary.get("curve_type")
            side_result = {
                "joint_id": joint.get("id"),
                "joint_type": joint.get("fusion_joint_type"),
                "side": side,
                "body_id": primary.get("body"),
                "primary_reference": {
                    "entity_type": primary.get("type"),
                    "entity_index": primary.get("index"),
                    "analytic_type": expected_type,
                },
                "primary_candidate_id": primary_candidate.get("candidate_id") if primary_candidate else None,
                "primary_analytic_type_match": bool(
                    primary_candidate and primary_candidate.get("analytic_type") == expected_type
                ),
                "equivalent_candidate_ids": [candidate["candidate_id"] for candidate in equivalent_candidates],
                "primary_or_equivalent_covered": selected is not None,
                "selected_candidate_has_direction": bool(selected and selected.get("direction")),
            }
            side_results.append(side_result)
            joint_sides.append(side_result)
        joint_results.append(
            {
                "joint_id": joint.get("id"),
                "joint_type": joint.get("fusion_joint_type"),
                "both_sides_covered": all(side["primary_or_equivalent_covered"] for side in joint_sides),
                "both_sides_directional": all(side["selected_candidate_has_direction"] for side in joint_sides),
            }
        )
    return {
        "assembly_id": master.get("joint_set_id"),
        "joint_count": len(joint_results),
        "side_count": len(side_results),
        "primary_reference_coverage": _ratio(
            sum(result["primary_candidate_id"] is not None for result in side_results), len(side_results)
        ),
        "primary_analytic_type_match_rate": _ratio(
            sum(result["primary_analytic_type_match"] for result in side_results), len(side_results)
        ),
        "primary_or_equivalent_coverage": _ratio(
            sum(result["primary_or_equivalent_covered"] for result in side_results), len(side_results)
        ),
        "joint_both_sides_coverage": _ratio(
            sum(result["both_sides_covered"] for result in joint_results), len(joint_results)
        ),
        "joint_both_sides_directional_rate": _ratio(
            sum(result["both_sides_directional"] for result in joint_results), len(joint_results)
        ),
        "sides": side_results,
        "joints": joint_results,
    }


def adapt_track_b_prediction(
    task: dict[str, Any],
    candidate_registry: dict[str, Any],
    model_prediction: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = []
    assembly_id = task["assembly_id"]
    if model_prediction.get("assembly_id") != assembly_id:
        errors.append("assembly_id does not match task")
    body_by_role = {body["body_role"]: body["body_id"] for body in task["bodies"]}
    candidates = {
        candidate["candidate_id"]: candidate
        for body in candidate_registry["bodies"]
        for candidate in body["candidates"]
    }
    raw_joints = model_prediction.get("predicted_joints")
    if not isinstance(raw_joints, list):
        errors.append("predicted_joints must be a list")
        raw_joints = []
    adapted_joints = []
    joint_reports = []
    for index, raw in enumerate(raw_joints):
        if not isinstance(raw, dict):
            errors.append(f"predicted_joints[{index}] must be an object")
            continue
        joint_type = raw.get("joint_type")
        if joint_type not in JOINT_TYPES:
            errors.append(f"predicted_joints[{index}] has unknown joint_type")
        candidate_a = candidates.get(raw.get("body_a_candidate_id"))
        candidate_b = candidates.get(raw.get("body_b_candidate_id"))
        if not candidate_a or candidate_a.get("body_role") != "body_a":
            errors.append(f"predicted_joints[{index}] body_a candidate is invalid")
        if not candidate_b or candidate_b.get("body_role") != "body_b":
            errors.append(f"predicted_joints[{index}] body_b candidate is invalid")
        if not candidate_a or not candidate_b or joint_type not in JOINT_TYPES:
            continue
        axis_status = (
            "unresolved_no_shared_assembly_frame"
            if joint_type in {
                "RevoluteJointType",
                "SliderJointType",
                "CylindricalJointType",
                "PlanarJointType",
                "PinSlotJointType",
            }
            else "not_applicable_single_axis"
        )
        adapted_joints.append(
            {
                "body_a": body_by_role["body_a"],
                "body_b": body_by_role["body_b"],
                "joint_type": joint_type,
                "axis": None,
                "axis_status": axis_status,
                "selected_entities": {
                    "body_a": {
                        "type": candidate_a["entity_type"],
                        "index": candidate_a["entity_index"],
                    },
                    "body_b": {
                        "type": candidate_b["entity_type"],
                        "index": candidate_b["entity_index"],
                    },
                },
                "confidence": raw.get("confidence"),
            }
        )
        joint_reports.append(
            {
                "prediction_index": index,
                "body_a_candidate_id": candidate_a["candidate_id"],
                "body_b_candidate_id": candidate_b["candidate_id"],
                "axis_status": axis_status,
                "axis_explanation": "Body candidates are in separate local frames; the assembled transforms are hidden joint labels.",
            }
        )
    adapted = {
        "schema": "cadengbench_a_pair_prediction_v1",
        "assembly_id": assembly_id,
        "predicted_joints": adapted_joints,
    }
    report = {
        "schema": "cadengbench_a_track_b_adapter_report_v1",
        "assembly_id": assembly_id,
        "valid": not errors,
        "errors": errors,
        "input_joint_count": len(raw_joints),
        "adapted_joint_count": len(adapted_joints),
        "joints": joint_reports,
        "ground_truth_accessed": False,
    }
    return adapted, report


def _candidate_record(
    node: dict[str, Any],
    candidate_id: str,
    body_role: str,
    body_id: str,
    entity_type: str,
    entity_index: int,
) -> dict[str, Any]:
    analytic_type = node.get("surface_type") or node.get("curve_type")
    point = _node_point(node, entity_type)
    direction, direction_kind = _node_direction(node, entity_type)
    record = {
        "candidate_id": candidate_id,
        "body_role": body_role,
        "body_id": body_id,
        "entity_type": entity_type,
        "entity_index": entity_index,
        "analytic_type": analytic_type,
        "key_point": point,
        "direction": direction,
        "direction_kind": direction_kind,
        "radius": _finite_or_none(node.get("radius")),
        "area": _finite_or_none(node.get("area")),
        "length": _finite_or_none(node.get("length")),
        "derivation": "source_body_graph_node",
        "authority": "D1_DERIVED",
        "display_label": ("F" if entity_type == "BRepFace" else "E") + f"{entity_index:04d}",
        "prompt_eligible": point is not None,
    }
    return record


def _lookup_entity_candidate(
    lookup: dict[tuple[Any, Any, Any], dict[str, Any]], entity: dict[str, Any]
) -> dict[str, Any] | None:
    return lookup.get((entity.get("body"), entity.get("type"), entity.get("index")))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _node_point(node: dict[str, Any], entity_type: str) -> list[float] | None:
    prefixes = ("centroid", "point_on_face", "origin") if entity_type == "BRepFace" else ("center", "start_point")
    for prefix in prefixes:
        point = [node.get(f"{prefix}_{axis}") for axis in "xyz"]
        if all(_is_finite(value) for value in point):
            return [float(value) for value in point]
    return None


def _node_direction(node: dict[str, Any], entity_type: str) -> tuple[list[float] | None, str | None]:
    if entity_type == "BRepFace":
        choices = (("axis", "analytic_axis"), ("normal", "surface_normal"))
    elif node.get("curve_type") in {"Circle3DCurveType", "Arc3DCurveType", "Ellipse3DCurveType"}:
        choices = (("normal", "curve_plane_normal"),)
    else:
        start = [node.get(f"start_point_{axis}") for axis in "xyz"]
        end = [node.get(f"end_point_{axis}") for axis in "xyz"]
        if all(_is_finite(value) for value in start + end):
            vector = [float(b) - float(a) for a, b in zip(start, end)]
            normalized = _normalize(vector)
            if normalized is not None:
                return normalized, "line_direction"
        choices = (("tangent", "curve_tangent"),)
    for prefix, kind in choices:
        vector = [node.get(f"{prefix}_{axis}") for axis in "xyz"]
        if all(_is_finite(value) for value in vector):
            normalized = _normalize([float(value) for value in vector])
            if normalized is not None:
                return normalized, kind
    return None, None


def _public_body_geometry(graph: dict[str, Any], body_role: str, body_id: str) -> dict[str, Any]:
    properties = graph.get("properties") or {}
    bounding_box = properties.get("bounding_box") or {}
    minimum = bounding_box.get("min_point") or {}
    maximum = bounding_box.get("max_point") or {}
    min_xyz = [minimum.get(axis) for axis in "xyz"]
    max_xyz = [maximum.get(axis) for axis in "xyz"]
    extents = [float(high) - float(low) for low, high in zip(min_xyz, max_xyz)]
    return {
        "body_role": body_role,
        "body_id": body_id,
        "bounding_box": {"min": min_xyz, "max": max_xyz, "extents": extents},
        "topology": {
            key: int(properties.get(key, 0))
            for key in ("body_count", "shell_count", "face_count", "edge_count", "vertex_count")
        },
        "area": _finite_or_none(properties.get("area")),
        "volume": _finite_or_none(properties.get("volume")),
        "body_to_normalized_transform": _normalization_transform(min_xyz, max_xyz),
        "normalized_to_body_transform": _inverse_normalization_transform(min_xyz, max_xyz),
    }


def _normalization_transform(minimum: list[float], maximum: list[float]) -> list[list[float]]:
    center = [(float(low) + float(high)) / 2.0 for low, high in zip(minimum, maximum)]
    scale = max(float(high) - float(low) for low, high in zip(minimum, maximum)) or 1.0
    return [
        [1.0 / scale, 0.0, 0.0, -center[0] / scale],
        [0.0, 1.0 / scale, 0.0, -center[1] / scale],
        [0.0, 0.0, 1.0 / scale, -center[2] / scale],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _inverse_normalization_transform(minimum: list[float], maximum: list[float]) -> list[list[float]]:
    center = [(float(low) + float(high)) / 2.0 for low, high in zip(minimum, maximum)]
    scale = max(float(high) - float(low) for low, high in zip(minimum, maximum)) or 1.0
    return [
        [scale, 0.0, 0.0, center[0]],
        [0.0, scale, 0.0, center[1]],
        [0.0, 0.0, scale, center[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _assign_face_bins(pool: list[dict[str, Any]]) -> None:
    ordered = sorted(row["total_face_count"] for row in pool)
    low_threshold = ordered[len(ordered) // 3]
    high_threshold = ordered[(2 * len(ordered)) // 3]
    for row in pool:
        count = row["total_face_count"]
        row["face_count_bin"] = "low" if count <= low_threshold else "medium" if count <= high_threshold else "high"


def _best_candidate(
    candidates: list[dict[str, Any]], selected: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidate set is empty")
    before = _selection_penalty(selected)
    return min(
        candidates,
        key=lambda row: (
            _selection_penalty(selected + [row]) - before,
            _tie_break(seed, row["assembly_id"]),
        ),
    )


def _local_improvement(
    selected: list[dict[str, Any]], pool: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    selected = list(selected)
    for _ in range(100):
        current_penalty = _selection_penalty(selected)
        best = None
        selected_ids = {row["assembly_id"] for row in selected}
        for remove_index, removed in enumerate(selected):
            for added in pool:
                if added["assembly_id"] in selected_ids:
                    continue
                proposal = selected[:remove_index] + [added] + selected[remove_index + 1 :]
                if any(
                    sum(row["selection_bucket"] == bucket for row in proposal) < 2
                    for bucket in SELECTION_BUCKETS
                ):
                    continue
                penalty = _selection_penalty(proposal)
                key = (penalty, _tie_break(seed, added["assembly_id"]), remove_index)
                if penalty < current_penalty and (best is None or key < best[0]):
                    best = (key, proposal)
        if best is None:
            break
        selected = best[1]
    return selected


def _selection_penalty(rows: list[dict[str, Any]]) -> int:
    coverage = summarize_coverage(rows)
    penalty = 0
    for bucket in SELECTION_BUCKETS:
        penalty += 1000 * max(0, 2 - coverage["selection_buckets"].get(bucket, 0)) ** 2
    for joint_type in JOINT_TYPES:
        penalty += 100 * max(0, 2 - coverage["joint_types"].get(joint_type, 0)) ** 2
    for key, target in COMPLEXITY_TARGETS.items():
        penalty += 20 * max(0, target - coverage["complexity"].get(key, 0)) ** 2
    for key, target in LOSSINESS_TARGETS.items():
        penalty += 20 * max(0, target - coverage["lossiness"].get(key, 0)) ** 2
    penalty += 15 * max(0, 6 - coverage["repeated_constraints"]) ** 2
    for key, target in FACE_BIN_TARGETS.items():
        penalty += 5 * max(0, target - coverage["face_bins"].get(key, 0)) ** 2
    return penalty


def _coverage_deficits(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_buckets": {
            key: max(0, 2 - coverage["selection_buckets"].get(key, 0)) for key in SELECTION_BUCKETS
        },
        "joint_types": {
            key: max(0, 2 - coverage["joint_types"].get(key, 0)) for key in JOINT_TYPES
        },
        "complexity": {
            key: max(0, target - coverage["complexity"].get(key, 0))
            for key, target in COMPLEXITY_TARGETS.items()
        },
        "lossiness": {
            key: max(0, target - coverage["lossiness"].get(key, 0))
            for key, target in LOSSINESS_TARGETS.items()
        },
        "repeated_constraints": max(0, 6 - coverage["repeated_constraints"]),
        "face_bins": {
            key: max(0, target - coverage["face_bins"].get(key, 0))
            for key, target in FACE_BIN_TARGETS.items()
        },
    }


def _tie_break(seed: int, assembly_id: str) -> str:
    return hashlib.sha256(f"{seed}:{assembly_id}".encode()).hexdigest()


def _candidate_ids_unique(candidate_bodies: list[dict[str, Any]]) -> bool:
    ids = [candidate["candidate_id"] for body in candidate_bodies for candidate in body["candidates"]]
    return len(ids) == len(set(ids))


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(child) for child in value)
    return isinstance(value, str) and value.startswith("/")


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_all_finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def _finite_or_none(value: Any) -> float | None:
    return float(value) if _is_finite(value) else None


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalize(vector: list[float]) -> list[float] | None:
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector] if length > 1e-12 else None


def _read_obj_mesh_with_face_groups(
    path: Path,
) -> tuple[list[list[float]], list[list[int]], list[int | None]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_entity_indices: list[int | None] = []
    current_face_entity_index: int | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            values = line.split()[1:4]
            if len(values) == 3:
                vertices.append([float(value) for value in values])
        elif line.startswith("g "):
            tokens = line.split()
            current_face_entity_index = (
                int(tokens[2])
                if len(tokens) == 3 and tokens[1] == "face" and tokens[2].isdigit()
                else None
            )
        elif line.startswith("f "):
            indices = []
            for token in line.split()[1:]:
                raw = int(token.split("/")[0])
                indices.append(raw - 1 if raw > 0 else len(vertices) + raw)
            if len(indices) >= 3 and all(0 <= index < len(vertices) for index in indices):
                faces.append(indices)
                face_entity_indices.append(current_face_entity_index)
    if not vertices or not faces:
        raise ValueError(f"OBJ does not contain a renderable mesh: {path}")
    return vertices, faces, face_entity_indices


def _read_obj_mesh(path: Path) -> tuple[list[list[float]], list[list[int]]]:
    vertices, faces, _ = _read_obj_mesh_with_face_groups(path)
    return vertices, faces


def _render_mesh_view(
    vertices: list[list[float]],
    faces: list[list[int]],
    output_path: Path,
    *,
    color: str,
    elevation: float,
    azimuth: float,
    candidates: list[dict[str, Any]] | None = None,
    face_entity_indices: list[int | None] | None = None,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertex_array = np.asarray(vertices, dtype=float)
    stride = max(1, math.ceil(len(faces) / 12000))
    sampled_faces = faces[::stride]
    sampled_groups = (face_entity_indices or [None] * len(faces))[::stride]
    polygons = [vertex_array[face] for face in sampled_faces]
    figure = plt.figure(
        figsize=(12.0, 7.68) if candidates else (7.68, 7.68),
        dpi=100,
        facecolor="white",
    )
    axis = figure.add_axes(
        [0.01, 0.02, 0.64, 0.91] if candidates else [0.01, 0.01, 0.98, 0.95],
        projection="3d",
    )
    palette = list(plt.get_cmap("tab20").colors)
    face_color_by_entity = {
        int(candidate["entity_index"]): palette[(local_index - 1) % len(palette)]
        for local_index, candidate in enumerate(candidates or [], start=1)
        if candidate.get("entity_type") == "BRepFace"
    }
    polygon_colors = [
        to_rgba(
            face_color_by_entity.get(group, color),
            1.0 if group in face_color_by_entity else (0.32 if candidates else 1.0),
        )
        for group in sampled_groups
    ]
    collection = Poly3DCollection(
        polygons,
        facecolor=polygon_colors,
        edgecolor="#27323a",
        linewidth=0.18,
    )
    axis.add_collection3d(collection)

    minimum = vertex_array.min(axis=0)
    maximum = vertex_array.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = float(max(maximum - minimum)) or 1.0
    half = span * 0.51
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()

    triad_origin = center - span * 0.43
    triad_length = span * 0.12
    for direction, triad_color, label in (
        ([1, 0, 0], "#d33f49", "X"),
        ([0, 1, 0], "#2b9348", "Y"),
        ([0, 0, 1], "#3066be", "Z"),
    ):
        axis.quiver(*triad_origin, *direction, length=triad_length, color=triad_color, linewidth=1.6)
        endpoint = triad_origin + np.asarray(direction) * triad_length * 1.15
        axis.text(*endpoint, label, color=triad_color, fontsize=8, weight="bold")
    scale_start = np.array([center[0] - span * 0.42, center[1] + span * 0.42, center[2] - span * 0.42])
    scale_end = scale_start + np.array([span * 0.2, 0.0, 0.0])
    axis.plot(*zip(scale_start, scale_end), color="#111111", linewidth=2.0)
    axis.text(*scale_start, f"{span * 0.2:.3g} units", fontsize=7, color="#111111")

    if candidates:
        figure.text(
            0.67,
            0.955,
            "Faces: matching color highlight. Edges: X marker at key point.",
            fontsize=8,
            color="#374151",
        )
    for local_index, candidate in enumerate(candidates or [], start=1):
        point = candidate.get("key_point")
        candidate_color = palette[(local_index - 1) % len(palette)]
        if not point:
            continue
        marker = "o" if candidate["entity_type"] == "BRepFace" else "X"
        axis.scatter(*point, color=candidate_color, marker=marker, s=30, depthshade=False)
        axis.text(*point, str(local_index), fontsize=6, color="#111111", weight="bold")
        analytic = str(candidate.get("analytic_type") or "unknown").replace("SurfaceType", "").replace("3DCurveType", "")
        figure.text(
            0.67,
            0.915 - (local_index - 1) * 0.041,
            f"{local_index:02d}  {candidate['candidate_id']}  {analytic}",
            fontsize=7.5,
            color=candidate_color,
            family="monospace",
        )
    if title:
        axis.set_title(title, fontsize=10, pad=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor="white")
    plt.close(figure)
