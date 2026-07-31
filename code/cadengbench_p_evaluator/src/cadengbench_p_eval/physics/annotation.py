"""Separate, auditable L3 structural-probe annotation workflow."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema

from cadengbench_p_eval.io import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)

ANNOTATION_SCHEMA = "cadengbench-p.l3-annotation-candidate.v1"
CARD_SCHEMA = "cadengbench-p.l3-physics-card.v2"
CORPUS_SCHEMA = "cadengbench-p.l3-annotation-corpus.v1"
ROLE_ID = "l3_physics_probe_annotator"
AXES = ("x", "y", "z")


class L3AnnotationError(RuntimeError):
    """The L3 annotation corpus or provider response is invalid."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _repo_path(repo_root: Path, raw: str, *, require_file: bool = False) -> Path:
    root = repo_root.resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise L3AnnotationError(f"configured path escapes repository: {raw}") from exc
    if require_file and not path.is_file():
        raise L3AnnotationError(f"configured file is missing: {path}")
    return path


def _load_profile(repo_root: Path, profile_path: Path) -> dict[str, Any]:
    profile = read_json(profile_path)
    if profile.get("schema_version") != "cadengbench-p.l3-probe-profile.v1":
        raise L3AnnotationError("unsupported L3 probe profile")
    configured = _repo_path(
        repo_root,
        str(profile["annotation"]["output_schema_path"]),
        require_file=True,
    )
    jsonschema.Draft202012Validator.check_schema(read_json(configured))
    return profile


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _source_step(item_root: Path) -> Path:
    candidates = (
        item_root / "source/generated_from_cadquery.step",
        item_root / "source/fusion360_occ_reconstruction.step",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise L3AnnotationError(f"item has no deterministic source STEP: {item_root.name}")


def _source_family(item_id: str) -> str:
    if item_id.startswith("p_benchcad_"):
        return "benchcad"
    if item_id.startswith("p_fusion360_"):
        return "fusion360_reconstruction"
    raise L3AnnotationError(f"unsupported item ID: {item_id}")


def _vector(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _face_inventory(step_path: Path, eligibility: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import cadquery as cq
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise L3AnnotationError(
            "CadQuery is required for L3 context preparation"
        ) from exc

    workplane = cq.importers.importStep(str(step_path))
    solids = workplane.solids().vals()
    faces = workplane.faces().vals()
    if not solids:
        raise L3AnnotationError(f"STEP contains no solids: {step_path}")
    combined = workplane if len(solids) == 1 else workplane.combine()
    bbox = combined.val().BoundingBox()
    lower = [float(bbox.xmin), float(bbox.ymin), float(bbox.zmin)]
    upper = [float(bbox.xmax), float(bbox.ymax), float(bbox.zmax)]
    lengths = [upper[index] - lower[index] for index in range(3)]
    characteristic = math.sqrt(sum(length * length for length in lengths))
    tolerance = max(
        characteristic * float(eligibility["plane_tolerance_relative"]),
        1e-7,
    )
    total_area = sum(float(face.Area()) for face in faces)
    normal_min = float(eligibility["normal_alignment_min"])
    region_records: dict[str, dict[str, Any]] = {}

    for axis_index, axis in enumerate(AXES):
        for side, target in (("min", lower[axis_index]), ("max", upper[axis_index])):
            matches: list[dict[str, Any]] = []
            for face_index, face in enumerate(faces):
                if face.geomType() != "PLANE":
                    continue
                normal = _vector(face.normalAt())
                if abs(normal[axis_index]) < normal_min:
                    continue
                face_bbox = face.BoundingBox()
                face_lower = [face_bbox.xmin, face_bbox.ymin, face_bbox.zmin]
                face_upper = [face_bbox.xmax, face_bbox.ymax, face_bbox.zmax]
                if (
                    abs(float(face_lower[axis_index]) - target) > tolerance
                    or abs(float(face_upper[axis_index]) - target) > tolerance
                ):
                    continue
                matches.append(
                    {
                        "face_index_diagnostic_only": face_index,
                        "area_mm2": round(float(face.Area()), 8),
                        "centroid_mm": [
                            round(value, 8) for value in _vector(face.Center())
                        ],
                        "normal": [round(value, 8) for value in normal],
                    }
                )
            if not matches:
                continue
            area = sum(record["area_mm2"] for record in matches)
            region_id = f"axis:{axis}:{side}"
            region_records[region_id] = {
                "region_id": region_id,
                "axis": axis,
                "side": side,
                "surface_count": len(matches),
                "area_mm2": round(area, 8),
                "area_fraction": round(area / total_area, 10) if total_area else 0.0,
                "surfaces": matches,
            }

    pairs = []
    minimum_area = float(eligibility["minimum_region_area_fraction"])
    maximum_surfaces = int(eligibility["maximum_region_surface_count"])
    for axis_index, axis in enumerate(AXES):
        minimum = region_records.get(f"axis:{axis}:min")
        maximum = region_records.get(f"axis:{axis}:max")
        if minimum is None or maximum is None:
            continue
        pair_eligible = (
            min(minimum["area_fraction"], maximum["area_fraction"]) >= minimum_area
            and max(minimum["surface_count"], maximum["surface_count"])
            <= maximum_surfaces
        )
        pairs.append(
            {
                "pair_id": f"opposed:{axis}",
                "axis": axis,
                "span_length_mm": round(lengths[axis_index], 8),
                "minimum_region": minimum,
                "maximum_region": maximum,
                "eligible": pair_eligible,
            }
        )

    reasons = []
    if len(solids) != int(eligibility["expected_solid_count"]):
        reasons.append("solid_count_not_one")
    eligible_pairs = [pair for pair in pairs if pair["eligible"]]
    if not eligible_pairs:
        reasons.append("no_qualified_opposed_planar_region_pair")
    return {
        "step_sha256": sha256_file(step_path),
        "solid_count": len(solids),
        "face_count": len(faces),
        "total_surface_area_mm2": round(total_area, 8),
        "bbox_mm": {
            "minimum": [round(value, 8) for value in lower],
            "maximum": [round(value, 8) for value in upper],
            "lengths": [round(value, 8) for value in lengths],
            "diagonal": round(characteristic, 8),
        },
        "eligible_opposed_region_pairs": eligible_pairs,
        "all_opposed_region_pairs": pairs,
        "deterministic_eligibility": {
            "eligible": not reasons,
            "reason_codes": reasons,
            "policy": {
                key: eligibility[key]
                for key in (
                    "expected_solid_count",
                    "normal_alignment_min",
                    "plane_tolerance_relative",
                    "minimum_region_area_fraction",
                    "maximum_region_surface_count",
                )
            },
        },
    }


def _semantic_projection(item_root: Path) -> dict[str, Any]:
    path = item_root / "benchmark/consumer_ready_item_v2.json"
    if not path.is_file():
        return {"status": "unavailable"}
    consumer = read_json(path)
    annotations = consumer.get("annotations") or {}
    engineering = annotations.get("engineering_task") or {}
    identity = engineering.get("part_identity") or {}
    design = engineering.get("design") or {}
    function = engineering.get("function") or {}
    visual = annotations.get("visual_grounding") or {}
    interfaces = function.get("interfaces") or []
    return {
        "status": "candidate_context_only",
        "authority_warning": (
            "LLM/VLM-authored context may be semantically wrong and is not selector "
            "or source-intent authority."
        ),
        "title": identity.get("title"),
        "category": identity.get("category"),
        "application_context": design.get("application_context"),
        "global_form": visual.get("global_form"),
        "interfaces": [
            {
                "kind": value.get("interface_kind"),
                "description": value.get("description"),
            }
            for value in interfaces[:2]
            if isinstance(value, Mapping)
        ],
    }


def _visual_source(item_root: Path) -> Path:
    preferred = (
        item_root / "visuals/v2/engineering_4view_visible_hidden.png",
        item_root / "visuals/v2/vlm_orbit_contact_sheet.png",
        item_root / "source/composite_png.png",
    )
    for path in preferred:
        if path.is_file():
            return path
    images = sorted((item_root / "source").glob("*.png"))
    if not images:
        raise L3AnnotationError(f"item has no render: {item_root.name}")
    return images[0]


def _project_image(source: Path, destination: Path, long_edge: int) -> None:
    from PIL import Image

    with Image.open(source) as image:
        converted = image.convert("RGB")
        converted.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        converted.save(destination, format="PNG", optimize=True)


def _selection_metadata(
    selection_records: Mapping[str, Mapping[str, Any]], item_id: str
) -> dict[str, Any]:
    record = selection_records.get(item_id) or {}
    return {
        "source_key": record.get("source_key"),
        "complexity": record.get("complexity"),
        "selection_stratum": record.get("selection_stratum"),
        "split": record.get("split"),
    }


def _annotation_input_projection(context: Mapping[str, Any]) -> dict[str, Any]:
    """Keep model context compact while retaining full evidence in context.json."""

    geometry = context["geometry"]

    def compact_region(region: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: region[key]
            for key in (
                "region_id",
                "axis",
                "side",
                "surface_count",
                "area_mm2",
                "area_fraction",
            )
        }

    pairs = [
        {
            "pair_id": pair["pair_id"],
            "axis": pair["axis"],
            "span_length_mm": pair["span_length_mm"],
            "minimum_region": compact_region(pair["minimum_region"]),
            "maximum_region": compact_region(pair["maximum_region"]),
        }
        for pair in geometry["eligible_opposed_region_pairs"]
    ]
    return {
        "item_id": context["item_id"],
        "source_family": context["source_family"],
        "selection": context["selection"],
        "geometry": {
            "solid_count": geometry["solid_count"],
            "face_count": geometry["face_count"],
            "bbox_lengths_mm": geometry["bbox_mm"]["lengths"],
            "deterministic_eligibility": geometry["deterministic_eligibility"],
            "eligible_opposed_region_pairs": pairs,
        },
        "semantic_context": context["semantic_context"],
        "authority": context["authority"],
    }


def prepare_annotation_corpus(
    *, repo_root: Path, profile_path: Path, output_root: Path | None = None
) -> dict[str, Any]:
    """Prepare all 300 compact multimodal annotation bundles without model calls."""

    repo_root = repo_root.resolve()
    profile_path = profile_path.resolve()
    profile = _load_profile(repo_root, profile_path)
    dataset = profile["dataset"]
    items_root = _repo_path(repo_root, str(dataset["items_root"]))
    item_ids_path = _repo_path(
        repo_root, str(dataset["item_ids_path"]), require_file=True
    )
    output_root = (
        output_root.resolve()
        if output_root is not None
        else _repo_path(repo_root, str(dataset["sidecar_root"]))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    item_ids = [
        line.strip() for line in item_ids_path.read_text().splitlines() if line.strip()
    ]
    if len(item_ids) != 300 or len(set(item_ids)) != len(item_ids):
        raise L3AnnotationError(
            "L3 profile must resolve to exactly 300 unique item IDs"
        )

    selection_path = item_ids_path.parent / "selected_items.jsonl"
    selection_records: dict[str, Mapping[str, Any]] = {}
    if selection_path.is_file():
        for line in selection_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                selection_records[str(value["item_id"])] = value

    requests = []
    eligibility_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for item_id in sorted(item_ids):
        item_root = items_root / item_id
        if not item_root.is_dir():
            raise L3AnnotationError(f"selected item is missing: {item_id}")
        step_path = _source_step(item_root)
        geometry = _face_inventory(step_path, profile["eligibility"])
        source = _source_family(item_id)
        source_counts[source] += 1
        eligibility_counts[
            "eligible"
            if geometry["deterministic_eligibility"]["eligible"]
            else "ineligible"
        ] += 1
        item_output = output_root / "items" / item_id
        image_path = item_output / "input_image.png"
        _project_image(
            _visual_source(item_root),
            image_path,
            int(profile["annotation"]["image_long_edge_px"]),
        )
        image_record = {
            "path": image_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(image_path),
            "bytes": image_path.stat().st_size,
        }
        context = {
            "schema_version": "cadengbench-p.l3-annotation-context.v1",
            "item_id": item_id,
            "source_family": source,
            "selection": _selection_metadata(selection_records, item_id),
            "geometry": geometry,
            "semantic_context": _semantic_projection(item_root),
            "visual": image_record,
            "authority": {
                "source_intent_claimed": False,
                "llm_output_authority": "candidate_only",
                "promotion_requires_deterministic_and_solver_validation": True,
            },
        }
        context_path = item_output / "context.json"
        write_json(context_path, context)
        bundle_body = {
            "schema_version": "cadengbench-p.l3-annotation-bundle.v1",
            "item_id": item_id,
            "role_id": ROLE_ID,
            "inputs": _annotation_input_projection(context),
        }
        bundle_id = f"sha256:{sha256_json(bundle_body)}"
        bundle = {**bundle_body, "bundle_id": bundle_id}
        bundle_path = item_output / "bundle.json"
        write_json(bundle_path, bundle)
        requests.append(
            {
                "item_id": item_id,
                "role_id": ROLE_ID,
                "bundle_id": bundle_id,
                "bundle_path": bundle_path.relative_to(repo_root).as_posix(),
                "bundle_sha256": sha256_file(bundle_path),
                "context_path": context_path.relative_to(repo_root).as_posix(),
                "context_sha256": sha256_file(context_path),
                "image_path": image_record["path"],
                "image_sha256": image_record["sha256"],
                "response_path": (item_output / "annotation_candidate.json")
                .relative_to(repo_root)
                .as_posix(),
                "metadata_path": (item_output / "annotation_metadata.json")
                .relative_to(repo_root)
                .as_posix(),
            }
        )

    request_path = output_root / "requests.jsonl"
    _write_jsonl(request_path, requests)
    manifest_body = {
        "schema_version": CORPUS_SCHEMA,
        "profile_id": profile["profile_id"],
        "profile_sha256": sha256_file(profile_path),
        "item_ids_sha256": sha256_file(item_ids_path),
        "item_count": len(item_ids),
        "source_counts": dict(sorted(source_counts.items())),
        "deterministic_eligibility_counts": dict(sorted(eligibility_counts.items())),
        "requests_path": request_path.relative_to(repo_root).as_posix(),
        "requests_sha256": sha256_file(request_path),
        "core_dataset_mutated": False,
    }
    manifest = {**manifest_body, "manifest_sha256": sha256_json(manifest_body)}
    write_json(output_root / "manifest.json", manifest)
    return manifest


def _fixed_schema(
    schema: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    fixed = deepcopy(schema)
    properties = fixed["properties"]
    properties["bundle_id"] = {"const": bundle["bundle_id"]}
    properties["item_id"] = {"const": bundle["item_id"]}
    properties["role_id"] = {"const": ROLE_ID}
    return fixed


def _guidance_schema(schema: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Project the strict schema into the pinned xgrammar-supported subset."""

    removed: list[str] = []

    def visit(value: Any, path: str) -> Any:
        if isinstance(value, Mapping):
            projected = {}
            for key, child in value.items():
                child_path = f"{path}/{key}"
                if key in {"uniqueItems", "allOf", "if", "then", "else"}:
                    removed.append(child_path)
                    continue
                projected[str(key)] = visit(child, child_path)
            return projected
        if isinstance(value, list):
            return [
                visit(child, f"{path}/{index}") for index, child in enumerate(value)
            ]
        return value

    projected = visit(schema, "#")
    if not isinstance(projected, dict):
        raise L3AnnotationError("guidance schema projection is not an object")
    return projected, sorted(removed)


def _messages(
    *,
    prompt: str,
    bundle: Mapping[str, Any],
    schema: Mapping[str, Any],
    image_path: Path,
) -> list[dict[str, Any]]:
    user_payload = canonical_json_bytes(
        {
            "bundle_identity": {
                "bundle_id": bundle["bundle_id"],
                "item_id": bundle["item_id"],
                "role_id": bundle["role_id"],
            },
            "response_schema": schema,
            "inputs": bundle["inputs"],
        }
    ).decode("utf-8")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_payload},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        },
    ]


def _parse_content(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise L3AnnotationError("provider content is not JSON text")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise L3AnnotationError("provider JSON is not an object")
    return parsed


def _region_ids(bundle: Mapping[str, Any]) -> set[str]:
    pairs = bundle["inputs"]["geometry"]["eligible_opposed_region_pairs"]
    result = set()
    for pair in pairs:
        result.add(str(pair["minimum_region"]["region_id"]))
        result.add(str(pair["maximum_region"]["region_id"]))
    return result


def _validate_candidate(
    candidate: Mapping[str, Any], bundle: Mapping[str, Any]
) -> None:
    eligible = bool(
        bundle["inputs"]["geometry"]["deterministic_eligibility"]["eligible"]
    )
    proposed = candidate["applicability"] == "proposed"
    if proposed != (candidate["output_status"] == "complete"):
        raise L3AnnotationError("output_status and applicability disagree")
    nullable = (
        "scenario_family",
        "fixed_region_id",
        "loaded_region_id",
        "load_direction",
    )
    if proposed:
        if not eligible:
            raise L3AnnotationError(
                "model proposed a deterministically ineligible item"
            )
        if any(candidate[field] is None for field in nullable):
            raise L3AnnotationError("proposed annotation has null execution fields")
        available = _region_ids(bundle)
        fixed = str(candidate["fixed_region_id"])
        loaded = str(candidate["loaded_region_id"])
        if fixed not in available or loaded not in available:
            raise L3AnnotationError("annotation references an unavailable region")
        _, fixed_axis, fixed_side = fixed.split(":")
        _, loaded_axis, loaded_side = loaded.split(":")
        if fixed_axis != loaded_axis or fixed_side == loaded_side:
            raise L3AnnotationError("fixed and loaded regions are not an opposed pair")
        direction = candidate["load_direction"]
        if candidate["scenario_family"] == "axial_tension":
            if direction["axis"] != fixed_axis:
                raise L3AnnotationError("axial load direction does not match span axis")
            outward_sign = 1 if loaded_side == "max" else -1
            if int(direction["sign"]) != outward_sign:
                raise L3AnnotationError("axial tension points into the solid")
        elif direction["axis"] == fixed_axis:
            raise L3AnnotationError("cantilever load must be transverse to the span")
    elif any(candidate[field] is not None for field in nullable):
        raise L3AnnotationError("abstaining annotation must use null execution fields")


def _semantic_validation(
    candidate: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, str | None]:
    try:
        _validate_candidate(candidate, bundle)
    except L3AnnotationError as exc:
        return {"status": "rejected", "reason": str(exc)}
    return {"status": "accepted", "reason": None}


@dataclass(frozen=True)
class _Endpoint:
    base_url: str
    served_model_name: str
    model_id: str
    revision: str
    chat_template_kwargs: Mapping[str, Any]


async def _post_with_retry(
    client: Any,
    url: str,
    payload: Mapping[str, Any],
    *,
    retries: int,
) -> tuple[dict[str, Any], int]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.post(url, json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                raise L3AnnotationError(f"transient HTTP {response.status_code}")
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise L3AnnotationError("provider response is not an object")
            return value, attempt
        except Exception as exc:  # provider boundary
            last = exc
            if attempt >= retries:
                break
            await asyncio.sleep(2**attempt + random.random() * 0.25)
    raise L3AnnotationError(f"provider request failed: {last}")


async def _annotate_one(
    *,
    semaphore: asyncio.Semaphore,
    client: Any,
    repo_root: Path,
    request: Mapping[str, Any],
    endpoint: _Endpoint,
    prompt: str,
    base_schema: Mapping[str, Any],
    annotation_config: Mapping[str, Any],
    profile_sha256: str,
) -> dict[str, Any]:
    bundle_path = _repo_path(repo_root, str(request["bundle_path"]), require_file=True)
    image_path = _repo_path(repo_root, str(request["image_path"]), require_file=True)
    response_path = _repo_path(repo_root, str(request["response_path"]))
    metadata_path = _repo_path(repo_root, str(request["metadata_path"]))
    if sha256_file(bundle_path) != request["bundle_sha256"]:
        raise L3AnnotationError(f"bundle changed: {request['item_id']}")
    if sha256_file(image_path) != request["image_sha256"]:
        raise L3AnnotationError(f"image changed: {request['item_id']}")
    bundle = read_json(bundle_path)
    schema = _fixed_schema(base_schema, bundle)
    guidance_schema, removed_guidance_keywords = _guidance_schema(schema)
    messages = _messages(
        prompt=prompt, bundle=bundle, schema=schema, image_path=image_path
    )
    request_identity = {
        "bundle_sha256": request["bundle_sha256"],
        "image_sha256": request["image_sha256"],
        "prompt_sha256": sha256_json(prompt),
        "schema_sha256": sha256_json(schema),
        "guidance_schema_sha256": sha256_json(guidance_schema),
        "profile_sha256": profile_sha256,
        "model_id": endpoint.model_id,
        "revision": endpoint.revision,
    }
    if response_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("request_identity") == request_identity:
            candidate = read_json(response_path)
            jsonschema.validate(candidate, schema)
            semantic_validation = _semantic_validation(candidate, bundle)
            return {
                "item_id": request["item_id"],
                "resumed": True,
                "input_tokens": int(metadata["usage"]["input_tokens"]),
                "output_tokens": int(metadata["usage"]["output_tokens"]),
                "applicability": candidate["applicability"],
                "semantic_status": semantic_validation["status"],
            }
        raise L3AnnotationError(
            f"existing annotation has different frozen inputs: {request['item_id']}"
        )

    async with semaphore:
        token_payload = {
            "model": endpoint.served_model_name,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": dict(endpoint.chat_template_kwargs),
        }
        tokenized, token_retries = await _post_with_retry(
            client,
            f"{endpoint.base_url.rsplit('/v1', 1)[0]}/tokenize",
            token_payload,
            retries=int(annotation_config["maximum_transport_retries"]),
        )
        input_tokens = int(tokenized["count"])
        if input_tokens > int(annotation_config["max_input_tokens"]):
            raise L3AnnotationError(
                f"input token cap exceeded for {request['item_id']}: {input_tokens}"
            )
        completion_payload = {
            "model": endpoint.served_model_name,
            "messages": messages,
            "temperature": float(annotation_config["temperature"]),
            "top_p": float(annotation_config["top_p"]),
            "max_tokens": int(annotation_config["max_output_tokens"]),
            "seed": int(sha256_json(bundle)[0:8], 16),
            "chat_template_kwargs": dict(endpoint.chat_template_kwargs),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cadengbench_p_l3_probe_annotation",
                    "schema": guidance_schema,
                    "strict": True,
                },
            },
        }
        provider, completion_retries = await _post_with_retry(
            client,
            f"{endpoint.base_url}/chat/completions",
            completion_payload,
            retries=int(annotation_config["maximum_transport_retries"]),
        )
    candidate = _parse_content(provider["choices"][0]["message"]["content"])
    jsonschema.validate(candidate, schema)
    semantic_validation = _semantic_validation(candidate, bundle)
    usage = provider.get("usage") or {}
    output_tokens = int(usage.get("completion_tokens") or 0)
    write_json(response_path, candidate)
    raw_path = response_path.with_name("provider_response.json")
    write_json(raw_path, provider)
    write_json(
        metadata_path,
        {
            "schema_version": "cadengbench-p.l3-annotation-metadata.v1",
            "item_id": request["item_id"],
            "created_at": _utc_now(),
            "request_identity": request_identity,
            "endpoint": {
                "base_url": endpoint.base_url,
                "served_model_name": endpoint.served_model_name,
                "model_id": endpoint.model_id,
                "revision": endpoint.revision,
            },
            "decoding": {
                key: annotation_config[key]
                for key in ("temperature", "top_p", "max_output_tokens")
            },
            "guidance_schema_projection": {
                "sha256": sha256_json(guidance_schema),
                "removed_keywords": removed_guidance_keywords,
            },
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "transport_retries": {
                "tokenize": token_retries,
                "completion": completion_retries,
            },
            "semantic_validation": semantic_validation,
            "provider_response_path": raw_path.relative_to(repo_root).as_posix(),
            "provider_response_sha256": sha256_file(raw_path),
            "candidate_sha256": sha256_file(response_path),
        },
    )
    return {
        "item_id": request["item_id"],
        "resumed": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "applicability": candidate["applicability"],
        "semantic_status": semantic_validation["status"],
    }


async def _run_annotation_async(
    *,
    repo_root: Path,
    profile_path: Path,
    output_root: Path,
    selected_item_ids: set[str] | None = None,
) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise L3AnnotationError("httpx is required for local vLLM annotation") from exc
    profile = _load_profile(repo_root, profile_path)
    annotation = profile["annotation"]
    model_profile_path = _repo_path(
        repo_root, str(annotation["model_profile_path"]), require_file=True
    )
    model_profile = read_json(model_profile_path)
    endpoint_raw = model_profile["endpoints"][annotation["endpoint_name"]]
    endpoint = _Endpoint(
        base_url=str(endpoint_raw["base_url"]).rstrip("/"),
        served_model_name=str(endpoint_raw["served_model_name"]),
        model_id=str(endpoint_raw["model_id"]),
        revision=str(endpoint_raw["revision"]),
        chat_template_kwargs=dict(endpoint_raw.get("chat_template_kwargs") or {}),
    )
    if (
        endpoint.model_id != annotation["model_id"]
        or endpoint.revision != annotation["revision"]
    ):
        raise L3AnnotationError("L3 annotation profile and endpoint identity disagree")
    prompt_path = _repo_path(
        repo_root, str(annotation["system_prompt_path"]), require_file=True
    )
    schema_path = _repo_path(
        repo_root, str(annotation["output_schema_path"]), require_file=True
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = read_json(schema_path)
    manifest = read_json(output_root / "manifest.json")
    requests_path = _repo_path(
        repo_root, str(manifest["requests_path"]), require_file=True
    )
    if sha256_file(requests_path) != manifest["requests_sha256"]:
        raise L3AnnotationError("annotation request manifest changed")
    requests = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(requests) != 300:
        raise L3AnnotationError("annotation request roster must contain 300 items")
    if selected_item_ids is not None:
        available = {str(request["item_id"]) for request in requests}
        missing = sorted(selected_item_ids - available)
        if missing:
            raise L3AnnotationError(f"unknown annotation item IDs: {missing}")
        requests = [
            request
            for request in requests
            if str(request["item_id"]) in selected_item_ids
        ]
    semaphore = asyncio.Semaphore(int(annotation["request_concurrency"]))
    async with httpx.AsyncClient(timeout=int(annotation["timeout_seconds"])) as client:
        models = await client.get(f"{endpoint.base_url}/models")
        models.raise_for_status()
        observed = {
            str(value.get("id"))
            for value in models.json().get("data", [])
            if isinstance(value, Mapping)
        }
        if (
            endpoint.served_model_name not in observed
            and endpoint.model_id not in observed
        ):
            raise L3AnnotationError("local vLLM endpoint exposes the wrong model")
        tasks = [
            asyncio.create_task(
                _annotate_one(
                    semaphore=semaphore,
                    client=client,
                    repo_root=repo_root,
                    request=request,
                    endpoint=endpoint,
                    prompt=prompt,
                    base_schema=schema,
                    annotation_config=annotation,
                    profile_sha256=sha256_file(profile_path),
                )
            )
            for request in requests
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [value for value in gathered if isinstance(value, BaseException)]
    results = [value for value in gathered if isinstance(value, dict)]
    tokens = sorted(int(value["input_tokens"]) for value in results)
    summary = {
        "schema_version": "cadengbench-p.l3-annotation-run-summary.v1",
        "created_at": _utc_now(),
        "item_count": len(requests),
        "completed_count": len(results),
        "failure_count": len(failures),
        "new_count": sum(not value["resumed"] for value in results),
        "resumed_count": sum(value["resumed"] for value in results),
        "applicability_counts": dict(
            sorted(Counter(value["applicability"] for value in results).items())
        ),
        "semantic_validation_counts": dict(
            sorted(Counter(value["semantic_status"] for value in results).items())
        ),
        "input_token_distribution": {
            "minimum": tokens[0] if tokens else None,
            "median": tokens[len(tokens) // 2] if tokens else None,
            "p95": tokens[round((len(tokens) - 1) * 0.95)] if tokens else None,
            "maximum": tokens[-1] if tokens else None,
            "hard_cap": int(annotation["max_input_tokens"]),
        },
        "output_token_total": sum(int(value["output_tokens"]) for value in results),
        "single_model": {
            "model_id": endpoint.model_id,
            "revision": endpoint.revision,
            "call_count": len(results),
        },
        "failures": [f"{type(value).__name__}: {value}" for value in failures[:50]],
    }
    write_json(output_root / "annotation_run_summary.json", summary)
    if failures:
        raise L3AnnotationError(
            f"L3 annotation wave failed for {len(failures)} items; see summary"
        )
    return summary


def run_annotation_corpus(
    *,
    repo_root: Path,
    profile_path: Path,
    output_root: Path | None = None,
    item_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    profile = _load_profile(repo_root.resolve(), profile_path.resolve())
    resolved_output = (
        output_root.resolve()
        if output_root is not None
        else _repo_path(repo_root.resolve(), str(profile["dataset"]["sidecar_root"]))
    )
    return asyncio.run(
        _run_annotation_async(
            repo_root=repo_root.resolve(),
            profile_path=profile_path.resolve(),
            output_root=resolved_output,
            selected_item_ids=set(item_ids) if item_ids is not None else None,
        )
    )


def _selector(region_id: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    _, axis, side = region_id.split(":")
    eligibility = profile["eligibility"]
    return {
        "kind": "axis_extreme_planar_surfaces_v2",
        "region_id": region_id,
        "axis": axis,
        "side": side,
        "normal_alignment_min": eligibility["normal_alignment_min"],
        "plane_tolerance_relative": eligibility["plane_tolerance_relative"],
        "minimum_count": 1,
        "maximum_count": eligibility["maximum_region_surface_count"],
        "minimum_area_fraction": eligibility["minimum_region_area_fraction"],
    }


def _compile_card(
    *,
    profile: Mapping[str, Any],
    profile_sha256: str,
    context: Mapping[str, Any],
    context_sha256: str,
    candidate: Mapping[str, Any],
    candidate_sha256: str,
) -> dict[str, Any]:
    probe = profile["probe"]
    direction = candidate["load_direction"]
    vector = [0.0, 0.0, 0.0]
    vector[AXES.index(direction["axis"])] = float(direction["sign"]) * float(
        probe["traction_magnitude_mpa"]
    )
    annotation = profile["annotation"]
    return {
        "schema_version": CARD_SCHEMA,
        "card_id": f"{context['item_id']}:linear-static-probe:v1",
        "item_id": context["item_id"],
        "status": "geometry_qualified",
        "source_intent_claimed": False,
        "scenario_family": candidate["scenario_family"],
        "geometry": {"units": "mm", "expected_solid_count": 1},
        "selectors": {
            "fixed": _selector(candidate["fixed_region_id"], profile),
            "loaded": _selector(candidate["loaded_region_id"], profile),
        },
        "material": dict(probe["material"]),
        "load": {
            "kind": "surface_traction_probe",
            "traction_vector_mpa": vector,
            "distribution": probe["load_distribution"],
        },
        "mesh": dict(probe["mesh"]),
        "analysis": dict(probe["analysis"]),
        "checks": list(probe["checks"]),
        "provenance": {
            "annotation_candidate_sha256": candidate_sha256,
            "context_sha256": context_sha256,
            "profile_sha256": profile_sha256,
            "annotator_model": annotation["model_id"],
            "annotator_revision": annotation["revision"],
        },
    }


def compile_physics_cards(
    *,
    repo_root: Path,
    profile_path: Path,
    output_root: Path | None = None,
    item_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Promote valid proposals to geometry-qualified cards; never silently repair."""

    repo_root = repo_root.resolve()
    profile_path = profile_path.resolve()
    profile = _load_profile(repo_root, profile_path)
    output_root = (
        output_root.resolve()
        if output_root is not None
        else _repo_path(repo_root, str(profile["dataset"]["sidecar_root"]))
    )
    manifest = read_json(output_root / "manifest.json")
    requests_path = _repo_path(
        repo_root, str(manifest["requests_path"]), require_file=True
    )
    card_schema_path = (
        repo_root / "cadengbench_p_evaluator/schemas/l3_physics_card.v2.schema.json"
    )
    card_schema = read_json(card_schema_path)
    requests = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(requests) != 300:
        raise L3AnnotationError("card compilation roster must contain 300 items")
    if item_ids is not None:
        selected = set(item_ids)
        available = {str(request["item_id"]) for request in requests}
        missing = sorted(selected - available)
        if missing:
            raise L3AnnotationError(f"unknown card-compilation item IDs: {missing}")
        requests = [
            request for request in requests if str(request["item_id"]) in selected
        ]
    records = []
    for request in requests:
        context_path = _repo_path(repo_root, request["context_path"], require_file=True)
        candidate_path = _repo_path(
            repo_root, request["response_path"], require_file=True
        )
        bundle_path = _repo_path(repo_root, request["bundle_path"], require_file=True)
        context = read_json(context_path)
        candidate = read_json(candidate_path)
        bundle = read_json(bundle_path)
        disposition = "rejected"
        reason = candidate["reason_code"]
        card_path: Path | None = None
        try:
            _validate_candidate(candidate, bundle)
            if candidate["applicability"] == "proposed":
                card = _compile_card(
                    profile=profile,
                    profile_sha256=sha256_file(profile_path),
                    context=context,
                    context_sha256=sha256_file(context_path),
                    candidate=candidate,
                    candidate_sha256=sha256_file(candidate_path),
                )
                jsonschema.validate(card, card_schema)
                card_path = candidate_path.with_name("physics_card.json")
                write_json(card_path, card)
                disposition = "geometry_qualified"
                reason = "candidate_matches_deterministic_region_contract"
        except (
            KeyError,
            TypeError,
            ValueError,
            L3AnnotationError,
            jsonschema.ValidationError,
        ) as exc:
            reason = f"invalid_candidate:{type(exc).__name__}:{exc}"
        validation = {
            "schema_version": "cadengbench-p.l3-annotation-validation.v1",
            "item_id": request["item_id"],
            "disposition": disposition,
            "reason": reason,
            "card_path": (
                card_path.relative_to(repo_root).as_posix() if card_path else None
            ),
            "card_sha256": sha256_file(card_path) if card_path else None,
            "llm_can_promote_solver_authority": False,
        }
        validation_path = candidate_path.with_name("annotation_validation.json")
        write_json(validation_path, validation)
        records.append(validation)
    summary = {
        "schema_version": "cadengbench-p.l3-card-compilation-summary.v1",
        "item_count": len(records),
        "disposition_counts": dict(
            sorted(Counter(record["disposition"] for record in records).items())
        ),
        "solver_qualified_count": 0,
        "core_dataset_mutated": False,
    }
    write_json(output_root / "card_compilation_summary.json", summary)
    return summary
