"""Deterministic Gmsh-to-CalculiX linear-static structural probe."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from cadengbench_p_eval.io import read_json, sha256_file, write_json
from cadengbench_p_eval.provenance import physics_evaluator_build_hash

CARD_SCHEMA = "cadengbench-p.l3-physics-card.v2"
RESULT_SCHEMA = "cadengbench-p.l3-calculix-result.v2"
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
# Gmsh Tetrahedron10 orders its final two midside nodes as edges 2-4, 1-4;
# CalculiX C3D10 follows the Abaqus convention 1-4, 2-4 for those slots.
GMSH_TO_CALCULIX_C3D10 = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)
STAGES = (
    "card",
    "runtime",
    "geometry",
    "selectors",
    "mesh",
    "deck",
    "solver",
    "result_sanity",
    "requirements",
    "visualization",
)
FLOAT_PATTERN = re.compile(r"[+-]?\d+\.\d+(?:[EeDd][+-]?\d+)")


class PhysicsCardError(ValueError):
    """A card is incomplete or outside the supported L3 scope."""


class L3StageError(RuntimeError):
    """A typed evaluator-stage failure."""

    def __init__(self, stage: str, message: str, *, owner: str = "model"):
        super().__init__(message)
        if owner not in {"model", "evaluator"}:
            raise ValueError(f"unsupported L3 failure owner: {owner}")
        self.stage = stage
        self.owner = owner


@dataclass(frozen=True)
class MeshData:
    nodes: tuple[tuple[int, float, float, float], ...]
    elements: tuple[tuple[int, ...], ...]
    element_type: str
    fixed_nodes: tuple[int, ...]
    loaded_nodes: tuple[int, ...]
    nodal_forces: Mapping[int, tuple[float, float, float]]
    applied_force: tuple[float, float, float]
    report: Mapping[str, Any]


@dataclass(frozen=True)
class ParsedFields:
    metrics: Mapping[str, Any]
    displacements: Mapping[int, tuple[float, float, float]]
    von_mises: Mapping[int, float]


def _card_schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[3] / "schemas/l3_physics_card.v2.schema.json"
    )


def _card_schema_document() -> dict[str, Any]:
    source_path = _card_schema_path()
    if source_path.is_file():
        value = read_json(source_path)
        if not isinstance(value, dict):
            raise PhysicsCardError("physics-card schema root is not an object")
        return value
    packaged = resources.files("cadengbench_p_eval").joinpath(
        "_schemas", "l3_physics_card.v2.schema.json"
    )
    if packaged.is_file():
        return json.loads(packaged.read_text(encoding="utf-8"))
    raise PhysicsCardError(f"physics-card schema is missing: {source_path}")


def validate_physics_card(card: Mapping[str, Any]) -> None:
    try:
        jsonschema.validate(dict(card), _card_schema_document())
    except jsonschema.ValidationError as exc:
        raise PhysicsCardError(f"invalid physics card: {exc.message}") from exc
    if card.get("schema_version") != CARD_SCHEMA:
        raise PhysicsCardError("unsupported physics-card schema")
    if card["status"] == "solver_qualified" and "qualification" not in card:
        raise PhysicsCardError(
            "solver-qualified card is missing qualification evidence"
        )
    if card["status"] != "solver_qualified" and "qualification" in card:
        raise PhysicsCardError("unpromoted card cannot carry qualification evidence")
    fixed = card["selectors"]["fixed"]
    loaded = card["selectors"]["loaded"]
    if fixed["axis"] != loaded["axis"] or fixed["side"] == loaded["side"]:
        raise PhysicsCardError("fixed and loaded selectors must be one opposed pair")
    vector = [float(value) for value in card["load"]["traction_vector_mpa"]]
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise PhysicsCardError("traction vector must be finite and nonzero")
    span_axis = AXIS_INDEX[str(fixed["axis"])]
    nonzero = [index for index, value in enumerate(vector) if abs(value) > 1e-15]
    if len(nonzero) != 1:
        raise PhysicsCardError("v2 probe requires one axis-aligned traction component")
    scenario = card["scenario_family"]
    if scenario == "axial_tension" and nonzero[0] != span_axis:
        raise PhysicsCardError("axial probe traction must follow the span axis")
    if scenario == "cantilever_transverse" and nonzero[0] == span_axis:
        raise PhysicsCardError("cantilever probe traction must be transverse")


def load_physics_card(path: Path) -> dict[str, Any]:
    card = read_json(path)
    if not isinstance(card, dict):
        raise PhysicsCardError("physics card root must be an object")
    validate_physics_card(card)
    return card


def _global_bbox(gmsh: Any, volumes: Sequence[tuple[int, int]]) -> tuple[float, ...]:
    boxes = [gmsh.model.getBoundingBox(3, tag) for _, tag in volumes]
    return tuple(
        min(box[index] for box in boxes)
        if index < 3
        else max(box[index] for box in boxes)
        for index in range(6)
    )


def _resolve_selector(
    gmsh: Any,
    selector: Mapping[str, Any],
    global_bbox: Sequence[float],
) -> tuple[list[int], list[dict[str, Any]]]:
    axis = AXIS_INDEX[str(selector["axis"])]
    side = str(selector["side"])
    diagonal = math.sqrt(
        sum((global_bbox[index + 3] - global_bbox[index]) ** 2 for index in range(3))
    )
    tolerance = max(
        diagonal * float(selector["plane_tolerance_relative"]),
        1e-7,
    )
    normal_min = float(selector["normal_alignment_min"])
    bbox_center = np.asarray(
        [
            (float(global_bbox[index]) + float(global_bbox[index + 3])) * 0.5
            for index in range(3)
        ],
        dtype=float,
    )
    candidates: list[dict[str, Any]] = []
    total_surface_area = sum(
        float(gmsh.model.occ.getMass(2, tag)) for _, tag in gmsh.model.getEntities(2)
    )
    for _, tag in gmsh.model.getEntities(2):
        entity_type = gmsh.model.getType(2, tag)
        if entity_type.lower() != "plane":
            continue
        bbox = gmsh.model.getBoundingBox(2, tag)
        try:
            lower, upper = gmsh.model.getParametrizationBounds(2, tag)
            parameters = [
                (float(lower[index]) + float(upper[index])) * 0.5
                for index in range(len(lower))
            ]
            normal = [float(value) for value in gmsh.model.getNormal(tag, parameters)]
            normal_source = "gmsh_parametric_surface_normal"
        except Exception:
            if abs(float(bbox[axis + 3]) - float(bbox[axis])) > tolerance:
                continue
            normal = [0.0, 0.0, 0.0]
            normal[axis] = 1.0
            normal_source = "planar_axis_collapsed_bbox_fallback"
        normal_array = np.asarray(normal[:3], dtype=float)
        magnitude = float(np.linalg.norm(normal_array))
        if not math.isfinite(magnitude) or magnitude <= 1e-12:
            continue
        normal_array /= magnitude
        if abs(float(normal_array[axis])) < normal_min:
            continue
        try:
            centroid = np.asarray(
                gmsh.model.occ.getCenterOfMass(2, tag), dtype=float
            )
        except Exception:
            centroid = np.asarray(
                [
                    (float(bbox[index]) + float(bbox[index + 3])) * 0.5
                    for index in range(3)
                ],
                dtype=float,
            )
        axis_intercept = float(
            bbox_center[axis]
            + np.dot(normal_array, centroid - bbox_center)
            / normal_array[axis]
        )
        area = float(gmsh.model.occ.getMass(2, tag))
        candidates.append(
            {
                "surface_tag": tag,
                "surface_type": entity_type,
                "bbox_mm": [float(value) for value in bbox],
                "area_mm2": area,
                "area_fraction": area / total_surface_area,
                "axis_intercept_mm": axis_intercept,
                "normal": normal_array.tolist(),
                "normal_source": normal_source,
                "selection_method": (
                    "extreme_axis_intercept_of_aligned_planar_surfaces_v1"
                ),
            }
        )
    target = (
        min(value["axis_intercept_mm"] for value in candidates)
        if candidates and side == "min"
        else max(value["axis_intercept_mm"] for value in candidates)
        if candidates
        else float(global_bbox[axis if side == "min" else axis + 3])
    )
    evidence = [
        {**value, "target_coordinate_mm": target}
        for value in candidates
        if abs(float(value["axis_intercept_mm"]) - target) <= tolerance
    ]
    selected = [int(value["surface_tag"]) for value in evidence]
    minimum = int(selector["minimum_count"])
    maximum = int(selector["maximum_count"])
    area_fraction = sum(value["area_fraction"] for value in evidence)
    if not minimum <= len(selected) <= maximum:
        raise L3StageError(
            "selectors",
            f"{selector['region_id']} resolved {len(selected)} surfaces, "
            f"outside [{minimum}, {maximum}]",
        )
    if area_fraction < float(selector["minimum_area_fraction"]):
        raise L3StageError(
            "selectors",
            f"{selector['region_id']} area fraction {area_fraction:.6g} is too small",
        )
    return sorted(selected), evidence


def _surface_nodes(gmsh: Any, surface_tags: Sequence[int]) -> tuple[int, ...]:
    result: set[int] = set()
    for tag in surface_tags:
        node_tags, _, _ = gmsh.model.mesh.getNodes(2, tag, True, False)
        result.update(int(value) for value in node_tags)
    if not result:
        raise L3StageError("mesh", "selected surface contains no mesh nodes")
    return tuple(sorted(result))


def _triangle_area(
    node_ids: Sequence[int], coordinates: Mapping[int, tuple[float, float, float]]
) -> float:
    first = np.asarray(coordinates[int(node_ids[0])], dtype=float)
    second = np.asarray(coordinates[int(node_ids[1])], dtype=float)
    third = np.asarray(coordinates[int(node_ids[2])], dtype=float)
    return float(np.linalg.norm(np.cross(second - first, third - first)) * 0.5)


def _surface_traction_forces(
    gmsh: Any,
    surface_tags: Sequence[int],
    coordinates: Mapping[int, tuple[float, float, float]],
    traction: Sequence[float],
) -> tuple[dict[int, tuple[float, float, float]], float]:
    accumulated: dict[int, np.ndarray] = {}
    integrated_area = 0.0
    for surface_tag in surface_tags:
        element_types, _, node_groups = gmsh.model.mesh.getElements(2, surface_tag)
        for element_type, flattened in zip(element_types, node_groups, strict=True):
            name, dimension, _, node_count, _, _ = gmsh.model.mesh.getElementProperties(
                element_type
            )
            if int(dimension) != 2 or "triangle" not in str(name).lower():
                continue
            values = [int(value) for value in flattened]
            for offset in range(0, len(values), int(node_count)):
                nodes = values[offset : offset + int(node_count)]
                if len(nodes) < 3:
                    continue
                area = _triangle_area(nodes[:3], coordinates)
                if not math.isfinite(area) or area <= 0.0:
                    raise L3StageError("mesh", "loaded surface has an invalid triangle")
                integrated_area += area
                if len(nodes) == 3:
                    weights = (1.0 / 3.0,) * 3
                elif len(nodes) == 6:
                    # Exact consistent nodal load for constant traction on a
                    # straight-sided quadratic triangle: corner integrals are
                    # zero and each midside shape function integrates to A/3.
                    weights = (0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
                else:
                    raise L3StageError(
                        "mesh",
                        f"unsupported loaded boundary triangle with {len(nodes)} nodes",
                    )
                for node_id, weight in zip(nodes, weights, strict=True):
                    if weight == 0.0:
                        continue
                    force = np.asarray(traction, dtype=float) * area * weight
                    accumulated.setdefault(node_id, np.zeros(3, dtype=float))
                    accumulated[node_id] += force
    if not accumulated or integrated_area <= 0.0:
        raise L3StageError("mesh", "loaded surface has no triangular boundary mesh")
    return {
        node_id: tuple(float(value) for value in force)
        for node_id, force in sorted(accumulated.items())
    }, integrated_area


def _mesh_size(global_bbox: Sequence[float], mesh: Mapping[str, Any]) -> float:
    lengths = [float(global_bbox[index + 3] - global_bbox[index]) for index in range(3)]
    positive = [value for value in lengths if value > 1e-9]
    if len(positive) != 3:
        raise L3StageError("geometry", "geometry bounding box is degenerate")
    return min(
        max(positive) / int(mesh["target_long_axis_divisions"]),
        min(positive) / int(mesh["target_short_axis_divisions"]),
    )


def mesh_step(card: Mapping[str, Any], step_path: Path, output_dir: Path) -> MeshData:
    """Import one solid, reacquire regions, and create a bounded tetrahedral mesh."""

    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise L3StageError(
            "mesh", "gmsh Python package is unavailable", owner="evaluator"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    logger_started = False
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.logger.start()
        logger_started = True
        gmsh.model.add(str(card["item_id"]))
        imported = gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        expected = int(card["geometry"]["expected_solid_count"])
        if not imported or len(volumes) != expected:
            raise L3StageError(
                "geometry", f"expected {expected} solid volume, found {len(volumes)}"
            )
        global_bbox = _global_bbox(gmsh, volumes)
        fixed_surfaces, fixed_evidence = _resolve_selector(
            gmsh, card["selectors"]["fixed"], global_bbox
        )
        loaded_surfaces, loaded_evidence = _resolve_selector(
            gmsh, card["selectors"]["loaded"], global_bbox
        )
        if set(fixed_surfaces) & set(loaded_surfaces):
            raise L3StageError("selectors", "fixed and loaded surfaces overlap")

        gmsh.model.addPhysicalGroup(3, [tag for _, tag in volumes], name="EALL")
        gmsh.model.addPhysicalGroup(2, fixed_surfaces, name="FIXED_SURFACES")
        gmsh.model.addPhysicalGroup(2, loaded_surfaces, name="LOADED_SURFACES")
        size = _mesh_size(global_bbox, card["mesh"])
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size)
        gmsh.option.setNumber("Mesh.Algorithm3D", int(card["mesh"]["algorithm_3d"]))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        started = time.monotonic()
        try:
            gmsh.model.mesh.generate(3)
            if card["mesh"]["element_type"] == "C3D10":
                gmsh.model.mesh.setOrder(2)
        except Exception as exc:
            raise L3StageError("mesh", f"Gmsh generation failed: {exc}") from exc
        mesh_runtime = time.monotonic() - started

        node_tags, raw_coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = {
            int(tag): (
                float(raw_coordinates[index * 3]),
                float(raw_coordinates[index * 3 + 1]),
                float(raw_coordinates[index * 3 + 2]),
            )
            for index, tag in enumerate(node_tags)
        }
        nodes = tuple(
            (node_id, *point) for node_id, point in sorted(coordinates.items())
        )
        expected_type = 11 if card["mesh"]["element_type"] == "C3D10" else 4
        expected_nodes = 10 if expected_type == 11 else 4
        element_types, element_tag_groups, node_tag_groups = (
            gmsh.model.mesh.getElements(3)
        )
        elements: list[tuple[int, ...]] = []
        element_tags: list[int] = []
        unexpected = []
        for element_type, tags, flattened in zip(
            element_types, element_tag_groups, node_tag_groups, strict=True
        ):
            name, dimension, order, node_count, _, _ = (
                gmsh.model.mesh.getElementProperties(element_type)
            )
            if int(element_type) != expected_type or int(node_count) != expected_nodes:
                unexpected.append(
                    {
                        "type": int(element_type),
                        "name": str(name),
                        "dimension": int(dimension),
                        "order": int(order),
                        "node_count": int(node_count),
                    }
                )
                continue
            values = [int(value) for value in flattened]
            for index, element_tag in enumerate(tags):
                offset = index * expected_nodes
                connectivity = values[offset : offset + expected_nodes]
                if expected_type == 11:
                    connectivity = [
                        connectivity[position] for position in GMSH_TO_CALCULIX_C3D10
                    ]
                elements.append((int(element_tag), *connectivity))
                element_tags.append(int(element_tag))
        if unexpected or not elements:
            raise L3StageError(
                "mesh", f"mesh has unsupported volume elements: {unexpected}"
            )
        maximum_elements = int(card["mesh"]["maximum_elements"])
        if len(elements) > maximum_elements:
            raise L3StageError(
                "mesh", f"element count {len(elements)} exceeds {maximum_elements}"
            )
        qualities = [
            float(value)
            for value in gmsh.model.mesh.getElementQualities(element_tags, "minSICN")
        ]
        minimum_quality = min(qualities)
        if minimum_quality < float(card["mesh"]["minimum_sicn"]):
            raise L3StageError(
                "mesh",
                f"minimum SICN {minimum_quality:.6g} is below the profile threshold",
            )
        fixed_nodes = _surface_nodes(gmsh, fixed_surfaces)
        traction = [float(value) for value in card["load"]["traction_vector_mpa"]]
        nodal_forces, triangulated_area = _surface_traction_forces(
            gmsh, loaded_surfaces, coordinates, traction
        )
        exact_loaded_area = sum(float(value["area_mm2"]) for value in loaded_evidence)
        area_scale = exact_loaded_area / triangulated_area
        nodal_forces = {
            node_id: tuple(component * area_scale for component in force)
            for node_id, force in nodal_forces.items()
        }
        loaded_nodes = tuple(sorted(nodal_forces))
        if set(fixed_nodes) & set(loaded_nodes):
            raise L3StageError("mesh", "fixed and loaded node sets overlap")
        applied = tuple(
            sum(force[index] for force in nodal_forces.values()) for index in range(3)
        )
        mesh_path = output_dir / "mesh.msh"
        gmsh.write(str(mesh_path))
        report = {
            "gmsh_version": gmsh.option.getString("General.Version"),
            "gmsh_threads": 1,
            "geometry_bbox_mm": [float(value) for value in global_bbox],
            "volume_entity_count": len(volumes),
            "fixed_selector": fixed_evidence,
            "loaded_selector": loaded_evidence,
            "node_count": len(nodes),
            "element_count": len(elements),
            "fixed_node_count": len(fixed_nodes),
            "loaded_node_count": len(loaded_nodes),
            "element_type": card["mesh"]["element_type"],
            "high_order_optimization": card["mesh"]["high_order_optimization"],
            "connectivity_mapping": (
                "gmsh_tetra10_to_calculix_c3d10_swap_nodes_9_10_v1"
                if expected_type == 11
                else "identity"
            ),
            "characteristic_length_mm": size,
            "quality_metric": "minSICN",
            "minimum_quality": minimum_quality,
            "mean_quality": float(np.mean(qualities)),
            "loaded_area_mm2": exact_loaded_area,
            "loaded_boundary_triangle_area_mm2": triangulated_area,
            "loaded_area_relative_discrepancy": abs(
                triangulated_area - exact_loaded_area
            )
            / max(exact_loaded_area, 1e-24),
            "traction_force_area_normalization": "exact_occ_planar_region_area_v1",
            "applied_force_n": list(applied),
            "mesh_runtime_seconds": mesh_runtime,
            "mesh_path": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
        }
        return MeshData(
            nodes=nodes,
            elements=tuple(sorted(elements)),
            element_type=str(card["mesh"]["element_type"]),
            fixed_nodes=fixed_nodes,
            loaded_nodes=loaded_nodes,
            nodal_forces=nodal_forces,
            applied_force=applied,
            report=report,
        )
    finally:
        if logger_started:
            messages = gmsh.logger.get()
            (output_dir / "gmsh.log").write_text(
                "\n".join(messages) + ("\n" if messages else ""),
                encoding="utf-8",
            )
            gmsh.logger.stop()
        gmsh.finalize()


def _write_ids(handle: Any, values: Sequence[int]) -> None:
    for index in range(0, len(values), 16):
        handle.write(", ".join(str(value) for value in values[index : index + 16]))
        handle.write("\n")


def write_calculix_deck(card: Mapping[str, Any], mesh: MeshData, path: Path) -> None:
    """Compile only validated structured data into a fixed CalculiX template."""

    material = card["material"]
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("*HEADING\n")
        handle.write(f"{card['card_id']}\n")
        handle.write("*NODE\n")
        for node_id, x, y, z in mesh.nodes:
            handle.write(f"{node_id}, {x:.12g}, {y:.12g}, {z:.12g}\n")
        handle.write(f"*ELEMENT, TYPE={mesh.element_type}, ELSET=EALL\n")
        for element in mesh.elements:
            handle.write(", ".join(str(value) for value in element) + "\n")
        handle.write("*NSET, NSET=FIXED\n")
        _write_ids(handle, mesh.fixed_nodes)
        handle.write("*NSET, NSET=LOADED\n")
        _write_ids(handle, mesh.loaded_nodes)
        handle.write("*SOLID SECTION, ELSET=EALL, MATERIAL=PROBE_MATERIAL\n")
        handle.write("*MATERIAL, NAME=PROBE_MATERIAL\n")
        handle.write("*ELASTIC\n")
        handle.write(
            f"{float(material['youngs_modulus_mpa']):.12g}, "
            f"{float(material['poisson_ratio']):.12g}\n"
        )
        handle.write("*BOUNDARY\n")
        handle.write("FIXED, 1, 3\n")
        handle.write("*STEP, NLGEOM=NO\n")
        handle.write("*STATIC\n")
        handle.write("1.0, 1.0\n")
        handle.write("*CLOAD\n")
        for node_id, force in mesh.nodal_forces.items():
            for component, value in enumerate(force, start=1):
                if abs(value) > 0.0:
                    handle.write(f"{node_id}, {component}, {value:.12g}\n")
        handle.write("*NODE FILE\n")
        handle.write("U, RF\n")
        handle.write("*EL FILE\n")
        handle.write("S, E\n")
        handle.write("*NODE PRINT, NSET=FIXED, TOTALS=YES\n")
        handle.write("RF\n")
        handle.write("*END STEP\n")


def calculix_version(ccx_path: Path) -> str:
    completed = subprocess.run(
        [str(ccx_path), "-v"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    value = (completed.stdout + "\n" + completed.stderr).strip()
    match = re.search(r"Version\s+([0-9.]+)", value, flags=re.IGNORECASE)
    return match.group(1) if match else value[:200]


def run_calculix(
    ccx_path: Path, deck_path: Path, *, timeout_seconds: int
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["CCX_NPROC_EQUATION_SOLVER"] = "1"
    environment["CCX_NPROC_RESULTS"] = "1"
    environment["NUMBER_OF_CPUS"] = "1"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(ccx_path), "-i", deck_path.stem],
            cwd=deck_path.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise L3StageError("solver", "CalculiX timed out") from exc
    runtime = time.monotonic() - started
    stdout_path = deck_path.parent / "calculix.stdout.txt"
    stderr_path = deck_path.parent / "calculix.stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    frd_path = deck_path.with_suffix(".frd")
    sta_path = deck_path.with_suffix(".sta")
    success = (
        completed.returncode == 0
        and "Job finished" in completed.stdout
        and frd_path.is_file()
        and sta_path.is_file()
    )
    report = {
        "calculix_version": calculix_version(ccx_path),
        "calculix_sha256": sha256_file(ccx_path),
        "calculix_threads": 1,
        "return_code": completed.returncode,
        "job_finished_marker": "Job finished" in completed.stdout,
        "frd_exists": frd_path.is_file(),
        "sta_exists": sta_path.is_file(),
        "runtime_seconds": runtime,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    if not success:
        raise L3StageError("solver", f"CalculiX execution failed: {report}")
    return report


def _frd_blocks(path: Path) -> dict[str, dict[int, tuple[float, ...]]]:
    blocks: dict[str, dict[int, tuple[float, ...]]] = {}
    active: str | None = None
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        header = re.match(r"\s*-4\s+(\S+)", line)
        if header:
            active = header.group(1)
            blocks[active] = {}
            continue
        if line.lstrip().startswith("-3"):
            active = None
            continue
        if active is None:
            continue
        record = re.match(r"\s*-1\s+(\d+)", line)
        if not record:
            continue
        values = tuple(
            float(value.replace("D", "E").replace("d", "e"))
            for value in FLOAT_PATTERN.findall(line[record.end() :])
        )
        blocks[active][int(record.group(1))] = values
    return blocks


def _von_mises(stress: Sequence[float]) -> float:
    sxx, syy, szz, sxy, syz, szx = stress[:6]
    return math.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy * sxy + syz * syz + szx * szx)
    )


def _regularized_stress_nodes(
    *,
    stress_nodes: set[int],
    coordinates: Mapping[int, tuple[float, float, float]],
    mesh: MeshData,
    card: Mapping[str, Any],
) -> tuple[list[int], str]:
    axis = AXIS_INDEX[str(card["selectors"]["fixed"]["axis"])]
    bbox = mesh.report["geometry_bbox_mm"]
    fixed_side = str(card["selectors"]["fixed"]["side"])
    loaded_side = str(card["selectors"]["loaded"]["side"])
    fixed_plane = float(bbox[axis if fixed_side == "min" else axis + 3])
    loaded_plane = float(bbox[axis if loaded_side == "min" else axis + 3])
    distance = float(card["analysis"]["stress_exclusion_mesh_lengths"]) * float(
        mesh.report["characteristic_length_mm"]
    )
    selected = [
        node_id
        for node_id in stress_nodes
        if abs(coordinates[node_id][axis] - fixed_plane) > distance
        and abs(coordinates[node_id][axis] - loaded_plane) > distance
    ]
    if len(selected) >= 20:
        return selected, "exclude_fixed_and_loaded_plane_neighborhoods"
    boundary = set(mesh.fixed_nodes) | set(mesh.loaded_nodes)
    selected = [node_id for node_id in stress_nodes if node_id not in boundary]
    if len(selected) >= 5:
        return selected, "fallback_exclude_boundary_nodes"
    return sorted(stress_nodes), "fallback_all_stress_nodes"


def parse_calculix_frd(
    path: Path, mesh: MeshData, card: Mapping[str, Any]
) -> ParsedFields:
    blocks = _frd_blocks(path)
    displacement = blocks.get("DISP", {})
    stress = blocks.get("STRESS", {})
    force = blocks.get("FORC", {})
    if not displacement or not stress or not force:
        raise L3StageError(
            "result_sanity", "FRD is missing DISP, STRESS, or FORC output"
        )
    displacements = {
        node: tuple(float(value) for value in values[:3])
        for node, values in displacement.items()
        if len(values) >= 3
    }
    von_mises = {
        node: _von_mises(values) for node, values in stress.items() if len(values) >= 6
    }
    if not displacements or not von_mises:
        raise L3StageError("result_sanity", "FRD result blocks are empty")
    coordinates = {node_id: (x, y, z) for node_id, x, y, z in mesh.nodes}
    magnitudes = {
        node: math.sqrt(sum(value * value for value in vector))
        for node, vector in displacements.items()
    }
    max_displacement_node = max(magnitudes, key=magnitudes.get)
    max_stress_node = max(von_mises, key=von_mises.get)
    reaction = [0.0, 0.0, 0.0]
    for node_id in mesh.fixed_nodes:
        values = force.get(node_id)
        if values is None or len(values) < 3:
            continue
        for index in range(3):
            reaction[index] += float(values[index])
    applied = [float(value) for value in mesh.applied_force]
    imbalance = [reaction[index] + applied[index] for index in range(3)]
    applied_norm = math.sqrt(sum(value * value for value in applied))
    equilibrium_error = math.sqrt(sum(value * value for value in imbalance)) / max(
        applied_norm, 1e-12
    )
    compliance = sum(
        sum(
            float(force_component) * float(displacements[node_id][index])
            for index, force_component in enumerate(nodal_force)
        )
        for node_id, nodal_force in mesh.nodal_forces.items()
        if node_id in displacements
    )
    span_axis = AXIS_INDEX[str(card["selectors"]["fixed"]["axis"])]
    bbox = mesh.report["geometry_bbox_mm"]
    span = float(bbox[span_axis + 3] - bbox[span_axis])
    youngs_modulus = float(card["material"]["youngs_modulus_mpa"])
    regular_nodes, regularization = _regularized_stress_nodes(
        stress_nodes=set(von_mises),
        coordinates=coordinates,
        mesh=mesh,
        card=card,
    )
    regular_values = np.asarray([von_mises[node] for node in regular_nodes])
    percentiles = {
        f"p{value:g}": float(np.percentile(regular_values, value))
        for value in card["analysis"]["regularized_stress_percentiles"]
    }
    traction = [float(value) for value in card["load"]["traction_vector_mpa"]]
    traction_magnitude = math.sqrt(sum(value * value for value in traction))
    stress_p95 = percentiles.get("p95")
    if stress_p95 is None:
        stress_p95 = float(np.percentile(regular_values, 95.0))
    metrics = {
        "max_displacement_mm": magnitudes[max_displacement_node],
        "max_displacement_node": max_displacement_node,
        "normalized_max_displacement": magnitudes[max_displacement_node] / span,
        "raw_max_nodal_von_mises_mpa": von_mises[max_stress_node],
        "raw_max_nodal_von_mises_node": max_stress_node,
        "regularized_von_mises_mpa": percentiles,
        "regularized_stress_node_count": len(regular_nodes),
        "stress_regularization": regularization,
        "stress_concentration_p95": stress_p95 / traction_magnitude,
        "compliance_n_mm": compliance,
        "normalized_compliance": compliance
        * youngs_modulus
        * span
        / max(applied_norm * applied_norm, 1e-24),
        "fixed_reaction_force_n": reaction,
        "applied_force_n": applied,
        "force_equilibrium_relative_error": equilibrium_error,
    }
    scalar_values = [
        metrics["max_displacement_mm"],
        metrics["raw_max_nodal_von_mises_mpa"],
        metrics["normalized_max_displacement"],
        metrics["stress_concentration_p95"],
        metrics["compliance_n_mm"],
        metrics["normalized_compliance"],
        metrics["force_equilibrium_relative_error"],
        *percentiles.values(),
    ]
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise L3StageError("result_sanity", "non-finite result metric")
    return ParsedFields(metrics, displacements, von_mises)


def _write_visualization(
    *, mesh: MeshData, fields: ParsedFields, output_dir: Path
) -> dict[str, Any]:
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise L3StageError("visualization", "VTK is unavailable") from exc

    point_index: dict[int, int] = {}
    points = vtk.vtkPoints()
    for index, (node_id, x, y, z) in enumerate(mesh.nodes):
        point_index[node_id] = index
        points.InsertNextPoint(x, y, z)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    for element in mesh.elements:
        if mesh.element_type == "C3D10":
            tetra = vtk.vtkQuadraticTetra()
            connectivity = element[1:11]
        else:
            tetra = vtk.vtkTetra()
            connectivity = element[1:5]
        for index, node_id in enumerate(connectivity):
            tetra.GetPointIds().SetId(index, point_index[node_id])
        grid.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())
    stress = vtk.vtkDoubleArray()
    stress.SetName("von_mises_mpa")
    stress.SetNumberOfComponents(1)
    displacement = vtk.vtkDoubleArray()
    displacement.SetName("displacement_mm")
    displacement.SetNumberOfComponents(3)
    for node_id, *_ in mesh.nodes:
        stress.InsertNextValue(float(fields.von_mises.get(node_id, 0.0)))
        displacement.InsertNextTuple3(
            *fields.displacements.get(node_id, (0.0, 0.0, 0.0))
        )
    grid.GetPointData().SetScalars(stress)
    grid.GetPointData().SetVectors(displacement)

    vtu_path = output_dir / "von_mises.vtu"
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(vtu_path))
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise L3StageError("visualization", "VTK result file could not be written")

    # The HPC VTK wheel requires an X display even when offscreen rendering is
    # requested. Keep VTK for the exact quadratic VTU artifact and render the
    # PNG with Matplotlib's display-independent Agg backend.
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import colormaps
    from matplotlib import colors as mpl_colors
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    values = np.asarray(list(fields.von_mises.values()), dtype=float)
    low, high = np.percentile(values, [2.0, 98.0])
    if high <= low:
        high = low + 1e-9
    boundary: dict[tuple[int, int, int], tuple[int, ...] | None] = {}
    for element in mesh.elements:
        nodes = element[1:]
        if mesh.element_type == "C3D10":
            faces = (
                (nodes[0], nodes[1], nodes[2], nodes[4], nodes[5], nodes[6]),
                (nodes[0], nodes[1], nodes[3], nodes[4], nodes[8], nodes[7]),
                (nodes[1], nodes[2], nodes[3], nodes[5], nodes[9], nodes[8]),
                (nodes[2], nodes[0], nodes[3], nodes[6], nodes[7], nodes[9]),
            )
        else:
            faces = (
                (nodes[0], nodes[1], nodes[2]),
                (nodes[0], nodes[1], nodes[3]),
                (nodes[1], nodes[2], nodes[3]),
                (nodes[2], nodes[0], nodes[3]),
            )
        for face in faces:
            key = tuple(sorted(face[:3]))
            boundary[key] = face if key not in boundary else None

    coordinates = {node_id: (x, y, z) for node_id, x, y, z in mesh.nodes}
    panels: list[list[tuple[float, float, float]]] = []
    panel_values: list[float] = []
    for face in boundary.values():
        if face is None:
            continue
        subdivisions = (
            (
                (face[0], face[3], face[5]),
                (face[3], face[1], face[4]),
                (face[5], face[4], face[2]),
                (face[3], face[4], face[5]),
            )
            if len(face) == 6
            else (face,)
        )
        for triangle in subdivisions:
            panels.append([coordinates[node_id] for node_id in triangle])
            panel_values.append(
                float(
                    np.mean(
                        [fields.von_mises.get(node_id, 0.0) for node_id in triangle]
                    )
                )
            )
    if not panels:
        raise L3StageError("visualization", "mesh has no exterior triangles")

    normalizer = mpl_colors.Normalize(vmin=float(low), vmax=float(high), clip=True)
    color_map = colormaps["turbo"]
    collection = Poly3DCollection(
        panels,
        facecolors=color_map(normalizer(np.asarray(panel_values))),
        edgecolors=(0.12, 0.12, 0.12, 0.35),
        linewidths=0.12,
        rasterized=True,
    )
    figure = plt.figure(figsize=(12, 9), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(collection)
    xyz = np.asarray([point for panel in panels for point in panel], dtype=float)
    minimum = xyz.min(axis=0)
    maximum = xyz.max(axis=0)
    center = (minimum + maximum) * 0.5
    half_span = max(float(np.max(maximum - minimum)) * 0.55, 1e-6)
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=24.0, azim=-55.0)
    axis.set_axis_off()
    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
    scalar.set_array([])
    color_bar = figure.colorbar(scalar, ax=axis, fraction=0.035, pad=0.02)
    color_bar.set_label("von Mises stress (MPa)")
    figure.suptitle(f"{mesh.element_type} linear-static response", fontsize=13)
    png_path = output_dir / "von_mises.png"
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise L3StageError("visualization", "von Mises PNG was not produced")
    return {
        "vtu_path": vtu_path.name,
        "vtu_sha256": sha256_file(vtu_path),
        "png_path": png_path.name,
        "png_sha256": sha256_file(png_path),
        "display_range_mpa": [float(low), float(high)],
    }


def _initial_stages() -> dict[str, dict[str, Any]]:
    return {stage: {"status": "not_reached"} for stage in STAGES}


def run_probe_card(
    card: Mapping[str, Any],
    *,
    step_path: Path,
    output_dir: Path,
    ccx_path: Path,
    timeout_seconds: int = 300,
    render: bool = False,
) -> dict[str, Any]:
    """Run one source or candidate STEP through the complete L3 stage cascade."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stages = _initial_stages()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "card_id": card.get("card_id"),
        "item_id": card.get("item_id"),
        "scoreable": card.get("status") == "solver_qualified",
        "source_intent_claimed": False,
        "evaluator_build_hash": physics_evaluator_build_hash(),
        "stages": stages,
        "final_status": "failed",
    }
    started = time.monotonic()
    try:
        validate_physics_card(card)
        write_json(output_dir / "physics_card.json", card)
        stages["card"] = {
            "status": "passed",
            "card_sha256": sha256_file(output_dir / "physics_card.json"),
        }
        runtime_lock = os.environ.get("CADENGBENCH_L3_RUNTIME_LOCK")
        if runtime_lock:
            runtime_lock_path = Path(runtime_lock)
            if not runtime_lock_path.is_file():
                raise L3StageError(
                    "runtime",
                    f"runtime lock is unavailable: {runtime_lock_path}",
                    owner="evaluator",
                )
            result["runtime_lock_sha256"] = sha256_file(runtime_lock_path)
            stages["runtime"] = {
                "status": "passed",
                "runtime_lock_sha256": result["runtime_lock_sha256"],
            }
        else:
            stages["runtime"] = {
                "status": "unpinned_development",
                "official": False,
            }
        if not step_path.is_file():
            raise L3StageError(
                "geometry",
                f"STEP file does not exist: {step_path}",
                owner="evaluator",
            )
        if not ccx_path.is_file() or not os.access(ccx_path, os.X_OK):
            raise L3StageError(
                "solver",
                f"CalculiX executable is unavailable: {ccx_path}",
                owner="evaluator",
            )
        stages["geometry"] = {
            "status": "passed",
            "step_sha256": sha256_file(step_path),
            "expected_solid_count": card["geometry"]["expected_solid_count"],
        }
        mesh = mesh_step(card, step_path, output_dir)
        stages["selectors"] = {
            "status": "passed",
            "fixed": mesh.report["fixed_selector"],
            "loaded": mesh.report["loaded_selector"],
        }
        stages["mesh"] = {"status": "passed", **mesh.report}
        write_json(output_dir / "mesh_report.json", mesh.report)
        deck_path = output_dir / "analysis.inp"
        write_calculix_deck(card, mesh, deck_path)
        stages["deck"] = {
            "status": "passed",
            "deck_sha256": sha256_file(deck_path),
            "template": "linear_static_structural_probe_v3",
        }
        solver_report = run_calculix(
            ccx_path, deck_path, timeout_seconds=timeout_seconds
        )
        stages["solver"] = {"status": "passed", **solver_report}
        fields = parse_calculix_frd(deck_path.with_suffix(".frd"), mesh, card)
        metrics = dict(fields.metrics)
        stages["result_sanity"] = {"status": "passed"}
        checks = {
            "force_equilibrium": {
                "passed": metrics["force_equilibrium_relative_error"]
                <= float(card["analysis"]["max_equilibrium_relative_error"]),
                "observed": metrics["force_equilibrium_relative_error"],
                "limit": card["analysis"]["max_equilibrium_relative_error"],
            },
            "small_deformation": {
                "passed": metrics["normalized_max_displacement"]
                <= float(card["analysis"]["max_normalized_displacement"]),
                "observed": metrics["normalized_max_displacement"],
                "limit": card["analysis"]["max_normalized_displacement"],
            },
            "regularized_von_mises": {
                "passed": True,
                "diagnostic_only": True,
                "observed": metrics["regularized_von_mises_mpa"],
            },
            "normalized_compliance": {
                "passed": True,
                "diagnostic_only": True,
                "observed": metrics["normalized_compliance"],
            },
        }
        required_pass = all(
            value["passed"]
            for value in checks.values()
            if not value.get("diagnostic_only", False)
        )
        stages["requirements"] = {
            "status": "passed" if required_pass else "failed",
            "checks": checks,
        }
        if render:
            try:
                stages["visualization"] = {
                    "status": "passed",
                    **_write_visualization(
                        mesh=mesh, fields=fields, output_dir=output_dir
                    ),
                }
            except Exception as exc:  # optional evidence must not alter physics
                stages["visualization"] = {
                    "status": "failed",
                    "required": False,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
        else:
            stages["visualization"] = {
                "status": "not_requested",
                "required": False,
            }
        result["metrics"] = metrics
        result["checks"] = checks
        result["final_status"] = "probe_pass" if required_pass else "requirement_fail"
    except PhysicsCardError as exc:
        stages["card"] = {"status": "failed", "message": str(exc)}
        result["failure"] = {
            "stage": "card",
            "owner": "evaluator",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    except L3StageError as exc:
        stages[exc.stage] = {"status": "failed", "message": str(exc)}
        result["failure"] = {
            "stage": exc.stage,
            "owner": exc.owner,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    except Exception as exc:  # pragma: no cover - defensive evidence boundary
        result["failure"] = {
            "stage": "internal",
            "owner": "evaluator",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }
    result["runtime_seconds"] = time.monotonic() - started
    write_json(output_dir / "result.json", result)
    return result
