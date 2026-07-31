"""Single-support CalculiX body-acceleration linear-static probe."""

from __future__ import annotations

import math
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from cadengbench_p_eval.physics.calculix import (
    GMSH_TO_CALCULIX_C3D10,
    L3StageError,
    MeshData,
    ParsedFields,
    _frd_blocks,
    _global_bbox,
    _mesh_size,
    _resolve_selector,
    _surface_nodes,
    _von_mises,
    _write_visualization,
    run_calculix,
)

from cadengbench_p_l3_extended.contracts import ContractError, validate_body_card
from cadengbench_p_l3_extended.io import sha256_file, write_json
from cadengbench_p_l3_extended.provenance import evaluator_build_hash

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
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


def _mesh_body(
    card: Mapping[str, Any], step_path: Path, output_dir: Path
) -> MeshData:
    try:
        import gmsh
    except ImportError as exc:
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
        if not imported or len(volumes) != 1:
            raise L3StageError(
                "geometry", f"expected one solid volume, found {len(volumes)}"
            )
        global_bbox = _global_bbox(gmsh, volumes)
        fixed_surfaces, fixed_evidence = _resolve_selector(
            gmsh, card["selector"], global_bbox
        )
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in volumes], name="EALL")
        gmsh.model.addPhysicalGroup(2, fixed_surfaces, name="FIXED_SURFACES")
        size = _mesh_size(global_bbox, card["mesh"])
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size)
        gmsh.option.setNumber("Mesh.Algorithm3D", int(card["mesh"]["algorithm_3d"]))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        started = time.monotonic()
        try:
            gmsh.model.mesh.generate(3)
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
        element_types, tag_groups, node_groups = gmsh.model.mesh.getElements(3)
        elements = []
        element_tags = []
        unexpected = []
        for element_type, tags, flattened in zip(
            element_types, tag_groups, node_groups, strict=True
        ):
            name, dimension, order, node_count, _, _ = (
                gmsh.model.mesh.getElementProperties(element_type)
            )
            if int(element_type) != 11 or int(node_count) != 10:
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
                offset = index * 10
                raw = values[offset : offset + 10]
                connectivity = [
                    raw[position] for position in GMSH_TO_CALCULIX_C3D10
                ]
                elements.append((int(element_tag), *connectivity))
                element_tags.append(int(element_tag))
        if unexpected or not elements:
            raise L3StageError(
                "mesh", f"mesh has unsupported volume elements: {unexpected}"
            )
        if len(elements) > int(card["mesh"]["maximum_elements"]):
            raise L3StageError(
                "mesh",
                f"element count {len(elements)} exceeds "
                f"{card['mesh']['maximum_elements']}",
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
        exact_volume_mm3 = sum(
            float(gmsh.model.occ.getMass(3, tag)) for _, tag in volumes
        )
        mesh_volume_mm3 = 0.0
        density = float(card["material"]["density_tonne_per_mm3"])
        acceleration = np.asarray(
            card["load"]["acceleration_vector_mm_s2"], dtype=float
        )
        accumulated: dict[int, np.ndarray] = {}
        consistent_weights = (-0.05,) * 4 + (0.2,) * 6
        for element in elements:
            points = [
                np.asarray(coordinates[node_id], dtype=float)
                for node_id in element[1:5]
            ]
            element_volume = abs(
                float(
                    np.dot(
                        points[1] - points[0],
                        np.cross(points[2] - points[0], points[3] - points[0]),
                    )
                )
            ) / 6.0
            mesh_volume_mm3 += element_volume
            element_force = density * element_volume * acceleration
            for node_id, weight in zip(
                element[1:11], consistent_weights, strict=True
            ):
                accumulated.setdefault(node_id, np.zeros(3, dtype=float))
                accumulated[node_id] += element_force * weight
        if not math.isfinite(mesh_volume_mm3) or mesh_volume_mm3 <= 0:
            raise L3StageError("mesh", "straight-sided mesh volume is invalid")
        nodal_forces = {
            node_id: tuple(float(value) for value in force)
            for node_id, force in sorted(accumulated.items())
        }
        applied = tuple(
            sum(force[index] for force in nodal_forces.values())
            for index in range(3)
        )
        mesh_path = output_dir / "mesh.msh"
        gmsh.write(str(mesh_path))
        report = {
            "gmsh_version": gmsh.option.getString("General.Version"),
            "gmsh_threads": 1,
            "geometry_bbox_mm": [float(value) for value in global_bbox],
            "volume_entity_count": 1,
            "exact_occ_volume_mm3": exact_volume_mm3,
            "straight_sided_mesh_volume_mm3": mesh_volume_mm3,
            "mesh_volume_relative_discrepancy": abs(
                mesh_volume_mm3 - exact_volume_mm3
            )
            / max(exact_volume_mm3, 1e-24),
            "fixed_selector": fixed_evidence,
            "node_count": len(nodes),
            "element_count": len(elements),
            "fixed_node_count": len(fixed_nodes),
            "element_type": "C3D10",
            "second_order_linear": True,
            "connectivity_mapping": (
                "gmsh_tetra10_to_calculix_c3d10_swap_nodes_9_10_v1"
            ),
            "characteristic_length_mm": size,
            "quality_metric": "minSICN",
            "minimum_quality": minimum_quality,
            "mean_quality": float(np.mean(qualities)),
            "density_tonne_per_mm3": density,
            "acceleration_vector_mm_s2": acceleration.tolist(),
            "body_load_distribution": "c3d10_consistent_nodal_body_force_v1",
            "loaded_node_count": len(nodal_forces),
            "applied_force_n": list(applied),
            "mesh_runtime_seconds": mesh_runtime,
            "mesh_path": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
        }
        return MeshData(
            nodes=nodes,
            elements=tuple(sorted(elements)),
            element_type="C3D10",
            fixed_nodes=fixed_nodes,
            loaded_nodes=tuple(sorted(nodal_forces)),
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


def _write_ids(stream: Any, values: Sequence[int]) -> None:
    for index in range(0, len(values), 16):
        stream.write(", ".join(str(value) for value in values[index : index + 16]))
        stream.write("\n")


def write_body_deck(
    card: Mapping[str, Any], mesh: MeshData, path: Path
) -> None:
    material = card["material"]
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("*HEADING\n")
        stream.write(f"{card['card_id']}\n")
        stream.write("*NODE\n")
        for node_id, x, y, z in mesh.nodes:
            stream.write(f"{node_id}, {x:.12g}, {y:.12g}, {z:.12g}\n")
        stream.write("*ELEMENT, TYPE=C3D10, ELSET=EALL\n")
        for element in mesh.elements:
            stream.write(", ".join(str(value) for value in element) + "\n")
        stream.write("*NSET, NSET=FIXED\n")
        _write_ids(stream, mesh.fixed_nodes)
        stream.write("*SOLID SECTION, ELSET=EALL, MATERIAL=PROBE_MATERIAL\n")
        stream.write("*MATERIAL, NAME=PROBE_MATERIAL\n")
        stream.write("*ELASTIC\n")
        stream.write(
            f"{float(material['youngs_modulus_mpa']):.12g}, "
            f"{float(material['poisson_ratio']):.12g}\n"
        )
        stream.write("*DENSITY\n")
        stream.write(f"{float(material['density_tonne_per_mm3']):.12g}\n")
        stream.write("*BOUNDARY\n")
        stream.write("FIXED, 1, 3\n")
        stream.write("*STEP, NLGEOM=NO\n")
        stream.write("*STATIC\n")
        stream.write("1.0, 1.0\n")
        stream.write("*CLOAD\n")
        for node_id, force in mesh.nodal_forces.items():
            for component, value in enumerate(force, start=1):
                if abs(value) > 0:
                    stream.write(f"{node_id}, {component}, {value:.12g}\n")
        stream.write("*NODE FILE\n")
        stream.write("U, RF\n")
        stream.write("*EL FILE\n")
        stream.write("S, E\n")
        stream.write("*NODE PRINT, NSET=FIXED, TOTALS=YES\n")
        stream.write("RF\n")
        stream.write("*END STEP\n")


def _parse_body_fields(
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
    coordinates = {node: (x, y, z) for node, x, y, z in mesh.nodes}
    magnitudes = {
        node: math.sqrt(sum(value * value for value in vector))
        for node, vector in displacements.items()
    }
    reported_reaction = [0.0, 0.0, 0.0]
    for node_id in mesh.fixed_nodes:
        values = force.get(node_id)
        if values is None or len(values) < 3:
            continue
        for index in range(3):
            reported_reaction[index] += float(values[index])
    constrained_external = [
        sum(
            float(mesh.nodal_forces.get(node_id, (0.0, 0.0, 0.0))[index])
            for node_id in mesh.fixed_nodes
        )
        for index in range(3)
    ]
    reaction = [
        reported_reaction[index] - constrained_external[index]
        for index in range(3)
    ]
    applied = [float(value) for value in mesh.applied_force]
    imbalance = [reaction[index] + applied[index] for index in range(3)]
    applied_norm = math.sqrt(sum(value * value for value in applied))
    equilibrium = math.sqrt(sum(value * value for value in imbalance)) / max(
        applied_norm, 1e-24
    )
    axis = AXIS_INDEX[str(card["selector"]["axis"])]
    bbox = mesh.report["geometry_bbox_mm"]
    span = float(bbox[axis + 3] - bbox[axis])
    fixed_plane = float(
        bbox[axis if card["selector"]["side"] == "min" else axis + 3]
    )
    exclusion = float(
        card["analysis"]["stress_exclusion_mesh_lengths"]
    ) * float(mesh.report["characteristic_length_mm"])
    regular_nodes = [
        node
        for node in von_mises
        if abs(coordinates[node][axis] - fixed_plane) > exclusion
    ]
    regularization = "exclude_fixed_plane_neighborhood"
    if len(regular_nodes) < 20:
        regular_nodes = [
            node for node in von_mises if node not in set(mesh.fixed_nodes)
        ]
        regularization = "fallback_exclude_fixed_nodes"
    if len(regular_nodes) < 5:
        regular_nodes = sorted(von_mises)
        regularization = "fallback_all_stress_nodes"
    regular_values = np.asarray([von_mises[node] for node in regular_nodes])
    percentiles = {
        f"p{value:g}": float(np.percentile(regular_values, value))
        for value in card["analysis"]["regularized_stress_percentiles"]
    }
    acceleration = [
        float(value) for value in card["load"]["acceleration_vector_mm_s2"]
    ]
    acceleration_norm = math.sqrt(sum(value * value for value in acceleration))
    characteristic_stress = (
        float(card["material"]["density_tonne_per_mm3"])
        * acceleration_norm
        * span
    )
    max_displacement_node = max(magnitudes, key=magnitudes.get)
    max_stress_node = max(von_mises, key=von_mises.get)
    stress_p95 = percentiles.get(
        "p95", float(np.percentile(regular_values, 95.0))
    )
    metrics = {
        "max_displacement_mm": magnitudes[max_displacement_node],
        "max_displacement_node": max_displacement_node,
        "normalized_max_displacement": magnitudes[max_displacement_node] / span,
        "raw_max_nodal_von_mises_mpa": von_mises[max_stress_node],
        "raw_max_nodal_von_mises_node": max_stress_node,
        "regularized_von_mises_mpa": percentiles,
        "regularized_stress_node_count": len(regular_nodes),
        "stress_regularization": regularization,
        "body_force_stress_concentration_p95": (
            stress_p95 / max(characteristic_stress, 1e-30)
        ),
        "reported_fixed_reaction_force_n": reported_reaction,
        "constrained_external_force_n": constrained_external,
        "fixed_reaction_force_n": reaction,
        "applied_force_n": applied,
        "force_equilibrium_relative_error": equilibrium,
    }
    scalars = [
        metrics["max_displacement_mm"],
        metrics["normalized_max_displacement"],
        metrics["raw_max_nodal_von_mises_mpa"],
        metrics["body_force_stress_concentration_p95"],
        metrics["force_equilibrium_relative_error"],
        *percentiles.values(),
    ]
    if not all(math.isfinite(float(value)) for value in scalars):
        raise L3StageError("result_sanity", "non-finite body-force result metric")
    return ParsedFields(metrics, displacements, von_mises)


def run_body_card(
    card: Mapping[str, Any],
    *,
    step_path: Path,
    output_dir: Path,
    ccx_path: Path,
    timeout_seconds: int = 300,
    render: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = {stage: {"status": "not_reached"} for stage in STAGES}
    result: dict[str, Any] = {
        "schema_version": "cadengbench-p.l3-body-acceleration-result.v1",
        "card_id": card.get("card_id"),
        "item_id": card.get("item_id"),
        "scoreable": card.get("status") == "solver_qualified",
        "source_intent_claimed": False,
        "evaluator_build_hash": evaluator_build_hash(),
        "stages": stages,
        "final_status": "failed",
    }
    started = time.monotonic()
    try:
        validate_body_card(card)
        write_json(output_dir / "body_card.json", card)
        stages["card"] = {
            "status": "passed",
            "card_sha256": sha256_file(output_dir / "body_card.json"),
        }
        runtime_lock = os.environ.get("CADENGBENCH_L3_RUNTIME_LOCK")
        if runtime_lock:
            lock_path = Path(runtime_lock)
            if not lock_path.is_file():
                raise L3StageError(
                    "runtime",
                    f"runtime lock is unavailable: {lock_path}",
                    owner="evaluator",
                )
            result["runtime_lock_sha256"] = sha256_file(lock_path)
            stages["runtime"] = {
                "status": "passed",
                "runtime_lock_sha256": result["runtime_lock_sha256"],
            }
        else:
            stages["runtime"] = {"status": "unpinned_development", "official": False}
        if not step_path.is_file():
            raise L3StageError("geometry", f"STEP does not exist: {step_path}")
        stages["geometry"] = {
            "status": "passed",
            "step_sha256": sha256_file(step_path),
            "expected_solid_count": 1,
        }
        mesh = _mesh_body(card, step_path, output_dir)
        stages["selectors"] = {
            "status": "passed",
            "fixed": mesh.report["fixed_selector"],
        }
        stages["mesh"] = {"status": "passed", **mesh.report}
        write_json(output_dir / "mesh_report.json", mesh.report)
        deck_path = output_dir / "analysis.inp"
        write_body_deck(card, mesh, deck_path)
        stages["deck"] = {
            "status": "passed",
            "deck_sha256": sha256_file(deck_path),
            "template": "single_support_consistent_body_acceleration_v1",
        }
        stages["solver"] = {
            "status": "passed",
            **run_calculix(ccx_path, deck_path, timeout_seconds=timeout_seconds),
        }
        fields = _parse_body_fields(deck_path.with_suffix(".frd"), mesh, card)
        metrics = dict(fields.metrics)
        stages["result_sanity"] = {"status": "passed"}
        checks = {
            "force_equilibrium": {
                "observed": metrics["force_equilibrium_relative_error"],
                "limit": card["analysis"]["max_equilibrium_relative_error"],
                "passed": metrics["force_equilibrium_relative_error"]
                <= float(card["analysis"]["max_equilibrium_relative_error"]),
            },
            "small_deformation": {
                "observed": metrics["normalized_max_displacement"],
                "limit": card["analysis"]["max_normalized_displacement"],
                "passed": metrics["normalized_max_displacement"]
                <= float(card["analysis"]["max_normalized_displacement"]),
            },
            "regularized_von_mises": {
                "observed": metrics["regularized_von_mises_mpa"],
                "diagnostic_only": True,
                "passed": True,
            },
        }
        required_pass = all(
            check["passed"]
            for check in checks.values()
            if not check.get("diagnostic_only", False)
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
            except Exception as exc:
                stages["visualization"] = {
                    "status": "failed",
                    "required": False,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
        else:
            stages["visualization"] = {"status": "not_requested", "required": False}
        result["metrics"] = metrics
        result["checks"] = checks
        result["final_status"] = "probe_pass" if required_pass else "requirement_fail"
    except ContractError as exc:
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
    except Exception as exc:
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
