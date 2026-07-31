#!/usr/bin/env python3
"""Validate supplement structure, anonymity, sample integrity, and L3 bindings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv-a",
    ".venv-p",
    ".solver",
    "__pycache__",
    "outputs",
}


def payload_files() -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(ROOT):
        directories[:] = sorted(
            name for name in directories if name not in EXCLUDED_DIRECTORIES
        )
        current_path = Path(current)
        files.extend(current_path / name for name in sorted(names))
    return sorted(files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_a_sample() -> None:
    sample = ROOT / "datasets/cadengbench_a_sample7_v2"
    manifest = json.loads((sample / "sample_manifest.json").read_text())
    require(manifest["benchmark_layers"] == ["A0", "A1", "A2"], "A layers differ")
    require(manifest["item_count"] == 7, "A sample must contain seven items")
    for row in manifest["files"]:
        path = sample / row["path"]
        require(path.is_file(), f"missing A sample file: {row['path']}")
        require(path.stat().st_size == row["bytes"], f"A byte mismatch: {row['path']}")
        require(sha256(path) == row["sha256"], f"A hash mismatch: {row['path']}")


def verify_p_sample() -> None:
    sample = ROOT / "datasets/cadengbench_p_sample10_v1"
    items = sorted((sample / "items").glob("p_*"))
    require(len(items) == 10, "P sample must contain ten items")
    item_ids = {item.name for item in items}
    for control_name in ("positive", "mutant_parameter_scale"):
        control_ids = {
            path.stem
            for path in (sample / "submissions" / control_name).glob("p_*.py")
        }
        require(
            control_ids == item_ids,
            f"P {control_name} controls differ from selected items",
        )
    sources = {"benchcad": 0, "fusion360": 0}
    for item in items:
        source = "benchcad" if item.name.startswith("p_benchcad_") else "fusion360"
        sources[source] += 1
        for relative in (
            "benchmark/consumer_ready_item_v2.json",
            "evaluation/qualification.v1.json",
            "item.json",
            "l3/annotation_candidate.json",
            "l3/annotation_metadata.json",
            "l3/annotation_validation.json",
            "l3/bundle.json",
            "l3/context.json",
            "l3/input_image.png",
            "l3/physics_card.json",
            "l3/physics_card.solver_qualified.json",
            "l3/provider_response.json",
            "l3/response_contract.json",
            "l3/source_qualification/evidence_manifest.json",
            "l3/source_qualification/mesh_report.json",
            "l3/source_qualification/qualification.json",
            "l3/source_qualification/result.json",
        ):
            require((item / relative).is_file(), f"missing P runtime file: {item.name}/{relative}")
        card_path = item / "l3/physics_card.solver_qualified.json"
        contract_path = item / "l3/response_contract.json"
        card = json.loads(card_path.read_text())
        contract = json.loads(contract_path.read_text())
        require(card["item_id"] == item.name, f"L3 card item mismatch: {item.name}")
        require(card["status"] == "solver_qualified", f"L3 card is not qualified: {item.name}")
        require(card["source_intent_claimed"] is False, f"invalid L3 intent claim: {item.name}")
        require(contract["item_id"] == item.name, f"L3 contract item mismatch: {item.name}")
        require(contract["status"] == "calibrated", f"L3 contract is not calibrated: {item.name}")
        require(
            sha256(card_path) == contract["bindings"]["base_card_sha256"],
            f"L3 card hash mismatch: {item.name}",
        )
        reference = item / "source" / (
            "generated_from_cadquery.step"
            if source == "benchcad"
            else "fusion360_occ_reconstruction.step"
        )
        require(reference.is_file(), f"missing L3 reference STEP: {item.name}")
        require(
            sha256(reference)
            == contract["bindings"]["nominal_reference_step_sha256"],
            f"L3 reference hash mismatch: {item.name}",
        )
    require(sources == {"benchcad": 6, "fusion360": 4}, f"unexpected P mix: {sources}")
    require(
        (ROOT / "schemas/cadengbench_p_fusion360_patch_submission_v1.schema.json").is_file(),
        "runtime Fusion functional-edit schema is missing",
    )
    require(
        not (sample / "l3_fixture").exists(),
        "standalone L3 fixture must not be present",
    )


def verify_scope_and_anonymity() -> None:
    require(
        not (ROOT / "code/cadengbench_a/exploratory").exists(),
        "discarded A graph code is present",
    )
    contract = json.loads(
        (
            ROOT
            / "code/cadengbench_a/dataset/cadengbench_a/benchmark_contract_v2/contract.json"
        ).read_text()
    )
    require(
        [row["id"] for row in contract["active_layers"]] == ["A0", "A1", "A2"],
        "A contract differs",
    )
    require(
        not any(path.is_symlink() for path in payload_files()),
        "payload symlinks are not allowed",
    )
    require(not (ROOT / ".env").exists(), "a populated .env must not be submitted")
    for relative in (
        "scripts/install_system_dependencies.sh",
        "scripts/resolve_ccx.sh",
        "scripts/run_l3_reference_controls.sh",
        "scripts/test_calculix.sh",
        "configs/l3/calculix_smoke.inp",
    ):
        require((ROOT / relative).is_file(), f"missing runtime dependency file: {relative}")
    forbidden = (
        "/" + "Users" + "/",
        "/" + "home" + "/",
    )
    skip_suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".step",
        ".stp",
        ".obj",
        ".msh",
        ".frd",
        ".pdf",
    }
    for path in payload_files():
        if path.suffix.lower() in skip_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden:
            require(
                token not in text,
                f"anonymity token {token!r} in {path.relative_to(ROOT)}",
            )


def verify_package_manifest() -> None:
    manifest_path = ROOT / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        path.relative_to(ROOT).as_posix(): path
        for path in payload_files()
        if path != manifest_path
    }
    declared = {row["path"]: row for row in manifest["files"]}
    require(set(declared) == set(expected), "package manifest file set differs")
    for relative, path in expected.items():
        row = declared[relative]
        require(path.stat().st_size == row["bytes"], f"byte mismatch: {relative}")
        require(sha256(path) == row["sha256"], f"hash mismatch: {relative}")


def main() -> int:
    verify_a_sample()
    verify_p_sample()
    verify_scope_and_anonymity()
    verify_package_manifest()
    print("Supplement verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
