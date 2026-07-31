"""End-to-end L0/L1/L2 orchestration and resume-safe batch execution."""

from __future__ import annotations

import math
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import __version__
from .aggregation import aggregate_run
from .case import LoadedCase, load_case
from .contracts import (
    CheckResult,
    EvaluationResult,
    LayerResult,
    Status,
    evaluation_result_from_dict,
    passed_value,
    reduce_required_status,
)
from .execution.static import inspect_submission
from .integrity import (
    RESULT_DIGEST_NAME,
    verify_evidence_manifest,
    write_evidence_manifest,
)
from .io import read_json, sha256_file, sha256_json, write_json, write_text
from .layers.l0 import evaluate_l0
from .layers.l1_isolated import evaluate_l1_isolated
from .layers.l2 import evaluate_l2
from .model_output import ModelOutputError, materialize_submission
from .parametrics.replay import (
    VariantEvaluationContext,
    VariantExecution,
    bind_parameter_manifest,
)
from .provenance import environment_record, evaluator_build_hash

RESULT_SCHEMA = "cadengbench-p.evaluation-result.v1"
RUN_MANIFEST_SCHEMA = "cadengbench-p.run-manifest.v2"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _not_reached(layer: str, upstream: str) -> LayerResult:
    return LayerResult.from_checks(
        layer,
        (
            CheckResult(
                f"{layer}.upstream",
                Status.NOT_REACHED,
                summary=f"{layer} was not reached because {upstream} did not pass",
            ),
        ),
    )


def _overall_status(layers: tuple[LayerResult, ...]) -> Status:
    if any(
        check.status is Status.EVALUATOR_ERROR
        for layer in layers
        for check in layer.checks
    ):
        return Status.EVALUATOR_ERROR
    statuses = {layer.status for layer in layers}
    if Status.EVALUATOR_ERROR in statuses:
        return Status.EVALUATOR_ERROR
    if Status.FAILED in statuses:
        return Status.FAILED
    if Status.NOT_SCOREABLE in statuses:
        return Status.NOT_SCOREABLE
    if Status.NOT_REACHED in statuses or Status.NOT_APPLICABLE in statuses:
        return Status.NOT_SCOREABLE
    return Status.PASSED


def _write_layer(output_root: Path, layer: LayerResult) -> None:
    write_json(output_root / f"{layer.layer.lower()}.json", layer.as_dict())


def _write_result(output_root: Path, result: EvaluationResult) -> None:
    """Publish immutable evidence, then the result and its commit digest."""

    output_root.mkdir(parents=True, exist_ok=True)
    write_evidence_manifest(output_root)
    result_path = output_root / "result.json"
    write_json(result_path, result.as_dict())
    write_text(output_root / RESULT_DIGEST_NAME, f"{sha256_file(result_path)}\n")


def _read_result_digest(output_root: Path) -> str:
    digest_path = output_root / RESULT_DIGEST_NAME
    if not digest_path.is_file():
        raise ValueError(f"resume result is missing {RESULT_DIGEST_NAME}")
    digest = digest_path.read_text(encoding="ascii").strip()
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("resume result digest is malformed")
    return digest


def _validate_serialized_result(
    value: Mapping[str, Any],
    *,
    case: LoadedCase,
    identity: Mapping[str, str],
    environment: Mapping[str, Any],
) -> EvaluationResult:
    invariants = {
        "schema_version": RESULT_SCHEMA,
        "evaluator_version": __version__,
        "item_id": case.item_id,
        "protocol": str(case.data["protocol"]),
        "identity": dict(identity),
        "environment": dict(environment),
        "llm_judge_used": False,
    }
    for field, expected in invariants.items():
        if value.get(field) != expected:
            raise ValueError(f"resume result invariant {field!r} does not match")

    try:
        result = evaluation_result_from_dict(dict(value))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"resume result is malformed: {exc}") from exc

    raw_layers = value.get("layers")
    if not isinstance(raw_layers, list) or len(raw_layers) != len(result.layers):
        raise ValueError("resume result has a malformed layer roster")
    if tuple(layer.layer for layer in result.layers) != ("L0", "L1", "L2"):
        raise ValueError("resume result must contain exactly L0, L1, and L2 in order")

    for raw_layer, layer in zip(raw_layers, result.layers, strict=True):
        if not isinstance(raw_layer, Mapping):
            raise ValueError("resume result contains a non-object layer")
        if raw_layer.get("passed") != passed_value(layer.status):
            raise ValueError(f"resume {layer.layer} passed field is inconsistent")
        reduced = reduce_required_status(layer.checks)
        if layer.status is not reduced:
            raise ValueError(
                f"resume {layer.layer} status does not reduce from its checks"
            )
        raw_checks = raw_layer.get("checks")
        if not isinstance(raw_checks, list) or len(raw_checks) != len(layer.checks):
            raise ValueError(f"resume {layer.layer} has a malformed check roster")
        check_ids: set[str] = set()
        for raw_check, check in zip(raw_checks, layer.checks, strict=True):
            if not isinstance(raw_check, Mapping):
                raise ValueError(f"resume {layer.layer} contains a malformed check")
            if check.check_id in check_ids:
                raise ValueError(
                    f"resume {layer.layer} repeats check {check.check_id!r}"
                )
            check_ids.add(check.check_id)
            if not isinstance(raw_check.get("required", True), bool):
                raise ValueError(
                    f"resume check {check.check_id!r} has a non-boolean required flag"
                )
            if raw_check.get("passed") != passed_value(check.status):
                raise ValueError(
                    f"resume check {check.check_id!r} has an inconsistent passed field"
                )

    recomputed_overall = _overall_status(result.layers)
    if result.overall_status is not recomputed_overall:
        raise ValueError("resume overall status does not reduce from its layers")
    if value.get("passed") != passed_value(result.overall_status):
        raise ValueError("resume overall passed field is inconsistent")
    return result


def _verify_result_artifacts(
    output_root: Path,
    *,
    value: Mapping[str, Any],
    case: LoadedCase,
    identity: Mapping[str, str],
    environment: Mapping[str, Any],
) -> None:
    case_path = output_root / "case.json"
    environment_path = output_root / "environment.json"
    if not case_path.is_file() or read_json(case_path) != case.data:
        raise ValueError("resume case evidence is absent or does not match")
    if not environment_path.is_file() or read_json(environment_path) != environment:
        raise ValueError("resume environment evidence is absent or does not match")

    raw_layers = value["layers"]
    for raw_layer in raw_layers:
        layer_name = str(raw_layer["layer"]).lower()
        layer_path = output_root / f"{layer_name}.json"
        if not layer_path.is_file() or read_json(layer_path) != raw_layer:
            raise ValueError(f"resume {layer_name} evidence is absent or inconsistent")

    submission_root = output_root / "submission"
    submission_manifest_path = submission_root / "manifest.json"
    if not submission_manifest_path.is_file():
        raise ValueError("resume submission manifest is missing")
    submission_manifest = read_json(submission_manifest_path)
    if not isinstance(submission_manifest, Mapping):
        raise ValueError("resume submission manifest is malformed")
    if submission_manifest.get("raw_sha256") != identity["raw_model_output_hash"]:
        raise ValueError("resume raw-output hash binding does not match")
    if submission_manifest.get("source_sha256") != identity["submission_source_hash"]:
        raise ValueError("resume source hash binding does not match")

    if case.data.get("protocol") == "functional_edit":
        filename = submission_manifest.get("filename")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename
        ):
            raise ValueError("resume functional submission filename is unsafe")
        submission_path = submission_root / filename
        if (
            not submission_path.is_file()
            or sha256_file(submission_path) != identity["submission_source_hash"]
        ):
            raise ValueError("resume functional submission failed SHA-256 validation")
        l0 = raw_layers[0]
        candidate_record = (l0.get("artifacts") or {}).get("candidate_step")
        if l0.get("status") == Status.PASSED.value and not isinstance(
            candidate_record, Mapping
        ):
            raise ValueError("resume passed Protocol-E L0 has no candidate STEP")
        if isinstance(candidate_record, Mapping):
            raw_candidate = candidate_record.get("path")
            if not isinstance(raw_candidate, str):
                raise ValueError("resume Protocol-E candidate path is malformed")
            candidate_path = Path(raw_candidate).resolve()
            try:
                candidate_path.relative_to(output_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    "resume Protocol-E candidate escapes its result root"
                ) from exc
            if (
                not candidate_path.is_file()
                or candidate_record.get("sha256") != sha256_file(candidate_path)
                or candidate_record.get("bytes") != candidate_path.stat().st_size
            ):
                raise ValueError(
                    "resume Protocol-E candidate failed SHA-256 validation"
                )
        verify_evidence_manifest(output_root)
        return

    source_path = submission_root / "submission.py"
    source_hash = identity["submission_source_hash"]
    if source_hash == "unavailable":
        if source_path.exists():
            raise ValueError("resume bundle has source code marked unavailable")
    elif not source_path.is_file() or sha256_file(source_path) != source_hash:
        raise ValueError("resume submission source artifact failed SHA-256 validation")

    raw_paths = sorted(
        path for path in submission_root.glob("raw_output*") if path.is_file()
    )
    raw_hash = identity["raw_model_output_hash"]
    if raw_hash == "missing":
        if raw_paths:
            raise ValueError("resume bundle unexpectedly contains a raw model output")
    elif len(raw_paths) != 1 or sha256_file(raw_paths[0]) != raw_hash:
        raise ValueError("resume raw model output artifact failed SHA-256 validation")

    l0 = raw_layers[0]
    candidate_record = (l0.get("artifacts") or {}).get("candidate_step")
    candidate_path = output_root / "l0" / "candidate.step"
    if candidate_record is None and candidate_path.exists():
        raise ValueError("resume candidate STEP is present without an artifact record")
    if candidate_record is not None:
        if not isinstance(candidate_record, Mapping):
            raise ValueError("resume candidate STEP artifact record is malformed")
        if candidate_record.get("path") != "candidate.step":
            raise ValueError("resume candidate STEP artifact path is not canonical")
        if (
            not candidate_path.is_file()
            or candidate_record.get("sha256") != sha256_file(candidate_path)
            or candidate_record.get("bytes") != candidate_path.stat().st_size
        ):
            raise ValueError("resume candidate STEP artifact failed validation")
    if l0.get("status") == Status.PASSED.value and candidate_record is None:
        raise ValueError("resume passed L0 result has no candidate STEP artifact")
    verify_evidence_manifest(output_root)


def _assert_safe_output_root(
    output_root: Path,
    *,
    protected_paths: tuple[Path, ...],
    protected_roots: tuple[Path, ...] = (),
) -> None:
    """Reject output locations whose replacement could destroy input or code."""

    output = output_root.expanduser().resolve()
    if output == Path("/"):
        raise ValueError("refusing to use the filesystem root as evaluator output")
    for protected_path in protected_paths:
        protected = protected_path.expanduser().resolve()
        if output == protected or output in protected.parents:
            raise ValueError(
                f"refusing evaluator output that can replace protected path {protected}"
            )
    for protected_root in protected_roots:
        protected = protected_root.expanduser().resolve()
        if (
            output == protected
            or output in protected.parents
            or protected in output.parents
        ):
            raise ValueError(
                f"refusing evaluator output that overlaps protected root {protected}"
            )

    package_root = Path(__file__).resolve().parents[2]
    for protected in (Path.cwd().resolve(), package_root):
        if output == protected or output in protected.parents:
            raise ValueError("refusing evaluator output that can replace the codebase")


def _prepare_item_output(output_root: Path, *, force: bool) -> None:
    """Recover interrupted item writes at the item transaction boundary."""

    if not output_root.exists():
        return
    if output_root.is_symlink():
        raise ValueError("evaluator output cannot be a symlink")
    if not output_root.is_dir():
        raise ValueError("evaluator output exists but is not a directory")
    if force:
        shutil.rmtree(output_root)
        return
    result_exists = (output_root / "result.json").is_file()
    digest_exists = (output_root / RESULT_DIGEST_NAME).is_file()
    if result_exists and digest_exists:
        return
    if any(output_root.iterdir()):
        shutil.rmtree(output_root)


def _l1_timeout(case: LoadedCase | Mapping[str, Any]) -> float:
    data = case.data if isinstance(case, LoadedCase) else case
    profile = data.get("evaluator_profile")
    if not isinstance(profile, Mapping):
        return 120.0
    geometry = profile.get("geometry")
    execution = profile.get("execution")
    fallback = 120.0
    if isinstance(execution, Mapping):
        fallback = float(execution.get("timeout_seconds", 60.0)) + float(
            execution.get("worker_startup_grace_seconds", 60.0)
        )
    timeout = float(
        geometry.get("evaluation_timeout_seconds", fallback)
        if isinstance(geometry, Mapping)
        else fallback
    )
    requirements = data.get("requirements")
    expected_solid_count = 1
    if isinstance(requirements, list):
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                continue
            contract = requirement.get("value_and_tolerance")
            body_count = (
                contract.get("body_count") if isinstance(contract, Mapping) else None
            )
            if (
                requirement.get("category") == "solid_contract"
                and isinstance(body_count, int)
                and not isinstance(body_count, bool)
                and body_count > expected_solid_count
            ):
                expected_solid_count = body_count
    if expected_solid_count > 8:
        return max(timeout, 1200.0)
    return timeout


def _analysis_from_l1(layer: LayerResult) -> tuple[str, dict[str, float]]:
    fingerprint = layer.metrics.get("geometry_fingerprint")
    measurements = layer.metrics.get("parameter_measurements")
    if (
        not isinstance(fingerprint, str)
        or _SHA256_PATTERN.fullmatch(fingerprint) is None
    ):
        raise ValueError("L1 did not produce a canonical geometry fingerprint")
    if not isinstance(measurements, Mapping):
        raise ValueError("L1 did not produce a parameter measurement packet")
    normalized: dict[str, float] = {}
    for key, value in measurements.items():
        if not isinstance(key, str) or not key:
            raise ValueError("L1 parameter measurement keys must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"L1 parameter measurement {key!r} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"L1 parameter measurement {key!r} is not finite")
        normalized[key] = number
    return fingerprint, normalized


def _missing_l1_analysis_status(layer: LayerResult) -> Status:
    """Preserve ownership when L1 cannot provide canonical geometry evidence."""

    if layer.status is Status.EVALUATOR_ERROR:
        return Status.EVALUATOR_ERROR
    return Status.FAILED


def _variant_case(
    case: LoadedCase, assignments: Mapping[str, float]
) -> tuple[dict[str, Any], list[str]]:
    """Build an in-memory parameter-matched requirement contract for one variant."""

    data = deepcopy(case.data)
    parameter_records = (data.get("parameter_system") or {}).get("parameters", [])
    parameters = {
        str(parameter.get("parameter_id")): parameter
        for parameter in parameter_records
        if isinstance(parameter, Mapping)
    }
    requirements = {
        str(requirement.get("requirement_id")): requirement
        for requirement in data.get("requirements", [])
        if isinstance(requirement, dict)
    }
    unbound: list[str] = []
    response_only: list[str] = []
    for parameter_id, value in assignments.items():
        parameter = parameters.get(parameter_id)
        variant_contract = (
            parameter.get("variant_requirement_contract")
            if isinstance(parameter, Mapping)
            else None
        )
        if (
            isinstance(variant_contract, Mapping)
            and variant_contract.get("kind") == "geometry_response_only"
        ):
            response_only.append(parameter_id)
            continue
        binding = (
            parameter.get("measurement_binding")
            if isinstance(parameter, Mapping)
            else None
        )
        if not isinstance(binding, Mapping):
            unbound.append(parameter_id)
            continue
        requirement = requirements.get(str(binding.get("requirement_id")))
        path = binding.get("requirement_value_path")
        if not isinstance(requirement, dict) or not isinstance(path, list) or not path:
            unbound.append(parameter_id)
            continue
        target: dict[str, Any] = requirement
        valid = True
        for component in path[:-1]:
            child = target.get(component)
            if not isinstance(component, str) or not isinstance(child, dict):
                valid = False
                break
            target = child
        final = path[-1]
        if not valid or not isinstance(final, str) or final not in target:
            unbound.append(parameter_id)
            continue
        target[final] = float(value)
    if assignments:
        dfm = data.get("dfm")
        families = dfm.get("families") if isinstance(dfm, Mapping) else None
        if isinstance(families, list):
            for family in families:
                if not isinstance(family, dict):
                    continue
                family["required"] = False
                family["scoreable"] = False
                family["applicability"] = "not_applicable_parameter_variant"
                family["unscoreable_reason"] = (
                    "the frozen source evidence does not prove this DFM threshold "
                    "invariant across the qualified parameter domain"
                )
            dfm["scoreable_family_count"] = 0
    if response_only:
        data["requirements"] = [
            requirement
            for requirement in data.get("requirements", [])
            if isinstance(requirement, Mapping)
            and requirement.get("category") == "solid_contract"
        ]
        data["variant_evaluation_scope"] = {
            "mode": "geometry_response_only",
            "parameter_ids": sorted(response_only),
            "required_invariants": ["valid_solid", "solid_count"],
            "parameter_response_required": True,
        }
    return data, sorted(unbound)


def _first_failure(layer: LayerResult) -> str | None:
    for check in layer.checks:
        if check.status is Status.FAILED:
            return check.failure_code or check.check_id
    return None


class _VariantBridge:
    def __init__(
        self,
        case: LoadedCase,
        submission: Path,
        output_root: Path,
        submission_contract: Mapping[str, Any],
    ) -> None:
        self.case = case
        self.submission = submission
        self.output_root = output_root
        self.l0_results: dict[str, LayerResult] = {}
        self.l1_results: dict[str, LayerResult] = {}
        self.binding_error: str | None = None
        try:
            self.bindings = bind_parameter_manifest(
                case.data["parameter_system"], submission_contract
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.bindings = {}
            self.binding_error = f"{type(exc).__name__}: {exc}"

    def execute(
        self, model_parameters: dict[str, float], variant_id: str
    ) -> VariantExecution:
        root = self.output_root / "variants" / variant_id
        l0 = evaluate_l0(
            case=self.case.data,
            submission_path=self.submission,
            output_dir=root,
            overrides=model_parameters,
        )
        self.l0_results[variant_id] = l0
        candidate = root / "candidate.step"
        if l0.status is not Status.PASSED or not candidate.is_file():
            failure_code = _first_failure(l0) if l0.status is Status.FAILED else None
            return VariantExecution(
                l0.status,
                failure_code=failure_code,
                details={"l0": l0.as_dict()},
            )
        if self.binding_error is not None:
            return VariantExecution(
                Status.EVALUATOR_ERROR,
                artifact=str(candidate),
                details={"parameter_binding_error": self.binding_error},
            )
        public_assignments = {
            parameter_id: float(model_parameters[binding.name])
            for parameter_id, binding in self.bindings.items()
        }
        variant_case, unbound = _variant_case(self.case, public_assignments)
        if unbound:
            l1 = LayerResult.from_checks(
                "L1",
                (
                    CheckResult(
                        "L1.variant.parameter_matched_contract",
                        Status.NOT_SCOREABLE,
                        summary=(
                            "variant requirements cannot be derived for public "
                            f"parameters: {unbound}"
                        ),
                        observed={"unbound_parameter_ids": unbound},
                    ),
                ),
            )
            self.l1_results[variant_id] = l1
            return VariantExecution(
                Status.NOT_SCOREABLE,
                artifact=str(candidate),
                details={"l0_output": str(root / "l0.json"), "l1": l1.as_dict()},
            )

        evaluated = evaluate_l1_isolated(
            variant_case,
            candidate,
            is_parameter_variant=True,
            l0_passed=True,
            timeout_seconds=_l1_timeout(variant_case),
            work_dir=root / "l1-worker",
        )
        variant_scope = variant_case.get("variant_evaluation_scope")
        response_only = (
            isinstance(variant_scope, Mapping)
            and variant_scope.get("mode") == "geometry_response_only"
        )
        binding_check = CheckResult(
            "L1.variant.parameter_matched_contract",
            Status.PASSED,
            expected={
                "public_assignments": dict(sorted(public_assignments.items())),
                "contract_mode": (
                    "geometry_response_only"
                    if response_only
                    else "direct_measurement_only"
                ),
            },
            summary=(
                "variant was evaluated for valid-solid, body-count, and non-no-op "
                "geometry response"
                if response_only
                else (
                    "variant was evaluated against parameter-matched deterministic "
                    "requirements without nominal target-shape comparison"
                )
            ),
        )
        l1 = LayerResult.from_checks(
            "L1",
            (binding_check, *evaluated.checks),
            metrics=evaluated.metrics,
            artifacts=evaluated.artifacts,
            runtime_seconds=evaluated.runtime_seconds,
        )
        self.l1_results[variant_id] = l1
        try:
            fingerprint, measurements = _analysis_from_l1(l1)
        except ValueError as exc:
            status = _missing_l1_analysis_status(l1)
            return VariantExecution(
                status,
                artifact=str(candidate),
                failure_code=(
                    "geometry_analysis_failed" if status is Status.FAILED else None
                ),
                details={
                    "l0_output": str(root / "l0.json"),
                    "l1": l1.as_dict(),
                    "analysis_error": str(exc),
                },
            )
        return VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint,
            artifact=str(candidate),
            measurements=measurements,
            details={
                "l0_output": str(root / "l0.json"),
                "l1_worker": str(root / "l1-worker"),
            },
        )

    def l0(
        self, execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult:
        del execution
        return self.l0_results[context.variant.variant_id]

    def l1(
        self, execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult:
        del execution
        return self.l1_results.get(
            context.variant.variant_id,
            LayerResult.from_checks(
                "L1",
                (
                    CheckResult(
                        "L1.process_isolation",
                        Status.EVALUATOR_ERROR,
                        summary="variant L1 result was not produced by the bridge",
                    ),
                ),
            ),
        )


def _identity(
    case: LoadedCase,
    *,
    raw_hash: str,
    source_hash: str,
    environment: dict[str, Any],
) -> dict[str, str]:
    parameters = (case.data.get("parameter_system") or {}).get("parameters", [])
    contract_modes = {
        str(contract.get("kind"))
        for parameter in parameters
        if isinstance(parameter, Mapping)
        and isinstance(
            contract := parameter.get("variant_requirement_contract"), Mapping
        )
    }
    l2_contract_mode = (
        next(iter(contract_modes))
        if len(contract_modes) == 1
        else "mixed"
        if contract_modes
        else "not_applicable"
    )
    return {
        "case_hash": str(case.data["case_hash"]),
        "profile_hash": str(case.data["profile_hash"]),
        "raw_model_output_hash": raw_hash,
        "submission_source_hash": source_hash,
        "evaluator_build_hash": evaluator_build_hash(),
        "environment_hash": str(environment["environment_hash"]),
        "source_family": str(case.data["source_family"]),
        "l2_contract_mode": l2_contract_mode,
    }


def _early_model_failure(
    case: LoadedCase,
    *,
    identity: dict[str, str],
    environment: dict[str, Any],
    failure_code: str,
    summary: str,
    started: float,
) -> EvaluationResult:
    l0 = LayerResult.from_checks(
        "L0",
        (
            CheckResult(
                "L0.submission.materialization",
                Status.FAILED,
                failure_code=failure_code,
                summary=summary,
            ),
        ),
    )
    layers = (l0, _not_reached("L1", "L0"), _not_reached("L2", "L0"))
    return EvaluationResult(
        schema_version=RESULT_SCHEMA,
        evaluator_version=__version__,
        item_id=case.item_id,
        protocol=str(case.data["protocol"]),
        identity=identity,
        layers=layers,
        overall_status=Status.FAILED,
        environment=environment,
        runtime_seconds=time.monotonic() - started,
    )


def _resume_result(
    output_root: Path,
    identity: dict[str, str],
    *,
    case: LoadedCase,
    environment: dict[str, Any],
) -> EvaluationResult | None:
    path = output_root / "result.json"
    if not path.is_file():
        return None
    recorded_digest = _read_result_digest(output_root)
    if sha256_file(path) != recorded_digest:
        raise ValueError("resume result failed its SHA-256 digest validation")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("resume result must be a JSON object")
    result = _validate_serialized_result(
        value,
        case=case,
        identity=identity,
        environment=environment,
    )
    _verify_result_artifacts(
        output_root,
        value=value,
        case=case,
        identity=identity,
        environment=environment,
    )
    return result


def evaluate_item(
    *,
    case_path: Path,
    submission_path: Path,
    output_root: Path,
    force: bool = False,
) -> EvaluationResult:
    """Evaluate one item and preserve a self-auditing evidence bundle."""

    started = time.monotonic()
    case = load_case(case_path, verify_artifacts=True)
    source_item_root = case.data.get("source_item_root")
    protected_roots = (
        (Path(source_item_root),)
        if isinstance(source_item_root, str) and source_item_root
        else ()
    )
    _assert_safe_output_root(
        output_root,
        protected_paths=(case_path, submission_path),
        protected_roots=protected_roots,
    )
    _prepare_item_output(output_root, force=force)
    environment = environment_record()
    execution_profile = case.data["evaluator_profile"]["execution"]
    max_bytes = int(execution_profile["max_source_bytes"])
    raw_hash = sha256_file(submission_path) if submission_path.is_file() else "missing"

    with tempfile.TemporaryDirectory(prefix="cadengbench-p-submission-") as raw:
        temporary_root = Path(raw)
        try:
            materialized = materialize_submission(
                submission_path,
                output_root=temporary_root,
                max_bytes=max_bytes,
            )
        except ModelOutputError as exc:
            identity = _identity(
                case,
                raw_hash=raw_hash,
                source_hash="unavailable",
                environment=environment,
            )
            if not force:
                existing = _resume_result(
                    output_root,
                    identity,
                    case=case,
                    environment=environment,
                )
                if existing is not None:
                    return existing
            if (output_root / "result.json").exists() and not force:
                raise ValueError(
                    "output contains a result for different inputs; use a new "
                    "directory or --force"
                ) from exc
            output_root.mkdir(parents=True, exist_ok=True)
            result = _early_model_failure(
                case,
                identity=identity,
                environment=environment,
                failure_code="invalid_model_output",
                summary=str(exc),
                started=started,
            )
            submission_root = output_root / "submission"
            submission_root.mkdir(parents=True, exist_ok=True)
            raw_suffix = submission_path.suffix if submission_path.suffix else ".txt"
            if submission_path.is_file():
                shutil.copyfile(
                    submission_path,
                    submission_root / f"raw_output{raw_suffix}",
                )
            write_json(
                submission_root / "manifest.json",
                {
                    "raw_sha256": raw_hash,
                    "source_sha256": "unavailable",
                    "extraction": "materialization_failed",
                },
            )
            write_json(output_root / "case.json", case.data)
            write_json(output_root / "environment.json", environment)
            for layer in result.layers:
                _write_layer(output_root, layer)
            _write_result(output_root, result)
            return result

        identity = _identity(
            case,
            raw_hash=materialized.raw_sha256,
            source_hash=materialized.source_sha256,
            environment=environment,
        )
        if not force:
            existing = _resume_result(
                output_root,
                identity,
                case=case,
                environment=environment,
            )
            if existing is not None:
                return existing
        if (output_root / "result.json").exists() and not force:
            raise ValueError(
                "output contains a result for different inputs; use a new "
                "directory or --force"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        submission_root = output_root / "submission"
        submission_root.mkdir(parents=True, exist_ok=True)
        final_submission = submission_root / "submission.py"
        write_text(
            final_submission,
            materialized.source_path.read_text(encoding="utf-8"),
        )
        raw_suffix = submission_path.suffix if submission_path.suffix else ".txt"
        if submission_path.is_file():
            shutil.copyfile(
                submission_path,
                submission_root / f"raw_output{raw_suffix}",
            )
        write_json(
            submission_root / "manifest.json",
            {
                "raw_sha256": materialized.raw_sha256,
                "source_sha256": materialized.source_sha256,
                "extraction": materialized.extraction,
            },
        )

    write_json(output_root / "case.json", case.data)
    write_json(output_root / "environment.json", environment)

    l0 = evaluate_l0(
        case=case.data,
        submission_path=final_submission,
        output_dir=output_root / "l0",
    )
    _write_layer(output_root, l0)
    if l0.status is Status.PASSED:
        candidate = output_root / "l0/candidate.step"
        l1 = evaluate_l1_isolated(
            case,
            candidate,
            target_step=case.artifact_path("target_step"),
            l0_passed=True,
            timeout_seconds=_l1_timeout(case),
            work_dir=output_root / "l1-worker",
        )
        _write_layer(output_root, l1)
    else:
        l1 = _not_reached("L1", "L0")
        _write_layer(output_root, l1)

    if l0.status is Status.PASSED:
        inspection = inspect_submission(final_submission, case=case.data).as_dict()
        candidate = output_root / "l0/candidate.step"
        try:
            fingerprint, measurements = _analysis_from_l1(l1)
        except ValueError as exc:
            status = _missing_l1_analysis_status(l1)
            l2 = LayerResult.from_checks(
                "L2",
                (
                    CheckResult(
                        "L2.nominal_geometry_analysis",
                        status,
                        failure_code=(
                            "geometry_analysis_failed"
                            if status is Status.FAILED
                            else None
                        ),
                        summary=str(exc),
                    ),
                ),
            )
        else:
            baseline = VariantExecution(
                Status.PASSED,
                fingerprint=fingerprint,
                artifact=str(candidate),
                measurements=measurements,
            )
            bridge = _VariantBridge(
                case,
                final_submission,
                output_root / "l2",
                inspection,
            )
            evaluated_l2 = evaluate_l2(
                case.data,
                inspection,
                execute=bridge.execute,
                evaluate_l0=bridge.l0,
                evaluate_l1=bridge.l1,
                baseline_execution=baseline,
            )
            nominal_l1_check = CheckResult(
                "L2.context.nominal_l1",
                (
                    Status.PASSED
                    if l1.status is Status.PASSED
                    else Status.FAILED
                ),
                required=False,
                failure_code=(
                    None
                    if l1.status is Status.PASSED
                    else "nominal_engineering_requirements_failed"
                ),
                observed={"nominal_l1_status": l1.status.value},
                summary=(
                    "nominal geometry passed L1"
                    if l1.status is Status.PASSED
                    else (
                        "L2 capability was evaluated independently after nominal "
                        "geometry failed one or more L1 requirements"
                    )
                ),
            )
            l2 = LayerResult.from_checks(
                "L2",
                (nominal_l1_check, *evaluated_l2.checks),
                metrics={
                    **evaluated_l2.metrics,
                    "nominal_l1_status": l1.status.value,
                },
                artifacts=evaluated_l2.artifacts,
                runtime_seconds=evaluated_l2.runtime_seconds,
            )
        _write_layer(output_root, l2)
    else:
        l2 = _not_reached("L2", "L0")
        _write_layer(output_root, l2)

    layers = (l0, l1, l2)
    result = EvaluationResult(
        schema_version=RESULT_SCHEMA,
        evaluator_version=__version__,
        item_id=case.item_id,
        protocol=str(case.data["protocol"]),
        identity=identity,
        layers=layers,
        overall_status=_overall_status(layers),
        environment=environment,
        runtime_seconds=time.monotonic() - started,
    )
    _write_result(output_root, result)
    return result


def _submission_for(submissions_root: Path, item_id: str) -> Path:
    candidates = (
        submissions_root / item_id / "submission.py",
        submissions_root / item_id / "fusion_patch.json",
        submissions_root / item_id / "submission.json",
        submissions_root / item_id / "response.txt",
        submissions_root / f"{item_id}.py",
        submissions_root / f"{item_id}.json",
        submissions_root / f"{item_id}.txt",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _validate_evalset_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != "cadengbench-p.evalset-manifest.v1":
        raise ValueError("unsupported evalset manifest schema")
    items = manifest.get("items")
    if not isinstance(items, list) or manifest.get("item_count") != len(items):
        raise ValueError("evalset manifest item_count does not match its records")
    for field in ("profile_id", "profile_hash"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"evalset manifest requires a nonempty {field}")
    protocol = manifest.get("protocol", "zero_to_cad")
    if protocol not in {"zero_to_cad", "functional_edit"}:
        raise ValueError("evalset manifest has an unsupported protocol")
    expected = manifest.get("manifest_hash")
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    if not isinstance(expected, str) or sha256_json(payload) != expected:
        raise ValueError("evalset manifest failed its semantic hash validation")


def _validate_evalset_case_binding(
    record: Mapping[str, Any],
    case: LoadedCase,
    manifest: Mapping[str, Any],
) -> None:
    item_id = str(record["item_id"])
    if case.item_id != item_id:
        raise ValueError(f"evalset item {item_id!r} resolves to case {case.item_id!r}")
    for field in ("case_hash", "source_family"):
        expected = record.get(field)
        observed = case.data.get(field)
        if not isinstance(expected, str) or expected != observed:
            raise ValueError(
                f"evalset item {item_id!r} has a mismatched {field} binding"
            )
    protocol = manifest.get("protocol", "zero_to_cad")
    if case.data.get("protocol") != protocol:
        raise ValueError(f"evalset item {item_id!r} has a mismatched protocol")
    if record.get("protocol", protocol) != protocol:
        raise ValueError(f"evalset item {item_id!r} record has a mismatched protocol")

    top_profile_hash = manifest["profile_hash"]
    case_profile_hash = case.data.get("profile_hash")
    if case_profile_hash != top_profile_hash:
        raise ValueError(
            f"evalset item {item_id!r} is bound to a different profile_hash"
        )
    record_profile_hash = record.get("profile_hash")
    if record_profile_hash is not None and record_profile_hash != top_profile_hash:
        raise ValueError(
            f"evalset item {item_id!r} record has a mismatched profile_hash"
        )
    evaluator_profile = case.data.get("evaluator_profile")
    if not isinstance(evaluator_profile, Mapping):
        raise ValueError(f"evalset item {item_id!r} has no evaluator profile")
    if evaluator_profile.get("profile_id") != manifest["profile_id"]:
        raise ValueError(f"evalset item {item_id!r} is bound to a different profile_id")
    if sha256_json(evaluator_profile) != top_profile_hash:
        raise ValueError(
            f"evalset item {item_id!r} evaluator profile failed its hash binding"
        )


def _submission_inventory_record(
    submissions_root: Path, item_id: str, index: int
) -> tuple[Path, dict[str, Any]]:
    selected = _submission_for(submissions_root, item_id)
    relative = selected.relative_to(submissions_root).as_posix()
    present = selected.is_file()
    return selected, {
        "index": index,
        "item_id": item_id,
        "path": relative,
        "state": "present" if present else "missing",
        "sha256": sha256_file(selected) if present else "missing",
        "size_bytes": selected.stat().st_size if present else None,
    }


def _validate_resume_manifest(
    existing: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    if existing.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError("resume output uses an unsupported run manifest")
    existing_identity = existing.get("identity")
    current_identity = current["identity"]
    if not isinstance(existing_identity, Mapping):
        raise ValueError("resume run manifest has no immutable identity")
    if existing.get("run_hash") != sha256_json(existing_identity):
        raise ValueError("resume run manifest failed its identity hash validation")
    inventory = existing.get("submission_inventory")
    if not isinstance(inventory, list):
        raise ValueError("resume run manifest has no submission inventory")
    if sha256_json(inventory) != existing_identity.get("submission_inventory_hash"):
        raise ValueError("resume submission inventory failed its hash validation")
    if existing_identity.get("model_id") != current_identity["model_id"]:
        raise ValueError("resume model_id differs from the original run")
    if (
        existing_identity.get("submission_inventory_hash")
        != current_identity["submission_inventory_hash"]
    ):
        raise ValueError("resume submission inventory differs from the original run")
    if dict(existing_identity) != current_identity:
        raise ValueError(
            "resume output belongs to a different evalset, evaluator, or environment"
        )
    if inventory != current["submission_inventory"]:
        raise ValueError("resume submission inventory records are inconsistent")
    if dict(existing) != dict(current):
        raise ValueError("resume run manifest differs from its immutable original")


def _result_has_evaluator_error(result: Mapping[str, Any]) -> bool:
    if result.get("overall_status") == Status.EVALUATOR_ERROR.value:
        return True
    for layer in result.get("layers", []):
        if not isinstance(layer, Mapping):
            continue
        if layer.get("status") == Status.EVALUATOR_ERROR.value:
            return True
        if any(
            isinstance(check, Mapping)
            and check.get("status") == Status.EVALUATOR_ERROR.value
            for check in layer.get("checks", [])
        ):
            return True
    return False


def _required_not_scoreable_count(result: Mapping[str, Any]) -> int:
    from .aggregation import benchmark_owned_not_scoreable

    return sum(
        1
        for layer in result.get("layers", [])
        if isinstance(layer, Mapping)
        for check in layer.get("checks", [])
        if isinstance(check, Mapping)
        and check.get("required", True) is True
        and check.get("status") == Status.NOT_SCOREABLE.value
        and not benchmark_owned_not_scoreable(check)
    )


def _benchmark_excluded_result(result: Mapping[str, Any]) -> bool:
    from .aggregation import benchmark_owned_not_scoreable

    checks = [
        check
        for layer in result.get("layers", [])
        if isinstance(layer, Mapping)
        for check in layer.get("checks", [])
        if isinstance(check, Mapping)
        and check.get("required", True) is True
        and check.get("status") == Status.NOT_SCOREABLE.value
    ]
    return (
        result.get("overall_status") == Status.NOT_SCOREABLE.value
        and bool(checks)
        and all(benchmark_owned_not_scoreable(check) for check in checks)
    )


def _result_roster(output_root: Path) -> set[str]:
    items_root = output_root / "items"
    if not items_root.is_dir():
        return set()
    return {
        path.parent.name for path in items_root.glob("*/result.json") if path.is_file()
    }


def _batch_job(
    case_path: str,
    submission_path: str,
    output_root: str,
    force: bool,
    repo_root: str | None = None,
) -> dict[str, Any]:
    case = load_case(Path(case_path), verify_artifacts=False)
    if case.data.get("protocol") == "functional_edit":
        if repo_root is None:
            raise ValueError("functional-edit evalsets require repo_root")
        from .functional_runner import evaluate_functional_item

        return evaluate_functional_item(
            repo_root=Path(repo_root),
            case_path=Path(case_path),
            submission_path=Path(submission_path),
            output_root=Path(output_root),
            force=force,
        ).as_dict()
    return evaluate_item(
        case_path=Path(case_path),
        submission_path=Path(submission_path),
        output_root=Path(output_root),
        force=force,
    ).as_dict()


def evaluate_evalset(
    *,
    evalset_root: Path,
    submissions_root: Path,
    output_root: Path,
    workers: int = 1,
    resume: bool = False,
    model_id: str = "unspecified",
    repo_root: Path | None = None,
    item_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate an evalset with deterministic process-level parallelism."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    model_id = model_id.strip()
    _assert_safe_output_root(
        output_root,
        protected_paths=(),
        protected_roots=(evalset_root, submissions_root),
    )
    manifest_path = evalset_root / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("items"), list):
        raise ValueError("evalset manifest is malformed")
    selected_item_ids = set(item_ids) if item_ids is not None else None
    if selected_item_ids is not None and (
        not selected_item_ids or len(selected_item_ids) != len(item_ids)
    ):
        raise ValueError("item_ids must be a nonempty unique list")
    records: list[tuple[Mapping[str, Any], Path]] = []
    seen_item_ids: set[str] = set()
    evalset_root_resolved = evalset_root.resolve()
    for record in manifest["items"]:
        if not isinstance(record, dict):
            raise ValueError("evalset items must be objects")
        item_id = record.get("item_id")
        case_relative = record.get("case_path")
        if (
            not isinstance(item_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", item_id) is None
            or item_id in seen_item_ids
            or not isinstance(case_relative, str)
            or not case_relative
        ):
            raise ValueError("evalset items require safe unique IDs and case paths")
        seen_item_ids.add(item_id)
        case_path = (evalset_root / case_relative).resolve()
        try:
            case_path.relative_to(evalset_root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"case path escapes the evalset root: {case_relative}"
            ) from exc
        if selected_item_ids is None or item_id in selected_item_ids:
            records.append((record, case_path))

    _validate_evalset_manifest(manifest)
    if selected_item_ids is not None:
        missing = selected_item_ids - seen_item_ids
        if missing:
            raise ValueError(
                f"requested evalset items are unavailable: {sorted(missing)}"
            )
    protocol = str(manifest.get("protocol", "zero_to_cad"))
    if protocol == "functional_edit" and repo_root is None:
        raise ValueError("functional-edit evalsets require --repo-root")
    for record, case_path in records:
        case = load_case(case_path, verify_artifacts=True)
        _validate_evalset_case_binding(record, case, manifest)

    jobs: list[tuple[Any, ...]] = []
    submission_inventory: list[dict[str, Any]] = []
    for index, (record, case_path) in enumerate(records):
        item_id = str(record["item_id"])
        submission, inventory_record = _submission_inventory_record(
            submissions_root,
            item_id,
            index,
        )
        submission_inventory.append(inventory_record)
        item_output = output_root / "items" / item_id
        job: tuple[Any, ...] = (
            item_id,
            str(case_path),
            str(submission),
            str(item_output),
            not resume,
        )
        if protocol == "functional_edit":
            assert repo_root is not None
            job = (*job, str(repo_root.resolve()))
        jobs.append(job)

    environment = environment_record()
    run_identity = {
        "evalset_manifest_hash": sha256_file(manifest_path),
        "evalset_semantic_hash": str(manifest["manifest_hash"]),
        "evaluator_build_hash": evaluator_build_hash(),
        "environment_hash": str(environment["environment_hash"]),
        "submission_inventory_hash": sha256_json(submission_inventory),
        "model_id": model_id,
        "protocol": protocol,
    }
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "identity": run_identity,
        "submission_inventory": submission_inventory,
        "item_count": len(records),
        "model_id": model_id,
    }
    run_manifest.update(run_identity)
    run_manifest["run_hash"] = sha256_json(run_identity)
    existing_manifest_path = output_root / "manifest.json"
    if resume and existing_manifest_path.is_file():
        existing_manifest = read_json(existing_manifest_path)
        if not isinstance(existing_manifest, dict):
            raise ValueError("resume run manifest is malformed")
        _validate_resume_manifest(existing_manifest, run_manifest)
    elif resume and output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("resume output is nonempty but has no run manifest")
    elif not resume and output_root.is_dir():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "manifest.json", run_manifest)
    write_json(output_root / "environment.json", environment)

    completed_results: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    if workers == 1:
        for job in jobs:
            item_id = str(job[0])
            case_path = str(job[1])
            try:
                result = _batch_job(*job[1:])
                if result.get("item_id") != item_id:
                    raise ValueError("batch result item_id differs from its job")
                completed_results[item_id] = result
            except Exception as exc:
                errors.append(
                    {
                        "item_id": item_id,
                        "case_path": case_path,
                        "exception": f"{type(exc).__name__}: {exc}",
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_batch_job, *job[1:]): (job[0], job[1]) for job in jobs
            }
            for future in as_completed(futures):
                item_id, case_path = futures[future]
                try:
                    result = future.result()
                    if result.get("item_id") != item_id:
                        raise ValueError("batch result item_id differs from its job")
                    completed_results[item_id] = result
                except Exception as exc:
                    errors.append(
                        {
                            "item_id": item_id,
                            "case_path": case_path,
                            "exception": f"{type(exc).__name__}: {exc}",
                        }
                    )
    write_json(output_root / "batch_errors.json", errors)
    expected_item_ids = {str(record["item_id"]) for record, _case_path in records}
    observed_item_ids = _result_roster(output_root)
    returned_item_ids = set(completed_results)
    execution_complete = not errors and returned_item_ids == expected_item_ids
    result_roster_exact = observed_item_ids == expected_item_ids
    evaluator_error_items = {
        item_id
        for item_id, result in completed_results.items()
        if _result_has_evaluator_error(result)
    }
    required_not_scoreable_by_item = {
        item_id: count
        for item_id, result in completed_results.items()
        if (count := _required_not_scoreable_count(result)) > 0
    }
    non_scoreable_outcome_items = {
        item_id
        for item_id, result in completed_results.items()
        if result.get("overall_status")
        not in {Status.PASSED.value, Status.FAILED.value}
        and not _benchmark_excluded_result(result)
    }
    benchmark_excluded_items = {
        item_id
        for item_id, result in completed_results.items()
        if _benchmark_excluded_result(result)
    }
    valid_complete_run = (
        execution_complete and result_roster_exact and not evaluator_error_items
    )
    summary: dict[str, Any] = {
        "run_hash": run_manifest["run_hash"],
        "requested_items": len(jobs),
        "completed_items": len(completed_results),
        "batch_error_count": len(errors),
        "execution_complete": execution_complete,
        "result_roster_exact": result_roster_exact,
        "missing_result_item_ids": sorted(expected_item_ids - observed_item_ids),
        "extra_result_item_ids": sorted(observed_item_ids - expected_item_ids),
        "item_evaluator_error_count": len(evaluator_error_items),
        "evaluator_error_item_ids": sorted(evaluator_error_items),
        "required_not_scoreable_item_count": len(required_not_scoreable_by_item),
        "required_not_scoreable_check_count": sum(
            required_not_scoreable_by_item.values()
        ),
        "non_scoreable_outcome_item_count": len(non_scoreable_outcome_items),
        "non_scoreable_outcome_item_ids": sorted(non_scoreable_outcome_items),
        "benchmark_excluded_item_count": len(benchmark_excluded_items),
        "benchmark_excluded_item_ids": sorted(benchmark_excluded_items),
        "valid_complete_run": valid_complete_run,
        "publication_valid": (
            valid_complete_run
            and not required_not_scoreable_by_item
            and not non_scoreable_outcome_items
        ),
    }
    if summary["valid_complete_run"]:
        try:
            summary["aggregate"] = aggregate_run(
                output_root,
                expected_item_ids=expected_item_ids,
            )
        except Exception as exc:
            summary["aggregation_error"] = f"{type(exc).__name__}: {exc}"
            summary["valid_complete_run"] = False
            summary["publication_valid"] = False
    write_json(output_root / "summary.json", summary)
    return summary
