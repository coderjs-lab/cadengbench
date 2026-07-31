"""Source-native Protocol-E execution and post-edit replay adapters."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cadengbench_p_eval.case import LoadedCase
from cadengbench_p_eval.contracts import CheckResult, LayerResult, Status
from cadengbench_p_eval.geometry.fingerprint import geometry_fingerprint
from cadengbench_p_eval.geometry.step import StepGeometryError, load_step
from cadengbench_p_eval.io import read_json, sha256_file, write_json, write_text
from cadengbench_p_eval.layers.l1_isolated import evaluate_l1_isolated
from cadengbench_p_eval.parametrics.replay import (
    VariantEvaluationContext,
    VariantExecution,
)


class FunctionalEditAdapterError(RuntimeError):
    """Protocol-E source execution cannot be completed deterministically."""


@dataclass(frozen=True)
class FunctionalSourceResult:
    normalized_result: dict[str, Any]
    source_result: dict[str, Any]
    acceptance_contract: dict[str, Any]
    candidate_step: Path | None
    baseline_step: Path

    @property
    def checks(self) -> dict[str, Mapping[str, Any]]:
        records = self.normalized_result.get("checks") or []
        return {
            str(record["check_id"]): record
            for record in records
            if isinstance(record, Mapping) and isinstance(record.get("check_id"), str)
        }


def expected_solid_count(source: FunctionalSourceResult) -> int | None:
    """Return the verified baseline solid count from the source adapter."""

    check = source.checks.get("core.solid_count_preserved")
    evidence = check.get("evidence") if isinstance(check, Mapping) else None
    raw = evidence.get("base_solid_count") if isinstance(evidence, Mapping) else None
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw

    # Some source adapters prove preservation without serializing the count.
    # The hidden baseline STEP remains authoritative for the nominal ledger.
    return len(load_step(source.baseline_step).solids)


def solid_count_requirement(source: FunctionalSourceResult) -> list[dict[str, Any]]:
    """Compile the source-bound Protocol-E nominal requirement ledger."""

    count = expected_solid_count(source)
    if count is None:
        return []
    return [
        {
            "requirement_id": "E-R-SOLID-COUNT",
            "category": "topology",
            "criticality": "required",
            "value_and_tolerance": {"body_count": count},
            "check_ids": ["hidden.check.solid_count.v1"],
        }
    ]


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FunctionalEditAdapterError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise FunctionalEditAdapterError(f"{field} must be finite")
    return number


def _load_backend(repo_root: Path) -> None:
    source_root = repo_root / "src"
    if not (source_root / "cadengbench").is_dir():
        raise FunctionalEditAdapterError(
            f"CADEngBench source backend is absent under {source_root}"
        )
    fusion_schema = (
        repo_root
        / "schemas"
        / "cadengbench_p_fusion360_patch_submission_v1.schema.json"
    )
    if not fusion_schema.is_file():
        raise FunctionalEditAdapterError(
            f"required evaluator runtime schema is absent: {fusion_schema}"
        )
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)


def _resolve_hidden_artifact(
    repo_root: Path,
    item_root: Path,
    item_id: str,
    record: Mapping[str, Any],
) -> Path:
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise FunctionalEditAdapterError("hidden artifact pointer is malformed")
    expected_hash = expected_hash.removeprefix("sha256:")
    candidate = (repo_root / raw_path).resolve()
    if not candidate.is_file():
        marker = f"/dataset/items/{item_id}/"
        if marker not in raw_path:
            raise FunctionalEditAdapterError(
                f"hidden artifact cannot be relocated: {raw_path}"
            )
        candidate = (item_root / raw_path.split(marker, 1)[1]).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise FunctionalEditAdapterError("hidden artifact escapes repository") from exc
    if not candidate.is_file() or sha256_file(candidate) != expected_hash:
        raise FunctionalEditAdapterError(
            f"hidden artifact failed its SHA-256 binding: {raw_path}"
        )
    return candidate


def _relocate_acceptance_contract(
    repo_root: Path,
    case: LoadedCase,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    raw = read_json(case.artifact_path("functional_edit_acceptance_contract"))
    if not isinstance(raw, Mapping):
        raise FunctionalEditAdapterError("acceptance contract must be an object")
    contract = deepcopy(dict(raw))
    references = contract.get("hidden_reference_artifacts")
    if not isinstance(references, Mapping):
        raise FunctionalEditAdapterError("acceptance contract has no hidden artifacts")
    relocated: dict[str, Path] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for role, record in references.items():
        if not isinstance(role, str) or not isinstance(record, Mapping):
            raise FunctionalEditAdapterError("hidden artifact roster is malformed")
        path = _resolve_hidden_artifact(
            repo_root,
            case.item_root,
            case.item_id,
            record,
        )
        relocated[role] = path
        normalized[role] = {
            **dict(record),
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    contract["hidden_reference_artifacts"] = normalized
    write_json(output_path, contract)
    return contract, relocated


class FunctionalEditAdapter:
    """Execute the frozen source adapter and replay its one qualified parameter."""

    def __init__(
        self,
        *,
        repo_root: Path,
        case: LoadedCase,
        submission_path: Path,
        output_root: Path,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.case = case
        self.submission_path = submission_path.resolve()
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        _load_backend(self.repo_root)
        self.acceptance, self.hidden = _relocate_acceptance_contract(
            self.repo_root,
            case,
            self.output_root / "relocated_acceptance_contract.json",
        )
        self.source_key = str(self.acceptance.get("source_key"))
        self.parameter = dict(self.acceptance.get("parameter_contract") or {})
        self.scale = _finite(
            self.parameter.get("scale_to_canonical"), "scale_to_canonical"
        )
        if self.scale <= 0.0:
            raise FunctionalEditAdapterError("scale_to_canonical must be positive")
        self.target_value = _finite(self.parameter.get("target_value"), "target_value")
        self.base_value = _finite(self.parameter.get("base_value"), "base_value")
        self.timeout_seconds = int(
            ((case.data.get("evaluator_profile") or {}).get("execution") or {}).get(
                "timeout_seconds", 120
            )
        )
        self.memory_mb = 4096
        self._l0_results: dict[str, LayerResult] = {}
        self._l1_results: dict[str, LayerResult] = {}
        self._source_result: FunctionalSourceResult | None = None

    def evaluate_source_submission(self) -> FunctionalSourceResult:
        """Run the existing certified BenchCAD/Fusion adapter once."""

        from cadengbench.evaluation import evaluate_functional_edit_submission

        root = self.output_root / "source_adapter"
        normalized = evaluate_functional_edit_submission(
            repo_root=self.repo_root,
            submission_path=self.submission_path,
            acceptance_contract_path=(
                self.output_root / "relocated_acceptance_contract.json"
            ),
            output_root=root,
            timeout_seconds=self.timeout_seconds,
            memory_mb=self.memory_mb,
        )
        source_path = root / "source_evaluator" / self.source_key / "evaluation.json"
        source = read_json(source_path) if source_path.is_file() else {}
        if not isinstance(source, Mapping):
            raise FunctionalEditAdapterError("source evaluator result is malformed")
        candidate = root / "source_evaluator" / self.source_key / "submission.step"
        result = FunctionalSourceResult(
            normalized_result=dict(normalized),
            source_result=dict(source),
            acceptance_contract=self.acceptance,
            candidate_step=candidate if candidate.is_file() else None,
            baseline_step=self.hidden["base_step"],
        )
        self._source_result = result
        return result

    def _analyze_variant(
        self,
        step_path: Path,
        *,
        value: float,
        details: Mapping[str, Any],
    ) -> VariantExecution:
        try:
            geometry = load_step(step_path)
            profile = self.case.data["evaluator_profile"]["geometry"]
            fingerprint = geometry_fingerprint(
                geometry,
                seed=int(profile["seed"]),
                surface_sample_count=int(profile["surface_sample_count"]),
            )
        except StepGeometryError as exc:
            return VariantExecution(
                Status.FAILED,
                artifact=str(step_path),
                failure_code="geometry_analysis_failed",
                details={
                    **dict(details),
                    "analysis_error": f"{type(exc).__name__}: {exc}",
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            return VariantExecution(
                Status.EVALUATOR_ERROR,
                artifact=str(step_path),
                details={
                    **dict(details),
                    "analysis_error": f"{type(exc).__name__}: {exc}",
                },
            )
        return VariantExecution(
            Status.PASSED,
            fingerprint=fingerprint,
            artifact=str(step_path),
            measurements={"E-P-001": value},
            details=dict(details),
        )

    def baseline_execution(self) -> VariantExecution:
        if self._source_result is None:
            raise FunctionalEditAdapterError("source submission has not been evaluated")
        step = self._source_result.candidate_step
        if step is None:
            return VariantExecution(
                Status.FAILED,
                failure_code="functional_edit_did_not_produce_step",
                details={"source_result": self._source_result.normalized_result},
            )
        return self._analyze_variant(
            step,
            value=self.target_value,
            details={"source_adapter_nominal": True},
        )

    def execute_variant(
        self, model_parameters: Mapping[str, float], variant_id: str
    ) -> VariantExecution:
        try:
            value = _finite(model_parameters.get("edit_value"), "edit_value")
            root = self.output_root / "post_edit_replay" / variant_id
            root.mkdir(parents=True, exist_ok=True)
            if self.source_key == "benchcad":
                execution, step = self._execute_benchcad_variant(root, value)
            elif self.source_key == "fusion360":
                execution, step = self._execute_fusion_variant(root, value)
            else:
                raise FunctionalEditAdapterError(
                    f"unsupported Protocol-E source: {self.source_key}"
                )
            succeeded = execution.get("status") == "success" and step.is_file()
            solid_valid = execution.get("solid_valid") is True
            if not succeeded or not solid_valid:
                failure = str(execution.get("failure_kind") or "invalid_solid")
                result = VariantExecution(
                    Status.FAILED,
                    artifact=str(step) if step.is_file() else None,
                    failure_code=failure,
                    details={"source_execution": execution},
                )
            else:
                result = self._analyze_variant(
                    step,
                    value=value,
                    details={"source_execution": execution},
                )
            self._l0_results[variant_id] = self._variant_l0(result)
            if self._l0_results[variant_id].status is Status.PASSED:
                self._l1_results[variant_id] = self._variant_l1(
                    result,
                    root / "l1-worker",
                )
            return result
        except FunctionalEditAdapterError as exc:
            return VariantExecution(
                Status.EVALUATOR_ERROR,
                details={"adapter_error": str(exc)},
            )
        except Exception as exc:  # external source backend boundary
            return VariantExecution(
                Status.EVALUATOR_ERROR,
                details={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    def _execute_benchcad_variant(
        self, root: Path, value: float
    ) -> tuple[dict[str, Any], Path]:
        from cadengbench.cadquery_ast import replace_numeric_call_argument
        from cadengbench.pipeline.cadexec import run_cadquery_source

        qualification = read_json(self.hidden["edit_binding_qualification"])
        selected = (
            qualification.get("selected")
            if isinstance(qualification, Mapping)
            else None
        )
        binding = selected.get("binding") if isinstance(selected, Mapping) else None
        if not isinstance(binding, Mapping):
            raise FunctionalEditAdapterError("BenchCAD binding qualification is absent")
        source = self.submission_path.read_text(encoding="utf-8")
        # The qualification span and base literal describe the unedited source.
        # A submission admitted by the certified Protocol-E adapter has already
        # proved exact non-target-source preservation and the requested target
        # assignment, so subsequent variants may reuse the frozen pointer and
        # call occurrence while allowing the target literal's length/value to
        # differ from the original source.
        replay_binding = dict(binding)
        replay_binding.pop("source_span", None)
        replay_binding.pop("base_value", None)
        edited = replace_numeric_call_argument(
            source, replay_binding, value / self.scale
        )
        source_path = root / "variant.py"
        step_path = root / "candidate.step"
        report_path = root / "execution.json"
        write_text(source_path, edited)
        execution = run_cadquery_source(
            source_path,
            step_path,
            report_path,
            timeout_seconds=self.timeout_seconds,
            memory_mb=self.memory_mb,
        )
        write_json(report_path, execution)
        return execution, step_path

    def _execute_fusion_variant(
        self, root: Path, value: float
    ) -> tuple[dict[str, Any], Path]:
        from cadengbench.fusion360 import (
            apply_fusion_patch_submission,
            run_fusion360_occ_reconstruction,
        )

        patch = read_json(self.submission_path)
        if not isinstance(patch, Mapping):
            raise FunctionalEditAdapterError("Fusion patch must be an object")
        variant_patch = deepcopy(dict(patch))
        edits = variant_patch.get("edits")
        if (
            not isinstance(edits, list)
            or len(edits) != 1
            or not isinstance(edits[0], Mapping)
        ):
            raise FunctionalEditAdapterError("Fusion patch must contain one edit")
        edits[0]["value"] = value / self.scale
        patch_path = root / "variant_patch.json"
        write_json(patch_path, variant_patch)
        edited, evidence = apply_fusion_patch_submission(
            variant_patch,
            base_history_path=self.hidden["base_history"],
            expected_pointer=str(self.parameter["source_pointer"]),
        )
        history_path = root / "edited_history.fusion360.json"
        step_path = root / "candidate.step"
        report_path = root / "execution.json"
        write_json(history_path, edited)
        execution = run_fusion360_occ_reconstruction(
            history_path,
            None,
            step_path,
            report_path,
            timeout_seconds=self.timeout_seconds,
        )
        execution["patch_evidence"] = evidence
        write_json(report_path, execution)
        return execution, step_path

    def _variant_l0(self, execution: VariantExecution) -> LayerResult:
        return LayerResult.from_checks(
            "L0",
            (
                CheckResult(
                    "L0.protocol_e.variant_source_replay",
                    execution.status,
                    failure_code=execution.failure_code,
                    observed=execution.details,
                ),
            ),
        )

    def _variant_l1(self, execution: VariantExecution, work_dir: Path) -> LayerResult:
        if execution.artifact is None:
            return LayerResult.from_checks(
                "L1",
                (
                    CheckResult(
                        "L1.protocol_e.variant_artifact",
                        Status.EVALUATOR_ERROR,
                        summary="post-edit replay produced no STEP artifact",
                    ),
                ),
            )
        case = deepcopy(self.case.data)
        assert self._source_result is not None
        case["requirements"] = solid_count_requirement(self._source_result)
        evaluated = evaluate_l1_isolated(
            case,
            Path(execution.artifact),
            l0_passed=True,
            is_parameter_variant=True,
            timeout_seconds=float(self.timeout_seconds * 2),
            work_dir=work_dir,
        )
        expected_count = expected_solid_count(self._source_result)
        observed_count = (
            (execution.details.get("source_execution") or {}).get("solid_count")
            if isinstance(execution.details, Mapping)
            else None
        )
        invariant = CheckResult(
            "L1.protocol_e.solid_count_preserved",
            (
                Status.PASSED
                if expected_count is None or observed_count == expected_count
                else Status.FAILED
            ),
            failure_code=(
                None
                if expected_count is None or observed_count == expected_count
                else "functional_edit_solid_count_changed"
            ),
            expected={"solid_count": expected_count},
            observed={"solid_count": observed_count},
            required=expected_count is not None,
        )
        return LayerResult.from_checks(
            "L1",
            (*evaluated.checks, invariant),
            metrics=evaluated.metrics,
            artifacts=evaluated.artifacts,
            runtime_seconds=evaluated.runtime_seconds,
        )

    def l0(
        self, execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult:
        del execution
        return self._l0_results.get(
            context.variant.variant_id,
            LayerResult.from_checks(
                "L0",
                (
                    CheckResult(
                        "L0.protocol_e.variant_result",
                        Status.EVALUATOR_ERROR,
                        summary="variant L0 result is absent",
                    ),
                ),
            ),
        )

    def l1(
        self, execution: VariantExecution, context: VariantEvaluationContext
    ) -> LayerResult:
        del execution
        return self._l1_results.get(
            context.variant.variant_id,
            LayerResult.from_checks(
                "L1",
                (
                    CheckResult(
                        "L1.protocol_e.variant_result",
                        Status.EVALUATOR_ERROR,
                        summary="variant L1 result is absent",
                    ),
                ),
            ),
        )
