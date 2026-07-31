"""Child process for bounded local CadQuery execution and STEP inspection.

This worker provides process and resource separation for development.  It is
not a security boundary; canonical evaluation of untrusted submissions must
run the same protocol inside the benchmark's pinned Linux container/job
sandbox.
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import resource
import signal
import socket
import traceback
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

WORKER_SCHEMA = "cadengbench-p.execution-worker.v1"


class ModelFailure(RuntimeError):
    def __init__(self, code: str, stage: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage


class _ModelDeadline(BaseException):
    """Private deadline signal that model ``except Exception`` cannot catch."""

    def __init__(self, seconds: int, stage: str) -> None:
        super().__init__(f"model construction exceeded {seconds} seconds")
        self.stage = stage


class _ModelTimer:
    """Apply the model wall limit after the worker and CAD kernel have started."""

    def __init__(self, seconds: int, stage: str) -> None:
        self.seconds = seconds
        self.stage = stage
        self.previous: Any = None

    def _timeout(self, _signum: int, _frame: Any) -> None:
        raise _ModelDeadline(self.seconds, self.stage)

    def __enter__(self) -> None:
        self.previous = signal.signal(signal.SIGALRM, self._timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *_unused: Any) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self.previous)


def _write_report(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_phase(path: Path, phase: str) -> None:
    _write_report(
        path,
        {
            "schema_version": WORKER_SCHEMA,
            "phase": phase,
        },
    )


def _deny_network(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise PermissionError("network access is disabled by the evaluator")


def _set_resource_limits(cpu_seconds: int, max_step_bytes: int) -> None:
    if cpu_seconds > 0:
        with suppress(OSError, ValueError):
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    for limit, value in (
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_FSIZE, max_step_bytes),
    ):
        with suppress(OSError, ValueError):
            resource.setrlimit(limit, (value, value))


def _safe_importer(allowed_roots: set[str]) -> Callable[..., Any]:
    def safe_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".", 1)[0]
        if level or root not in allowed_roots:
            raise ImportError(f"import is not permitted: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    return safe_import


def _safe_builtins(allowed_roots: set[str]) -> dict[str, Any]:
    return {
        "__build_class__": builtins.__build_class__,
        "__import__": _safe_importer(allowed_roots),
        "abs": abs,
        "all": all,
        "any": any,
        "ArithmeticError": ArithmeticError,
        "AssertionError": AssertionError,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "filter": filter,
        "float": float,
        "IndexError": IndexError,
        "int": int,
        "isinstance": isinstance,
        "KeyError": KeyError,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "object": object,
        "OverflowError": OverflowError,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "RuntimeError": RuntimeError,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "staticmethod": staticmethod,
        "str": str,
        "sum": sum,
        "super": super,
        "tuple": tuple,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "zip": zip,
        "ZeroDivisionError": ZeroDivisionError,
    }


def _expected_geometry_exception(exc: Exception) -> bool:
    module = type(exc).__module__
    return isinstance(
        exc, (OSError, RuntimeError, StopIteration, TypeError, ValueError)
    ) or module.startswith(("OCP.", "cadquery."))


def _collect_solids(value: Any) -> list[Any]:
    objects = list(value.vals()) if hasattr(value, "vals") else [value]
    solids: list[Any] = []
    for obj in objects:
        if hasattr(obj, "Solids"):
            solids.extend(list(obj.Solids()))
    return solids


def _geometry_facts(value: Any, *, failure_code: str) -> dict[str, Any]:
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer

        solids = _collect_solids(value)
        if not solids:
            raise ModelFailure(
                failure_code, "in_memory_geometry", "result has no solids"
            )

        volumes: list[float] = []
        areas: list[float] = []
        boxes: list[Any] = []
        face_count = 0
        edge_count = 0
        exact_valid = True
        ordinary_valid = True
        solid_valid = True
        for solid in solids:
            volume = abs(float(solid.Volume()))
            area = float(solid.Area())
            if not math.isfinite(volume) or volume <= 0.0:
                raise ModelFailure(
                    failure_code,
                    "in_memory_geometry",
                    "result has a non-positive or non-finite solid volume",
                )
            if not math.isfinite(area) or area <= 0.0:
                raise ModelFailure(
                    failure_code,
                    "in_memory_geometry",
                    "result has a non-positive or non-finite surface area",
                )
            volumes.append(volume)
            areas.append(area)
            boxes.append(solid.BoundingBox())
            face_count += len(solid.Faces())
            edge_count += len(solid.Edges())
            solid_valid = solid_valid and bool(solid.isValid())
            ordinary_valid = ordinary_valid and bool(
                BRepCheck_Analyzer(solid.wrapped, True, False, False).IsValid()
            )
            exact_valid = exact_valid and bool(
                BRepCheck_Analyzer(solid.wrapped, True, False, True).IsValid()
            )

        if not (solid_valid and ordinary_valid and exact_valid):
            raise ModelFailure(
                failure_code,
                "in_memory_geometry",
                "result contains an invalid B-Rep solid",
            )
        bounds = {
            "xmin": min(float(box.xmin) for box in boxes),
            "ymin": min(float(box.ymin) for box in boxes),
            "zmin": min(float(box.zmin) for box in boxes),
            "xmax": max(float(box.xmax) for box in boxes),
            "ymax": max(float(box.ymax) for box in boxes),
            "zmax": max(float(box.zmax) for box in boxes),
        }
        bounds["xlen"] = bounds["xmax"] - bounds["xmin"]
        bounds["ylen"] = bounds["ymax"] - bounds["ymin"]
        bounds["zlen"] = bounds["zmax"] - bounds["zmin"]
        if not all(math.isfinite(value) for value in bounds.values()):
            raise ModelFailure(
                failure_code, "in_memory_geometry", "result bounds are not finite"
            )
        return {
            "solid_count": len(solids),
            "solid_valid": solid_valid,
            "ordinary_brep_valid": ordinary_valid,
            "exact_brep_valid": exact_valid,
            "face_count": face_count,
            "edge_count": edge_count,
            "volume_mm3": sum(volumes),
            "surface_area_mm2": sum(areas),
            "bbox_mm": bounds,
        }
    except ModelFailure:
        raise
    except Exception as exc:
        if not _expected_geometry_exception(exc):
            raise
        raise ModelFailure(
            failure_code, "in_memory_geometry", f"geometry validation failed: {exc}"
        ) from exc


def _load_override(raw: str | None) -> dict[str, float]:
    if raw is None:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("parameter override must be an object")
    normalized: dict[str, float] = {}
    for name, number in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(float(number))
        ):
            raise ValueError(
                "parameter overrides require string keys and finite numbers"
            )
        normalized[name] = float(number)
    return normalized


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    import cadquery as cq

    source_path = Path(args.source).resolve(strict=True)
    phase_path = Path(args.phase_marker).resolve()
    sandbox = source_path.parent.resolve()
    output_path = sandbox / "model.step"
    output_path.unlink(missing_ok=True)
    allowed_roots = set(json.loads(args.allowed_import_roots))
    namespace: dict[str, Any] = {
        "__name__": "__cadengbench_submission__",
        "__builtins__": _safe_builtins(allowed_roots),
        "cq": cq,
        "show_object": lambda *unused_args, **unused_kwargs: None,
    }
    override = _load_override(args.override_json)
    try:
        source = source_path.read_text(encoding="utf-8")
        code = compile(source, str(source_path), "exec")
        _write_phase(phase_path, "source_execution")
        with _ModelTimer(args.model_wall_seconds, "source_execution"):
            exec(code, namespace, namespace)
    except _ModelDeadline as exc:
        raise ModelFailure("execution_timeout", exc.stage, str(exc)) from None
    except ModelFailure:
        raise
    except BaseException as exc:
        raise ModelFailure(
            "submission_runtime_error",
            "source_execution",
            f"submission raised {type(exc).__name__}: {exc}",
        ) from exc

    if not output_path.is_file() or output_path.is_symlink():
        raise ModelFailure(
            "model_step_missing",
            "output_contract",
            "submission did not create a regular model.step file",
        )
    authored_bytes = output_path.stat().st_size
    if authored_bytes <= 0:
        raise ModelFailure(
            "model_step_empty",
            "output_contract",
            "submission created an empty model.step",
        )
    if authored_bytes > args.max_step_bytes:
        raise ModelFailure(
            "step_size_limit_exceeded",
            "output_contract",
            f"model.step exceeds {args.max_step_bytes} bytes",
        )

    result = namespace.get("result")
    if result is None:
        raise ModelFailure(
            "result_missing", "source_execution", "submission did not define result"
        )
    if override:
        build = namespace.get("build")
        try:
            _write_phase(phase_path, "parameter_execution")
            with _ModelTimer(args.model_wall_seconds, "parameter_execution"):
                result = build(dict(override))
        except _ModelDeadline as exc:
            raise ModelFailure("execution_timeout", exc.stage, str(exc)) from None
        except ModelFailure:
            raise
        except BaseException as exc:
            raise ModelFailure(
                "parameter_override_runtime_error",
                "parameter_execution",
                f"build(overrides) raised {type(exc).__name__}: {exc}",
            ) from exc

    _write_phase(phase_path, "evaluator_postprocess")
    facts = _geometry_facts(result, failure_code="invalid_candidate_geometry")
    if override:
        variant_path = sandbox / "variant.step"
        variant_path.unlink(missing_ok=True)
        try:
            cq.exporters.export(result, str(variant_path), exportType="STEP")
            variant_path.replace(output_path)
        except Exception as exc:
            raise ModelFailure(
                "variant_step_export_failed",
                "output_contract",
                f"evaluator could not export build(overrides): {exc}",
            ) from exc
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise ModelFailure(
            "model_step_missing",
            "output_contract",
            "model.step disappeared after execution",
        )
    if output_path.stat().st_size > args.max_step_bytes:
        raise ModelFailure(
            "step_size_limit_exceeded",
            "output_contract",
            f"model.step exceeds {args.max_step_bytes} bytes",
        )
    _write_phase(phase_path, "complete")
    return {
        "schema_version": WORKER_SCHEMA,
        "status": "passed",
        "ownership": None,
        "failed_stage": None,
        "failure_code": None,
        "summary": "submission executed and produced valid in-memory solids",
        "source_executed": True,
        "model_step_authored": True,
        "evaluator_exported_override": bool(override),
        "override": override,
        "facts": facts,
        "step_bytes": output_path.stat().st_size,
        "authored_nominal_step_bytes": authored_bytes,
    }


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    import cadquery as cq

    step_path = Path(args.step).resolve(strict=True)
    if step_path.stat().st_size > args.max_step_bytes:
        raise ModelFailure(
            "step_size_limit_exceeded",
            "step_reimport",
            f"candidate STEP exceeds {args.max_step_bytes} bytes",
        )
    try:
        imported = cq.importers.importStep(str(step_path))
    except Exception as exc:
        raise ModelFailure(
            "step_reimport_failed", "step_reimport", f"STEP import failed: {exc}"
        ) from exc
    facts = _geometry_facts(imported, failure_code="invalid_reimported_geometry")
    return {
        "schema_version": WORKER_SCHEMA,
        "status": "passed",
        "ownership": None,
        "failed_stage": None,
        "failure_code": None,
        "summary": "STEP independently reimported as valid solids",
        "facts": facts,
        "step_bytes": step_path.stat().st_size,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--source", required=True)
    execute.add_argument("--report", required=True)
    execute.add_argument("--phase-marker", required=True)
    execute.add_argument("--allowed-import-roots", required=True)
    execute.add_argument("--override-json")
    execute.add_argument("--cpu-seconds", type=int, required=True)
    execute.add_argument("--model-wall-seconds", type=int, required=True)
    execute.add_argument("--max-step-bytes", type=int, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--step", required=True)
    inspect.add_argument("--report", required=True)
    inspect.add_argument("--cpu-seconds", type=int, required=True)
    inspect.add_argument("--max-step-bytes", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report_path = Path(args.report).resolve()
    _set_resource_limits(args.cpu_seconds, args.max_step_bytes)
    socket.socket = _deny_network  # type: ignore[assignment]
    socket.create_connection = _deny_network  # type: ignore[assignment]
    try:
        report = _execute(args) if args.mode == "execute" else _inspect(args)
        _write_report(report_path, report)
        return 0
    except ModelFailure as exc:
        _write_report(
            report_path,
            {
                "schema_version": WORKER_SCHEMA,
                "status": "failed",
                "ownership": "model",
                "failed_stage": exc.stage,
                "failure_code": exc.code,
                "summary": str(exc),
            },
        )
        return 0
    except BaseException as exc:
        _write_report(
            report_path,
            {
                "schema_version": WORKER_SCHEMA,
                "status": "evaluator_error",
                "ownership": "evaluator",
                "failed_stage": "worker_internal",
                "failure_code": "worker_internal_error",
                "summary": f"worker raised {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=20),
            },
        )
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
