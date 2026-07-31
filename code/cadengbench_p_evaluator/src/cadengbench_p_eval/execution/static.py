"""Static validation for the public CadQuery submission contract.

The checks in this module are deliberately conservative.  They reject source
which falls outside the benchmark's small public interface before any model
code is executed.  This is policy validation, not a security boundary.
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

INTERFACE_VERSION = "cadengbench-p.cadquery-submission.v1"
REQUIRED_SYMBOLS = ("DEFAULT_PARAMS", "PARAMETER_MANIFEST", "build", "result")
MAX_AST_NODES = 100_000

_DANGEROUS_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
_DANGEROUS_METHODS = {
    "chmod",
    "connect",
    "create_connection",
    "download",
    "from_file",
    "importBrep",
    "importDXF",
    "importShape",
    "importStep",
    "load",
    "mkdir",
    "open",
    "popen",
    "read_bytes",
    "read_text",
    "remove",
    "rename",
    "replace",
    "request",
    "rmdir",
    "socket",
    "system",
    "unlink",
    "urlopen",
    "write_bytes",
    "write_text",
}
_EXPORT_METHODS = {"export", "exportShape", "exportStep"}
_DEADLINE_SWALLOWING_EXCEPTIONS = {
    "BaseException",
    "GeneratorExit",
    "KeyboardInterrupt",
    "SystemExit",
}


class StaticValidationError(ValueError):
    """The model source violates the frozen public submission contract."""

    def __init__(self, failure_code: str, message: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class StaticInspection:
    interface_version: str
    source_sha256: str
    source_bytes: int
    default_params: dict[str, float]
    parameter_manifest: dict[str, dict[str, str]]
    allowed_import_roots: tuple[str, ...]
    ast_node_count: int
    step_output_path: str = "model.step"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_import_roots"] = list(self.allowed_import_roots)
        return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StaticValidationError(
            "invalid_evaluator_case", f"{name} is missing or malformed"
        )
    return value


def _case_contract(
    case: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    task = _mapping(case.get("task_contract"), "task_contract")
    submission = _mapping(task.get("submission"), "task_contract.submission")
    code = _mapping(
        submission.get("code_submission_contract"),
        "task_contract.submission.code_submission_contract",
    )
    parameter_system = _mapping(case.get("parameter_system"), "parameter_system")
    parameters = parameter_system.get("parameters")
    if (
        not isinstance(parameters, list)
        or not parameters
        or not all(isinstance(record, Mapping) for record in parameters)
    ):
        raise StaticValidationError(
            "invalid_evaluator_case", "parameter_system.parameters is malformed"
        )
    if code.get("interface_version") != INTERFACE_VERSION:
        raise StaticValidationError(
            "unsupported_submission_interface",
            f"expected interface {INTERFACE_VERSION}",
        )
    if parameter_system.get("interface_version") != INTERFACE_VERSION:
        raise StaticValidationError(
            "invalid_evaluator_case",
            "task and parameter-system interface versions differ",
        )
    if code.get("build_signature") != "build(params=None)":
        raise StaticValidationError(
            "invalid_evaluator_case", "the case does not require build(params=None)"
        )
    if code.get("step_output_path") != "model.step":
        raise StaticValidationError(
            "invalid_evaluator_case", "the case STEP output must be model.step"
        )
    symbols = code.get("required_top_level_symbols")
    if not isinstance(symbols, list) or set(symbols) != set(REQUIRED_SYMBOLS):
        raise StaticValidationError(
            "invalid_evaluator_case", "the case has an unsupported symbol contract"
        )
    return list(parameters), dict(code)


def _top_level_values(tree: ast.Module, name: str) -> list[ast.AST]:
    values: list[ast.AST] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            values.append(node.value)
    return values


def _literal_mapping(tree: ast.Module, name: str) -> dict[str, Any]:
    nodes = _top_level_values(tree, name)
    if len(nodes) != 1 or not isinstance(nodes[0], ast.Dict):
        raise StaticValidationError(
            "invalid_submission_interface",
            f"{name} must have exactly one top-level literal-dictionary assignment",
        )
    node = nodes[0]
    keys: list[str] = []
    for key_node in node.keys:
        if key_node is None:
            raise StaticValidationError(
                "invalid_submission_interface",
                f"{name} may not use dictionary unpacking",
            )
        try:
            key = ast.literal_eval(key_node)
        except (TypeError, ValueError) as exc:
            raise StaticValidationError(
                "invalid_submission_interface", f"{name} keys must be literal strings"
            ) from exc
        if not isinstance(key, str):
            raise StaticValidationError(
                "invalid_submission_interface", f"{name} keys must be literal strings"
            )
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise StaticValidationError(
            "invalid_submission_interface", f"{name} contains duplicate keys"
        )
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError) as exc:
        raise StaticValidationError(
            "invalid_submission_interface", f"{name} must contain literal values"
        ) from exc
    if not isinstance(value, dict):
        raise StaticValidationError(
            "invalid_submission_interface", f"{name} must be a dictionary"
        )
    return value


def _validate_build(tree: ast.Module) -> None:
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build"
    ]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise StaticValidationError(
            "invalid_submission_interface",
            "build must be defined exactly once as a synchronous function",
        )
    function = functions[0]
    arguments = function.args
    valid = (
        not function.decorator_list
        and not arguments.posonlyargs
        and len(arguments.args) == 1
        and arguments.args[0].arg == "params"
        and len(arguments.defaults) == 1
        and isinstance(arguments.defaults[0], ast.Constant)
        and arguments.defaults[0].value is None
        and not arguments.kwonlyargs
        and arguments.vararg is None
        and arguments.kwarg is None
    )
    if not valid:
        raise StaticValidationError(
            "invalid_submission_interface",
            "build must use the exact undecorated signature build(params=None)",
        )


def _validate_result(tree: ast.Module) -> None:
    values = _top_level_values(tree, "result")
    if len(values) != 1:
        raise StaticValidationError(
            "invalid_submission_interface",
            "result must have exactly one top-level assignment",
        )
    value = values[0]
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "build"
        and not value.args
        and not value.keywords
    ):
        raise StaticValidationError(
            "invalid_submission_interface",
            "result must be assigned exactly as result = build()",
        )


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    value: ast.AST = call.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _caught_exception_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for element in node.elts:
            names.update(_caught_exception_names(element))
        return names
    return set()


def _is_exact_step_export(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    name = _call_name(call)
    if name.split(".")[-1] != "export" or len(call.args) < 2:
        return False
    first, second = call.args[:2]
    if not (isinstance(first, ast.Name) and first.id == "result"):
        return False
    return isinstance(second, ast.Constant) and second.value == "model.step"


def _validate_step_export(tree: ast.Module) -> None:
    export_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node).split(".")[-1] in _EXPORT_METHODS
    ]
    top_level = [node for node in tree.body if _is_exact_step_export(node)]
    if len(export_calls) != 1 or len(top_level) != 1:
        raise StaticValidationError(
            "invalid_step_export",
            "source must contain exactly one top-level export of result to model.step",
        )


def _validate_policy(tree: ast.Module, *, allowed_import_roots: set[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                raise StaticValidationError(
                    "forbidden_exception_handler",
                    "bare except handlers are not permitted",
                )
            caught = _caught_exception_names(node.type)
            forbidden = sorted(caught & _DEADLINE_SWALLOWING_EXCEPTIONS)
            if forbidden:
                raise StaticValidationError(
                    "forbidden_exception_handler",
                    "catching deadline-swallowing exceptions is not permitted: "
                    f"{forbidden}",
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_import_roots:
                    raise StaticValidationError(
                        "forbidden_import", f"import is not permitted: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise StaticValidationError(
                    "forbidden_import", "relative imports are not permitted"
                )
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_import_roots:
                raise StaticValidationError(
                    "forbidden_import", f"import is not permitted: {node.module}"
                )
            if any(alias.name == "*" for alias in node.names):
                raise StaticValidationError(
                    "forbidden_import", "star imports are not permitted"
                )
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise StaticValidationError(
                "forbidden_source_operation", "dunder attribute access is not permitted"
            )
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            leaf = name.split(".")[-1]
            if leaf in _DANGEROUS_CALLS or leaf in _DANGEROUS_METHODS:
                raise StaticValidationError(
                    "forbidden_source_operation",
                    f"call is not permitted: {name or leaf}",
                )


def _validate_parameter_bindings(
    defaults: Mapping[str, Any],
    manifest: Mapping[str, Any],
    parameters: list[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    expected: dict[str, Mapping[str, Any]] = {}
    for record in parameters:
        parameter_id = record.get("parameter_id")
        if (
            not isinstance(parameter_id, str)
            or not parameter_id
            or parameter_id in expected
        ):
            raise StaticValidationError(
                "invalid_evaluator_case",
                "public parameter IDs are missing or duplicated",
            )
        expected[parameter_id] = record
    if set(manifest) != set(expected):
        missing = sorted(set(expected) - set(manifest))
        extra = sorted(set(manifest) - set(expected))
        raise StaticValidationError(
            "parameter_manifest_mismatch",
            f"manifest IDs differ from the case; missing={missing}, extra={extra}",
        )

    normalized: dict[str, dict[str, str]] = {}
    names: list[str] = []
    for parameter_id in sorted(expected):
        raw = manifest[parameter_id]
        case_record = expected[parameter_id]
        if not isinstance(raw, dict) or set(raw) != {"name", "unit", "role"}:
            raise StaticValidationError(
                "parameter_manifest_mismatch",
                f"manifest entry {parameter_id} must contain exactly "
                "name, unit, and role",
            )
        engineering_name = case_record.get("engineering_name")
        unit = case_record.get("unit")
        role = case_record.get("role")
        expected_values = (engineering_name, unit, role)
        if not all(isinstance(value, str) and value for value in expected_values):
            raise StaticValidationError(
                "invalid_evaluator_case", f"parameter {parameter_id} is incomplete"
            )
        if raw != {"name": engineering_name, "unit": unit, "role": role}:
            raise StaticValidationError(
                "parameter_manifest_mismatch",
                f"manifest entry {parameter_id} does not match its public "
                "name, unit, and role",
            )
        names.append(engineering_name)
        normalized[parameter_id] = dict(raw)

    if len(names) != len(set(names)):
        raise StaticValidationError(
            "parameter_binding_not_independent",
            "multiple public parameters bind to the same DEFAULT_PARAMS key",
        )
    if set(defaults) != set(names):
        missing = sorted(set(names) - set(defaults))
        extra = sorted(set(defaults) - set(names))
        raise StaticValidationError(
            "parameter_default_mismatch",
            "DEFAULT_PARAMS keys differ from the manifest; "
            f"missing={missing}, extra={extra}",
        )

    normalized_defaults: dict[str, float] = {}
    by_name = {str(record["engineering_name"]): record for record in parameters}
    for name in sorted(defaults):
        value = defaults[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise StaticValidationError(
                "invalid_parameter_default",
                f"DEFAULT_PARAMS[{name!r}] must be a finite number",
            )
        nominal = by_name[name].get("nominal")
        if (
            not isinstance(nominal, (int, float))
            or isinstance(nominal, bool)
            or not math.isfinite(float(nominal))
        ):
            raise StaticValidationError(
                "invalid_evaluator_case", f"parameter {name} has no finite nominal"
            )
        tolerance = max(1e-9, abs(float(nominal)) * 1e-9)
        if not math.isclose(
            float(value), float(nominal), rel_tol=0.0, abs_tol=tolerance
        ):
            raise StaticValidationError(
                "parameter_default_mismatch",
                f"DEFAULT_PARAMS[{name!r}] does not equal nominal {nominal}",
            )
        normalized_defaults[name] = float(value)
    return normalized_defaults, normalized


def inspect_submission(
    source_path: Path,
    *,
    case: Mapping[str, Any],
) -> StaticInspection:
    """Validate and bind one model source file without executing it."""

    parameters, _ = _case_contract(case)
    execution = _mapping(
        _mapping(case.get("evaluator_profile"), "evaluator_profile").get("execution"),
        "evaluator_profile.execution",
    )
    max_source_bytes = execution.get("max_source_bytes")
    roots = execution.get("allowed_import_roots")
    if (
        not isinstance(max_source_bytes, int)
        or max_source_bytes <= 0
        or not isinstance(roots, list)
        or not roots
        or not all(isinstance(root, str) and root for root in roots)
    ):
        raise StaticValidationError(
            "invalid_evaluator_case", "execution policy is incomplete"
        )
    if not source_path.is_file():
        raise StaticValidationError("submission_missing", "submission.py is missing")
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise StaticValidationError(
            "submission_unreadable", "submission.py could not be read"
        ) from exc
    if len(raw) > max_source_bytes:
        raise StaticValidationError(
            "source_size_limit_exceeded",
            f"submission.py exceeds {max_source_bytes} bytes",
        )
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StaticValidationError(
            "invalid_source_encoding", "submission.py must be UTF-8"
        ) from exc
    try:
        tree = ast.parse(source, filename=str(source_path), mode="exec")
    except SyntaxError as exc:
        raise StaticValidationError(
            "syntax_error", f"submission.py has invalid Python syntax: {exc.msg}"
        ) from exc
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_AST_NODES:
        raise StaticValidationError(
            "source_complexity_limit_exceeded",
            f"submission AST exceeds {MAX_AST_NODES} nodes",
        )

    allowed_roots = set(roots) | {"__future__"}
    _validate_policy(tree, allowed_import_roots=allowed_roots)
    defaults = _literal_mapping(tree, "DEFAULT_PARAMS")
    manifest = _literal_mapping(tree, "PARAMETER_MANIFEST")
    _validate_build(tree)
    _validate_result(tree)
    _validate_step_export(tree)
    normalized_defaults, normalized_manifest = _validate_parameter_bindings(
        defaults, manifest, parameters
    )
    return StaticInspection(
        interface_version=INTERFACE_VERSION,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
        default_params=normalized_defaults,
        parameter_manifest=normalized_manifest,
        allowed_import_roots=tuple(sorted(allowed_roots)),
        ast_node_count=node_count,
    )
