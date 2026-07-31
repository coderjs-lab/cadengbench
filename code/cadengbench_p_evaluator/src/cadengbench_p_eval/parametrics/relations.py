"""Safe typed arithmetic relations and optional bounded Z3 implication checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

from .domains import ContinuousDomain, DiscreteDomain, ParameterDomain


class RelationError(ValueError):
    """A typed relation is malformed or cannot be evaluated."""


class MissingRelationValue(RelationError):
    """A declared relation input is absent from the replay evidence."""

    def __init__(self, source: str, key: str) -> None:
        self.source = source
        self.key = key
        super().__init__(f"relation value is missing: {source}.{key}")


class SymbolicRelationUnsupported(RelationError):
    """A valid runtime relation is outside the deliberately small Z3 subset."""


_VARIABLE_SOURCES = {
    "parameter",
    "measurement",
    "baseline_parameter",
    "baseline_measurement",
    "metadata",
}


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelationError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RelationError(f"{field_name} must be a finite number")
    return number


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationError(f"{field_name} must be a nonempty string")
    return value


def _check_keys(raw: Mapping[str, Any], allowed: set[str], label: str) -> None:
    extras = set(raw) - allowed
    if extras:
        raise RelationError(f"unsupported {label} fields: {sorted(extras)}")


@dataclass(frozen=True)
class RelationContext:
    parameters: Mapping[str, float] = field(default_factory=dict)
    measurements: Mapping[str, float] = field(default_factory=dict)
    baseline_parameters: Mapping[str, float] = field(default_factory=dict)
    baseline_measurements: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, float] = field(default_factory=dict)

    def resolve(self, source: str, key: str) -> float:
        values = {
            "parameter": self.parameters,
            "measurement": self.measurements,
            "baseline_parameter": self.baseline_parameters,
            "baseline_measurement": self.baseline_measurements,
            "metadata": self.metadata,
        }.get(source)
        if values is None or key not in values:
            raise MissingRelationValue(source, key)
        return _number(values[key], f"{source}.{key}")


class Expression(Protocol):
    def evaluate(self, context: RelationContext) -> float: ...


@dataclass(frozen=True)
class ConstantExpression:
    value: float

    def evaluate(self, context: RelationContext) -> float:
        del context
        return self.value


@dataclass(frozen=True)
class VariableExpression:
    source: str
    key: str

    def evaluate(self, context: RelationContext) -> float:
        return context.resolve(self.source, self.key)


@dataclass(frozen=True)
class ArithmeticExpression:
    op: str
    args: tuple[Expression, ...]

    def evaluate(self, context: RelationContext) -> float:
        values = [argument.evaluate(context) for argument in self.args]
        if self.op == "add":
            result = sum(values)
        elif self.op == "sub":
            result = values[0] - values[1]
        elif self.op == "mul":
            result = math.prod(values)
        elif self.op == "div":
            if values[1] == 0.0:
                raise RelationError("relation division by zero")
            result = values[0] / values[1]
        elif self.op == "neg":
            result = -values[0]
        elif self.op == "abs":
            result = abs(values[0])
        else:  # pragma: no cover - parser guards this branch
            raise RelationError(f"unsupported arithmetic operator: {self.op}")
        if not math.isfinite(result):
            raise RelationError("relation arithmetic produced a non-finite value")
        return result


NumericExpression: TypeAlias = (
    ConstantExpression | VariableExpression | ArithmeticExpression
)


class Predicate(Protocol):
    def evaluate(self, context: RelationContext) -> bool: ...


@dataclass(frozen=True)
class ComparisonPredicate:
    op: str
    left: Expression
    right: Expression
    tolerance: float = 0.0

    def evaluate(self, context: RelationContext) -> bool:
        left = self.left.evaluate(context)
        right = self.right.evaluate(context)
        difference = left - right
        if self.op == "eq":
            return abs(difference) <= self.tolerance
        if self.op == "ne":
            return abs(difference) > self.tolerance
        if self.op == "lt":
            return difference < -self.tolerance
        if self.op == "le":
            return difference <= self.tolerance
        if self.op == "gt":
            return difference > self.tolerance
        if self.op == "ge":
            return difference >= -self.tolerance
        raise RelationError(f"unsupported comparison operator: {self.op}")


@dataclass(frozen=True)
class LogicalPredicate:
    op: str
    args: tuple[Predicate, ...]

    def evaluate(self, context: RelationContext) -> bool:
        if self.op == "and":
            return all(argument.evaluate(context) for argument in self.args)
        if self.op == "or":
            return any(argument.evaluate(context) for argument in self.args)
        if self.op == "not":
            return not self.args[0].evaluate(context)
        raise RelationError(f"unsupported logical operator: {self.op}")


TypedPredicate: TypeAlias = ComparisonPredicate | LogicalPredicate


@dataclass(frozen=True)
class TypedRelation:
    relation_id: str
    predicate: TypedPredicate
    authority: str
    evidence_ref: str
    required: bool = True

    def evaluate(self, context: RelationContext) -> bool:
        return self.predicate.evaluate(context)


def parse_expression(raw: Mapping[str, Any]) -> NumericExpression:
    if not isinstance(raw, Mapping):
        raise RelationError("expression must be an object")
    node_type = raw.get("type")
    if node_type == "constant":
        _check_keys(raw, {"type", "value"}, "constant expression")
        return ConstantExpression(_number(raw.get("value"), "constant.value"))
    if node_type == "variable":
        _check_keys(raw, {"type", "source", "key"}, "variable expression")
        source = _string(raw.get("source"), "variable.source")
        if source not in _VARIABLE_SOURCES:
            raise RelationError(f"unsupported variable source: {source}")
        return VariableExpression(source, _string(raw.get("key"), "variable.key"))
    if node_type == "arithmetic":
        _check_keys(raw, {"type", "op", "args"}, "arithmetic expression")
        op = _string(raw.get("op"), "arithmetic.op")
        arguments = raw.get("args")
        if not isinstance(arguments, list):
            raise RelationError("arithmetic.args must be a list")
        arity = len(arguments)
        if op in {"add", "mul"} and arity < 2:
            raise RelationError(f"{op} requires at least two arguments")
        if op in {"sub", "div"} and arity != 2:
            raise RelationError(f"{op} requires exactly two arguments")
        if op in {"neg", "abs"} and arity != 1:
            raise RelationError(f"{op} requires exactly one argument")
        if op not in {"add", "sub", "mul", "div", "neg", "abs"}:
            raise RelationError(f"unsupported arithmetic operator: {op}")
        return ArithmeticExpression(
            op, tuple(parse_expression(value) for value in arguments)
        )
    raise RelationError("expression.type must be constant, variable, or arithmetic")


def parse_predicate(raw: Mapping[str, Any]) -> TypedPredicate:
    if not isinstance(raw, Mapping):
        raise RelationError("predicate must be an object")
    node_type = raw.get("type")
    if node_type == "comparison":
        _check_keys(raw, {"type", "op", "left", "right", "tolerance"}, "comparison")
        op = _string(raw.get("op"), "comparison.op")
        if op not in {"eq", "ne", "lt", "le", "gt", "ge"}:
            raise RelationError(f"unsupported comparison operator: {op}")
        tolerance = _number(raw.get("tolerance", 0.0), "comparison.tolerance")
        if tolerance < 0.0:
            raise RelationError("comparison tolerance cannot be negative")
        return ComparisonPredicate(
            op=op,
            left=parse_expression(raw.get("left")),
            right=parse_expression(raw.get("right")),
            tolerance=tolerance,
        )
    if node_type == "logical":
        _check_keys(raw, {"type", "op", "args"}, "logical predicate")
        op = _string(raw.get("op"), "logical.op")
        arguments = raw.get("args")
        if not isinstance(arguments, list):
            raise RelationError("logical.args must be a list")
        if op in {"and", "or"} and len(arguments) < 2:
            raise RelationError(f"{op} requires at least two predicates")
        if op == "not" and len(arguments) != 1:
            raise RelationError("not requires exactly one predicate")
        if op not in {"and", "or", "not"}:
            raise RelationError(f"unsupported logical operator: {op}")
        return LogicalPredicate(
            op, tuple(parse_predicate(value) for value in arguments)
        )
    raise RelationError("predicate.type must be comparison or logical")


def parse_relation(raw: Mapping[str, Any]) -> TypedRelation:
    if not isinstance(raw, Mapping):
        raise RelationError("relation must be an object")
    _check_keys(
        raw,
        {"relation_id", "predicate", "authority", "evidence_ref", "required"},
        "relation",
    )
    required = raw.get("required", True)
    if not isinstance(required, bool):
        raise RelationError("relation.required must be a boolean")
    return TypedRelation(
        relation_id=_string(raw.get("relation_id"), "relation.relation_id"),
        predicate=parse_predicate(raw.get("predicate")),
        authority=_string(raw.get("authority"), "relation.authority"),
        evidence_ref=_string(raw.get("evidence_ref"), "relation.evidence_ref"),
        required=required,
    )


@dataclass(frozen=True)
class ImplicationProof:
    status: str
    counterexample: dict[str, str] = field(default_factory=dict)
    reason: str | None = None

    @property
    def proved(self) -> bool:
        return self.status == "proved"


def _z3_expression(
    expression: Expression, variables: Mapping[str, Any], z3: Any
) -> Any:
    if isinstance(expression, ConstantExpression):
        return z3.RealVal(str(expression.value))
    if isinstance(expression, VariableExpression):
        if expression.source != "parameter":
            raise SymbolicRelationUnsupported(
                "Z3 implication supports public parameter variables only"
            )
        if expression.key not in variables:
            raise SymbolicRelationUnsupported(
                f"Z3 relation references an unbounded parameter: {expression.key}"
            )
        return variables[expression.key]
    if not isinstance(expression, ArithmeticExpression):
        raise SymbolicRelationUnsupported("unsupported symbolic expression")
    arguments = [
        _z3_expression(argument, variables, z3) for argument in expression.args
    ]
    if expression.op == "add":
        return sum(arguments)
    if expression.op == "sub":
        return arguments[0] - arguments[1]
    if expression.op == "mul":
        result = arguments[0]
        for argument in arguments[1:]:
            result *= argument
        return result
    if expression.op == "div":
        denominator = expression.args[1]
        if not isinstance(denominator, ConstantExpression) or denominator.value == 0.0:
            raise SymbolicRelationUnsupported(
                "symbolic division requires a nonzero constant denominator"
            )
        return arguments[0] / arguments[1]
    if expression.op == "neg":
        return -arguments[0]
    if expression.op == "abs":
        return z3.If(arguments[0] >= 0, arguments[0], -arguments[0])
    raise SymbolicRelationUnsupported(f"unsupported symbolic operator: {expression.op}")


def _z3_predicate(predicate: Predicate, variables: Mapping[str, Any], z3: Any) -> Any:
    if isinstance(predicate, ComparisonPredicate):
        left = _z3_expression(predicate.left, variables, z3)
        right = _z3_expression(predicate.right, variables, z3)
        tolerance = z3.RealVal(str(predicate.tolerance))
        difference = left - right
        if predicate.op == "eq":
            return z3.And(difference <= tolerance, difference >= -tolerance)
        if predicate.op == "ne":
            return z3.Or(difference > tolerance, difference < -tolerance)
        if predicate.op == "lt":
            return difference < -tolerance
        if predicate.op == "le":
            return difference <= tolerance
        if predicate.op == "gt":
            return difference > tolerance
        if predicate.op == "ge":
            return difference >= -tolerance
    if isinstance(predicate, LogicalPredicate):
        arguments = [
            _z3_predicate(argument, variables, z3) for argument in predicate.args
        ]
        if predicate.op == "and":
            return z3.And(*arguments)
        if predicate.op == "or":
            return z3.Or(*arguments)
        if predicate.op == "not":
            return z3.Not(arguments[0])
    raise SymbolicRelationUnsupported("unsupported symbolic predicate")


def _parameter_dependencies(expression: Expression) -> set[str]:
    if isinstance(expression, ConstantExpression):
        return set()
    if isinstance(expression, VariableExpression):
        if expression.source != "parameter":
            raise SymbolicRelationUnsupported(
                "derived symbolic expressions support parameter variables only"
            )
        return {expression.key}
    if isinstance(expression, ArithmeticExpression):
        return set().union(
            *(_parameter_dependencies(argument) for argument in expression.args)
        )
    raise SymbolicRelationUnsupported("unsupported derived symbolic expression")


def _extend_symbolic_parameters(
    variables: Mapping[str, Any],
    derived_expressions: Mapping[str, Mapping[str, Any]] | None,
    z3: Any,
) -> dict[str, Any]:
    values = dict(variables)
    if not derived_expressions:
        return values
    pending: dict[str, Expression] = {}
    for parameter_id, raw in derived_expressions.items():
        if (
            not isinstance(parameter_id, str)
            or not parameter_id
            or parameter_id in values
        ):
            raise RelationError("derived symbolic parameter IDs must be unique")
        pending[parameter_id] = parse_expression(raw)
    while pending:
        progressed = False
        for parameter_id in sorted(tuple(pending)):
            expression = pending[parameter_id]
            dependencies = _parameter_dependencies(expression)
            if not dependencies.issubset(values):
                continue
            values[parameter_id] = _z3_expression(expression, values, z3)
            pending.pop(parameter_id)
            progressed = True
        if not progressed:
            raise RelationError(
                "derived symbolic parameters are cyclic or reference unknown IDs: "
                f"{sorted(pending)}"
            )
    return values


def _z3_domain_constraints(
    domains: Mapping[str, ParameterDomain], variables: Mapping[str, Any], z3: Any
) -> list[Any]:
    constraints: list[Any] = []
    for parameter_id in sorted(domains):
        domain = domains[parameter_id]
        variable = variables[parameter_id]
        if isinstance(domain, ContinuousDomain):
            lower = z3.RealVal(str(domain.minimum))
            upper = z3.RealVal(str(domain.maximum))
            constraints.append(
                variable >= lower if domain.minimum_inclusive else variable > lower
            )
            constraints.append(
                variable <= upper if domain.maximum_inclusive else variable < upper
            )
        elif isinstance(domain, DiscreteDomain):
            constraints.append(
                z3.Or(*[variable == z3.RealVal(str(value)) for value in domain.values])
            )
        else:
            raise RelationError(
                f"unsupported domain for parameter {parameter_id}: "
                f"{type(domain).__name__}"
            )
    return constraints


def _solver_result(
    constraints: Sequence[Any], *, timeout_ms: int, z3: Any
) -> tuple[Any, Any]:
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*constraints)
    return solver, solver.check()


def prove_implication(
    domains: Mapping[str, ParameterDomain],
    assumptions: Sequence[TypedPredicate],
    conclusion: TypedPredicate,
    *,
    timeout_ms: int = 1000,
    derived_expressions: Mapping[str, Mapping[str, Any]] | None = None,
) -> ImplicationProof:
    """Prove ``domains AND assumptions => conclusion`` without vacuous success.

    The domain and then the complete antecedent are checked for satisfiability
    before the conclusion is translated or negated. Invalid frozen contracts
    therefore remain evaluator-owned, unscoreable evidence rather than a model
    failure or a vacuously proved implication.
    """

    if timeout_ms < 1:
        raise RelationError("Z3 timeout must be positive")
    try:
        import z3  # type: ignore[import-not-found]
    except ImportError:
        return ImplicationProof("unavailable", reason="z3-solver is not installed")

    if not domains:
        return ImplicationProof(
            "invalid_contract", reason="SMT obligations require at least one domain"
        )
    if not all(isinstance(key, str) and key for key in domains):
        return ImplicationProof(
            "invalid_contract", reason="domain parameter IDs must be nonempty strings"
        )

    try:
        independent_variables = {
            parameter_id: z3.Real(parameter_id) for parameter_id in sorted(domains)
        }
        variables = _extend_symbolic_parameters(
            independent_variables, derived_expressions, z3
        )
        domain_constraints = _z3_domain_constraints(domains, independent_variables, z3)
    except (RelationError, TypeError, ValueError) as exc:
        return ImplicationProof("invalid_contract", reason=str(exc))

    domain_solver, domain_result = _solver_result(
        domain_constraints, timeout_ms=timeout_ms, z3=z3
    )
    if domain_result == z3.unsat:
        return ImplicationProof(
            "invalid_contract", reason="frozen parameter domains are unsatisfiable"
        )
    if domain_result == z3.unknown:
        return ImplicationProof(
            "unknown",
            reason=f"domain satisfiability: {domain_solver.reason_unknown()}",
        )

    try:
        assumption_constraints = [
            _z3_predicate(assumption, variables, z3) for assumption in assumptions
        ]
    except SymbolicRelationUnsupported as exc:
        return ImplicationProof("unsupported", reason=str(exc))
    except (RelationError, TypeError, ValueError) as exc:
        return ImplicationProof("invalid_contract", reason=str(exc))

    antecedent_constraints = [*domain_constraints, *assumption_constraints]
    antecedent_solver, antecedent_result = _solver_result(
        antecedent_constraints, timeout_ms=timeout_ms, z3=z3
    )
    if antecedent_result == z3.unsat:
        return ImplicationProof(
            "infeasible_assumptions",
            reason="domain and assumptions are jointly unsatisfiable",
        )
    if antecedent_result == z3.unknown:
        return ImplicationProof(
            "unknown",
            reason=f"antecedent satisfiability: {antecedent_solver.reason_unknown()}",
        )

    try:
        negated_conclusion = z3.Not(_z3_predicate(conclusion, variables, z3))
    except SymbolicRelationUnsupported as exc:
        return ImplicationProof("unsupported", reason=str(exc))
    except (RelationError, TypeError, ValueError) as exc:
        return ImplicationProof("invalid_contract", reason=str(exc))

    solver, result = _solver_result(
        [*antecedent_constraints, negated_conclusion],
        timeout_ms=timeout_ms,
        z3=z3,
    )
    if result == z3.unsat:
        return ImplicationProof("proved")
    if result == z3.sat:
        model = solver.model()
        counterexample = {
            parameter_id: str(model.eval(variable, model_completion=True))
            for parameter_id, variable in variables.items()
        }
        return ImplicationProof("counterexample", counterexample=counterexample)
    return ImplicationProof("unknown", reason=solver.reason_unknown())
