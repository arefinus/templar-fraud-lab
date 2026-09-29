"""Allowlisted predicate AST with a tiny interpreter. Never `eval`.

A node is a JSON object. Allowed shapes:

  {"const": <number|string>}
  {"feat": "<feature name>"}
  {"op": "<comparison|arith|logic|aggregate>", "args": [node, ...]}

Comparisons: > >= < <= == != in            (in: [feat, {"const": [..]}])
Arithmetic:  + - * /                        (2 args; / of 0 -> None -> False)
Logic:       and or not
Aggregates:  sum mean max min               over feature-ref args only
Anything else raises PredicateError. Depth is bounded.
"""

from __future__ import annotations

from typing import Any

COMPARISONS = ("==", "!=", ">", ">=", "<", "<=", "in")
ARITHMETIC = ("+", "-", "*", "/")
LOGIC = ("and", "or", "not")
AGGREGATES = ("sum", "mean", "max", "min")
ALLOWED_OPS = frozenset(COMPARISONS + ARITHMETIC + LOGIC + AGGREGATES)
MAX_DEPTH = 8
MAX_CONJUNCTS = 3


class PredicateError(ValueError):
    """The AST contains a node outside the allowlist or is malformed."""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(node: Any, depth: int = 0, allowed_features: set[str] | None = None) -> None:
    if depth > MAX_DEPTH:
        raise PredicateError(f"predicate deeper than {MAX_DEPTH}")
    if not isinstance(node, dict):
        raise PredicateError(f"node must be an object, got {type(node).__name__}")
    keys = set(node)
    if keys == {"const"}:
        v = node["const"]
        if not (_is_number(v) or isinstance(v, str)
                or (isinstance(v, list) and all(_is_number(x) or isinstance(x, str) for x in v))):
            raise PredicateError(f"constant of unsupported type: {v!r}")
        return
    if keys == {"feat"}:
        if not isinstance(node["feat"], str):
            raise PredicateError("feature reference must be a string")
        if allowed_features is not None and node["feat"] not in allowed_features:
            raise PredicateError(f"unknown feature {node['feat']!r}")
        return
    if keys != {"op", "args"}:
        raise PredicateError(f"node keys must be {{const}}, {{feat}} or {{op,args}}, got {sorted(keys)}")
    op, args = node["op"], node["args"]
    if op not in ALLOWED_OPS:
        raise PredicateError(f"operator {op!r} is not allowlisted")
    if not isinstance(args, list) or not args:
        raise PredicateError("args must be a non-empty list")
    if op == "not" and len(args) != 1:
        raise PredicateError("not takes exactly one argument")
    if op in COMPARISONS + ARITHMETIC and len(args) != 2:
        raise PredicateError(f"{op} takes exactly two arguments")
    if op in AGGREGATES and any(set(a) != {"feat"} for a in args if isinstance(a, dict)):
        raise PredicateError("aggregates accept feature references only")
    for a in args:
        validate(a, depth + 1, allowed_features)


def referenced_features(node: dict[str, Any]) -> list[str]:
    if "feat" in node:
        return [node["feat"]]
    if "const" in node:
        return []
    out: list[str] = []
    for a in node["args"]:
        out.extend(referenced_features(a))
    return list(dict.fromkeys(out))


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool) or isinstance(v, str):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN -> None


def evaluate(node: dict[str, Any], row: dict[str, Any]) -> Any:
    """Evaluate a VALIDATED node against a row mapping. Missing -> None -> False."""
    if "const" in node:
        return node["const"]
    if "feat" in node:
        return row.get(node["feat"])
    op, args = node["op"], node["args"]
    if op == "not":
        return not bool(evaluate(args[0], row))
    if op == "and":
        return all(bool(evaluate(a, row)) for a in args)
    if op == "or":
        return any(bool(evaluate(a, row)) for a in args)
    if op in AGGREGATES:
        vals = [x for x in (_num(evaluate(a, row)) for a in args) if x is not None]
        if not vals:
            return None
        return {"sum": sum, "mean": lambda v: sum(v) / len(v), "max": max, "min": min}[op](vals)
    left, right = evaluate(args[0], row), evaluate(args[1], row)
    if op == "in":
        return left is not None and left in (right if isinstance(right, list) else [right])
    if op in ("==", "!="):
        if left is None or right is None:
            return False
        return (left == right) if op == "==" else (left != right)
    a, b = _num(left), _num(right)
    if a is None or b is None:
        return None if op in ARITHMETIC else False
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return None if b == 0 else a / b
    return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]


def count_conjuncts(node: dict[str, Any]) -> int:
    if "op" in node and node["op"] == "and":
        return sum(count_conjuncts(a) for a in node["args"])
    return 1


def depth(node: dict[str, Any]) -> int:
    if "op" not in node:
        return 0
    return 1 + max(depth(a) for a in node["args"])
