"""Deterministic evaluation of canonical isotope-component flux projections."""

from __future__ import annotations

from collections.abc import Mapping
import math

from fluxemu.exceptions import MappingError
from fluxemu.model import FluxProjectionExpression, FluxProjectionRule


def evaluate_expression(expression: FluxProjectionExpression, flux: Mapping[str, float]) -> float:
    try:
        value = sum(float(term.coefficient) * float(flux[term.reaction_id]) for term in expression.terms)
    except KeyError as error:
        raise MappingError(f"projection references missing physical reaction {error.args[0]!r}") from error
    if not math.isfinite(value):
        raise MappingError("projection expression produced a non-finite value")
    return value


def evaluate_flux_projection(rule: FluxProjectionRule, flux: Mapping[str, float]) -> float:
    """Evaluate ``max(sum(a_j*v_j), 0)`` and enforce declared lump equalities."""

    raw = evaluate_expression(rule.expression, flux)
    tolerance = float(rule.zero_tolerance)
    for equivalent in rule.equivalent_expressions:
        candidate = evaluate_expression(equivalent, flux)
        if abs(candidate - raw) > tolerance:
            raise MappingError(
                f"projection {rule.projection_id!r} equivalent-expression residual "
                f"{abs(candidate - raw)} exceeds {tolerance}"
            )
    if abs(raw) <= tolerance:
        return 0.0
    if rule.transform != "positive_part":
        raise MappingError(f"unsupported projection transform {rule.transform!r}")
    return max(raw, 0.0)


__all__ = ["evaluate_expression", "evaluate_flux_projection"]
