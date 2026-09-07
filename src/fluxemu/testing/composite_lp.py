"""Numerically guarded exact finite-family composite minimax linear programme."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from ..exceptions import InputValidationError
from .composite import (
    DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES,
    CompositeBinaryTestingProblem,
    CompositeOptimizationError,
    FiniteCompositeMinimaxResult,
    _enumerate_outcomes,
    _family_mass_matrix,
)
from .simple import NumericalLimitError, SimpleBinaryTestingConstraint


MIN_EXACT_COMPOSITE_EPSILON = 1e-12
_HIGHS_FEASIBILITY_TOLERANCE = 1e-10
_RESULT_RELATIVE_TOLERANCE = 5e-10


def _accurate_errors(matrix: np.ndarray, rejection: np.ndarray) -> tuple[float, ...]:
    return tuple(
        math.fsum(
            float(probability) * float(decision)
            for probability, decision in zip(row, rejection, strict=True)
        )
        for row in matrix
    )


def exact_finite_composite_minimax(
    problem: CompositeBinaryTestingProblem,
    *,
    epsilon: Real,
    max_outcomes: int = DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES,
) -> FiniteCompositeMinimaxResult:
    """Solve the complete finite-class randomised minimax test with guarded LP numerics.

    Null constraints are divided by the declared Type-I budget before they are
    passed to HiGHS. This keeps the right-hand side at one and prevents the LP's
    absolute primal-feasibility tolerance from becoming the statistical Type-I
    tolerance. Budgets below ``MIN_EXACT_COMPOSITE_EPSILON`` fail explicitly;
    no larger budget is substituted.

    The complete count space is still enumerated exactly as represented by the
    multinomial laws. Structural zeros are retained and ``max_outcomes`` is a
    hard failure boundary, not an approximation switch.
    """

    if not isinstance(problem, CompositeBinaryTestingProblem):
        raise InputValidationError("problem must be CompositeBinaryTestingProblem")
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    if constraint.epsilon < MIN_EXACT_COMPOSITE_EPSILON:
        raise NumericalLimitError(
            "exact composite minimax Type-I budget is below the supported "
            f"LP numerical floor {MIN_EXACT_COMPOSITE_EPSILON:g}; no budget "
            "substitution was made"
        )

    outcomes = _enumerate_outcomes(problem, max_outcomes)
    null_mass = _family_mass_matrix(problem.null, outcomes)
    alternative_mass = _family_mass_matrix(problem.alternative, outcomes)

    try:
        from scipy.optimize import linprog
    except ImportError as error:  # pragma: no cover - full CI installs the extra
        raise CompositeOptimizationError(
            "exact composite minimax testing requires the 'testing' SciPy extra"
        ) from error

    outcome_count = len(outcomes)
    variable_count = outcome_count + 1
    objective = np.zeros(variable_count, dtype=float)
    objective[-1] = 1.0

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    inverse_budget = 1.0 / constraint.epsilon
    if not math.isfinite(inverse_budget):
        raise NumericalLimitError(
            "exact composite minimax Type-I budget cannot be represented in the scaled LP"
        )
    for probabilities in null_mass:
        row = np.zeros(variable_count, dtype=float)
        row[:outcome_count] = probabilities * inverse_budget
        if not np.isfinite(row).all():
            raise NumericalLimitError(
                "scaled composite Type-I constraint exceeds finite LP representability"
            )
        rows.append(row)
        rhs.append(1.0)
    for probabilities in alternative_mass:
        # beta >= 1 - E_Q[phi]  <=>  -E_Q[phi] - beta <= -1.
        row = np.zeros(variable_count, dtype=float)
        row[:outcome_count] = -probabilities
        row[-1] = -1.0
        rows.append(row)
        rhs.append(-1.0)

    result = linprog(
        objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs, dtype=float),
        bounds=[(0.0, 1.0)] * variable_count,
        method="highs",
        options={
            "primal_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE,
            "dual_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE,
            "ipm_optimality_tolerance": 1e-12,
        },
    )
    if not result.success or result.x is None:
        raise CompositeOptimizationError(
            "exact finite composite minimax LP failed: "
            + (result.message or "unknown HiGHS failure")
        )

    phi = np.asarray(result.x[:outcome_count], dtype=float)
    beta_variable = float(result.x[-1])
    if not np.isfinite(phi).all() or np.any(phi < 0) or np.any(phi > 1):
        raise CompositeOptimizationError(
            "exact composite minimax LP returned invalid rejection probabilities"
        )
    if not math.isfinite(beta_variable) or not 0 <= beta_variable <= 1:
        raise CompositeOptimizationError(
            "exact composite minimax LP returned an invalid Type-II objective"
        )

    null_errors = _accurate_errors(null_mass, phi)
    alternative_power = _accurate_errors(alternative_mass, phi)
    alternative_errors = tuple(1.0 - value for value in alternative_power)
    worst_alpha = max(null_errors)
    worst_beta = max(alternative_errors)

    alpha_tolerance = _RESULT_RELATIVE_TOLERANCE * constraint.epsilon
    if worst_alpha > constraint.epsilon + alpha_tolerance:
        raise CompositeOptimizationError(
            "exact composite minimax LP violates the declared Type-I constraint: "
            f"worst alpha={worst_alpha:.17g}, epsilon={constraint.epsilon:.17g}"
        )
    beta_tolerance = _RESULT_RELATIVE_TOLERANCE * max(1.0, abs(worst_beta))
    if abs(worst_beta - beta_variable) > beta_tolerance:
        raise CompositeOptimizationError(
            "exact composite minimax LP objective disagrees with evaluated worst-case Type II"
        )

    active_tolerance = 1e-9
    active_null = tuple(
        index
        for index, value in enumerate(null_errors)
        if abs(value - worst_alpha) <= active_tolerance
    )
    active_alternative = tuple(
        index
        for index, value in enumerate(alternative_errors)
        if abs(value - worst_beta) <= active_tolerance
    )
    return FiniteCompositeMinimaxResult(
        problem=problem,
        constraint=constraint,
        outcomes=outcomes,
        rejection_probabilities=tuple(float(value) for value in phi),
        null_type_i_errors=null_errors,
        alternative_type_ii_errors=alternative_errors,
        worst_type_i_error=worst_alpha,
        minimax_type_ii_error=worst_beta,
        active_null_members=active_null,
        active_alternative_members=active_alternative,
    )


__all__ = ["MIN_EXACT_COMPOSITE_EPSILON", "exact_finite_composite_minimax"]
