"""Finite-sample testing for declared observation laws.

Simple testing keeps fixed H0=P0 and H1=P1 roles. Composite testing extends
those conventions to explicit finite law classes, with a projected Rényi test
and an unrestricted finite minimax LP. MFA fitting remains separate.
"""

from .simple import (
    BrunoOrderBound,
    BrunoTheoremAssumptionError,
    NumericalLimitError,
    SimpleBinaryLawPair,
    SimpleBinaryTestingConstraint,
    validate_bruno_assumptions,
    validate_renyi_order,
)
from .bruno import bruno_converse_at_order
from .stationary import (
    SimpleBinaryFluxHypotheses,
    StationarySimpleTestingResult,
    evaluate_stationary_simple_hypotheses,
)
from .likelihood import (
    DEFAULT_EXACT_P_VALUE_MAX_OUTCOMES,
    ExactPValueEnumerationLimitError,
    LikelihoodRatioPValue,
    UndefinedLikelihoodRatioError,
    likelihood_ratio_p_value,
    log_likelihood_ratio,
)
from .composite import (
    DEFAULT_COMPOSITE_MAX_OUTCOMES,
    CompositeAchievabilityConditionError,
    CompositeMinimaxSolverError,
    CompositeRenyiConverse,
    FiniteCompositeEnumerationLimitError,
    FiniteCompositeHypotheses,
    FiniteMinimaxTestResult,
    FiniteObservationLaw,
    ProjectedRenyiTestResult,
    composite_renyi_converse_at_order,
    projected_renyi_test,
    solve_finite_minimax_test,
)

__all__ = [
    "BrunoOrderBound", "BrunoTheoremAssumptionError", "NumericalLimitError",
    "SimpleBinaryLawPair", "SimpleBinaryTestingConstraint",
    "SimpleBinaryFluxHypotheses", "StationarySimpleTestingResult",
    "DEFAULT_EXACT_P_VALUE_MAX_OUTCOMES", "ExactPValueEnumerationLimitError",
    "LikelihoodRatioPValue", "UndefinedLikelihoodRatioError",
    "bruno_converse_at_order", "evaluate_stationary_simple_hypotheses",
    "likelihood_ratio_p_value", "log_likelihood_ratio",
    "validate_bruno_assumptions", "validate_renyi_order",
    "DEFAULT_COMPOSITE_MAX_OUTCOMES", "CompositeAchievabilityConditionError",
    "CompositeMinimaxSolverError", "CompositeRenyiConverse",
    "FiniteCompositeEnumerationLimitError", "FiniteCompositeHypotheses",
    "FiniteMinimaxTestResult", "FiniteObservationLaw", "ProjectedRenyiTestResult",
    "composite_renyi_converse_at_order", "projected_renyi_test",
    "solve_finite_minimax_test",
]
