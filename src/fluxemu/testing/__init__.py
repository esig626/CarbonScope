"""Simple and finite-composite testing over declared genuine-count laws.

Simple testing fixes one H0/P0 law and one H1/P1 law. Composite testing uses
explicit finite H0/H1 law families and keeps minimax worst-case Type I/II roles
separate from MFA fitting. Composite classes are never silently convexified.
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
    DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES,
    CalibratedCompositeProjectedTest,
    CompositeBinaryTestingProblem,
    CompositeEnumerationLimitError,
    CompositeMIDLawFamily,
    CompositeOptimizationError,
    CompositeProjectedBound,
    CompositeProjectionError,
    CompositeRenyiConverseBound,
    FiniteCompositeMinimaxResult,
    VerifiedCompositeRenyiProjection,
    calibrate_composite_projected_test,
    composite_renyi_converse_at_order,
    projected_composite_bound_at_order,
    verified_composite_renyi_projection,
)
from .composite_lp import (
    MIN_EXACT_COMPOSITE_EPSILON,
    exact_finite_composite_minimax,
)
from .composite_stationary import (
    CompositeFluxHypotheses,
    StationaryCompositeTestingResult,
    evaluate_stationary_composite_hypotheses,
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
    "DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES", "MIN_EXACT_COMPOSITE_EPSILON",
    "CalibratedCompositeProjectedTest", "CompositeBinaryTestingProblem",
    "CompositeEnumerationLimitError", "CompositeFluxHypotheses",
    "CompositeMIDLawFamily", "CompositeOptimizationError",
    "CompositeProjectedBound", "CompositeProjectionError",
    "CompositeRenyiConverseBound", "FiniteCompositeMinimaxResult",
    "StationaryCompositeTestingResult", "VerifiedCompositeRenyiProjection",
    "calibrate_composite_projected_test", "composite_renyi_converse_at_order",
    "evaluate_stationary_composite_hypotheses", "exact_finite_composite_minimax",
    "projected_composite_bound_at_order", "verified_composite_renyi_projection",
]
