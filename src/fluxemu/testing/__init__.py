"""Simple and finite-composite testing over declared genuine-count laws.

Simple testing fixes one H0/P0 law and one H1/P1 law. Composite testing uses
explicit finite H0/H1 families of complete independent MID product laws and
keeps minimax worst-case Type I/II roles separate from MFA fitting. Finite
classes are never silently convexified.
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
    MIN_EXACT_COMPOSITE_EPSILON,
    CalibratedCompositeScoreTest,
    CompositeBinaryTestingProblem,
    CompositeEnumerationLimitError,
    CompositeMIDLawFamily,
    CompositeOptimizationError,
    CompositeRenyiConverseBound,
    CompositeRenyiScoreCandidate,
    CompositeScoreBound,
    CompositeScoreTestEvaluation,
    CompositeScoreVerificationError,
    FiniteCompositeMinimaxResult,
    IndependentMIDProductLaw,
    calibrate_composite_score_test,
    composite_renyi_converse_at_order,
    composite_renyi_score_candidate,
    composite_score_bound_at_order,
    evaluate_composite_score_test,
    exact_finite_composite_minimax,
    verified_composite_renyi_score,
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
    "CalibratedCompositeScoreTest", "CompositeBinaryTestingProblem",
    "CompositeEnumerationLimitError", "CompositeFluxHypotheses",
    "CompositeMIDLawFamily", "CompositeOptimizationError",
    "CompositeRenyiConverseBound", "CompositeRenyiScoreCandidate",
    "CompositeScoreBound", "CompositeScoreTestEvaluation",
    "CompositeScoreVerificationError", "FiniteCompositeMinimaxResult",
    "IndependentMIDProductLaw", "StationaryCompositeTestingResult",
    "calibrate_composite_score_test", "composite_renyi_converse_at_order",
    "composite_renyi_score_candidate", "composite_score_bound_at_order",
    "evaluate_composite_score_test", "evaluate_stationary_composite_hypotheses",
    "exact_finite_composite_minimax", "verified_composite_renyi_score",
]
