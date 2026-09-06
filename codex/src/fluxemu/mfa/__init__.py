"""Native stationary MFA with exact KL/Rényi MID-divergence fitting.

Pure data and divergence functionality does not import optional optimization
backends. Fitting loads SciPy only when optimization is requested.
"""

from .divergence import kl_divergence, renyi_divergence, validate_mid
from .normalisation import normalise_mid
from .schema import (
    DivergenceObjectiveConfig,
    MFAFitError,
    MFAObjectiveEvaluation,
    MFAObservationDivergence,
    MFAOptimizationConfig,
    MFAStartDiagnostic,
    StationaryMFAExperiment,
    StationaryMFAProblem,
    StationaryMFAResult,
    StationaryMIDObservation,
    mfa_fit_fingerprint,
    mfa_problem_fingerprint,
    validate_stationary_mfa_problem,
)
from .stationary import evaluate_stationary_mfa, fit_stationary_mfa

__all__ = [
    "DivergenceObjectiveConfig",
    "MFAFitError",
    "MFAObjectiveEvaluation",
    "MFAObservationDivergence",
    "MFAOptimizationConfig",
    "MFAStartDiagnostic",
    "StationaryMFAExperiment",
    "StationaryMFAProblem",
    "StationaryMFAResult",
    "StationaryMIDObservation",
    "evaluate_stationary_mfa",
    "fit_stationary_mfa",
    "kl_divergence",
    "mfa_fit_fingerprint",
    "mfa_problem_fingerprint",
    "normalise_mid",
    "renyi_divergence",
    "validate_mid",
    "validate_stationary_mfa_problem",
]
