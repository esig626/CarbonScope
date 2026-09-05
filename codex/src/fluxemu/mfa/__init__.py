"""Stationary MFA scientific records and exact KL/Rényi MID divergences.

Pure data and divergence functionality does not import optional optimization
backends. A later fit action loads its backend only when it is needed.
"""

from .divergence import kl_divergence, renyi_divergence, validate_mid
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
    "kl_divergence",
    "mfa_fit_fingerprint",
    "mfa_problem_fingerprint",
    "renyi_divergence",
    "validate_mid",
    "validate_stationary_mfa_problem",
]
