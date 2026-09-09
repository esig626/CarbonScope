"""Explicit observation laws, separate from MID-divergence MFA fitting.

Use multinomial laws only for genuine count measurements. Normalized MIDs,
percentages, peak areas, and arbitrary intensities cannot supply count totals.
Externally corrected continuous MIDs may instead use an explicitly calibrated
Dirichlet law.  Its precision is not a count.  No observation-law action
changes the existing MFA objective.
"""

from .schema import (
    MAX_MULTINOMIAL_TOTAL,
    MIDCountObservation,
    StationaryCountSample,
    StationaryCountSpecification,
    StationaryObservationExperiment,
    StationaryObservationLawComponent,
    StationaryObservationLawResult,
    StationaryObservationSpecification,
    stationary_observation_specification_fingerprint,
    validate_stationary_observation_specification,
)
from .multinomial import (
    MAX_MULTINOMIAL_LOG_MASS_ERROR,
    MultinomialMIDLaw,
    independent_product_kl,
    independent_product_renyi,
    kl_multinomial,
    multinomial_count_constant,
    renyi_multinomial,
)
from .stationary import (
    evaluate_stationary_observation_laws,
    sample_stationary_observations,
)
from .dirichlet import (
    DIRICHLET_NEAR_ONE_REFUSAL_RADIUS,
    DIRICHLET_PRECISION_SOURCES,
    DIRICHLET_RENYI_ROUNDOFF_RELATIVE_BUDGET,
    DIRICHLET_REPLICATE_SEMANTICS,
    DIRICHLET_SIMPLEX_TOLERANCE,
    ComponentPrecisionEstimate,
    DirichletCovarianceDiagnostic,
    DirichletLogLikelihoodScore,
    DirichletMIDLaw,
    DirichletNumericalError,
    ImpliedPrecisionDiagnostic,
    MIDCorrectionProvenance,
    PrecisionDispersion,
    ReplicateDirichletCalibration,
    diagnose_dirichlet_covariance,
    dirichlet_pairwise_log_likelihood_score,
    dirichlet_score_log_moment,
    estimate_dirichlet_precision_from_replicates,
    implied_dirichlet_precision_from_uncertainty,
    kl_dirichlet,
    renyi_dirichlet,
)
from .dirichlet_stationary import (
    StationaryDirichletBlockSpecification,
    StationaryDirichletLawComponent,
    StationaryDirichletObservationExperiment,
    StationaryDirichletObservationLawResult,
    StationaryDirichletObservationSpecification,
    evaluate_stationary_dirichlet_observation_families,
    stationary_dirichlet_observation_specification_fingerprint,
    validate_stationary_dirichlet_observation_specification,
)

__all__ = [
    "MAX_MULTINOMIAL_LOG_MASS_ERROR", "MAX_MULTINOMIAL_TOTAL",
    "MIDCountObservation", "MultinomialMIDLaw",
    "StationaryCountSample", "StationaryCountSpecification",
    "StationaryObservationExperiment", "StationaryObservationLawComponent",
    "StationaryObservationLawResult", "StationaryObservationSpecification",
    "evaluate_stationary_observation_laws", "sample_stationary_observations",
    "independent_product_kl", "independent_product_renyi", "kl_multinomial",
    "multinomial_count_constant", "renyi_multinomial",
    "stationary_observation_specification_fingerprint",
    "validate_stationary_observation_specification",
    "DIRICHLET_NEAR_ONE_REFUSAL_RADIUS", "DIRICHLET_PRECISION_SOURCES",
    "DIRICHLET_RENYI_ROUNDOFF_RELATIVE_BUDGET",
    "DIRICHLET_REPLICATE_SEMANTICS", "DIRICHLET_SIMPLEX_TOLERANCE",
    "ComponentPrecisionEstimate", "DirichletCovarianceDiagnostic",
    "DirichletLogLikelihoodScore", "DirichletMIDLaw", "DirichletNumericalError",
    "ImpliedPrecisionDiagnostic", "MIDCorrectionProvenance", "PrecisionDispersion",
    "ReplicateDirichletCalibration", "diagnose_dirichlet_covariance",
    "dirichlet_pairwise_log_likelihood_score", "dirichlet_score_log_moment",
    "estimate_dirichlet_precision_from_replicates",
    "implied_dirichlet_precision_from_uncertainty", "kl_dirichlet",
    "renyi_dirichlet",
    "StationaryDirichletBlockSpecification", "StationaryDirichletLawComponent",
    "StationaryDirichletObservationExperiment",
    "StationaryDirichletObservationLawResult",
    "StationaryDirichletObservationSpecification",
    "evaluate_stationary_dirichlet_observation_families",
    "stationary_dirichlet_observation_specification_fingerprint",
    "validate_stationary_dirichlet_observation_specification",
]
