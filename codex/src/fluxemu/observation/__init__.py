"""Explicit fixed-total count laws, separate from MID-divergence MFA fitting.

Use multinomial laws only for genuine count measurements. Normalized MIDs,
percentages, peak areas, and arbitrary intensities cannot supply count totals.
No observation-law action changes the existing MFA objective.
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

__all__ = [
    "MAX_MULTINOMIAL_LOG_MASS_ERROR", "MAX_MULTINOMIAL_TOTAL",
    "MIDCountObservation", "MultinomialMIDLaw",
    "StationaryCountSample", "StationaryCountSpecification",
    "StationaryObservationExperiment", "StationaryObservationLawComponent",
    "StationaryObservationLawResult", "StationaryObservationSpecification",
    "independent_product_kl", "independent_product_renyi", "kl_multinomial",
    "multinomial_count_constant", "renyi_multinomial",
    "stationary_observation_specification_fingerprint",
    "validate_stationary_observation_specification",
]
