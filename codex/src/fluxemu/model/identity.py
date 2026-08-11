"""Separate deterministic identities for models and stationary experiments."""

from __future__ import annotations

from hashlib import sha256

from .schema import CanonicalModel, StationaryExperimentSemantics, TransientExperimentSemantics
from .serialisation import deterministic_serialise
from .validation import (
    validate_canonical_model,
    validate_stationary_experiment,
    validate_transient_experiment,
)


def _fingerprint(value: object) -> str:
    return sha256(deterministic_serialise(value).encode("utf-8")).hexdigest()


def model_fingerprint(model: CanonicalModel) -> str:
    """Fingerprint only the flux and isotope scientific model state."""

    validate_canonical_model(model)
    return _fingerprint(model)


def experiment_fingerprint(
    model: CanonicalModel,
    experiment: StationaryExperimentSemantics,
) -> str:
    """Validate against ``model`` then fingerprint only experiment semantics."""

    validate_stationary_experiment(model, experiment)
    return _fingerprint(experiment)


def transient_experiment_fingerprint(
    model: CanonicalModel,
    experiment: TransientExperimentSemantics,
) -> str:
    """Fingerprint the complete, explicitly dynamic experiment contract."""

    validate_transient_experiment(model, experiment)
    return _fingerprint(experiment)


fingerprint_model = model_fingerprint
fingerprint_experiment = experiment_fingerprint
