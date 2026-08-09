"""Projection of validated runtime configuration into stationary science."""

from __future__ import annotations

from fluxemu.configuration import ExperimentConfig
from fluxemu.model import StationaryExperimentSemantics, Target, Tracer


def project_stationary_experiment(
    config: ExperimentConfig,
) -> StationaryExperimentSemantics:
    """Retain only the ordered scientific identity of a stationary experiment."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be a validated ExperimentConfig")
    return StationaryExperimentSemantics(
        tracers=tuple(
            Tracer(
                tracer.metabolite_id,
                tuple(tracer.isotopomer_fractions.items()),
                tracer.correction,
            )
            for tracer in config.tracers
        ),
        targets=tuple(
            Target(
                target.fragment_id,
                target.metabolite_id,
                target.atom_positions,
                target.analytical_method,
                target.formula,
                target.correction,
            )
            for target in config.targets
        ),
    )
