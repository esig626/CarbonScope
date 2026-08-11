"""Projection of validated runtime configuration into stationary science."""

from __future__ import annotations

from fluxemu.configuration import ExperimentConfig, TransientExperimentConfig
from fluxemu.model import (
    PoolQuantity,
    StationaryExperimentSemantics,
    Target,
    Tracer,
    TransientExperimentSemantics,
)


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


def project_transient_experiment(
    config: TransientExperimentConfig,
) -> TransientExperimentSemantics:
    """Project validated YAML settings into canonical transient science."""

    if not isinstance(config, TransientExperimentConfig):
        raise TypeError("config must be a validated TransientExperimentConfig")
    return TransientExperimentSemantics(
        tracers=tuple(Tracer(x.metabolite_id, tuple(x.isotopomer_fractions.items()), x.correction)
                      for x in config.tracers),
        targets=tuple(Target(x.fragment_id, x.metabolite_id, x.atom_positions,
                             x.analytical_method, x.formula, x.correction)
                      for x in config.targets),
        time_points=config.time_points,
        pool_quantities=tuple(PoolQuantity(x.metabolite_id, x.quantity)
                              for x in config.pool_quantities),
        initial_internal_mids=config.initial_internal_mids,
    )
