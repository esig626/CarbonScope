"""Native FBA/FVA to stationary-MID orchestration over one canonical model."""

from __future__ import annotations

from dataclasses import dataclass

from fluxemu.emu import NativeStationaryResult, compile_emu_plan, evaluate_stationary
from fluxemu.execution import CanonicalFluxState
from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis import (
    FBAResult,
    FVAResult,
    prepare_highs_flux_region,
    run_highs_fba,
    run_highs_vffva,
    run_prepared_highs_vffva,
)
from fluxemu.model import (
    CanonicalModel,
    StationaryExperimentSemantics,
    validate_canonical_model,
    validate_stationary_experiment,
)


@dataclass(frozen=True, slots=True)
class NativeStationaryAnalysisResult:
    """Results from one deterministic native flux-to-MID analysis.

    ``flux_state`` is built exclusively from ``fba.fluxes``.  The independent
    endpoint ranges in ``fva`` are diagnostic and are never treated as a
    jointly feasible flux vector.
    """

    fba: FBAResult
    fva: FVAResult
    flux_state: CanonicalFluxState
    mids: NativeStationaryResult


def run_native_fba(model: CanonicalModel) -> FBAResult:
    """Validate ``model`` and delegate its flux component to native HiGHS FBA."""

    validate_canonical_model(model)
    return run_highs_fba(model.flux_model)


def run_native_fva(
    model: CanonicalModel, fraction_of_optimum: float = 1.0, *, workers: int | None = None
) -> FVAResult:
    """Validate ``model`` and delegate to reusable, dynamically scheduled FVA."""

    validate_canonical_model(model)
    return run_highs_vffva(model.flux_model, fraction_of_optimum, workers=workers)


def _fba_flux_state(model: CanonicalModel, fba: FBAResult) -> CanonicalFluxState:
    """Copy an FBA primal into the canonical declared order without alteration."""

    reaction_order = tuple(reaction.reaction_id for reaction in model.flux_model.reactions)
    if tuple(fba.fluxes.index) != reaction_order:
        raise AnalysisError("native FBA result does not preserve canonical reaction order")
    return CanonicalFluxState(
        "fba-optimum",
        tuple((reaction_id, value) for reaction_id, value in fba.fluxes.items()),
    )


def run_native_stationary_analysis(
    model: CanonicalModel,
    experiment: StationaryExperimentSemantics,
    *,
    fva_fraction_of_optimum: float = 1.0,
) -> NativeStationaryAnalysisResult:
    """Run native HiGHS FBA/FVA and native stationary EMU over one model.

    The exact complete FBA primal is the only flux state supplied to EMU.  FVA
    is calculated and returned only to characterize independent reaction
    ranges; its endpoints do not participate in isotope propagation.
    """

    validate_stationary_experiment(model, experiment)
    prepared = prepare_highs_flux_region(model.flux_model, fva_fraction_of_optimum)
    fba = prepared.fba
    fva = run_prepared_highs_vffva(prepared)
    flux_state = _fba_flux_state(model, fba)
    plan = compile_emu_plan(model, experiment)
    mids = evaluate_stationary(plan, (flux_state,))
    return NativeStationaryAnalysisResult(fba, fva, flux_state, mids)


__all__ = [
    "NativeStationaryAnalysisResult",
    "run_native_fba",
    "run_native_fva",
    "run_native_stationary_analysis",
]
