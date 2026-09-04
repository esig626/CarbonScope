"""Native FBA/FVA to stationary-MID orchestration over one canonical model."""

from __future__ import annotations

from dataclasses import dataclass

from fluxemu.emu import NativeStationaryResult, compile_emu_plan, evaluate_stationary
from fluxemu.execution import CanonicalFluxState
from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis import (
    FBAResult,
    FVAResult,
    NativeFluxSamplingResult,
    prepare_highs_flux_region,
    run_highs_fba,
    run_highs_vffva,
    run_prepared_highs_vffva,
    sample_prepared_highs_flux_states,
)
from fluxemu.flux_analysis.sampling import (
    DEFAULT_BURN_IN,
    DEFAULT_MAX_DIRECTION_ATTEMPTS,
    DEFAULT_THINNING,
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


@dataclass(frozen=True, slots=True)
class NativeStationaryEnsembleResult:
    """Distinct products of one native sampled Stage 1 analysis.

    ``fva`` contains independent reaction-wise extrema. ``flux_sampling``
    contains the complete jointly feasible states supplied, as one ordered
    batch, to the stationary EMU evaluator. ``mid_ensemble`` contains the MID
    predictions conditional on those exact states.
    """

    fba: FBAResult
    fva: FVAResult
    flux_sampling: NativeFluxSamplingResult
    mid_ensemble: NativeStationaryResult


def _validate_ensemble_integer(
    value: object, name: str, *, allow_zero: bool
) -> None:
    minimum = 0 if allow_zero else 1
    description = "a nonnegative integer" if allow_zero else "a positive integer"
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisError(f"{name} must be {description}")


def _validate_ensemble_controls(
    sample_count: object,
    seed: object,
    burn_in: object,
    thinning: object,
    fva_workers: object,
    max_direction_attempts: object,
) -> None:
    """Reject malformed controls before compiling or optimizing the flux LP."""

    _validate_ensemble_integer(sample_count, "sample_count", allow_zero=False)
    _validate_ensemble_integer(seed, "seed", allow_zero=True)
    _validate_ensemble_integer(burn_in, "burn_in", allow_zero=True)
    _validate_ensemble_integer(thinning, "thinning", allow_zero=False)
    _validate_ensemble_integer(
        max_direction_attempts, "max_direction_attempts", allow_zero=False
    )
    if fva_workers is not None:
        _validate_ensemble_integer(fva_workers, "fva_workers", allow_zero=False)


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


def run_native_stationary_ensemble(
    model: CanonicalModel,
    experiment: StationaryExperimentSemantics,
    *,
    sample_count: int,
    seed: int = 0,
    fraction_of_optimum: float = 1.0,
    burn_in: int = DEFAULT_BURN_IN,
    thinning: int = DEFAULT_THINNING,
    fva_workers: int | None = 1,
    max_direction_attempts: int = DEFAULT_MAX_DIRECTION_ATTEMPTS,
) -> NativeStationaryEnsembleResult:
    """Run the complete native Stage 1 sampled stationary pipeline.

    The canonical LP and biological optimum are prepared exactly once. Fast
    FVA and sampling consume that same retained-objective region, and one
    compiled stationary EMU plan evaluates the sampler's exact ordered tuple
    of complete :class:`CanonicalFluxState` records in a single batch.
    """

    validate_stationary_experiment(model, experiment)
    _validate_ensemble_controls(
        sample_count,
        seed,
        burn_in,
        thinning,
        fva_workers,
        max_direction_attempts,
    )
    prepared = prepare_highs_flux_region(model.flux_model, fraction_of_optimum)
    fva = run_prepared_highs_vffva(prepared, workers=fva_workers)
    flux_sampling = sample_prepared_highs_flux_states(
        prepared,
        fva,
        sample_count,
        seed=seed,
        burn_in=burn_in,
        thinning=thinning,
        max_direction_attempts=max_direction_attempts,
    )
    plan = compile_emu_plan(model, experiment)
    mid_ensemble = evaluate_stationary(plan, flux_sampling.states)
    return NativeStationaryEnsembleResult(
        fba=prepared.fba,
        fva=fva,
        flux_sampling=flux_sampling,
        mid_ensemble=mid_ensemble,
    )


__all__ = [
    "NativeStationaryAnalysisResult",
    "NativeStationaryEnsembleResult",
    "run_native_fba",
    "run_native_fva",
    "run_native_stationary_analysis",
    "run_native_stationary_ensemble",
]
