"""Native FBA/FVA to stationary-MID orchestration over one canonical model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

from fluxemu.emu import (
    CompiledEMUPlan,
    DEFAULT_NATIVE_MID_TOLERANCE,
    NativeStationaryResult,
    compile_emu_plan,
    evaluate_stationary,
)
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
    sample_prepared_flux_states,
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

    ``fva`` contains independent reaction-wise extrema and is never a flux
    state. ``flux_sampling`` contains the ordered complete, jointly feasible
    states supplied as one batch to stationary EMU. ``mid_ensemble`` contains
    predictions conditional on those exact states. The fingerprints bind the
    compiled isotope calculation to its canonical model and experiment.
    """

    fba: FBAResult
    fva: FVAResult
    flux_sampling: NativeFluxSamplingResult
    mid_ensemble: NativeStationaryResult
    model_fingerprint: str
    experiment_fingerprint: str


def _validate_ensemble_integer(
    value: object, name: str, *, allow_zero: bool
) -> int:
    minimum = 0 if allow_zero else 1
    description = "a nonnegative integer" if allow_zero else "a positive integer"
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise AnalysisError(f"{name} must be {description}")
    return int(value)


def _validate_ensemble_controls(
    sample_count: object,
    seed: object,
    burn_in: object,
    thinning: object,
    fva_workers: object,
    max_direction_attempts: object,
) -> tuple[int, int, int, int, int | None, int]:
    """Reject malformed controls before compiling or optimizing the flux LP."""

    count_value = _validate_ensemble_integer(
        sample_count, "sample_count", allow_zero=False
    )
    seed_value = _validate_ensemble_integer(seed, "seed", allow_zero=True)
    burn_in_value = _validate_ensemble_integer(
        burn_in, "burn_in", allow_zero=True
    )
    thinning_value = _validate_ensemble_integer(
        thinning, "thinning", allow_zero=False
    )
    attempts_value = _validate_ensemble_integer(
        max_direction_attempts,
        "max_direction_attempts",
        allow_zero=False,
    )
    workers_value = None
    if fva_workers is not None:
        workers_value = _validate_ensemble_integer(
            fva_workers, "fva_workers", allow_zero=False
        )
    return (
        count_value,
        seed_value,
        burn_in_value,
        thinning_value,
        workers_value,
        attempts_value,
    )


def _validate_mid_tolerance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError("mid_tolerance must be a finite nonnegative number")
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise AnalysisError("mid_tolerance must be a finite nonnegative number")
    return tolerance


def run_native_fba(model: CanonicalModel) -> FBAResult:
    """Validate ``model`` and delegate its flux component to native HiGHS FBA."""

    validate_canonical_model(model)
    return run_highs_fba(model.flux_model)


def run_native_fva(
    model: CanonicalModel, fraction_of_optimum: float = 1.0, *, workers: int | None = None
) -> FVAResult:
    """Validate ``model`` and delegate to reusable native FVA.

    Omitted or ``None`` ``workers`` selects one solver-owning thread. An
    explicit value greater than one enables dynamically scheduled shared-memory
    worker threads.
    """

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


def _validate_mid_ensemble_order(
    flux_sampling: NativeFluxSamplingResult,
    plan: CompiledEMUPlan,
    mid_ensemble: NativeStationaryResult,
) -> None:
    """Enforce the public sample-major, declared-target ordering contract."""

    target_order = tuple(target_id for target_id, _ in plan.targets) + tuple(
        target_id for target_id, _ in plan.observations
    )
    expected = tuple(
        (state.sample_id, target_id)
        for state in flux_sampling.states
        for target_id in target_order
    )
    actual = tuple(
        (prediction.sample_id, prediction.target_id)
        for prediction in mid_ensemble.forward.predictions
    )
    if actual != expected:
        raise AnalysisError(
            "native stationary MID ensemble does not preserve sampled-state and "
            "declared-target order"
        )


def run_native_stationary_ensemble(
    model: CanonicalModel,
    experiment: StationaryExperimentSemantics,
    *,
    sample_count: int,
    seed: int = 0,
    fraction_of_optimum: float = 1.0,
    burn_in: int = 100,
    thinning: int = 10,
    fva_workers: int | None = 1,
    max_direction_attempts: int = 100,
    mid_tolerance: float = DEFAULT_NATIVE_MID_TOLERANCE,
) -> NativeStationaryEnsembleResult:
    """Run the complete native Stage 1 sampled stationary pipeline.

    The canonical LP and biological optimum are prepared exactly once. Fast
    FVA and sampling consume that same retained-objective region, with the
    exact FVA result passed to the sampler. One compiled stationary EMU plan
    then evaluates the sampler's exact ordered tuple of complete
    :class:`CanonicalFluxState` records in a single batch.
    """

    validate_stationary_experiment(model, experiment)
    (
        count_value,
        seed_value,
        burn_in_value,
        thinning_value,
        workers_value,
        attempts_value,
    ) = _validate_ensemble_controls(
        sample_count,
        seed,
        burn_in,
        thinning,
        fva_workers,
        max_direction_attempts,
    )
    tolerance_value = _validate_mid_tolerance(mid_tolerance)
    prepared = prepare_highs_flux_region(model.flux_model, fraction_of_optimum)
    fva = run_prepared_highs_vffva(prepared, workers=workers_value)
    flux_sampling = sample_prepared_flux_states(
        prepared,
        count_value,
        seed=seed_value,
        fva=fva,
        burn_in=burn_in_value,
        thinning=thinning_value,
        max_direction_attempts=attempts_value,
    )
    plan = compile_emu_plan(model, experiment)
    mid_ensemble = evaluate_stationary(
        plan, flux_sampling.states, mid_tolerance=tolerance_value
    )
    _validate_mid_ensemble_order(flux_sampling, plan, mid_ensemble)
    return NativeStationaryEnsembleResult(
        fba=prepared.fba,
        fva=fva,
        flux_sampling=flux_sampling,
        mid_ensemble=mid_ensemble,
        model_fingerprint=plan.model_fingerprint,
        experiment_fingerprint=plan.experiment_fingerprint,
    )


# The shorter remote-compatible spelling is canonical. These explicit aliases
# preserve the equally clear ``*_analysis`` vocabulary without a second path.
NativeStationaryEnsembleAnalysisResult = NativeStationaryEnsembleResult
run_native_stationary_ensemble_analysis = run_native_stationary_ensemble


__all__ = [
    "NativeStationaryAnalysisResult",
    "NativeStationaryEnsembleAnalysisResult",
    "NativeStationaryEnsembleResult",
    "run_native_fba",
    "run_native_fva",
    "run_native_stationary_analysis",
    "run_native_stationary_ensemble",
    "run_native_stationary_ensemble_analysis",
]
