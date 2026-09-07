"""Native fixed-flux transient EMU compilation and integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from fluxemu.exceptions import ForwardEMUError, MappingError
from fluxemu.execution import (
    CanonicalFluxState,
    TransientForwardResult,
    TransientMID,
    TransientMIDValue,
    _validate_states,
)
from fluxemu.model import (
    CanonicalModel,
    StationaryExperimentSemantics,
    TransientExperimentSemantics,
    transient_experiment_fingerprint,
    validate_transient_experiment,
)

from .graph import CompiledEMUPlan, EMU, EMUContribution, compile_emu_plan
from .stationary import _contribution_mid, _contribution_rate, _turnover, _validate_mid
from .tracers import source_emu_mid


DEFAULT_TRANSIENT_RTOL = 1e-9
DEFAULT_TRANSIENT_ATOL = 1e-12
DEFAULT_TRANSIENT_MID_TOLERANCE = 1e-8
DEFAULT_TRANSIENT_METHOD = "RK45"


@dataclass(frozen=True, slots=True)
class TransientStateBlock:
    """Location of one complete EMU MID in the global ODE vector."""

    emu: EMU
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class CompiledTransientEMUPlan:
    """Reusable transient topology and deterministic dynamic-state layout."""

    stationary_plan: CompiledEMUPlan
    experiment: TransientExperimentSemantics
    model_fingerprint: str
    experiment_fingerprint: str
    target_order: tuple[str, ...]
    emu_order: tuple[EMU, ...]
    dynamic_state_order: tuple[tuple[EMU, int], ...]
    state_blocks: tuple[TransientStateBlock, ...]
    time_point_order: tuple[float, ...]
    pool_quantity_order: tuple[tuple[str, float], ...]

    @property
    def model(self) -> CanonicalModel:
        return self.stationary_plan.model


@dataclass(frozen=True, slots=True)
class TransientDiagnostics:
    sample_id: Any
    solver_success: bool
    solver_method: str
    requested_time_count: int
    internal_time_count: int
    rhs_evaluation_count: int
    minimum_mid_component: float
    maximum_normalization_error: float
    maximum_absolute_mid_sum_derivative: float
    maximum_stationary_limit_discrepancy: float | None = None


@dataclass(frozen=True, slots=True)
class NativeTransientResult:
    forward: TransientForwardResult
    diagnostics: tuple[TransientDiagnostics, ...]


def compile_transient_emu_plan(
    model: CanonicalModel,
    experiment: TransientExperimentSemantics,
) -> CompiledTransientEMUPlan:
    """Compile explicit mappings once and add the V1 global transient layout."""

    validate_transient_experiment(model, experiment)
    stationary_experiment = StationaryExperimentSemantics(
        experiment.tracers, experiment.targets
    )
    stationary_plan = compile_emu_plan(model, stationary_experiment)
    unknowns = tuple(emu for layer in stationary_plan.layers for emu in layer.unknowns)
    required_pools = tuple(dict.fromkeys(emu.metabolite_id for emu in unknowns))
    supplied = {item.metabolite_id: float(item.quantity) for item in experiment.pool_quantities}
    missing = tuple(item for item in required_pools if item not in supplied)
    extras = tuple(item.metabolite_id for item in experiment.pool_quantities if item.metabolite_id not in required_pools)
    if missing:
        raise MappingError("missing pool quantity for dynamic metabolite(s): " + ", ".join(missing))
    if extras:
        raise MappingError(
            "pool quantities are permitted only for required balanced dynamic metabolites: "
            + ", ".join(extras)
        )

    blocks: list[TransientStateBlock] = []
    state_order: list[tuple[EMU, int]] = []
    offset = 0
    for emu in unknowns:
        stop = offset + emu.size + 1
        blocks.append(TransientStateBlock(emu, offset, stop))
        state_order.extend((emu, index) for index in range(emu.size + 1))
        offset = stop
    return CompiledTransientEMUPlan(
        stationary_plan=stationary_plan,
        experiment=experiment,
        model_fingerprint=stationary_plan.model_fingerprint,
        experiment_fingerprint=transient_experiment_fingerprint(model, experiment),
        target_order=tuple(item[0] for item in stationary_plan.targets),
        emu_order=stationary_plan.emus,
        dynamic_state_order=tuple(state_order),
        state_blocks=tuple(blocks),
        time_point_order=tuple(float(item) for item in experiment.time_points),
        pool_quantity_order=tuple((item, supplied[item]) for item in required_pools),
    )


def _validate_numerical_setting(value: float, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MappingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise MappingError(f"{name} must be finite and {qualifier}")
    return result


def _initial_state(plan: CompiledTransientEMUPlan) -> np.ndarray:
    state = np.zeros(len(plan.dynamic_state_order), dtype=float)
    for block in plan.state_blocks:
        state[block.start] = 1.0
    return state


def _evaluate_sample(
    plan: CompiledTransientEMUPlan,
    state: CanonicalFluxState,
    *,
    method: str,
    rtol: float,
    atol: float,
    mid_tolerance: float,
) -> tuple[tuple[TransientMID, ...], TransientDiagnostics]:
    stationary_plan = plan.stationary_plan
    flux = dict(state.values)
    pools = dict(plan.pool_quantity_order)
    isotope_counts = {
        item.metabolite_id: item.carbon_count
        for item in plan.model.isotope_model.metabolites
    }
    tracers = {item.metabolite_id: item for item in plan.experiment.tracers}
    source_mids = {
        emu: source_emu_mid(tracers[emu.metabolite_id], emu, isotope_counts[emu.metabolite_id])
        for emu in stationary_plan.source_emus
    }
    by_product: dict[EMU, list[EMUContribution]] = {}
    for contribution in stationary_plan.contributions:
        by_product.setdefault(contribution.product, []).append(contribution)

    def unpack(vector: np.ndarray) -> dict[EMU, np.ndarray]:
        mids = dict(source_mids)
        for block in plan.state_blocks:
            mids[block.emu] = vector[block.start:block.stop]
        return mids

    def rhs(_time: float, vector: np.ndarray) -> np.ndarray:
        mids = unpack(vector)
        derivative = np.empty_like(vector)
        for block in plan.state_blocks:
            emu = block.emu
            production = np.zeros(emu.size + 1, dtype=float)
            for contribution in by_product.get(emu, ()):
                effective = (
                    _contribution_rate(stationary_plan, contribution, flux)
                    * contribution.branch_weight
                )
                production += effective * _contribution_mid(contribution, mids)
            loss = _turnover(stationary_plan, emu, flux) * mids[emu]
            derivative[block.start:block.stop] = (production - loss) / pools[emu.metabolite_id]
        return derivative

    initial = _initial_state(plan)
    final_time = plan.time_point_order[-1]
    if final_time == 0.0 or initial.size == 0:
        requested_states = np.repeat(
            initial[:, np.newaxis], len(plan.time_point_order), axis=1
        )
        internal_time_count = 1
        nfev = 0
    else:
        try:
            from scipy.integrate import solve_ivp
        except ImportError as exc:  # pragma: no cover - exercised by isolation tests
            raise ForwardEMUError(
                "native transient execution requires the 'transient' SciPy extra"
            ) from exc

        solution = solve_ivp(
            rhs,
            (0.0, final_time),
            initial,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        if not solution.success or solution.sol is None:
            raise ForwardEMUError(
                f"transient solver failed for sample {state.sample_id!r}: {solution.message}"
            )
        requested_states = np.asarray(solution.sol(plan.time_point_order), dtype=float)
        internal_time_count = int(solution.t.size)
        nfev = int(solution.nfev)

    balanced = {
        item.metabolite_id
        for item in plan.model.flux_model.metabolites
        if item.steady_state_balanced
    }
    predictions: list[TransientMID] = []
    normalization_errors: list[float] = []
    minimum_component = math.inf
    max_sum_derivative = 0.0
    for column, time in enumerate(plan.time_point_order):
        mids = unpack(requested_states[:, column])
        for block in plan.state_blocks:
            error = _validate_mid(mids[block.emu], block.emu, mid_tolerance)
            normalization_errors.append(error)
            minimum_component = min(minimum_component, float(np.min(mids[block.emu])))
        derivative = rhs(time, requested_states[:, column])
        for block in plan.state_blocks:
            max_sum_derivative = max(
                max_sum_derivative,
                abs(float(np.sum(derivative[block.start:block.stop]))),
            )
        for target_id, emu in stationary_plan.targets:
            if emu.metabolite_id in balanced or emu in source_mids:
                vector = mids[emu]
            else:
                numerator = np.zeros(emu.size + 1, dtype=float)
                denominator = 0.0
                for contribution in by_product.get(emu, ()):
                    effective = (
                        _contribution_rate(stationary_plan, contribution, flux)
                        * contribution.branch_weight
                    )
                    numerator += effective * _contribution_mid(contribution, mids)
                    denominator += effective
                if denominator <= 1e-15:
                    raise ForwardEMUError(
                        f"terminal target {target_id!r} has zero productive flux"
                    )
                vector = numerator / denominator
            error = _validate_mid(vector, emu, mid_tolerance)
            normalization_errors.append(error)
            minimum_component = min(minimum_component, float(np.min(vector)))
            predictions.append(
                TransientMID(
                    state.sample_id,
                    time,
                    target_id,
                    tuple(float(item) for item in vector),
                )
            )
    maximum_normalization = max(normalization_errors, default=0.0)
    diagnostics = TransientDiagnostics(
        sample_id=state.sample_id,
        solver_success=True,
        solver_method=method,
        requested_time_count=len(plan.time_point_order),
        internal_time_count=internal_time_count,
        rhs_evaluation_count=nfev,
        minimum_mid_component=minimum_component,
        maximum_normalization_error=maximum_normalization,
        maximum_absolute_mid_sum_derivative=max_sum_derivative,
    )
    return tuple(predictions), diagnostics


def evaluate_transient(
    plan: CompiledTransientEMUPlan,
    fluxes: Mapping[str, float] | Sequence[CanonicalFluxState],
    *,
    method: str = DEFAULT_TRANSIENT_METHOD,
    rtol: float = DEFAULT_TRANSIENT_RTOL,
    atol: float = DEFAULT_TRANSIENT_ATOL,
    mid_tolerance: float = DEFAULT_TRANSIENT_MID_TOLERANCE,
) -> NativeTransientResult:
    """Integrate an ordered batch of complete fixed flux states from time zero."""

    if not isinstance(plan, CompiledTransientEMUPlan):
        raise MappingError("plan must be a CompiledTransientEMUPlan")
    if not isinstance(method, str) or not method:
        raise MappingError("transient solver method must be a nonempty string")
    relative = _validate_numerical_setting(rtol, "rtol", positive=True)
    absolute = _validate_numerical_setting(atol, "atol", positive=True)
    tolerance = _validate_numerical_setting(mid_tolerance, "MID tolerance", positive=False)
    states = _validate_states(plan.model, fluxes)
    predictions: list[TransientMID] = []
    diagnostics: list[TransientDiagnostics] = []
    maximum_normalization = 0.0
    for state in states:
        sample_predictions, sample_diagnostics = _evaluate_sample(
            plan,
            state,
            method=method,
            rtol=relative,
            atol=absolute,
            mid_tolerance=tolerance,
        )
        predictions.extend(sample_predictions)
        diagnostics.append(sample_diagnostics)
        maximum_normalization = max(
            maximum_normalization, sample_diagnostics.maximum_normalization_error
        )
    values = tuple(
        TransientMIDValue(
            item.sample_id,
            item.time,
            item.target_id,
            index,
            fraction,
        )
        for item in predictions
        for index, fraction in enumerate(item.fractions)
    )
    return NativeTransientResult(
        TransientForwardResult(
            tuple(predictions), values, tolerance, maximum_normalization
        ),
        tuple(diagnostics),
    )


__all__ = [
    "CompiledTransientEMUPlan",
    "DEFAULT_TRANSIENT_ATOL",
    "DEFAULT_TRANSIENT_METHOD",
    "DEFAULT_TRANSIENT_MID_TOLERANCE",
    "DEFAULT_TRANSIENT_RTOL",
    "NativeTransientResult",
    "TransientDiagnostics",
    "TransientStateBlock",
    "compile_transient_emu_plan",
    "evaluate_transient",
]
