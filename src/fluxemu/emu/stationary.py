"""Flux-dependent numerical evaluation of a compiled stationary EMU plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from fluxemu.exceptions import ForwardEMUError, MappingError, ValidationError
from fluxemu.execution import (
    CanonicalFluxState,
    StationaryForwardResult,
    StationaryMID,
    StationaryMIDValue,
    _validate_states,
)

from .graph import CompiledEMUPlan, EMU, EMUContribution
from .projection import evaluate_flux_projection
from .tracers import convolve_mids, source_emu_mid


DEFAULT_NATIVE_MID_TOLERANCE = 1e-10
DEFAULT_RANK_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class LayerDiagnostics:
    sample_id: Any
    emu_size: int
    matrix_dimension: int
    rank: int
    condition_number: float
    max_absolute_residual: float
    minimum_component: float
    max_normalization_error: float


@dataclass(frozen=True, slots=True)
class NativeStationaryResult:
    forward: StationaryForwardResult
    layer_diagnostics: tuple[LayerDiagnostics, ...]


def _directed_flux(plan: CompiledEMUPlan, reaction_id: str, flux: dict[str, float]) -> float:
    reaction = next(item for item in plan.model.flux_model.reactions if item.reaction_id == reaction_id)
    value = flux[reaction_id]
    if float(reaction.lower_bound) >= 0.0:
        directed = value
    elif float(reaction.upper_bound) <= 0.0:
        directed = -value
    else:  # compilation normally rejects this first
        raise MappingError(f"reaction {reaction_id!r} has ambiguous physical direction")
    if directed < -1e-12:
        raise MappingError(f"reaction {reaction_id!r} has flux opposite its supported direction")
    return max(directed, 0.0)


def _contribution_rate(plan: CompiledEMUPlan, contribution: EMUContribution, flux: dict[str, float]) -> float:
    if contribution.flux_projection is not None:
        return evaluate_flux_projection(contribution.flux_projection, flux)
    return _directed_flux(plan, contribution.reaction_id, flux)


def _turnover(plan: CompiledEMUPlan, emu: EMU, flux: dict[str, float]) -> float:
    total = 0.0
    for reaction in plan.model.flux_model.reactions:
        coefficient = sum(
            float(term.coefficient)
            for term in reaction.stoichiometric_terms
            if term.metabolite_id == emu.metabolite_id
        )
        consumption = -coefficient * float(flux[reaction.reaction_id])
        if consumption > 1e-12:
            total += consumption
    return total


def _contribution_mid(contribution: EMUContribution, mids: dict[EMU, np.ndarray]) -> np.ndarray:
    missing = tuple(item for item in contribution.precursors if item not in mids)
    if missing:
        raise ForwardEMUError(
            f"precursor EMU(s) are unavailable for {contribution.reaction_id!r}: {missing!r}"
        )
    result = convolve_mids(tuple(mids[item] for item in contribution.precursors))
    if len(result) != contribution.product.size + 1:
        raise ForwardEMUError(
            f"condensation for {contribution.reaction_id!r} produced the wrong MID length"
        )
    return result


def _validate_mid(mid: np.ndarray, emu: EMU, tolerance: float) -> float:
    if mid.shape != (emu.size + 1,) or not np.all(np.isfinite(mid)):
        raise ValidationError(f"native MID for {emu!r} has invalid length or non-finite values")
    minimum = float(np.min(mid))
    if minimum < -tolerance:
        raise ValidationError(f"native MID for {emu!r} contains negative component {minimum}")
    error = abs(float(np.sum(mid)) - 1.0)
    if error > tolerance:
        raise ValidationError(f"native MID for {emu!r} normalization error {error} exceeds {tolerance}")
    return error


def _evaluate_state(
    plan: CompiledEMUPlan, state: CanonicalFluxState, tolerance: float
) -> tuple[tuple[StationaryMID, ...], tuple[LayerDiagnostics, ...], float]:
    flux = dict(state.values)
    isotope_counts = {item.metabolite_id: item.carbon_count for item in plan.model.isotope_model.metabolites}
    tracers = {item.metabolite_id: item for item in plan.experiment.tracers}
    mids: dict[EMU, np.ndarray] = {
        emu: source_emu_mid(tracers[emu.metabolite_id], emu, isotope_counts[emu.metabolite_id])
        for emu in plan.source_emus
    }
    by_product: dict[EMU, list[EMUContribution]] = {}
    for contribution in plan.contributions:
        by_product.setdefault(contribution.product, []).append(contribution)

    # A compiled plan is flux-independent and therefore contains every
    # admissible producer.  For a particular complete state, trace only
    # components whose projected rate is nonzero.  This prevents inactive
    # certified pathways from introducing mathematically undefined zero-flow
    # pools into an otherwise well-posed stationary system.
    active: set[EMU] = set(plan.source_emus)
    def activate(emu: EMU) -> None:
        if emu in active:
            return
        active.add(emu)
        for contribution in by_product.get(emu, ()):
            if _contribution_rate(plan, contribution, flux) * contribution.branch_weight > 1e-12:
                for precursor in contribution.precursors:
                    activate(precursor)
    for _, emu in plan.targets:
        activate(emu)
    for _, precursors in plan.observations:
        for emu in precursors:
            activate(emu)

    diagnostics: list[LayerDiagnostics] = []
    normalization_errors: list[float] = []
    for layer in plan.layers:
        unknowns = tuple(emu for emu in layer.unknowns if emu in active)
        index = {emu: position for position, emu in enumerate(unknowns)}
        dimension = len(unknowns)
        if dimension == 0:
            continue
        matrix = np.zeros((dimension, dimension), dtype=float)
        rhs = np.zeros((dimension, layer.size + 1), dtype=float)
        for row, emu in enumerate(unknowns):
            turnover = _turnover(plan, emu, flux)
            matrix[row, row] = turnover
            for contribution in by_product.get(emu, ()):
                effective = _contribution_rate(plan, contribution, flux) * contribution.branch_weight
                if effective <= 1e-12:
                    continue
                if (
                    len(contribution.precursors) == 1
                    and contribution.precursors[0].size == layer.size
                    and contribution.precursors[0] in index
                ):
                    matrix[row, index[contribution.precursors[0]]] -= effective
                else:
                    rhs[row] += effective * _contribution_mid(contribution, mids)
        rank = int(np.linalg.matrix_rank(matrix, tol=DEFAULT_RANK_TOLERANCE))
        condition = float(np.linalg.cond(matrix))
        if rank < dimension or not math.isfinite(condition):
            raise ForwardEMUError(
                f"stationary EMU layer {layer.size} is singular: dimension={dimension}, "
                f"rank={rank}, condition_number={condition}, unknowns={unknowns!r}"
            )
        solution = np.linalg.solve(matrix, rhs)
        residual = float(np.max(np.abs(matrix @ solution - rhs), initial=0.0))
        layer_errors = []
        for emu, vector in zip(unknowns, solution):
            error = _validate_mid(vector, emu, tolerance)
            layer_errors.append(error)
            normalization_errors.append(error)
            mids[emu] = vector
        diagnostics.append(
            LayerDiagnostics(
                state.sample_id, layer.size, dimension, rank, condition, residual,
                float(np.min(solution)), max(layer_errors, default=0.0),
            )
        )

    balanced = {
        item.metabolite_id for item in plan.model.flux_model.metabolites if item.steady_state_balanced
    }
    predictions: list[StationaryMID] = []
    for target_id, emu in plan.targets:
        if emu.metabolite_id in balanced:
            vector = mids[emu]
        else:
            numerator = np.zeros(emu.size + 1, dtype=float)
            denominator = 0.0
            for contribution in by_product.get(emu, ()):
                effective = _contribution_rate(plan, contribution, flux) * contribution.branch_weight
                numerator += effective * _contribution_mid(contribution, mids)
                denominator += effective
            if denominator <= 1e-15:
                raise ForwardEMUError(f"terminal target {target_id!r} has zero productive flux")
            vector = numerator / denominator
        error = _validate_mid(vector, emu, tolerance)
        normalization_errors.append(error)
        predictions.append(StationaryMID(state.sample_id, target_id, tuple(float(x) for x in vector)))
    for target_id, precursors in plan.observations:
        vector = convolve_mids(tuple(mids[item] for item in precursors))
        synthetic = EMU(target_id, tuple(range(1, sum(item.size for item in precursors) + 1)))
        error = _validate_mid(vector, synthetic, tolerance)
        normalization_errors.append(error)
        predictions.append(StationaryMID(state.sample_id, target_id, tuple(float(x) for x in vector)))
    return tuple(predictions), tuple(diagnostics), max(normalization_errors, default=0.0)


def evaluate_stationary(
    plan: CompiledEMUPlan,
    fluxes: Mapping[str, float] | Sequence[CanonicalFluxState],
    *,
    mid_tolerance: float = DEFAULT_NATIVE_MID_TOLERANCE,
) -> NativeStationaryResult:
    """Evaluate one compiled topology for an ordered batch of complete flux states."""

    tolerance = float(mid_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise MappingError("MID tolerance must be finite and nonnegative")
    states = _validate_states(plan.model, fluxes)
    predictions: list[StationaryMID] = []
    diagnostics: list[LayerDiagnostics] = []
    normalization = 0.0
    for state in states:
        try:
            state_predictions, state_diagnostics, state_normalization = _evaluate_state(
                plan, state, tolerance
            )
        except np.linalg.LinAlgError as error:
            raise ForwardEMUError(
                f"flux state {state.sample_id!r}: stationary EMU linear solve failed: "
                f"{error}"
            ) from error
        except (ForwardEMUError, ValidationError, MappingError) as error:
            exception_type = type(error)
            raise exception_type(f"flux state {state.sample_id!r}: {error}") from error
        predictions.extend(state_predictions)
        diagnostics.extend(state_diagnostics)
        normalization = max(normalization, state_normalization)
    values = tuple(
        StationaryMIDValue(item.sample_id, item.target_id, index, fraction)
        for item in predictions
        for index, fraction in enumerate(item.fractions)
    )
    forward = StationaryForwardResult(tuple(predictions), values, tolerance, normalization)
    return NativeStationaryResult(forward, tuple(diagnostics))


__all__ = [
    "DEFAULT_NATIVE_MID_TOLERANCE", "LayerDiagnostics", "NativeStationaryResult",
    "evaluate_stationary",
]
