"""Canonical stationary forward execution owned by FluxEMU."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from .exceptions import MappingError
from .model import (
    CanonicalModel,
    StationaryExperimentSemantics,
    validate_canonical_model,
    validate_stationary_experiment,
)


DEFAULT_MID_TOLERANCE = 1e-8
FLUX_BOUND_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class CanonicalFluxState:
    """One complete flux vector in canonical reaction order."""

    sample_id: Any
    values: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(
        cls, sample_id: Any, values: Mapping[str, float]
    ) -> "CanonicalFluxState":
        if not isinstance(values, Mapping):
            raise MappingError("canonical flux values must be a mapping")
        return cls(sample_id, tuple(values.items()))


@dataclass(frozen=True, slots=True)
class StationaryMID:
    """One target MID predicted for one canonical flux state."""

    sample_id: Any
    target_id: str
    fractions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StationaryMIDValue:
    """One long-form isotopologue prediction."""

    sample_id: Any
    target_id: str
    isotopologue_index: int
    predicted_fraction: float


@dataclass(frozen=True, slots=True)
class StationaryForwardResult:
    """Validated, ordered stationary predictions with no backend objects."""

    predictions: tuple[StationaryMID, ...]
    values: tuple[StationaryMIDValue, ...]
    mid_tolerance: float
    max_normalization_error: float


@dataclass(frozen=True, slots=True)
class TransientMID:
    """One target MID at one requested time for one fixed flux state."""

    sample_id: Any
    time: float
    target_id: str
    fractions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TransientMIDValue:
    """One long-form time-resolved isotopologue prediction."""

    sample_id: Any
    time: float
    target_id: str
    isotopologue_index: int
    predicted_fraction: float


@dataclass(frozen=True, slots=True)
class TransientForwardResult:
    """Validated ordered trajectories containing no numerical backend objects."""

    predictions: tuple[TransientMID, ...]
    values: tuple[TransientMIDValue, ...]
    mid_tolerance: float
    max_normalization_error: float


def _validate_states(
    model: CanonicalModel,
    fluxes: Mapping[str, float] | Sequence[CanonicalFluxState],
) -> tuple[CanonicalFluxState, ...]:
    if isinstance(fluxes, Mapping):
        states = (CanonicalFluxState.from_mapping("state-0", fluxes),)
    elif isinstance(fluxes, Sequence) and not isinstance(fluxes, (str, bytes)):
        states = tuple(fluxes)
        if not states or not all(isinstance(item, CanonicalFluxState) for item in states):
            raise MappingError("flux batch must contain CanonicalFluxState records")
    else:
        raise MappingError("fluxes must be a mapping or a sequence of CanonicalFluxState records")

    reaction_order = tuple(item.reaction_id for item in model.flux_model.reactions)
    required = set(reaction_order)
    sample_ids: list[Any] = []
    ordered_states: list[CanonicalFluxState] = []
    for state in states:
        sample_ids.append(state.sample_id)
        pairs = state.values
        if not isinstance(pairs, tuple):
            raise MappingError("CanonicalFluxState.values must be a tuple")
        ids = [item[0] for item in pairs if isinstance(item, tuple) and len(item) == 2]
        if len(ids) != len(pairs) or len(ids) != len(set(ids)):
            raise MappingError(f"flux state {state.sample_id!r} contains malformed or duplicate IDs")
        supplied = set(ids)
        missing = [item for item in reaction_order if item not in supplied]
        unknown = [item for item in ids if item not in required]
        if missing:
            raise MappingError(
                f"flux state {state.sample_id!r} is missing reaction ID(s): " + ", ".join(missing)
            )
        if unknown:
            raise MappingError(
                f"flux state {state.sample_id!r} contains unknown reaction ID(s): " + ", ".join(unknown)
            )
        raw = dict(pairs)
        ordered: list[tuple[str, float]] = []
        for reaction in model.flux_model.reactions:
            value = raw[reaction.reaction_id]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MappingError(f"flux for reaction {reaction.reaction_id!r} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise MappingError(f"flux for reaction {reaction.reaction_id!r} must be finite")
            if numeric < float(reaction.lower_bound) - FLUX_BOUND_TOLERANCE or numeric > float(
                reaction.upper_bound
            ) + FLUX_BOUND_TOLERANCE:
                raise MappingError(
                    f"flux {numeric} for reaction {reaction.reaction_id!r} is outside "
                    f"[{reaction.lower_bound}, {reaction.upper_bound}]"
                )
            ordered.append((reaction.reaction_id, numeric))
        ordered_states.append(CanonicalFluxState(state.sample_id, tuple(ordered)))
    if len(sample_ids) != len(set(sample_ids)):
        raise MappingError("canonical flux-state sample IDs must be unique")
    return tuple(ordered_states)


def run_stationary_forward(
    model: CanonicalModel,
    experiment: StationaryExperimentSemantics,
    fluxes: Mapping[str, float] | Sequence[CanonicalFluxState],
    *,
    mid_tolerance: float = DEFAULT_MID_TOLERANCE,
) -> StationaryForwardResult:
    """Execute stationary MIDs from canonical science and complete fluxes.

    Backend imports are deliberately local so importing this public contract,
    and especially :mod:`fluxemu.model`, does not import mfapy or SciPy.
    """

    validate_canonical_model(model)
    validate_stationary_experiment(model, experiment)
    if isinstance(mid_tolerance, bool) or not isinstance(mid_tolerance, (int, float)):
        raise MappingError("MID tolerance must be numeric")
    tolerance = float(mid_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise MappingError("MID tolerance must be finite and nonnegative")
    states = _validate_states(model, fluxes)

    from .backends.mfapy import execute_stationary

    return execute_stationary(model, experiment, states, tolerance)


__all__ = [
    "CanonicalFluxState",
    "DEFAULT_MID_TOLERANCE",
    "StationaryForwardResult",
    "StationaryMID",
    "StationaryMIDValue",
    "TransientForwardResult",
    "TransientMID",
    "TransientMIDValue",
    "run_stationary_forward",
]
