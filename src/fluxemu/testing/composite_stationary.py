"""Stationary flux-family bridge for the finite composite testing core.

The bridge maps two explicitly supplied finite tuples of complete feasible flux
states through the existing stationary observation layer.  The resulting law
classes can be passed to the projected Rényi and exact finite minimax routines.

This is intentionally named and documented as a finite-family construction. A
sampled flux ensemble is not a certificate for the full continuous feasible
flux family.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import observation as _observation
from ..exceptions import FluxEMUError, InputValidationError
from ..execution import CanonicalFluxState
from ..observation import (
    StationaryObservationLawResult,
    StationaryObservationSpecification,
)
from ..observation.schema import _validate_state_record
from .composite import FiniteCompositeHypotheses, FiniteObservationLaw


def _declared_block_identities(
    specification: StationaryObservationSpecification,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (block.experiment_id, item.target_id, item.replicate_id)
        for block in specification.experiments
        for item in block.specifications
    )


def _validate_states(
    states: tuple[CanonicalFluxState, ...], role: str,
) -> None:
    if not isinstance(states, tuple) or not states:
        raise InputValidationError(
            f"{role}_states must be a nonempty immutable tuple of CanonicalFluxState records"
        )
    for state in states:
        _validate_state_record(state)


def _finite_laws_from_source(
    source: StationaryObservationLawResult,
    states: tuple[CanonicalFluxState, ...],
    identities: tuple[tuple[str, str, str], ...],
    *,
    role: str,
    independent_blocks: bool,
) -> tuple[FiniteObservationLaw, ...]:
    if not isinstance(source, StationaryObservationLawResult):
        raise InputValidationError(f"{role} source must be StationaryObservationLawResult")
    if source.states != states:
        raise InputValidationError(f"{role} source does not retain the declared state tuple")
    block_count = len(identities)
    expected_components = len(states) * block_count
    if len(source.components) != expected_components:
        raise InputValidationError(
            f"{role} source has {len(source.components)} components; expected {expected_components}"
        )

    laws = []
    for state_index, state in enumerate(states):
        start = state_index * block_count
        stop = start + block_count
        components = source.components[start:stop]
        actual_identities = tuple(
            (item.experiment_id, item.target_id, item.replicate_id)
            for item in components
        )
        if actual_identities != identities:
            raise InputValidationError(
                f"{role} observation block order differs from the declared specification"
            )
        if any(item.state != state for item in components):
            raise InputValidationError(
                f"{role} observation components do not retain source state {state_index}"
            )
        laws.append(FiniteObservationLaw(
            blocks=tuple(item.law for item in components),
            independent=independent_blocks,
            label=f"{role}[{state_index}]:{state.sample_id}",
        ))
    return tuple(laws)


@dataclass(frozen=True, slots=True, kw_only=True)
class StationaryFiniteCompositeResult:
    """Finite flux-state families and their aligned complete observation laws."""

    null_states: tuple[CanonicalFluxState, ...]
    alternative_states: tuple[CanonicalFluxState, ...]
    null_observation_laws: StationaryObservationLawResult
    alternative_observation_laws: StationaryObservationLawResult
    hypotheses: FiniteCompositeHypotheses
    block_identities: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        _validate_states(self.null_states, "null")
        _validate_states(self.alternative_states, "alternative")
        if not isinstance(self.hypotheses, FiniteCompositeHypotheses):
            raise InputValidationError("hypotheses must be FiniteCompositeHypotheses")
        if not isinstance(self.block_identities, tuple) or not self.block_identities:
            raise InputValidationError("block_identities must be a nonempty immutable tuple")
        if (
            self.null_observation_laws.states != self.null_states
            or self.alternative_observation_laws.states != self.alternative_states
        ):
            raise InputValidationError("stationary composite sources must retain their state tuples")
        if (
            self.null_observation_laws.model_fingerprint
            != self.alternative_observation_laws.model_fingerprint
            or self.null_observation_laws.specification_fingerprint
            != self.alternative_observation_laws.specification_fingerprint
            or self.null_observation_laws.experiment_fingerprints
            != self.alternative_observation_laws.experiment_fingerprints
        ):
            raise InputValidationError("null and alternative stationary observation provenance must align")
        if len(self.hypotheses.null) != len(self.null_states):
            raise InputValidationError("finite null law class does not align with null flux states")
        if len(self.hypotheses.alternative) != len(self.alternative_states):
            raise InputValidationError("finite alternative law class does not align with alternative flux states")

    @property
    def observation_specification_fingerprint(self) -> str:
        return self.null_observation_laws.specification_fingerprint


def evaluate_stationary_finite_composite_hypotheses(
    specification: StationaryObservationSpecification,
    *,
    null_states: tuple[CanonicalFluxState, ...],
    alternative_states: tuple[CanonicalFluxState, ...],
    independent_blocks: bool = False,
) -> StationaryFiniteCompositeResult:
    """Map two finite feasible flux-state families to finite observation classes.

    Each supplied state is independently validated by the existing stationary
    observation layer and mapped through EMU to genuine-count laws. Multiple
    experiment/target/replicate blocks require an explicit declaration of
    independence.

    The returned hypotheses are the finite laws induced by exactly the supplied
    states.  They can be passed directly to ``projected_renyi_test``,
    ``composite_renyi_converse_at_order`` and ``solve_finite_minimax_test``.
    No claim is made that the supplied states exhaust a continuous flux family.
    """
    if not isinstance(specification, StationaryObservationSpecification):
        raise InputValidationError("specification must be StationaryObservationSpecification")
    if type(independent_blocks) is not bool:
        raise InputValidationError("independent_blocks must be an explicit bool")
    _validate_states(null_states, "null")
    _validate_states(alternative_states, "alternative")

    identities = _declared_block_identities(specification)
    if not identities:
        raise InputValidationError("stationary observation specification has no declared count blocks")
    if len(identities) > 1 and not independent_blocks:
        raise InputValidationError(
            "multiple observation blocks require explicit independent_blocks=True"
        )
    if len(identities) == 1 and independent_blocks:
        raise InputValidationError(
            "a single observation block must use independent_blocks=False"
        )

    sources = []
    for role, states in (("null", null_states), ("alternative", alternative_states)):
        try:
            source = _observation.evaluate_stationary_observation_laws(
                specification, states,
            )
        except FluxEMUError as error:
            raise type(error)(f"{role} finite flux family: {error}") from error
        sources.append(source)
    null_source, alternative_source = sources

    null_laws = _finite_laws_from_source(
        null_source,
        null_states,
        identities,
        role="H0",
        independent_blocks=independent_blocks,
    )
    alternative_laws = _finite_laws_from_source(
        alternative_source,
        alternative_states,
        identities,
        role="H1",
        independent_blocks=independent_blocks,
    )
    hypotheses = FiniteCompositeHypotheses(
        null=null_laws,
        alternative=alternative_laws,
    )
    return StationaryFiniteCompositeResult(
        null_states=null_states,
        alternative_states=alternative_states,
        null_observation_laws=null_source,
        alternative_observation_laws=alternative_source,
        hypotheses=hypotheses,
        block_identities=identities,
    )


__all__ = [
    "StationaryFiniteCompositeResult",
    "evaluate_stationary_finite_composite_hypotheses",
]
