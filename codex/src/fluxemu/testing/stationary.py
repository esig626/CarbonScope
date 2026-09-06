"""Two fixed complete flux states joined to existing stationary count laws.

Each hypothesis is evaluated through ``fluxemu.observation`` with independent
original-model feasibility validation. The fixed roles preserve source state
identities, including identical states or distinct states sharing a sample ID.
No fitting or additional biological objective restriction occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import observation as _observation
from ..exceptions import FluxEMUError, InputValidationError
from ..execution import CanonicalFluxState
from ..observation import (
    StationaryObservationLawComponent,
    StationaryObservationLawResult,
    StationaryObservationSpecification,
)
from ..observation.schema import _validate_state_record
from .simple import SimpleBinaryLawPair, _testing_digest


def _identity(component: StationaryObservationLawComponent) -> tuple[str, str, str]:
    return component.experiment_id, component.target_id, component.replicate_id


@dataclass(frozen=True, slots=True, kw_only=True)
class SimpleBinaryFluxHypotheses:
    """H0 is ``null_state``; H1 is ``alternative_state``, with no estimation.

    This immutable record validates native state structure. Original-model
    completeness, scientific reaction order, and feasibility are validated by
    :func:`evaluate_stationary_simple_hypotheses` through the observation layer.
    """

    null_state: CanonicalFluxState
    alternative_state: CanonicalFluxState

    def __post_init__(self) -> None:
        _validate_state_record(self.null_state)
        _validate_state_record(self.alternative_state)

    @property
    def null_state_fingerprint(self) -> str:
        return _testing_digest(("simple-binary-flux-state-v1", self.null_state))

    @property
    def alternative_state_fingerprint(self) -> str:
        return _testing_digest(("simple-binary-flux-state-v1", self.alternative_state))

    @property
    def fingerprint(self) -> str:
        return _testing_digest((
            "simple-binary-flux-hypotheses-v1",
            ("H0=P0", self.null_state_fingerprint),
            ("H1=P1", self.alternative_state_fingerprint),
        ))


def _validate_source(
    source: StationaryObservationLawResult, state: CanonicalFluxState, role: str,
) -> None:
    if not isinstance(source, StationaryObservationLawResult):
        raise InputValidationError(f"{role} source must be StationaryObservationLawResult")
    if source.states != (state,):
        raise InputValidationError(f"{role} observation source must retain exactly its flux state")
    experiments = dict(source.experiment_fingerprints)
    if len(experiments) != len(source.experiment_fingerprints):
        raise InputValidationError(f"{role} source contains duplicate experiment fingerprints")
    for component in source.components:
        if component.state != state:
            raise InputValidationError(f"{role} observation component has a different flux state")
        if (
            component.model_fingerprint != source.model_fingerprint
            or component.specification_fingerprint != source.specification_fingerprint
            or component.experiment_fingerprint != experiments.get(component.experiment_id)
        ):
            raise InputValidationError(f"{role} observation component provenance differs from its source")


@dataclass(frozen=True, slots=True, kw_only=True)
class StationarySimpleTestingResult:
    """Aligned H0/P0 and H1/P1 laws with both native feasibility reports.

    Source components retain state, experiment, target, replicate, mass-class,
    genuine-count, and model provenance. Support mismatch remains observable
    here; Bruno certification subsequently enforces its theorem assumption.
    """

    hypotheses: SimpleBinaryFluxHypotheses
    null_observation_laws: StationaryObservationLawResult
    alternative_observation_laws: StationaryObservationLawResult
    law_pair: SimpleBinaryLawPair

    def __post_init__(self) -> None:
        if not isinstance(self.hypotheses, SimpleBinaryFluxHypotheses):
            raise InputValidationError("hypotheses must be SimpleBinaryFluxHypotheses")
        if not isinstance(self.law_pair, SimpleBinaryLawPair):
            raise InputValidationError("law_pair must be SimpleBinaryLawPair")
        null, alternative = self.null_observation_laws, self.alternative_observation_laws
        _validate_source(null, self.hypotheses.null_state, "null")
        _validate_source(alternative, self.hypotheses.alternative_state, "alternative")
        if (
            null.model_fingerprint != alternative.model_fingerprint
            or null.specification_fingerprint != alternative.specification_fingerprint
            or null.experiment_fingerprints != alternative.experiment_fingerprints
        ):
            raise InputValidationError("null and alternative observation provenance must align")
        null_identities = tuple(_identity(item) for item in null.components)
        alternative_identities = tuple(_identity(item) for item in alternative.components)
        if null_identities != alternative_identities or null_identities != self.law_pair.block_identities:
            raise InputValidationError("null and alternative experiment/target/replicate block order must align")
        if (
            tuple(item.law for item in null.components) != self.law_pair.null_laws
            or tuple(item.law for item in alternative.components) != self.law_pair.alternative_laws
        ):
            raise InputValidationError("law_pair must retain the exact null and alternative source laws")
        if (
            self.law_pair.null_state_fingerprint != self.hypotheses.null_state_fingerprint
            or self.law_pair.alternative_state_fingerprint != self.hypotheses.alternative_state_fingerprint
            or self.law_pair.observation_specification_fingerprint != null.specification_fingerprint
        ):
            raise InputValidationError("law_pair must bind both flux states and the observation specification")

    @property
    def null_components(self) -> tuple[StationaryObservationLawComponent, ...]:
        return self.null_observation_laws.components

    @property
    def alternative_components(self) -> tuple[StationaryObservationLawComponent, ...]:
        return self.alternative_observation_laws.components

    @property
    def null_predicted_mids(self) -> tuple[tuple[float, ...], ...]:
        return tuple(item.predicted_mid for item in self.null_components)

    @property
    def alternative_predicted_mids(self) -> tuple[tuple[float, ...], ...]:
        return tuple(item.predicted_mid for item in self.alternative_components)

    @property
    def fingerprint(self) -> str:
        return _testing_digest((
            "stationary-simple-testing-result-v1", self.hypotheses.fingerprint,
            self.law_pair.fingerprint,
            tuple(item.fingerprint for item in self.null_components),
            tuple(item.fingerprint for item in self.alternative_components),
        ))


def evaluate_stationary_simple_hypotheses(
    specification: StationaryObservationSpecification,
    *,
    null_state: CanonicalFluxState,
    alternative_state: CanonicalFluxState,
    independent_blocks: bool = False,
) -> StationarySimpleTestingResult:
    """Map two fixed feasible states to aligned genuine-count observation laws.

    Multiple experiment/target/replicate blocks require the caller's explicit
    ``independent_blocks=True`` declaration. Each block retains its own count
    total. A single block needs no independence declaration.

    The existing observation bridge evaluates each fixed role separately,
    validating complete native states against the original model without an
    FBA-optimality restriction. Separate calls preserve sample IDs even when
    the roles use the same state or distinct states with a shared source ID.
    The returned object can be passed directly to ``bruno_converse_at_order``.
    """
    if type(independent_blocks) is not bool:
        raise InputValidationError("independent_blocks must be an explicit bool")
    hypotheses = SimpleBinaryFluxHypotheses(
        null_state=null_state, alternative_state=alternative_state,
    )
    specification_fingerprint = _observation.stationary_observation_specification_fingerprint(specification)
    declarations = tuple(
        ((block.experiment_id, item.target_id, item.replicate_id), item.total_count)
        for block in specification.experiments for item in block.specifications
    )
    if len(declarations) > 1 and not independent_blocks:
        raise InputValidationError("multiple observation blocks require explicit independent_blocks=True")

    sources = []
    for role, state in (("null", null_state), ("alternative", alternative_state)):
        try:
            source = _observation.evaluate_stationary_observation_laws(specification, (state,))
        except FluxEMUError as error:
            raise type(error)(f"{role} flux hypothesis: {error}") from error
        actual = tuple((_identity(item), item.law.n) for item in source.components)
        if actual != declarations:
            raise InputValidationError(f"{role} observation laws differ from declared block identities or count totals")
        sources.append(source)
    null_source, alternative_source = sources
    null_laws = tuple(item.law for item in null_source.components)
    alternative_laws = tuple(item.law for item in alternative_source.components)
    product = len(declarations) > 1
    law_pair = SimpleBinaryLawPair(
        null=null_laws if product else null_laws[0],
        alternative=alternative_laws if product else alternative_laws[0],
        independent=product,
        block_identities=tuple(identity for identity, _ in declarations),
        null_state_fingerprint=hypotheses.null_state_fingerprint,
        alternative_state_fingerprint=hypotheses.alternative_state_fingerprint,
        observation_specification_fingerprint=specification_fingerprint,
    )
    return StationarySimpleTestingResult(
        hypotheses=hypotheses, null_observation_laws=null_source,
        alternative_observation_laws=alternative_source, law_pair=law_pair,
    )


__all__ = [
    "SimpleBinaryFluxHypotheses", "StationarySimpleTestingResult",
    "evaluate_stationary_simple_hypotheses",
]
