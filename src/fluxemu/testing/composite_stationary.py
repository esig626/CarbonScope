"""Finite composite flux hypotheses joined to stationary genuine-count laws."""

from __future__ import annotations

from dataclasses import dataclass

from .. import observation as _observation
from ..exceptions import FluxEMUError, InputValidationError
from ..execution import CanonicalFluxState
from ..observation import StationaryObservationLawResult, StationaryObservationSpecification
from ..observation.schema import _validate_state_record
from .composite import CompositeBinaryTestingProblem, CompositeMIDLawFamily
from .simple import _testing_digest


def _state_ids(role: str, states: tuple[CanonicalFluxState, ...]) -> tuple[str, ...]:
    return tuple(
        f"{role}-{index}:{_testing_digest(('composite-flux-state-v1', state))}"
        for index, state in enumerate(states)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeFluxHypotheses:
    """Explicit finite H0/H1 sets of complete canonical flux states."""

    null_states: tuple[CanonicalFluxState, ...]
    alternative_states: tuple[CanonicalFluxState, ...]

    def __post_init__(self) -> None:
        for name, states in (
            ("null_states", self.null_states),
            ("alternative_states", self.alternative_states),
        ):
            if not isinstance(states, tuple) or not states:
                raise InputValidationError(f"{name} must be a nonempty immutable tuple")
            for state in states:
                _validate_state_record(state)
            sample_ids = tuple(state.sample_id for state in states)
            if len(set(sample_ids)) != len(sample_ids):
                raise InputValidationError(f"{name} sample IDs must be unique within the family")

    @property
    def null_member_ids(self) -> tuple[str, ...]:
        return _state_ids("H0", self.null_states)

    @property
    def alternative_member_ids(self) -> tuple[str, ...]:
        return _state_ids("H1", self.alternative_states)

    @property
    def fingerprint(self) -> str:
        return _testing_digest(
            (
                "composite-flux-hypotheses-v1",
                ("H0", self.null_member_ids),
                ("H1", self.alternative_member_ids),
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StationaryCompositeTestingResult:
    """Finite flux families and their aligned stationary count-law classes."""

    hypotheses: CompositeFluxHypotheses
    null_observation_laws: StationaryObservationLawResult
    alternative_observation_laws: StationaryObservationLawResult
    problem: CompositeBinaryTestingProblem

    def __post_init__(self) -> None:
        if not isinstance(self.hypotheses, CompositeFluxHypotheses):
            raise InputValidationError("hypotheses must be CompositeFluxHypotheses")
        if not isinstance(self.problem, CompositeBinaryTestingProblem):
            raise InputValidationError("problem must be CompositeBinaryTestingProblem")
        null = self.null_observation_laws
        alternative = self.alternative_observation_laws
        if null.states != self.hypotheses.null_states:
            raise InputValidationError("null observation source does not retain the null state family")
        if alternative.states != self.hypotheses.alternative_states:
            raise InputValidationError(
                "alternative observation source does not retain the alternative state family"
            )
        if (
            null.model_fingerprint != alternative.model_fingerprint
            or null.specification_fingerprint != alternative.specification_fingerprint
            or null.experiment_fingerprints != alternative.experiment_fingerprints
        ):
            raise InputValidationError("composite null and alternative observation provenance must align")
        if tuple(item.law for item in null.components) != self.problem.null.members:
            raise InputValidationError("composite null family does not retain source laws in state order")
        if tuple(item.law for item in alternative.components) != self.problem.alternative.members:
            raise InputValidationError(
                "composite alternative family does not retain source laws in state order"
            )
        if self.problem.null.member_ids != self.hypotheses.null_member_ids:
            raise InputValidationError("composite null member provenance is not bound to flux states")
        if self.problem.alternative.member_ids != self.hypotheses.alternative_member_ids:
            raise InputValidationError(
                "composite alternative member provenance is not bound to flux states"
            )

    @property
    def fingerprint(self) -> str:
        return _testing_digest(
            (
                "stationary-composite-testing-result-v1",
                self.hypotheses.fingerprint,
                self.problem.fingerprint,
                self.null_observation_laws.specification_fingerprint,
            )
        )


def evaluate_stationary_composite_hypotheses(
    specification: StationaryObservationSpecification,
    *,
    null_states: tuple[CanonicalFluxState, ...],
    alternative_states: tuple[CanonicalFluxState, ...],
) -> StationaryCompositeTestingResult:
    """Map two finite feasible flux-state classes to one-block multinomial laws.

    The composite theory implemented here is the common-i.i.d.-categorical
    finite-class problem. The observation specification must therefore declare
    exactly one experiment/target/replicate count block with one genuine total.
    No product-block composite theorem is inferred from the simple-testing API.
    """

    hypotheses = CompositeFluxHypotheses(
        null_states=null_states, alternative_states=alternative_states
    )
    declarations = tuple(
        (block.experiment_id, item.target_id, item.replicate_id, item.total_count)
        for block in specification.experiments
        for item in block.specifications
    )
    if len(declarations) != 1:
        raise InputValidationError(
            "stationary composite testing currently requires exactly one "
            "experiment/target/replicate genuine-count block"
        )

    sources = []
    for role, states in (
        ("null", hypotheses.null_states),
        ("alternative", hypotheses.alternative_states),
    ):
        try:
            source = _observation.evaluate_stationary_observation_laws(specification, states)
        except FluxEMUError as error:
            raise type(error)(f"{role} composite flux hypothesis: {error}") from error
        if len(source.components) != len(states):
            raise InputValidationError(
                f"{role} composite observation source does not contain one law per state"
            )
        for state, component in zip(states, source.components, strict=True):
            if component.state != state:
                raise InputValidationError(
                    f"{role} composite observation source changed scientific state order"
                )
        sources.append(source)

    null_source, alternative_source = sources
    problem = CompositeBinaryTestingProblem(
        null=CompositeMIDLawFamily(
            members=tuple(item.law for item in null_source.components),
            member_ids=hypotheses.null_member_ids,
        ),
        alternative=CompositeMIDLawFamily(
            members=tuple(item.law for item in alternative_source.components),
            member_ids=hypotheses.alternative_member_ids,
        ),
    )
    return StationaryCompositeTestingResult(
        hypotheses=hypotheses,
        null_observation_laws=null_source,
        alternative_observation_laws=alternative_source,
        problem=problem,
    )


__all__ = [
    "CompositeFluxHypotheses",
    "StationaryCompositeTestingResult",
    "evaluate_stationary_composite_hypotheses",
]
