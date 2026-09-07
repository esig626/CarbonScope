"""Finite composite flux hypotheses joined to joint stationary count laws."""

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
from .composite import (
    CompositeBinaryTestingProblem,
    CompositeMIDLawFamily,
    IndependentMIDProductLaw,
)
from .simple import _testing_digest


def _identity(component: StationaryObservationLawComponent) -> tuple[str, str, str]:
    return component.experiment_id, component.target_id, component.replicate_id


def _state_ids(role: str, states: tuple[CanonicalFluxState, ...]) -> tuple[str, ...]:
    return tuple(
        f"{role}-{index}:{_testing_digest(('composite-flux-state-v2', state))}"
        for index, state in enumerate(states)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeFluxHypotheses:
    """Explicit finite H0/H1 sets of complete canonical flux states.

    Member IDs are provenance only. Composite decision rules consume the
    generated observable laws and never state IDs, flux coordinates or sampling
    frequencies.
    """

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
                "composite-flux-hypotheses-v2",
                ("H0", self.null_member_ids),
                ("H1", self.alternative_member_ids),
            )
        )


def _product_members(
    source: StationaryObservationLawResult,
    states: tuple[CanonicalFluxState, ...],
    declarations: tuple[tuple[tuple[str, str, str], int], ...],
    role: str,
) -> tuple[IndependentMIDProductLaw, ...]:
    block_count = len(declarations)
    expected_components = len(states) * block_count
    if len(source.components) != expected_components:
        raise InputValidationError(
            f"{role} composite observation source contains {len(source.components)} blocks; "
            f"expected {expected_components} from {len(states)} states x {block_count} declarations"
        )
    identities = tuple(identity for identity, _ in declarations)
    products: list[IndependentMIDProductLaw] = []
    for state_index, state in enumerate(states):
        start = state_index * block_count
        chunk = source.components[start : start + block_count]
        if len(chunk) != block_count:
            raise InputValidationError(f"{role} composite block grouping is incomplete")
        actual = tuple((_identity(item), item.law.n) for item in chunk)
        if actual != declarations:
            raise InputValidationError(
                f"{role} composite observation block identity/total order differs from the declaration"
            )
        if any(item.state != state for item in chunk):
            raise InputValidationError(
                f"{role} composite observation source changed scientific state order"
            )
        products.append(
            IndependentMIDProductLaw(
                blocks=tuple(item.law for item in chunk),
                block_identities=identities,
            )
        )
    return tuple(products)


@dataclass(frozen=True, slots=True, kw_only=True)
class StationaryCompositeTestingResult:
    """Finite flux families and aligned complete product observation-law classes."""

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
            raise InputValidationError(
                "composite null and alternative observation provenance must align"
            )
        if self.problem.null.member_ids != self.hypotheses.null_member_ids:
            raise InputValidationError(
                "composite null member provenance is not bound to the null flux family"
            )
        if self.problem.alternative.member_ids != self.hypotheses.alternative_member_ids:
            raise InputValidationError(
                "composite alternative member provenance is not bound to the alternative flux family"
            )

    @property
    def fingerprint(self) -> str:
        return _testing_digest(
            (
                "stationary-composite-testing-result-v2",
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
    independent_blocks: bool = False,
) -> StationaryCompositeTestingResult:
    """Map finite feasible flux-state classes to joint product count laws.

    Multiple experiment/target/replicate blocks require explicit
    ``independent_blocks=True``. The declaration is statistical semantics, not
    inferred from the appearance of the data. Every complete flux state is
    validated by the existing observation/EMU chain before its blocks are
    grouped into one complete observable-law member.

    The returned finite family is exactly the represented class. State IDs,
    sampling frequencies and flux coordinates are not inputs to any decision
    rule, and no finite family is treated as a prior or an implicit convex hull.
    """

    if type(independent_blocks) is not bool:
        raise InputValidationError("independent_blocks must be an explicit bool")
    hypotheses = CompositeFluxHypotheses(
        null_states=null_states, alternative_states=alternative_states
    )
    declarations = tuple(
        ((block.experiment_id, item.target_id, item.replicate_id), item.total_count)
        for block in specification.experiments
        for item in block.specifications
    )
    if not declarations:
        raise InputValidationError(
            "stationary composite testing requires at least one genuine-count block"
        )
    if len(declarations) > 1 and not independent_blocks:
        raise InputValidationError(
            "multiple composite observation blocks require explicit independent_blocks=True"
        )

    sources: list[StationaryObservationLawResult] = []
    product_families: list[tuple[IndependentMIDProductLaw, ...]] = []
    for role, states in (
        ("null", hypotheses.null_states),
        ("alternative", hypotheses.alternative_states),
    ):
        try:
            source = _observation.evaluate_stationary_observation_laws(
                specification, states
            )
        except FluxEMUError as error:
            raise type(error)(f"{role} composite flux hypothesis: {error}") from error
        products = _product_members(source, states, declarations, role)
        sources.append(source)
        product_families.append(products)

    null_source, alternative_source = sources
    null_products, alternative_products = product_families
    problem = CompositeBinaryTestingProblem(
        null=CompositeMIDLawFamily(
            members=null_products,
            member_ids=hypotheses.null_member_ids,
        ),
        alternative=CompositeMIDLawFamily(
            members=alternative_products,
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
