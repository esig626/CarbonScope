"""Finite flux hypotheses joined to continuous stationary Dirichlet laws."""

from __future__ import annotations

from dataclasses import dataclass

from .. import observation as _observation
from ..exceptions import InputValidationError
from ..execution import CanonicalFluxState
from ..observation import (
    StationaryDirichletObservationLawResult,
    StationaryDirichletObservationSpecification,
)
from .composite_stationary import CompositeFluxHypotheses
from .dirichlet_composite import (
    DirichletCompositeBinaryTestingProblem,
    DirichletCompositeMIDLawFamily,
    IndependentDirichletMIDProductLaw,
)
from .simple import _testing_digest


def _identity(component) -> tuple[str, str, str]:
    return component.experiment_id, component.target_id, component.replicate_id


def _product_members(
    source: StationaryDirichletObservationLawResult,
    states: tuple[CanonicalFluxState, ...],
    identities: tuple[tuple[str, str, str], ...],
    role: str,
) -> tuple[IndependentDirichletMIDProductLaw, ...]:
    block_count = len(identities)
    if len(source.components) != len(states) * block_count:
        raise InputValidationError(
            f"{role} Dirichlet source has an inconsistent state/block component count"
        )
    products = []
    for state_index, state in enumerate(states):
        chunk = source.components[
            state_index * block_count : (state_index + 1) * block_count
        ]
        if tuple(_identity(item) for item in chunk) != identities:
            raise InputValidationError(
                f"{role} Dirichlet source changed observation block order"
            )
        if any(item.state != state for item in chunk):
            raise InputValidationError(
                f"{role} Dirichlet source changed scientific state order"
            )
        products.append(IndependentDirichletMIDProductLaw(
            blocks=tuple(item.law for item in chunk),
        ))
    return tuple(products)


def _validate_source(
    source: StationaryDirichletObservationLawResult,
    states: tuple[CanonicalFluxState, ...],
    family: DirichletCompositeMIDLawFamily,
    role: str,
) -> None:
    if not isinstance(source, StationaryDirichletObservationLawResult):
        raise InputValidationError(
            f"{role} source must be StationaryDirichletObservationLawResult"
        )
    if source.states != states:
        raise InputValidationError(f"{role} Dirichlet source changed state family")
    experiments = dict(source.experiment_fingerprints)
    if len(experiments) != len(source.experiment_fingerprints):
        raise InputValidationError(f"{role} source has duplicate experiment fingerprints")
    for component in source.components:
        if (
            component.model_fingerprint != source.model_fingerprint
            or component.specification_fingerprint != source.specification_fingerprint
            or component.experiment_fingerprint != experiments.get(component.experiment_id)
        ):
            raise InputValidationError(
                f"{role} Dirichlet component provenance differs from its source"
            )
    identities = family.members[0].block_identities
    if _product_members(source, states, identities, role) != family.members:
        raise InputValidationError(
            f"{role} Dirichlet family must retain the exact ordered source laws"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StationaryDirichletCompositeTestingResult:
    hypotheses: CompositeFluxHypotheses
    null_observation_laws: StationaryDirichletObservationLawResult
    alternative_observation_laws: StationaryDirichletObservationLawResult
    problem: DirichletCompositeBinaryTestingProblem

    def __post_init__(self) -> None:
        if not isinstance(self.hypotheses, CompositeFluxHypotheses):
            raise InputValidationError("hypotheses must be CompositeFluxHypotheses")
        if not isinstance(self.problem, DirichletCompositeBinaryTestingProblem):
            raise InputValidationError(
                "problem must be DirichletCompositeBinaryTestingProblem"
            )
        null, alternative = self.null_observation_laws, self.alternative_observation_laws
        _validate_source(null, self.hypotheses.null_states, self.problem.null, "null")
        _validate_source(
            alternative,
            self.hypotheses.alternative_states,
            self.problem.alternative,
            "alternative",
        )
        if (
            null.model_fingerprint != alternative.model_fingerprint
            or null.specification_fingerprint != alternative.specification_fingerprint
            or null.experiment_fingerprints != alternative.experiment_fingerprints
            or null.common_active_supports != alternative.common_active_supports
        ):
            raise InputValidationError(
                "Dirichlet null/alternative observation provenance and active faces must align"
            )
        if self.problem.null.member_ids != self.hypotheses.null_member_ids:
            raise InputValidationError("Dirichlet null member IDs are not bound to H0 states")
        if self.problem.alternative.member_ids != self.hypotheses.alternative_member_ids:
            raise InputValidationError("Dirichlet alternative member IDs are not bound to H1 states")

    @property
    def fingerprint(self) -> str:
        return _testing_digest((
            "stationary-dirichlet-composite-testing-result-v1",
            self.hypotheses.fingerprint,
            self.problem.fingerprint,
            tuple(
                (
                    source.model_fingerprint,
                    source.experiment_fingerprints,
                    source.specification_fingerprint,
                    source.common_active_supports,
                    tuple(component.fingerprint for component in source.components),
                )
                for source in (
                    self.null_observation_laws,
                    self.alternative_observation_laws,
                )
            ),
        ))


def evaluate_stationary_dirichlet_composite_hypotheses(
    specification: StationaryDirichletObservationSpecification,
    *,
    null_states: tuple[CanonicalFluxState, ...],
    alternative_states: tuple[CanonicalFluxState, ...],
    independent_blocks: bool = False,
) -> StationaryDirichletCompositeTestingResult:
    """Map finite feasible states to common-face continuous product laws."""

    if type(independent_blocks) is not bool:
        raise InputValidationError("independent_blocks must be an explicit bool")
    _observation.validate_stationary_dirichlet_observation_specification(specification)
    hypotheses = CompositeFluxHypotheses(
        null_states=null_states, alternative_states=alternative_states,
    )
    identities = tuple(
        (experiment.experiment_id, item.target_id, item.replicate_id)
        for experiment in specification.experiments
        for item in experiment.specifications
    )
    if len(identities) > 1 and not independent_blocks:
        raise InputValidationError(
            "multiple Dirichlet MID blocks require explicit independent_blocks=True"
        )
    null_source, alternative_source = (
        _observation.evaluate_stationary_dirichlet_observation_families(
            specification,
            null_states=hypotheses.null_states,
            alternative_states=hypotheses.alternative_states,
        )
    )
    null_products = _product_members(
        null_source, hypotheses.null_states, identities, "null",
    )
    alternative_products = _product_members(
        alternative_source, hypotheses.alternative_states, identities, "alternative",
    )
    problem = DirichletCompositeBinaryTestingProblem(
        null=DirichletCompositeMIDLawFamily(
            members=null_products, member_ids=hypotheses.null_member_ids,
        ),
        alternative=DirichletCompositeMIDLawFamily(
            members=alternative_products, member_ids=hypotheses.alternative_member_ids,
        ),
    )
    return StationaryDirichletCompositeTestingResult(
        hypotheses=hypotheses,
        null_observation_laws=null_source,
        alternative_observation_laws=alternative_source,
        problem=problem,
    )


__all__ = [
    "StationaryDirichletCompositeTestingResult",
    "evaluate_stationary_dirichlet_composite_hypotheses",
]
