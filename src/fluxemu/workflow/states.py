"""Constrained, jointly feasible represented finite H0/H1 state families."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.flux_analysis import (
    FBAResult, FVAResult, NativeFluxSamplingResult, PreparedFluxRegion,
    prepare_highs_flux_region, run_prepared_highs_vffva,
    sample_prepared_flux_states, validate_flux_states,
)
from fluxemu.flux_analysis.sampling import BOUND_TOLERANCE
from fluxemu.model import FluxModel, validate_flux_model

from .schema import (
    HypothesisSpecification, WorkflowSpecification, scientific_fingerprint,
    validate_hypothesis_testing_specification,
)


@dataclass(frozen=True, slots=True)
class FeasibleRegionWitness:
    """An audited coordinate extremum excluded by the opposing region's box.

    This is a region distinction diagnostic, never a generated flux vector.
    Both regions share the same stoichiometry and have no objective-retention
    row, so these extrema suffice to detect distinct regions up to tolerance.
    """

    source_role: str
    excluded_by_role: str
    reaction_id: str
    endpoint: str
    value: float
    opposing_bound: float
    separation: float
    tolerance: float


@dataclass(frozen=True, slots=True)
class HypothesisStateFamily:
    role: str
    hypothesis: HypothesisSpecification
    model: FluxModel
    sampling: NativeFluxSamplingResult
    fva: FVAResult
    fba: FBAResult
    fingerprint: str
    hypothesis_fingerprint: str
    region_witnesses: tuple[FeasibleRegionWitness, ...]

    @property
    def states(self) -> tuple[CanonicalFluxState, ...]:
        return self.sampling.states


def constrain_flux_model(model: FluxModel, hypothesis: HypothesisSpecification) -> FluxModel:
    """Intersect explicit biological bounds with immutable common-model bounds.

    A fixed reaction is an interval whose two endpoints coincide.  Broader
    declarations cannot relax the common model, and redundant declarations
    remain visible in provenance.  No reaction, stoichiometry, ordering, or
    biological-objective coefficient is replaced.
    """

    validate_flux_model(model)
    if not isinstance(hypothesis, HypothesisSpecification):
        raise InputValidationError("hypothesis must be HypothesisSpecification")
    constraints = {item.reaction_id: item for item in hypothesis.reaction_bounds}
    unknown = set(constraints) - {item.reaction_id for item in model.reactions}
    if unknown:
        raise InputValidationError("constraint references unknown reaction(s): " + ", ".join(sorted(unknown)))
    reactions = []
    for reaction in model.reactions:
        constraint = constraints.get(reaction.reaction_id)
        lower, upper = float(reaction.lower_bound), float(reaction.upper_bound)
        if constraint is not None:
            if constraint.lower_bound is not None:
                lower = max(lower, constraint.lower_bound)
            if constraint.upper_bound is not None:
                upper = min(upper, constraint.upper_bound)
        if lower > upper:
            raise InputValidationError(
                f"infeasible bound intersection for reaction {reaction.reaction_id!r}: [{lower}, {upper}]"
            )
        reactions.append(replace(reaction, lower_bound=lower, upper_bound=upper))
    constrained = replace(model, reactions=tuple(reactions))
    validate_flux_model(constrained)
    return constrained


def _region_witnesses(
    null: PreparedFluxRegion, alternative: PreparedFluxRegion,
    null_fva: FVAResult, alternative_fva: FVAResult,
) -> tuple[FeasibleRegionWitness, ...]:
    witnesses = []
    for source_role, opposing_role, source_fva, opposing in (
        ("H0", "H1", null_fva, alternative),
        ("H1", "H0", alternative_fva, null),
    ):
        for index, reaction_id in enumerate(opposing.lp.reaction_ids):
            minimum = float(source_fva.ranges.loc[reaction_id, "minimum"])
            maximum = float(source_fva.ranges.loc[reaction_id, "maximum"])
            for endpoint, value, bound, separation in (
                ("minimum", minimum, opposing.lp.lower_bounds[index], opposing.lp.lower_bounds[index] - minimum),
                ("maximum", maximum, opposing.lp.upper_bounds[index], maximum - opposing.lp.upper_bounds[index]),
            ):
                if separation > BOUND_TOLERANCE:
                    witnesses.append(FeasibleRegionWitness(
                        source_role, opposing_role, reaction_id, endpoint,
                        value, bound, separation, BOUND_TOLERANCE,
                    ))
    return tuple(witnesses)


def generate_hypothesis_state_families(
    specification: WorkflowSpecification,
) -> tuple[HypothesisStateFamily, HypothesisStateFamily]:
    """Prepare both feasible regions before sampling in fixed H0, H1 order.

    FVA extrema certify geometry and distinguish regions; the native sampler
    produces every complete state.  Coincident or numerically indistinguishable
    feasible regions fail explicitly even if their declarations differ.
    """

    validate_hypothesis_testing_specification(specification)
    regions = []
    for role, hypothesis in (("H0", specification.null), ("H1", specification.alternative)):
        try:
            model = constrain_flux_model(specification.model.flux_model, hypothesis)
            prepared = prepare_highs_flux_region(model, fraction_of_optimum=None)
        except (AnalysisError, InputValidationError) as error:
            raise type(error)(f"{role} feasible-region construction failed: {error}") from error
        regions.append((role, hypothesis, model, prepared))
    fvas = []
    for role, _, _, prepared in regions:
        try:
            fvas.append(run_prepared_highs_vffva(prepared, workers=1, audit_endpoints=True))
        except AnalysisError as error:
            raise AnalysisError(f"{role} feasible-region FVA failed: {error}") from error
    witnesses = _region_witnesses(regions[0][3], regions[1][3], fvas[0], fvas[1])
    if not witnesses:
        raise InputValidationError(
            "H0 and H1 feasible regions are identical or not distinguishable at the "
            f"native bound tolerance {BOUND_TOLERANCE:g}; redundant declarations "
            "and different sampling seeds do not define distinct hypotheses"
        )
    families = []
    for (role, hypothesis, model, prepared), fva in zip(regions, fvas, strict=True):
        policy = hypothesis.state_generation
        try:
            sampling = sample_prepared_flux_states(
                prepared, policy.count, seed=policy.seed, fva=fva,
                burn_in=policy.burn_in, thinning=policy.thinning,
                max_direction_attempts=policy.max_direction_attempts,
            )
            validation = validate_flux_states(prepared, sampling.states)
            if not validation.valid:
                raise AnalysisError("returned complete states failed independent feasibility validation: " + "; ".join(validation.errors))
        except AnalysisError as error:
            raise AnalysisError(f"{role} state generation failed: {error}") from error
        hypothesis_fingerprint = scientific_fingerprint((
            "constrained-flux-hypothesis-v1", role, specification.model,
            hypothesis.description, hypothesis.reaction_bounds, "full-constrained-region",
        ))
        fingerprint = scientific_fingerprint((
            "represented-finite-flux-family-v1", hypothesis_fingerprint,
            policy, sampling.states, sampling.provenance.algorithm,
            sampling.provenance.algorithm_version,
            sampling.provenance.random_bit_generator,
        ))
        families.append(HypothesisStateFamily(
            role, hypothesis, model, sampling, fva, prepared.fba,
            fingerprint, hypothesis_fingerprint, witnesses,
        ))
    return families[0], families[1]


__all__ = [
    "FeasibleRegionWitness", "HypothesisStateFamily", "constrain_flux_model",
    "generate_hypothesis_state_families",
]
