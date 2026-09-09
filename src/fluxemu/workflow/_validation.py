"""Validate retained workflow bindings without repeating scientific solvers.

These checks detect inconsistent replacements of independently retained
records. They do not authenticate an entirely, coherently fabricated result
or repeat the numerical certification performed by the testing primitives.
"""

from __future__ import annotations

import math

from fluxemu import testing
from fluxemu.exceptions import InputValidationError
from fluxemu.flux_analysis.highs import compile_flux_lp
from fluxemu.flux_analysis.results import FBAResult, FVAResult, _fva_ranges_sha256
from fluxemu.flux_analysis.sampling import (
    ALGORITHM_NAME, ALGORITHM_VERSION, BOUND_TOLERANCE, RANDOM_BIT_GENERATOR,
    FluxSampleValidationReport, FluxSamplingProvenance, NativeFluxSamplingResult,
)
from fluxemu.model import experiment_fingerprint, model_fingerprint
from fluxemu.observation.schema import _validate_state_record

from .schema import scientific_fingerprint, validate_hypothesis_testing_specification
from .states import FeasibleRegionWitness, HypothesisStateFamily, constrain_flux_model


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputValidationError("inconsistent workflow result: " + message)


def _family(specification, family, role, hypothesis) -> None:
    _require(isinstance(family, HypothesisStateFamily), f"{role} state family type")
    _require(family.role == role and family.hypothesis == hypothesis, f"{role} hypothesis binding")
    _require(family.model == constrain_flux_model(specification.model.flux_model, hypothesis),
             f"{role} constrained model binding")
    sampling = family.sampling
    _require(isinstance(sampling, NativeFluxSamplingResult), f"{role} sampling record type")
    _require(isinstance(sampling.states, tuple) and bool(sampling.states), f"{role} immutable state tuple")
    for state in sampling.states:
        _validate_state_record(state)
    _require(len({state.sample_id for state in sampling.states}) == len(sampling.states),
             f"{role} unique state identities")
    policy = hypothesis.state_generation
    _require(len(sampling.states) == sampling.sample_count == policy.count, f"{role} family size")
    provenance = sampling.provenance
    _require(isinstance(provenance, FluxSamplingProvenance), f"{role} sampling provenance type")
    for key in ("seed", "burn_in", "thinning", "max_direction_attempts"):
        _require(getattr(provenance, key) == getattr(policy, key), f"{role} sampling {key}")
    order = tuple(reaction.reaction_id for reaction in family.model.reactions)
    _require(provenance.sample_count == policy.count and provenance.reaction_order == order,
             f"{role} sampling count/reaction order")
    _require((provenance.algorithm, provenance.algorithm_version, provenance.random_bit_generator)
             == (ALGORITHM_NAME, ALGORITHM_VERSION, RANDOM_BIT_GENERATOR), f"{role} sampling algorithm")
    _require(provenance.fraction_of_optimum is None and provenance.retained_objective_bound is None,
             f"{role} full constrained-region policy")
    fingerprint = compile_flux_lp(family.model).fingerprint
    _require(provenance.model_fingerprint == fingerprint, f"{role} sampling model fingerprint")
    validation = sampling.validation
    _require(isinstance(validation, FluxSampleValidationReport) and validation.valid,
             f"{role} accepted feasibility validation")
    _require(validation.model_fingerprint == fingerprint and validation.reaction_order == order
             and validation.sample_count == policy.count, f"{role} feasibility validation identity")
    _require(tuple(item.sample_id for item in validation.diagnostics)
             == tuple(state.sample_id for state in sampling.states), f"{role} feasibility state order")
    _require(isinstance(family.fva, FVAResult) and isinstance(family.fba, FBAResult),
             f"{role} native FBA/FVA result types")
    _require(family.fva.model_fingerprint == fingerprint and family.fva.fraction_of_optimum is None,
             f"{role} FVA model/policy binding")
    _require(family.fva.ranges_sha256 == provenance.fva_ranges_sha256
             == _fva_ranges_sha256(family.fva.ranges, fingerprint), f"{role} FVA diagnostic integrity")
    _require(family.fba.objective_value == family.fva.objective_value == provenance.biological_optimum
             and family.fba.objective_direction == family.fva.objective_direction
             == provenance.objective_direction
             == {"maximise": "max", "minimise": "min"}[family.model.objective.direction],
             f"{role} FBA objective provenance")
    _require(tuple(family.fba.fluxes.index) == order, f"{role} FBA reaction order")
    _require(all(math.isfinite(value)
                 and reaction.lower_bound - BOUND_TOLERANCE <= value <= reaction.upper_bound + BOUND_TOLERANCE
                 for reaction, value in zip(family.model.reactions, family.fba.fluxes, strict=True)),
             f"{role} FBA constrained bounds")
    hypothesis_digest = scientific_fingerprint((
        "constrained-flux-hypothesis-v1", role, specification.model,
        hypothesis.description, hypothesis.reaction_bounds, "full-constrained-region",
    ))
    _require(family.hypothesis_fingerprint == hypothesis_digest, f"{role} hypothesis fingerprint")
    family_digest = scientific_fingerprint((
        "represented-finite-flux-family-v1", hypothesis_digest, policy, sampling.states,
        provenance.algorithm, provenance.algorithm_version, provenance.random_bit_generator,
    ))
    _require(family.fingerprint == family_digest, f"{role} finite family fingerprint")


def _region_evidence(result) -> None:
    expected = []
    for source, opposing in ((result.null_family, result.alternative_family),
                             (result.alternative_family, result.null_family)):
        for reaction in opposing.model.reactions:
            minimum = float(source.fva.ranges.loc[reaction.reaction_id, "minimum"])
            maximum = float(source.fva.ranges.loc[reaction.reaction_id, "maximum"])
            for endpoint, value, bound, separation in (
                ("minimum", minimum, reaction.lower_bound, reaction.lower_bound - minimum),
                ("maximum", maximum, reaction.upper_bound, maximum - reaction.upper_bound),
            ):
                if separation > BOUND_TOLERANCE:
                    expected.append(FeasibleRegionWitness(
                        source.role, opposing.role, reaction.reaction_id, endpoint,
                        value, bound, separation, BOUND_TOLERANCE,
                    ))
    _require(bool(expected), "distinct constrained-region evidence")
    for family in (result.null_family, result.alternative_family):
        _require(isinstance(family.region_witnesses, tuple)
                 and family.region_witnesses == tuple(expected), "region distinction witness binding")


def _testing_records(result) -> None:
    from .runner import PROCEDURES, ProcedureEvaluation, TestingRefusal

    settings = result.specification.testing
    records = result.testing_results
    _require(isinstance(records, tuple) and all(isinstance(item, ProcedureEvaluation) for item in records),
             "immutable typed testing results")
    requested = set(settings.procedures)
    required = set(requested)
    if requested & {"analytical_score_bound", "deterministic_score_error", "calibrated_score_error"}:
        required.add("candidate_score")
    if "deterministic_score_error" in requested:
        required.add("analytical_score_bound")
    expected = []
    for procedure in (*settings.procedures, *(name for name in PROCEDURES if name not in requested)):
        orders = (settings.converse_orders if procedure == "composite_converse" else
                  (None,) if procedure == "exact_minimax" else settings.score_orders)
        expected.extend((procedure, order) for order in orders or (None,))
    _require(tuple((item.procedure, item.order) for item in records) == tuple(expected),
             "procedure and supplied-order coverage/order")
    lookup = {(item.procedure, item.order): item for item in records}
    types = {
        "composite_converse": testing.CompositeRenyiConverseBound,
        "exact_minimax": testing.FiniteCompositeMinimaxResult,
        "candidate_score": testing.CompositeRenyiScoreCandidate,
        "analytical_score_bound": testing.CompositeScoreBound,
        "deterministic_score_error": testing.CompositeScoreTestEvaluation,
        "calibrated_score_error": testing.CalibratedCompositeScoreTest,
    }
    for item in records:
        _require(type(item.requested) is bool and item.requested == (item.procedure in requested),
                 "requested procedure marker")
        if item.procedure not in required:
            _require(item.status == "not_requested" and item.value is None and item.refusal is None,
                     "unrequested quantity status")
            continue
        _require(item.status in {"evaluated", "refused"}, "requested quantity outcome")
        dependency_name = (
            "analytical_score_bound" if item.procedure == "deterministic_score_error" else
            "candidate_score" if item.procedure in {"analytical_score_bound", "calibrated_score_error"} else None
        )
        dependency = lookup.get((dependency_name, item.order)) if dependency_name else None
        if item.status == "refused":
            refusal = item.refusal
            _require(item.value is None and isinstance(refusal, TestingRefusal), "typed statistical refusal")
            _require((refusal.procedure, refusal.order) == (item.procedure, item.order), "refusal identity")
            _require(all(isinstance(text, str) and text for text in
                         (refusal.category, refusal.exception_type, refusal.reason)), "explicit refusal reason")
            if dependency is not None and dependency.status == "refused":
                _require(refusal.category == "prerequisite_refused", "refused prerequisite status")
            continue
        _require(isinstance(item.value, types[item.procedure]) and item.refusal is None,
                 f"{item.procedure} numerical result type/status")
        value = item.value
        if dependency is not None:
            _require(dependency.status == "evaluated", "evaluated score prerequisite")
        if item.procedure == "deterministic_score_error":
            _require(value.bound == dependency.value, "deterministic score bound binding")
            candidate = value.bound.candidate
            constraint = value.bound.constraint
        elif item.procedure in {"analytical_score_bound", "calibrated_score_error"}:
            _require(value.candidate == dependency.value, "score candidate binding")
            candidate, constraint = value.candidate, value.constraint
        elif item.procedure == "candidate_score":
            candidate, constraint = value, None
        else:
            candidate, constraint = None, value.constraint
        problem = candidate.problem if candidate is not None else value.problem
        _require(problem == result.problem, f"{item.procedure} observable problem binding")
        if constraint is not None:
            _require(constraint.epsilon == settings.epsilon, f"{item.procedure} Type-I budget binding")
        if candidate is not None:
            _require(candidate.order == item.order, "candidate supplied order")
            if item.procedure in {"analytical_score_bound", "deterministic_score_error"}:
                _require(candidate.uniform_moment_bounds_verified, "analytical score verification")
        elif item.procedure == "composite_converse":
            _require(value.order == item.order, "converse supplied order")
        if hasattr(value, "outcomes"):
            _require(isinstance(value.outcomes, tuple) and len(value.outcomes) <= settings.max_outcomes,
                     "declared exact enumeration cap")


def validate_workflow_result(result) -> None:
    """Check all retained scientific identities without sampling, EMU or testing."""
    from .runner import (
        HypothesisTestingWorkflowResult, RelationshipCheck, WorkflowOutputPaths,
        WorkflowProvenance, WorkflowSoftwareProvenance, _check_relationships,
    )

    _require(isinstance(result, HypothesisTestingWorkflowResult), "public result type")
    specification = result.specification
    validate_hypothesis_testing_specification(specification)
    _family(specification, result.null_family, "H0", specification.null)
    _family(specification, result.alternative_family, "H1", specification.alternative)
    _region_evidence(result)
    stationary = result.stationary
    _require(isinstance(stationary, testing.StationaryCompositeTestingResult), "stationary source result type")
    stationary.__post_init__()
    _require(stationary.hypotheses.null_states == result.null_family.states
             and stationary.hypotheses.alternative_states == result.alternative_family.states,
             "stationary H0/H1 state family binding")
    observation = specification.observation
    model_digest = model_fingerprint(specification.model)
    experiment_digests = tuple((block.experiment_id, experiment_fingerprint(specification.model, block.experiment))
                               for block in observation.experiments)
    declarations = tuple(((block.experiment_id, item.target_id, item.replicate_id), item.total_count)
                         for block in observation.experiments for item in block.specifications)
    for source, family in ((stationary.null_observation_laws, result.problem.null),
                           (stationary.alternative_observation_laws, result.problem.alternative)):
        _require(source.model_fingerprint == model_digest
                 and source.experiment_fingerprints == experiment_digests
                 and source.specification_fingerprint == observation.fingerprint,
                 "stationary observation specification binding")
        for member in family.members:
            _require(tuple(zip(member.block_identities, member.block_totals, strict=True)) == declarations,
                     "stationary block/count declaration binding")
    _testing_records(result)
    _require(isinstance(result.relationship_checks, tuple)
             and all(isinstance(check, RelationshipCheck) for check in result.relationship_checks),
             "immutable typed relationship evidence")
    _require(result.relationship_checks == _check_relationships(result.testing_results),
             "recomputed numerical relationship evidence")
    provenance = result.provenance
    _require(isinstance(provenance, WorkflowProvenance)
             and isinstance(provenance.software, WorkflowSoftwareProvenance), "typed provenance")
    identities = {
        "model_fingerprint": model_digest,
        "experiment_fingerprints": experiment_digests,
        "specification_fingerprint": specification.fingerprint,
        "null_hypothesis_fingerprint": result.null_family.hypothesis_fingerprint,
        "alternative_hypothesis_fingerprint": result.alternative_family.hypothesis_fingerprint,
        "null_finite_family_fingerprint": result.null_family.fingerprint,
        "alternative_finite_family_fingerprint": result.alternative_family.fingerprint,
        "null_observation_family_fingerprint": result.problem.null.fingerprint,
        "alternative_observation_family_fingerprint": result.problem.alternative.fingerprint,
        "observation_specification_fingerprint": observation.fingerprint,
        "composite_problem_fingerprint": result.problem.fingerprint,
        "stationary_bridge_fingerprint": stationary.fingerprint,
    }
    for field, expected in identities.items():
        _require(getattr(provenance, field) == expected, f"provenance {field}")
    if result.output_paths is not None:
        _require(isinstance(result.output_paths, WorkflowOutputPaths), "output-path record type")


__all__ = ["validate_workflow_result"]
