"""Complete constrained-state construction and distinct-region controls."""

from dataclasses import replace
from pathlib import Path

import pytest

from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.flux_analysis import prepare_highs_flux_region, validate_flux_states
from fluxemu.model import load_sbml_flux_model
from fluxemu.native_io import load_native_stationary_spec
from fluxemu.observation import (
    StationaryCountSpecification, StationaryObservationExperiment,
    StationaryObservationSpecification,
)
from fluxemu.workflow import states as states_module
from fluxemu.workflow.schema import (
    HypothesisSpecification, ReactionBoundConstraint, StateGenerationSpecification,
    TestingSpecification as WorkflowTestingSpecification, WorkflowSpecification,
)
from fluxemu.workflow.states import constrain_flux_model, generate_hypothesis_state_families


FIXTURES = Path(__file__).parent / "fixtures" / "native_portability"


def specification():
    flux = load_sbml_flux_model(FIXTURES / "model.xml")
    model, experiment, _ = load_native_stationary_spec(FIXTURES / "experiment.yaml", flux)
    observation = StationaryObservationSpecification(model, (
        StationaryObservationExperiment("tracer", experiment, (StationaryCountSpecification("portable_mid", 2, "r0"),)),
    ))
    hypotheses = tuple(HypothesisSpecification(
        f"{role} interval", (ReactionBoundConstraint("foreign_hx", lower, upper),),
        StateGenerationSpecification("hit_and_run", 4, seed, 3, 2),
    ) for role, lower, upper, seed in (("H0", 1, 3, 10), ("H1", 7, 9, 20)))
    return WorkflowSpecification(observation, *hypotheses, WorkflowTestingSpecification(0.1, ("exact_minimax",)), False)


def test_intersections_preserve_common_model_and_objective():
    spec = specification()
    constrained = constrain_flux_model(spec.model.flux_model, spec.null)
    assert spec.model.flux_model.reactions[0].lower_bound == 0
    assert constrained.reactions[0].lower_bound == 1
    assert constrained.reactions[0].upper_bound == 3
    assert constrained.objective is spec.model.flux_model.objective
    broad = replace(spec.null, reaction_bounds=(ReactionBoundConstraint("foreign_hx", -100, 100),))
    unchanged = constrain_flux_model(spec.model.flux_model, broad)
    assert unchanged.reactions[0].lower_bound == 0
    assert unchanged.reactions[0].upper_bound == 10


def test_families_are_jointly_feasible_reproducible_and_not_fva_endpoint_vectors():
    spec = specification()
    null, alternative = generate_hypothesis_state_families(spec)
    again = generate_hypothesis_state_families(spec)
    assert (null.role, alternative.role) == ("H0", "H1")
    assert null.region_witnesses and null.region_witnesses == alternative.region_witnesses
    assert [(family.states, family.fingerprint) for family in (null, alternative)] == [(family.states, family.fingerprint) for family in again]
    for family, lower, upper in ((null, 1, 3), (alternative, 7, 9)):
        assert family.sampling.sample_count == 4
        assert family.sampling.provenance.fraction_of_optimum is None
        assert validate_flux_states(prepare_highs_flux_region(family.model, None), family.states).valid
        for state in family.states:
            assert [item[0] for item in state.values] == ["foreign_hx", "foreign_sink"]
            hx, sink = [item[1] for item in state.values]
            assert hx == pytest.approx(sink, abs=1e-9)
            assert lower < hx < upper
        assert family.fba.objective_value == pytest.approx(upper)
        assert any(state.values[1][1] < upper for state in family.states)


def test_redundant_different_bounds_do_not_create_distinct_regions():
    spec = specification()
    null = replace(spec.null, reaction_bounds=(ReactionBoundConstraint("foreign_hx", upper_bound=3),))
    alternative = replace(spec.alternative, reaction_bounds=(ReactionBoundConstraint("foreign_sink", upper_bound=3),))
    with pytest.raises(InputValidationError, match="identical or not distinguishable"):
        generate_hypothesis_state_families(replace(spec, null=null, alternative=alternative))


def test_overlapping_regions_are_allowed_and_near_identical_regions_refuse():
    spec = specification()
    null = replace(spec.null, reaction_bounds=(ReactionBoundConstraint("foreign_hx", 1, 6),))
    alternative = replace(spec.alternative, reaction_bounds=(ReactionBoundConstraint("foreign_hx", 4, 9),))
    assert len(generate_hypothesis_state_families(replace(spec, null=null, alternative=alternative))) == 2
    alternative = replace(spec.alternative, reaction_bounds=(ReactionBoundConstraint("foreign_hx", 1, 6 + 1e-9),))
    with pytest.raises(InputValidationError, match="tolerance"):
        generate_hypothesis_state_families(replace(spec, null=null, alternative=alternative))


@pytest.mark.parametrize("role", ["null", "alternative"])
def test_infeasibility_is_detected_before_either_family_is_sampled(monkeypatch, role):
    spec = specification()
    impossible = replace(getattr(spec, role), reaction_bounds=(
        ReactionBoundConstraint("foreign_hx", 1, 2),
        ReactionBoundConstraint("foreign_sink", 3, 4),
    ))
    monkeypatch.setattr(states_module, "sample_prepared_flux_states", lambda *args, **kwargs: pytest.fail("sampling began before both feasibility checks"))
    with pytest.raises(AnalysisError, match="H0|H1"):
        generate_hypothesis_state_families(replace(spec, **{role: impossible}))


def test_empty_intersection_fails_with_hypothesis_context():
    spec = specification()
    null = replace(spec.null, reaction_bounds=(ReactionBoundConstraint("foreign_hx", 20, 30),))
    with pytest.raises(InputValidationError, match="H0.*infeasible bound intersection"):
        generate_hypothesis_state_families(replace(spec, null=null))


def test_seed_changes_family_identity_without_changing_biological_hypothesis():
    spec = specification()
    original = generate_hypothesis_state_families(spec)[0]
    null = replace(spec.null, state_generation=replace(spec.null.state_generation, seed=42))
    changed = generate_hypothesis_state_families(replace(spec, null=null))[0]
    assert changed.hypothesis_fingerprint == original.hypothesis_fingerprint
    assert changed.fingerprint != original.fingerprint
    assert changed.states != original.states
    null = replace(spec.null, reaction_bounds=(ReactionBoundConstraint("foreign_hx", 1, 4),))
    changed = generate_hypothesis_state_families(replace(spec, null=null))[0]
    assert changed.hypothesis_fingerprint != original.hypothesis_fingerprint


def test_fixed_regions_return_requested_ordered_count_of_complete_states():
    spec = specification()
    null = replace(spec.null, reaction_bounds=(ReactionBoundConstraint("foreign_hx", 1, 1),))
    alternative = replace(spec.alternative, reaction_bounds=(ReactionBoundConstraint("foreign_hx", 2, 2),))
    for family, value in zip(generate_hypothesis_state_families(replace(spec, null=null, alternative=alternative)), (1, 2), strict=True):
        assert len(family.states) == 4
        assert len({state.sample_id for state in family.states}) == 4
        assert all(state.values == (("foreign_hx", value), ("foreign_sink", value)) for state in family.states)
