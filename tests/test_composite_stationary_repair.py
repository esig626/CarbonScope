"""Adversarial stationary provenance regressions, reproduced before repair."""

from dataclasses import replace

import pytest

from fluxemu import mfa
from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.observation import MultinomialMIDLaw, StationaryCountSpecification
from fluxemu.observation import stationary as observation_module
from fluxemu.testing import evaluate_stationary_composite_hypotheses
from test_testing_stationary import _ordered_specification, _specification, _state


@pytest.fixture
def stationary_result():
    return evaluate_stationary_composite_hypotheses(
        _specification(4),
        null_states=(_state("p0", 2.0), _state("p1", 3.0)),
        alternative_states=(_state("q0", 6.0), _state("q1", 7.0)),
    )


def test_composite_result_rejects_law_role_swap_with_original_member_ids(stationary_result):
    result = stationary_result
    problem = replace(
        result.problem,
        null=replace(result.problem.null, members=result.problem.alternative.members),
        alternative=replace(result.problem.alternative, members=result.problem.null.members),
    )
    with pytest.raises(InputValidationError):
        replace(result, problem=problem)


@pytest.mark.parametrize("field", [
    "model_fingerprint", "experiment_fingerprint", "specification_fingerprint",
])
def test_composite_result_rejects_component_provenance_divergence(stationary_result, field):
    result = stationary_result
    source = result.null_observation_laws
    component = replace(source.components[0], **{field: "different-scientific-source"})
    source = replace(source, components=(component,) + source.components[1:])
    with pytest.raises(InputValidationError):
        replace(result, null_observation_laws=source)


@pytest.mark.parametrize("change", ["state_order", "count_total", "law", "missing_component"])
def test_composite_result_rejects_source_law_or_state_misalignment(stationary_result, change):
    result = stationary_result
    source = result.null_observation_laws
    if change == "state_order":
        components = source.components[::-1]
    elif change == "missing_component":
        components = source.components[:-1]
    else:
        first = source.components[0]
        law = (MultinomialMIDLaw(first.law.n + 1, first.law.probabilities)
               if change == "count_total" else MultinomialMIDLaw(first.law.n, (0.5, 0.5)))
        components = (replace(first, law=law),) + source.components[1:]
    source = replace(source, components=components)
    with pytest.raises(InputValidationError):
        replace(result, null_observation_laws=source)


def test_composite_result_rejects_duplicate_experiment_provenance(stationary_result):
    source = stationary_result.null_observation_laws
    source = replace(source, experiment_fingerprints=source.experiment_fingerprints * 2)
    alternative = replace(
        stationary_result.alternative_observation_laws,
        experiment_fingerprints=source.experiment_fingerprints,
    )
    with pytest.raises(InputValidationError):
        replace(stationary_result, null_observation_laws=source, alternative_observation_laws=alternative)


def test_composite_result_fingerprint_retains_all_source_provenance(stationary_result):
    result = stationary_result

    def update_model(source):
        return replace(
            source,
            model_fingerprint="different-model",
            components=tuple(replace(item, model_fingerprint="different-model")
                             for item in source.components),
        )

    changed = replace(
        result,
        null_observation_laws=update_model(result.null_observation_laws),
        alternative_observation_laws=update_model(result.alternative_observation_laws),
    )
    assert changed.fingerprint != result.fingerprint


def test_composite_entrypoint_validates_specification_type():
    with pytest.raises(InputValidationError):
        evaluate_stationary_composite_hypotheses(
            object(), null_states=(_state("p0", 2.0),),
            alternative_states=(_state("q0", 6.0),),
        )


@pytest.mark.parametrize("role", ["null", "alternative"])
@pytest.mark.parametrize("invalid", [
    CanonicalFluxState("incomplete", (("Z_IN", 2.0), ("A_IN", 8.0))),
    CanonicalFluxState("reordered", (("A_IN", 8.0), ("Z_IN", 2.0), ("M_OUT", 10.0))),
    CanonicalFluxState("unbalanced", (("Z_IN", 2.0), ("A_IN", 8.0), ("M_OUT", 9.0))),
    CanonicalFluxState("outside", (("Z_IN", 11.0), ("A_IN", 0.0), ("M_OUT", 11.0))),
])
def test_composite_validates_each_state_before_its_emu_batch(monkeypatch, role, invalid):
    evaluated = []
    original = observation_module.evaluate_stationary

    def record(plan, states):
        evaluated.extend(states)
        return original(plan, states)

    monkeypatch.setattr(observation_module, "evaluate_stationary", record)
    states = {
        "null_states": (_state("p0", 2.0), _state("p1", 3.0)),
        "alternative_states": (_state("q0", 6.0), _state("q1", 7.0)),
    }
    states[f"{role}_states"] = (states[f"{role}_states"][0], invalid)
    with pytest.raises((AnalysisError, InputValidationError), match=f"{role} composite flux hypothesis"):
        evaluate_stationary_composite_hypotheses(_specification(), **states)
    assert invalid not in evaluated


def test_composite_preserves_state_and_observation_order_without_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("stationary composite bridge invoked MFA fitting")

    monkeypatch.setattr(mfa, "fit_stationary_mfa", forbidden)
    specification = _ordered_specification()
    states = (_state("z", 2.0), _state("a", 7.0))
    result = evaluate_stationary_composite_hypotheses(
        specification, null_states=states, alternative_states=states[::-1],
        independent_blocks=True,
    )
    identities = tuple(
        (block.experiment_id, item.target_id, item.replicate_id)
        for block in specification.experiments for item in block.specifications
    )
    totals = tuple(item.total_count for block in specification.experiments
                   for item in block.specifications)
    for source, family, ordered_states in (
        (result.null_observation_laws, result.problem.null, states),
        (result.alternative_observation_laws, result.problem.alternative, states[::-1]),
    ):
        assert source.states == ordered_states
        assert tuple(item.state for item in source.components) == tuple(
            state for state in ordered_states for _ in identities
        )
        assert tuple(item.sample_id for item in source.validation.diagnostics) == tuple(
            state.sample_id for state in ordered_states
        )
        assert all(member.block_identities == identities for member in family.members)
        assert all(member.block_totals == totals for member in family.members)


def test_composite_structural_zeros_and_changed_state_fingerprints_are_exact():
    states = (_state("same", 0.0), _state("changed", 10.0))
    result = evaluate_stationary_composite_hypotheses(
        _specification(), null_states=(states[0],), alternative_states=(states[1],),
    )
    assert result.problem.null.members[0].blocks[0].probabilities == (0.0, 1.0)
    assert result.problem.alternative.members[0].blocks[0].probabilities == (1.0, 0.0)
    changed = evaluate_stationary_composite_hypotheses(
        _specification(), null_states=(_state("same", 10.0),),
        alternative_states=(states[1],),
    )
    assert changed.hypotheses.null_member_ids != result.hypotheses.null_member_ids
    assert changed.fingerprint != result.fingerprint


@pytest.mark.parametrize("ordinary_mid_or_intensity", [0.5, 50.0, 1000.0, True])
def test_composite_count_declarations_never_infer_genuine_counts(ordinary_mid_or_intensity):
    with pytest.raises(InputValidationError):
        StationaryCountSpecification("O-mid", ordinary_mid_or_intensity)
