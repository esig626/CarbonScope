"""Finite composite flux hypotheses mapped through native stationary EMU."""

from __future__ import annotations

import pytest

from fluxemu import observation as observation_api
from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.testing import (
    CompositeFluxHypotheses,
    composite_renyi_converse_at_order,
    evaluate_stationary_composite_hypotheses,
    exact_finite_composite_minimax,
)
from test_testing_stationary import _ordered_specification, _specification, _state


def test_single_block_stationary_composite_bridge_preserves_flux_and_law_order():
    null_states = (_state("p0", 2.0), _state("p1", 3.0))
    alternative_states = (_state("q0", 6.0), _state("q1", 7.0))
    result = evaluate_stationary_composite_hypotheses(
        _specification(4),
        null_states=null_states,
        alternative_states=alternative_states,
    )
    assert result.hypotheses.null_states == null_states
    assert result.hypotheses.alternative_states == alternative_states
    assert result.null_observation_laws.states == null_states
    assert result.alternative_observation_laws.states == alternative_states
    assert result.problem.null.block_count == result.problem.alternative.block_count == 1
    null_products = result.problem.null.members
    alternative_products = result.problem.alternative.members
    assert null_products[0].blocks[0].probabilities == pytest.approx((0.2, 0.8), abs=2e-15)
    assert null_products[1].blocks[0].probabilities == pytest.approx((0.3, 0.7), abs=2e-15)
    assert alternative_products[0].blocks[0].probabilities == pytest.approx((0.6, 0.4), abs=2e-15)
    assert alternative_products[1].blocks[0].probabilities == pytest.approx((0.7, 0.3), abs=2e-15)
    assert result.problem.null.member_ids == result.hypotheses.null_member_ids
    assert result.problem.alternative.member_ids == result.hypotheses.alternative_member_ids
    assert len(result.fingerprint) == 64


def test_stationary_composite_problem_runs_exact_minimax_on_small_native_flux_families():
    result = evaluate_stationary_composite_hypotheses(
        _specification(4),
        null_states=(_state("p0", 2.0), _state("p1", 3.0)),
        alternative_states=(_state("q0", 6.0), _state("q1", 7.0)),
    )
    minimax = exact_finite_composite_minimax(result.problem, epsilon=0.05)
    assert minimax.worst_type_i_error <= 0.05 + 1e-10
    assert 0 <= minimax.minimax_type_ii_error <= 1
    assert minimax.active_null_member_ids
    assert minimax.active_alternative_member_ids


def test_multiple_stationary_blocks_form_one_explicit_joint_product_law_per_state():
    specification = _ordered_specification()
    null_state, alternative_state = _state("p0", 2.0), _state("q0", 6.0)
    result = evaluate_stationary_composite_hypotheses(
        specification,
        null_states=(null_state,),
        alternative_states=(alternative_state,),
        independent_blocks=True,
    )
    identities = tuple(
        (block.experiment_id, item.target_id, item.replicate_id)
        for block in specification.experiments
        for item in block.specifications
    )
    totals = tuple(
        item.total_count
        for block in specification.experiments
        for item in block.specifications
    )
    assert result.problem.null.block_count == 8
    assert result.problem.null.members[0].block_identities == identities
    assert result.problem.alternative.members[0].block_identities == identities
    assert result.problem.null.members[0].block_totals == totals
    assert result.problem.alternative.members[0].block_totals == totals
    assert totals == (37, 59, 11, 23, 23, 11, 59, 37)
    expected = observation_api.independent_product_renyi(
        result.problem.alternative.members[0].blocks,
        result.problem.null.members[0].blocks,
        1.5,
    )
    converse = composite_renyi_converse_at_order(
        result.problem, epsilon=0.05, order=1.5,
    )
    assert converse.reverse_renyi == pytest.approx(expected, abs=2e-12)


def test_multiple_composite_blocks_require_explicit_independence_before_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("missing independence declaration reached observation evaluation")

    monkeypatch.setattr(
        observation_api,
        "evaluate_stationary_observation_laws",
        forbidden,
    )
    with pytest.raises(InputValidationError, match="independent_blocks=True"):
        evaluate_stationary_composite_hypotheses(
            _ordered_specification(),
            null_states=(_state("p0", 2.0),),
            alternative_states=(_state("q0", 6.0),),
        )


def test_composite_flux_family_rejects_duplicate_member_sample_ids():
    with pytest.raises(InputValidationError, match="sample IDs must be unique"):
        CompositeFluxHypotheses(
            null_states=(_state("same", 2.0), _state("same", 3.0)),
            alternative_states=(_state("q0", 6.0),),
        )


def test_each_composite_role_validates_every_complete_flux_state_before_emu():
    invalid = CanonicalFluxState(
        "bad",
        (("Z_IN", 2.0), ("A_IN", 8.0), ("M_OUT", 9.0)),
    )
    with pytest.raises((AnalysisError, InputValidationError), match="null composite flux hypothesis"):
        evaluate_stationary_composite_hypotheses(
            _specification(4),
            null_states=(_state("p0", 2.0), invalid),
            alternative_states=(_state("q0", 6.0),),
        )


def test_state_ids_are_provenance_only_and_same_source_id_across_roles_does_not_conflate_laws():
    result = evaluate_stationary_composite_hypotheses(
        _specification(4),
        null_states=(_state("shared", 2.0),),
        alternative_states=(_state("shared", 6.0),),
    )
    assert result.null_observation_laws.components[0].sample_id == "shared"
    assert result.alternative_observation_laws.components[0].sample_id == "shared"
    assert result.problem.null.member_ids != result.problem.alternative.member_ids
    assert result.problem.null.members[0].blocks[0].probabilities == pytest.approx(
        (0.2, 0.8), abs=2e-15,
    )
    assert result.problem.alternative.members[0].blocks[0].probabilities == pytest.approx(
        (0.6, 0.4), abs=2e-15,
    )
