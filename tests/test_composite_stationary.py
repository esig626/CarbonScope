"""Finite composite flux hypotheses mapped through native stationary EMU."""

from __future__ import annotations

import pytest

from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.testing import (
    CompositeFluxHypotheses,
    evaluate_stationary_composite_hypotheses,
    exact_finite_composite_minimax,
)
from test_testing_stationary import _ordered_specification, _specification, _state


def test_stationary_composite_bridge_preserves_flux_family_and_law_order():
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
    null_mids = tuple(item.predicted_mid for item in result.null_observation_laws.components)
    alternative_mids = tuple(
        item.predicted_mid for item in result.alternative_observation_laws.components
    )
    assert null_mids[0] == pytest.approx((0.2, 0.8), abs=2e-15)
    assert null_mids[1] == pytest.approx((0.3, 0.7), abs=2e-15)
    assert alternative_mids[0] == pytest.approx((0.6, 0.4), abs=2e-15)
    assert alternative_mids[1] == pytest.approx((0.7, 0.3), abs=2e-15)
    assert tuple(item.law for item in result.null_observation_laws.components) == (
        result.problem.null.members
    )
    assert tuple(item.law for item in result.alternative_observation_laws.components) == (
        result.problem.alternative.members
    )
    assert result.problem.null.member_ids == result.hypotheses.null_member_ids
    assert result.problem.alternative.member_ids == result.hypotheses.alternative_member_ids
    assert result.problem.n == 4
    assert result.problem.mass_classes == (0, 1)
    assert len(result.fingerprint) == 64


def test_stationary_composite_problem_runs_exact_minimax_on_native_flux_families():
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


def test_stationary_composite_bridge_rejects_multiple_observation_blocks_explicitly():
    with pytest.raises(InputValidationError, match="exactly one.*genuine-count block"):
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


def test_composite_flux_roles_may_reuse_sample_id_across_families_without_conflation():
    result = evaluate_stationary_composite_hypotheses(
        _specification(4),
        null_states=(_state("shared", 2.0),),
        alternative_states=(_state("shared", 6.0),),
    )
    assert result.null_observation_laws.components[0].sample_id == "shared"
    assert result.alternative_observation_laws.components[0].sample_id == "shared"
    assert result.problem.null.member_ids != result.problem.alternative.member_ids
    assert result.problem.null.members[0].probabilities == pytest.approx((0.2, 0.8), abs=2e-15)
    assert result.problem.alternative.members[0].probabilities == pytest.approx((0.6, 0.4), abs=2e-15)
