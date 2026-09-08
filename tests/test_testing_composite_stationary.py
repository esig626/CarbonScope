"""Finite flux-state families mapped into the composite testing core."""

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.testing import (
    composite_renyi_converse_at_order,
    evaluate_stationary_finite_composite_hypotheses,
    projected_renyi_test,
    solve_finite_minimax_test,
)
from test_observation_stationary import _specification as _ordered_specification
from test_testing_stationary import _specification, _state


def test_stationary_bridge_maps_each_supplied_flux_state_to_one_finite_law():
    null_states = (_state("n0", 2.5), _state("n1", 3.0))
    alternative_states = (_state("a0", 5.0), _state("a1", 6.0))

    result = evaluate_stationary_finite_composite_hypotheses(
        _specification(1),
        null_states=null_states,
        alternative_states=alternative_states,
    )

    assert result.null_states == null_states
    assert result.alternative_states == alternative_states
    assert len(result.hypotheses.null) == 2
    assert len(result.hypotheses.alternative) == 2
    assert result.block_identities == (("fixed-tracer", "O-mid", "genuine-counts"),)
    assert result.hypotheses.null[0].blocks[0].probabilities == (0.25, 0.75)
    assert result.hypotheses.null[1].blocks[0].probabilities == (0.3, 0.7)
    assert result.hypotheses.alternative[0].blocks[0].probabilities == (0.5, 0.5)
    assert result.hypotheses.alternative[1].blocks[0].probabilities == (0.6, 0.4)
    assert result.hypotheses.null[0].label == "H0[0]:n0"
    assert result.hypotheses.alternative[1].label == "H1[1]:a1"
    assert result.observation_specification_fingerprint == _specification(1).fingerprint


def test_stationary_finite_family_runs_both_testing_paths_without_flux_fitting():
    result = evaluate_stationary_finite_composite_hypotheses(
        _specification(1),
        null_states=(_state("n0", 2.5), _state("n1", 3.0)),
        alternative_states=(_state("a0", 5.0), _state("a1", 6.0)),
    )

    exact = solve_finite_minimax_test(result.hypotheses, epsilon=0.1)
    projected = projected_renyi_test(result.hypotheses, epsilon=0.1, order=0.5)
    converse = composite_renyi_converse_at_order(
        result.hypotheses, epsilon=0.1, order=2.0,
    )

    assert converse.type_ii_lower_bound <= exact.beta_star + 1e-9
    assert exact.beta_star <= projected.best_achievable_upper_bound + 1e-9
    assert max(exact.null_type_i_errors) <= 0.1 + 1e-9
    assert max(projected.calibrated_null_type_i_errors) <= 0.1 + 1e-9


def test_multiple_stationary_blocks_require_explicit_independence():
    specification = _ordered_specification()
    with pytest.raises(InputValidationError, match="independent_blocks=True"):
        evaluate_stationary_finite_composite_hypotheses(
            specification,
            null_states=(_state("n0"),),
            alternative_states=(_state("a0", 5.0),),
        )

    result = evaluate_stationary_finite_composite_hypotheses(
        specification,
        null_states=(_state("n0"),),
        alternative_states=(_state("a0", 5.0),),
        independent_blocks=True,
    )
    assert result.hypotheses.null[0].independent is True
    assert result.hypotheses.alternative[0].independent is True
    assert len(result.hypotheses.null[0].blocks) == len(result.block_identities) == 8


def test_single_stationary_block_rejects_spurious_independence_declaration():
    with pytest.raises(InputValidationError, match="single observation block"):
        evaluate_stationary_finite_composite_hypotheses(
            _specification(1),
            null_states=(_state("n0"),),
            alternative_states=(_state("a0", 5.0),),
            independent_blocks=True,
        )
