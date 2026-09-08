"""Finite composite testing: projected tests, converses, and exact minimax LPs."""

import math

import pytest

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing.composite import (
    CompositeAchievabilityConditionError,
    FiniteCompositeEnumerationLimitError,
    FiniteCompositeHypotheses,
    FiniteObservationLaw,
    composite_renyi_converse_at_order,
    projected_renyi_test,
    solve_finite_minimax_test,
)


def _single(probabilities, *, label=None):
    return FiniteObservationLaw.single(
        MultinomialMIDLaw(1, tuple(probabilities)), label=label,
    )


def _singleton_problem():
    return FiniteCompositeHypotheses(
        null=(_single((0.8, 0.2), label="P0"),),
        alternative=(_single((0.2, 0.8), label="Q1"),),
    )


def test_finite_minimax_singleton_reduces_to_randomised_np_value():
    result = solve_finite_minimax_test(_singleton_problem(), epsilon=0.1)

    assert result.solver_status == "Optimal"
    assert result.beta_star == pytest.approx(0.6, abs=1e-9)
    assert result.null_type_i_errors == pytest.approx((0.1,), abs=1e-9)
    assert result.alternative_type_ii_errors == pytest.approx((0.6,), abs=1e-9)
    assert result.solver_objective == pytest.approx(result.beta_star, abs=1e-9)

    # Enumeration order is (0,1), then (1,0).  The high LLR outcome (0,1)
    # is randomised with probability 1/2 to spend the exact Type-I budget.
    assert result.outcomes == (((0, 1),), ((1, 0),))
    assert result.rejection_probabilities == pytest.approx((0.5, 0.0), abs=1e-9)


def test_finite_minimax_overlapping_classes_optimises_worst_case_errors():
    hypotheses = FiniteCompositeHypotheses(
        null=(
            _single((0.8, 0.2), label="P0a"),
            _single((0.7, 0.3), label="P0b"),
        ),
        alternative=(
            _single((0.2, 0.8), label="Q1a"),
            _single((0.3, 0.7), label="Q1b"),
        ),
    )
    result = solve_finite_minimax_test(hypotheses, epsilon=0.1)

    assert max(result.null_type_i_errors) <= 0.1 + 1e-9
    assert result.beta_star == pytest.approx(max(result.alternative_type_ii_errors), abs=1e-9)
    assert result.beta_star == pytest.approx(23.0 / 30.0, abs=1e-9)
    assert result.worst_null_indices == (1,)
    assert result.worst_alternative_indices == (1,)


def test_identical_law_in_both_classes_forces_beta_star_one_minus_epsilon():
    shared = _single((0.7, 0.3), label="shared")
    hypotheses = FiniteCompositeHypotheses(null=(shared,), alternative=(shared,))

    result = solve_finite_minimax_test(hypotheses, epsilon=0.1)
    assert result.beta_star == pytest.approx(0.9, abs=1e-9)


def test_exact_minimax_preserves_disjoint_support_and_can_separate_perfectly():
    hypotheses = FiniteCompositeHypotheses(
        null=(_single((1.0, 0.0), label="P0"),),
        alternative=(_single((0.0, 1.0), label="Q1"),),
    )

    result = solve_finite_minimax_test(hypotheses, epsilon=0.05)
    assert result.beta_star == pytest.approx(0.0, abs=1e-12)
    assert result.null_type_i_errors == pytest.approx((0.0,), abs=1e-12)
    assert result.alternative_type_ii_errors == pytest.approx((0.0,), abs=1e-12)

    converse = composite_renyi_converse_at_order(hypotheses, epsilon=0.05, order=2.0)
    assert math.isinf(converse.reverse_renyi)
    assert math.isinf(converse.forward_renyi)
    assert converse.type_ii_lower_bound == 0.0


def test_projected_singleton_score_is_certified_and_calibration_matches_np():
    hypotheses = _singleton_problem()
    projected = projected_renyi_test(hypotheses, epsilon=0.1, order=0.5)
    optimum = solve_finite_minimax_test(hypotheses, epsilon=0.1)

    assert projected.projected_null_index == 0
    assert projected.projected_alternative_index == 0
    assert projected.projected_formula_certified is True
    assert projected.projected_formula_type_ii_upper_bound is not None
    assert projected.calibrated_boundary_randomisation == pytest.approx(0.5, abs=1e-9)
    assert projected.calibrated_type_ii_error == pytest.approx(0.6, abs=1e-9)
    assert projected.best_achievable_upper_bound == pytest.approx(optimum.beta_star, abs=1e-9)


def test_projected_test_is_only_an_achievable_upper_bound_for_general_classes():
    hypotheses = FiniteCompositeHypotheses(
        null=(
            _single((0.82, 0.18), label="P0a"),
            _single((0.68, 0.32), label="P0b"),
        ),
        alternative=(
            _single((0.25, 0.75), label="Q1a"),
            _single((0.36, 0.64), label="Q1b"),
        ),
    )
    projected = projected_renyi_test(hypotheses, epsilon=0.1, order=0.5)
    optimum = solve_finite_minimax_test(hypotheses, epsilon=0.1)

    assert max(projected.calibrated_null_type_i_errors) <= 0.1 + 1e-9
    assert projected.best_achievable_upper_bound + 1e-9 >= optimum.beta_star


def test_composite_renyi_converse_is_a_lower_bound_on_finite_minimax_optimum():
    hypotheses = FiniteCompositeHypotheses(
        null=(
            _single((0.8, 0.2), label="P0a"),
            _single((0.7, 0.3), label="P0b"),
        ),
        alternative=(
            _single((0.2, 0.8), label="Q1a"),
            _single((0.3, 0.7), label="Q1b"),
        ),
    )
    optimum = solve_finite_minimax_test(hypotheses, epsilon=0.1)
    converse = composite_renyi_converse_at_order(hypotheses, epsilon=0.1, order=2.0)

    assert 0 <= converse.type_ii_lower_bound <= optimum.beta_star + 1e-9
    assert converse.reverse_pair_indices == (1, 1)
    assert converse.forward_pair_indices == (1, 1)


def test_outcome_enumeration_limit_fails_without_changing_procedure():
    law = FiniteObservationLaw.single(
        MultinomialMIDLaw(10, (0.2, 0.3, 0.5)), label="large-space",
    )
    hypotheses = FiniteCompositeHypotheses(null=(law,), alternative=(law,))

    with pytest.raises(FiniteCompositeEnumerationLimitError, match="max_outcomes=10"):
        solve_finite_minimax_test(hypotheses, epsilon=0.1, max_outcomes=10)


def test_projected_score_fails_if_every_pair_has_disjoint_support():
    hypotheses = FiniteCompositeHypotheses(
        null=(_single((1.0, 0.0), label="P0"),),
        alternative=(_single((0.0, 1.0), label="Q1"),),
    )
    with pytest.raises(CompositeAchievabilityConditionError, match="infinite projected"):
        projected_renyi_test(hypotheses, epsilon=0.1, order=0.5)
