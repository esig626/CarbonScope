"""Finite composite minimax, Rényi converse, projection, and calibration tests."""

from __future__ import annotations

import math

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    CompositeBinaryTestingProblem,
    CompositeEnumerationLimitError,
    CompositeMIDLawFamily,
    CompositeProjectionError,
    calibrate_composite_projected_test,
    composite_renyi_converse_at_order,
    exact_finite_composite_minimax,
    projected_composite_bound_at_order,
    verified_composite_renyi_projection,
)


def _family(n, probabilities, prefix):
    return CompositeMIDLawFamily(
        members=tuple(MultinomialMIDLaw(n, tuple(value)) for value in probabilities),
        member_ids=tuple(f"{prefix}-{index}" for index in range(len(probabilities))),
    )


def _problem(n, null, alternative):
    return CompositeBinaryTestingProblem(
        null=_family(n, null, "P"),
        alternative=_family(n, alternative, "Q"),
    )


def test_composite_family_preserves_declared_members_without_convexifying():
    family = _family(4, ((0.8, 0.2), (0.7, 0.3)), "P")
    assert len(family.members) == 2
    assert family.member_ids == ("P-0", "P-1")
    assert tuple(item.probabilities for item in family.members) == (
        (0.8, 0.2), (0.7, 0.3),
    )
    assert family.n == 4
    assert family.mass_classes == (0, 1)
    assert family.full_support
    assert len(family.fingerprint) == 64


def test_composite_problem_requires_common_count_and_mass_class_spaces():
    with pytest.raises(InputValidationError, match="count total"):
        CompositeBinaryTestingProblem(
            null=_family(2, ((0.8, 0.2),), "P"),
            alternative=_family(3, ((0.2, 0.8),), "Q"),
        )
    with pytest.raises(InputValidationError, match="mass-class"):
        CompositeBinaryTestingProblem(
            null=_family(2, ((0.8, 0.2),), "P"),
            alternative=_family(2, ((0.2, 0.3, 0.5),), "Q"),
        )


def test_singleton_composite_converse_recovers_mixer_reverse_bound():
    problem = _problem(4, ((0.25, 0.75),), ((0.5, 0.5),))
    result = composite_renyi_converse_at_order(
        problem, epsilon=0.05, order=2.0,
    )
    assert result.order == 2.0
    assert result.rate == pytest.approx(-math.log(0.05) / 4)
    assert result.reverse_renyi == pytest.approx(math.log(4 / 3), abs=3e-14)
    assert result.full_law_reverse_renyi == pytest.approx(4 * math.log(4 / 3), abs=3e-14)
    assert result.type_ii_lower_bound == pytest.approx(0.602476804, abs=5e-10)
    assert result.null_member_id == "P-0"
    assert result.alternative_member_id == "Q-0"
    assert not result.global_order_envelope_evaluated


def test_exact_two_by_two_finite_composite_minimax_is_point_six():
    problem = _problem(
        1,
        ((0.9, 0.1), (0.8, 0.2)),
        ((0.1, 0.9), (0.2, 0.8)),
    )
    result = exact_finite_composite_minimax(problem, epsilon=0.1)
    assert result.outcomes == ((0, 1), (1, 0))
    assert result.minimax_type_ii_error == pytest.approx(0.6, abs=2e-12)
    assert result.worst_type_i_error <= 0.1 + 1e-12
    assert result.randomized
    assert max(result.null_type_i_errors) == pytest.approx(0.1, abs=2e-12)
    assert max(result.alternative_type_ii_errors) == pytest.approx(0.6, abs=2e-12)


def test_identical_composite_law_has_exact_one_minus_epsilon_value():
    law = ((0.25, 0.75),)
    problem = _problem(3, law, law)
    result = exact_finite_composite_minimax(problem, epsilon=0.2)
    assert result.worst_type_i_error == pytest.approx(0.2, abs=2e-10)
    assert result.minimax_type_ii_error == pytest.approx(0.8, abs=2e-10)


def test_exact_minimax_retains_structural_zeros_without_repair():
    problem = _problem(2, ((1.0, 0.0),), ((0.5, 0.5),))
    result = exact_finite_composite_minimax(problem, epsilon=0.1)
    assert result.outcomes == ((0, 2), (1, 1), (2, 0))
    assert 0 <= result.minimax_type_ii_error <= 1
    assert problem.null.members[0].probabilities == (1.0, 0.0)
    converse = composite_renyi_converse_at_order(problem, epsilon=0.1, order=2.0)
    assert converse.reverse_renyi == math.inf
    assert converse.type_ii_lower_bound == 0.0
    with pytest.raises(CompositeProjectionError, match="full support"):
        verified_composite_renyi_projection(problem, order=0.5)
    assert problem.null.members[0].probabilities == (1.0, 0.0)


def test_exact_composite_enumeration_cap_fails_without_approximation():
    problem = _problem(
        20,
        ((0.2, 0.3, 0.5),),
        ((0.4, 0.2, 0.4),),
    )
    # C(22,2)=231 complete count vectors.
    with pytest.raises(CompositeEnumerationLimitError, match="231.*max_outcomes=100"):
        exact_finite_composite_minimax(problem, epsilon=0.1, max_outcomes=100)


def _ordered_problem(n=4):
    return _problem(
        n,
        ((0.8, 0.2), (0.7, 0.3)),
        ((0.3, 0.7), (0.2, 0.8)),
    )


def test_ordered_family_projection_selects_adjacent_pair_and_verifies_uniform_moments():
    problem = _ordered_problem()
    projection = verified_composite_renyi_projection(problem, order=0.5)
    assert projection.null_member_index == 1
    assert projection.alternative_member_index == 0
    assert projection.null_member_id == "P-1"
    assert projection.alternative_member_id == "Q-0"
    expected_z = 2 * math.sqrt(0.21)
    assert projection.hellinger_integral == pytest.approx(expected_z, abs=2e-15)
    assert projection.maximum_null_moment <= expected_z + 1e-12
    assert projection.maximum_alternative_moment <= expected_z + 1e-12
    assert not projection.finite_n_least_favourable_claimed


def test_projected_closed_form_rule_separates_its_bound_from_constant_test_bound():
    projection = verified_composite_renyi_projection(_ordered_problem(), order=0.5)
    result = projected_composite_bound_at_order(projection, epsilon=0.05)
    assert result.actual_worst_type_i_error <= 0.05 + 1e-12
    assert 0 <= result.actual_worst_type_ii_error <= 1
    # Here r>D_lambda, so the theorem's projected exponential expression is
    # vacuous (>1) and the deterministic threshold rejects no outcomes. The
    # separate constant randomized test gives the global 1-epsilon upper bound.
    assert result.raw_exponential_upper_bound > 1
    assert result.actual_worst_type_ii_error == 1.0
    assert result.type_ii_upper_bound == pytest.approx(0.95)
    assert result.actual_worst_type_ii_error <= result.raw_exponential_upper_bound
    assert result.outcomes == 5


def test_ordered_projected_calibration_exhausts_budget_and_matches_exact_minimax():
    problem = _ordered_problem()
    projection = verified_composite_renyi_projection(problem, order=0.5)
    calibrated = calibrate_composite_projected_test(projection, epsilon=0.05)
    exact = exact_finite_composite_minimax(problem, epsilon=0.05)
    assert calibrated.exhausts_type_i_budget
    assert calibrated.worst_type_i_error == pytest.approx(0.05, abs=2e-12)
    assert calibrated.worst_type_ii_error == pytest.approx(
        exact.minimax_type_ii_error, abs=2e-11
    )
    assert exact.minimax_type_ii_error == pytest.approx(
        0.5317777777777777, abs=2e-11
    )
    assert 0 < calibrated.boundary_randomization < 1


def test_nonordered_finite_family_refuses_pairwise_projection_without_uniform_guarantee():
    problem = _problem(
        2,
        (
            (0.48893532731995315, 0.36528534865685425, 0.14577932402319255),
            (0.6075735673559299, 0.14243110644188922, 0.24999532620218082),
        ),
        (
            (0.2023671236848312, 0.11575760984712621, 0.6818752664680426),
            (0.05866884903608609, 0.6433094511677528, 0.298021699796161),
        ),
    )
    # Pairwise order-1/2 minimization alone is not enough for composite
    # achievability. This class violates the alternative-side uniform moment
    # inequality at its finite-family minimizer and must fail closed.
    with pytest.raises(CompositeProjectionError, match="uniform moment inequalities"):
        verified_composite_renyi_projection(problem, order=0.5)


def test_projection_order_and_composite_converse_order_are_distinct_domains():
    problem = _ordered_problem()
    with pytest.raises(InputValidationError, match="0 < lambda < 1"):
        verified_composite_renyi_projection(problem, order=2.0)
    with pytest.raises(InputValidationError, match="strictly greater than 1"):
        composite_renyi_converse_at_order(problem, epsilon=0.05, order=0.5)
