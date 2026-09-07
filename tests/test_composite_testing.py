"""Finite joint-product composite minimax, converse, score, and calibration tests."""

from __future__ import annotations

import math

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import MultinomialMIDLaw, independent_product_renyi
from fluxemu.testing import (
    CompositeBinaryTestingProblem,
    CompositeEnumerationLimitError,
    CompositeMIDLawFamily,
    CompositeScoreVerificationError,
    IndependentMIDProductLaw,
    calibrate_composite_score_test,
    composite_renyi_converse_at_order,
    composite_renyi_score_candidate,
    composite_score_bound_at_order,
    evaluate_composite_score_test,
    exact_finite_composite_minimax,
    verified_composite_renyi_score,
)


ONE_BLOCK_ID = (("experiment", "target", "counts"),)


def _product_law(*blocks, identities=None):
    laws = tuple(MultinomialMIDLaw(n, tuple(probabilities)) for n, probabilities in blocks)
    return IndependentMIDProductLaw(
        blocks=laws,
        block_identities=identities or tuple(
            ("experiment", f"target-{index}", "counts")
            for index in range(len(laws))
        ),
    )


def _family(members, prefix):
    products = tuple(members)
    return CompositeMIDLawFamily(
        members=products,
        member_ids=tuple(f"{prefix}-{index}" for index in range(len(products))),
    )


def _one_block_problem(n, null, alternative):
    null_members = tuple(
        _product_law((n, probabilities), identities=ONE_BLOCK_ID)
        for probabilities in null
    )
    alternative_members = tuple(
        _product_law((n, probabilities), identities=ONE_BLOCK_ID)
        for probabilities in alternative
    )
    return CompositeBinaryTestingProblem(
        null=_family(null_members, "P"),
        alternative=_family(alternative_members, "Q"),
    )


def test_product_law_requires_ordered_matching_block_semantics():
    law = _product_law((2, (0.5, 0.5)), (3, (0.25, 0.75)))
    assert law.block_totals == (2, 3)
    assert law.block_mass_classes == ((0, 1), (0, 1))
    assert len(law.fingerprint) == 64
    with pytest.raises(InputValidationError, match="identify every"):
        IndependentMIDProductLaw(
            blocks=law.blocks,
            block_identities=(("only", "one", "identity"),),
        )


def test_composite_family_preserves_declared_product_members_without_convexifying():
    family = _family(
        (
            _product_law((4, (0.8, 0.2)), identities=ONE_BLOCK_ID),
            _product_law((4, (0.7, 0.3)), identities=ONE_BLOCK_ID),
        ),
        "P",
    )
    assert len(family.members) == 2
    assert family.member_ids == ("P-0", "P-1")
    assert tuple(member.blocks[0].probabilities for member in family.members) == (
        (0.8, 0.2), (0.7, 0.3),
    )
    assert family.block_count == 1
    assert len(family.fingerprint) == 64


def test_composite_problem_rejects_misaligned_product_block_structure():
    null = _family(
        (_product_law((2, (0.8, 0.2)), identities=ONE_BLOCK_ID),), "P"
    )
    different_total = _family(
        (_product_law((3, (0.2, 0.8)), identities=ONE_BLOCK_ID),), "Q"
    )
    with pytest.raises(InputValidationError, match="observation-block structure"):
        CompositeBinaryTestingProblem(null=null, alternative=different_total)

    different_identity = _family(
        (
            _product_law(
                (2, (0.2, 0.8)),
                identities=(("other", "target", "counts"),),
            ),
        ),
        "Q",
    )
    with pytest.raises(InputValidationError, match="observation-block structure"):
        CompositeBinaryTestingProblem(null=null, alternative=different_identity)


def test_singleton_product_composite_converse_recovers_mixer_reverse_bound():
    problem = _one_block_problem(4, ((0.25, 0.75),), ((0.5, 0.5),))
    result = composite_renyi_converse_at_order(problem, epsilon=0.05, order=2.0)
    expected_divergence = 4 * math.log(4 / 3)
    assert result.reverse_renyi == pytest.approx(expected_divergence, abs=3e-14)
    assert result.type_ii_lower_bound == pytest.approx(0.602476804, abs=5e-10)
    assert result.null_member_id == "P-0"
    assert result.alternative_member_id == "Q-0"
    assert not result.global_order_envelope_evaluated


def test_multiblock_composite_converse_uses_full_product_renyi_additivity():
    identities = (("e", "a", "r"), ("e", "b", "r"))
    null_law = _product_law(
        (2, (0.5, 0.5)), (3, (1.0, 0.0)), identities=identities,
    )
    alternative_law = _product_law(
        (2, (0.25, 0.75)), (3, (1.0, 0.0)), identities=identities,
    )
    problem = CompositeBinaryTestingProblem(
        null=_family((null_law,), "P"),
        alternative=_family((alternative_law,), "Q"),
    )
    result = composite_renyi_converse_at_order(problem, epsilon=0.05, order=2)
    expected = independent_product_renyi(alternative_law.blocks, null_law.blocks, 2)
    assert result.reverse_renyi == pytest.approx(expected, abs=2e-14)
    assert expected == pytest.approx(2 * math.log(1.25), abs=2e-14)


def test_exact_two_by_two_finite_composite_minimax_is_point_six():
    problem = _one_block_problem(
        1,
        ((0.9, 0.1), (0.8, 0.2)),
        ((0.1, 0.9), (0.2, 0.8)),
    )
    result = exact_finite_composite_minimax(problem, epsilon=0.1)
    assert result.outcomes == (((0, 1),), ((1, 0),))
    assert result.minimax_type_ii_error == pytest.approx(0.6, abs=2e-12)
    assert result.worst_type_i_error <= 0.1 + 1e-12
    assert result.randomised
    assert max(result.null_type_i_errors) == pytest.approx(0.1, abs=2e-12)
    assert max(result.alternative_type_ii_errors) == pytest.approx(0.6, abs=2e-12)


def test_adding_an_identical_independent_nuisance_block_does_not_change_exact_minimax_value():
    identities = (("e", "signal", "r"), ("e", "nuisance", "r"))
    null = tuple(
        _product_law(
            (1, p), (1, (0.4, 0.6)), identities=identities,
        )
        for p in ((0.9, 0.1), (0.8, 0.2))
    )
    alternative = tuple(
        _product_law(
            (1, q), (1, (0.4, 0.6)), identities=identities,
        )
        for q in ((0.1, 0.9), (0.2, 0.8))
    )
    problem = CompositeBinaryTestingProblem(
        null=_family(null, "P"), alternative=_family(alternative, "Q")
    )
    result = exact_finite_composite_minimax(problem, epsilon=0.1)
    assert len(result.outcomes) == 4
    assert result.minimax_type_ii_error == pytest.approx(0.6, abs=3e-11)


def test_identical_composite_product_law_has_exact_one_minus_epsilon_value():
    law = _product_law((3, (0.25, 0.75)), identities=ONE_BLOCK_ID)
    problem = CompositeBinaryTestingProblem(
        null=_family((law,), "P"), alternative=_family((law,), "Q")
    )
    result = exact_finite_composite_minimax(problem, epsilon=0.2)
    assert result.worst_type_i_error == pytest.approx(0.2, abs=2e-10)
    assert result.minimax_type_ii_error == pytest.approx(0.8, abs=2e-10)


def test_exact_minimax_retains_product_structural_zeros_without_repair():
    problem = _one_block_problem(2, ((1.0, 0.0),), ((0.5, 0.5),))
    result = exact_finite_composite_minimax(problem, epsilon=0.1)
    assert result.outcomes == (((0, 2),), ((1, 1),), ((2, 0),))
    assert 0 <= result.minimax_type_ii_error <= 1
    assert problem.null.members[0].blocks[0].probabilities == (1.0, 0.0)


def test_exact_joint_composite_enumeration_cap_fails_without_approximation():
    identities = (("e", "a", "r"), ("e", "b", "r"))
    p = _product_law(
        (2, (0.5, 0.5)), (2, (0.5, 0.5)), identities=identities,
    )
    q = _product_law(
        (2, (0.25, 0.75)), (2, (0.25, 0.75)), identities=identities,
    )
    problem = CompositeBinaryTestingProblem(
        null=_family((p,), "P"), alternative=_family((q,), "Q")
    )
    # Three count outcomes per binary n=2 block -> 9 joint outcomes.
    with pytest.raises(CompositeEnumerationLimitError, match="max_outcomes=8"):
        exact_finite_composite_minimax(problem, epsilon=0.1, max_outcomes=8)


def _ordered_problem(n=4):
    return _one_block_problem(
        n,
        ((0.8, 0.2), (0.7, 0.3)),
        ((0.3, 0.7), (0.2, 0.8)),
    )


def test_ordered_family_candidate_selects_adjacent_pair_and_verifies_uniform_moments():
    problem = _ordered_problem()
    candidate = composite_renyi_score_candidate(problem, order=0.5)
    assert candidate.null_member_index == 1
    assert candidate.alternative_member_index == 0
    assert candidate.null_member_id == "P-1"
    assert candidate.alternative_member_id == "Q-0"
    expected_single_z = 2 * math.sqrt(0.21)
    assert math.exp(candidate.log_hellinger_integral) == pytest.approx(
        expected_single_z**4, abs=3e-15
    )
    assert candidate.maximum_log_null_moment <= candidate.log_hellinger_integral + 1e-12
    assert candidate.maximum_log_alternative_moment <= candidate.log_hellinger_integral + 1e-12
    assert candidate.uniform_moment_bounds_verified
    assert not candidate.finite_n_least_favourable_claimed
    assert not candidate.joint_convex_projection_claimed


def test_structural_zero_support_can_still_produce_a_verified_candidate_score():
    problem = _one_block_problem(1, ((1.0, 0.0),), ((0.5, 0.5),))
    candidate = verified_composite_renyi_score(problem, order=0.5)
    assert candidate.uniform_moment_bounds_verified
    assert candidate.score_blocks[0][1] == math.inf
    assert candidate.verification_failures == ()


def test_shared_selected_zero_used_by_another_member_fails_candidate_support_gate():
    problem = _one_block_problem(
        1,
        ((0.7, 0.3, 0.0), (0.6, 0.3, 0.1)),
        ((0.3, 0.7, 0.0),),
    )
    candidate = composite_renyi_score_candidate(problem, order=0.5)
    assert not candidate.uniform_moment_bounds_verified
    assert any("p*=q*=0" in item for item in candidate.verification_failures)
    with pytest.raises(CompositeScoreVerificationError, match="candidate score"):
        verified_composite_renyi_score(problem, order=0.5)


def test_score_bound_is_analytical_and_does_not_require_joint_outcome_enumeration():
    candidate = verified_composite_renyi_score(_ordered_problem(), order=0.5)
    bound = composite_score_bound_at_order(candidate, epsilon=0.05)
    assert bound.raw_exponential_upper_bound > 1
    assert bound.constant_randomised_upper_bound == pytest.approx(0.95)
    assert bound.minimax_type_ii_upper_bound == pytest.approx(0.95)


def test_enumerated_score_rule_separates_its_error_from_constant_test_bound():
    candidate = verified_composite_renyi_score(_ordered_problem(), order=0.5)
    bound = composite_score_bound_at_order(candidate, epsilon=0.05)
    evaluated = evaluate_composite_score_test(bound)
    assert evaluated.worst_type_i_error <= 0.05 + 1e-12
    assert evaluated.worst_type_ii_error == pytest.approx(1.0)
    assert evaluated.worst_type_ii_error <= bound.raw_exponential_upper_bound
    assert len(evaluated.outcomes) == 5


def test_ordered_score_calibration_exhausts_budget_and_matches_exact_minimax():
    problem = _ordered_problem()
    candidate = verified_composite_renyi_score(problem, order=0.5)
    calibrated = calibrate_composite_score_test(candidate, epsilon=0.05)
    exact = exact_finite_composite_minimax(problem, epsilon=0.05)
    assert calibrated.exhausts_type_i_budget
    assert calibrated.worst_type_i_error == pytest.approx(0.05, abs=2e-12)
    assert calibrated.worst_type_ii_error == pytest.approx(
        exact.minimax_type_ii_error, abs=2e-11
    )
    assert exact.minimax_type_ii_error == pytest.approx(
        0.5317777777777777, abs=2e-11
    )
    assert 0 < calibrated.boundary_randomisation < 1


def test_nonordered_finite_family_reports_pairwise_candidate_but_refuses_uniform_certification():
    problem = _one_block_problem(
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
    candidate = composite_renyi_score_candidate(problem, order=0.5)
    assert not candidate.uniform_moment_bounds_verified
    assert candidate.verification_failures
    with pytest.raises(CompositeScoreVerificationError, match="uniform composite"):
        verified_composite_renyi_score(problem, order=0.5)


def test_score_and_converse_orders_are_distinct_domains():
    problem = _ordered_problem()
    with pytest.raises(InputValidationError, match="0 < lambda < 1"):
        composite_renyi_score_candidate(problem, order=2.0)
    with pytest.raises(InputValidationError, match="strictly greater than 1"):
        composite_renyi_converse_at_order(problem, epsilon=0.05, order=0.5)
