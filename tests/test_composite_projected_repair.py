"""Adversarial controls for honest finite-family projected-score certification."""
from decimal import Decimal, localcontext
import math

import pytest

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    CompositeBinaryTestingProblem,
    CompositeMIDLawFamily,
    CompositeScoreVerificationError,
    IndependentMIDProductLaw,
    calibrate_composite_score_test,
    composite_renyi_score_candidate,
    composite_score_bound_at_order,
    evaluate_composite_score_test,
    exact_finite_composite_minimax,
    verified_composite_renyi_score,
)


def _problem(null, alternative):
    def family(rows):
        return CompositeMIDLawFamily(members=tuple(
            IndependentMIDProductLaw(blocks=(MultinomialMIDLaw(1, tuple(row)),))
            for row in rows
        ))
    return CompositeBinaryTestingProblem(null=family(null), alternative=family(alternative))


def _independent_log_moment(probabilities, p_star, q_star, exponent):
    with localcontext() as ctx:
        ctx.prec = 80
        exponent = Decimal.from_float(exponent)
        result = sum(
            Decimal.from_float(probability)
            * (exponent * (Decimal.from_float(q).ln() - Decimal.from_float(p).ln())).exp()
            for probability, p, q in zip(probabilities, p_star, q_star, strict=True)
        )
        return result.ln()


@pytest.mark.parametrize("order,tolerance", [(1e-12, 1e-10), (0.5, 2.0)])
def test_tolerance_cannot_certify_a_false_uniform_moment_inequality(order, tolerance):
    problem = _problem(((0.9, 0.1), (0.1, 0.9)), ((0.5, 0.5),))
    candidate = composite_renyi_score_candidate(problem, order=order, tolerance=tolerance)
    p = problem.null.members[candidate.null_member_index].blocks[0].probabilities
    q = problem.alternative.members[candidate.alternative_member_index].blocks[0].probabilities
    selected = _independent_log_moment(p, p, q, order)
    worst = max(_independent_log_moment(law.blocks[0].probabilities, p, q, order)
                for law in problem.null.members)
    assert worst > selected + Decimal("1e-12")
    assert not candidate.uniform_moment_bounds_verified
    with pytest.raises(CompositeScoreVerificationError):
        verified_composite_renyi_score(problem, order=order, tolerance=tolerance)


def test_disjoint_support_analytical_score_does_not_reject_null_exclusive_outcomes():
    problem = _problem(((1.0, 0.0),), ((0.0, 1.0),))
    candidate = verified_composite_renyi_score(problem, order=0.5)
    evaluated = evaluate_composite_score_test(composite_score_bound_at_order(candidate, epsilon=0.1))
    assert evaluated.worst_type_i_error == 0.0
    assert evaluated.worst_type_ii_error == 0.0
    assert evaluated.rejection_probabilities == (1.0, 0.0)


def test_calibrated_score_reports_tiny_positive_complement_mass_directly():
    problem = _problem(((0.5, 0.5),), ((1.0, 1e-20),))
    candidate = verified_composite_renyi_score(problem, order=0.5)
    result = calibrate_composite_score_test(candidate, epsilon=0.5)
    assert result.rejection_probabilities == (0.0, 1.0)
    assert result.worst_type_ii_error > 0
    assert result.worst_type_ii_error == pytest.approx(1e-20, rel=1e-13, abs=0)


def test_finite_positive_likelihood_ratios_produce_finite_score_coordinates():
    problem = _problem(((1e-320, 1.0),), ((1.0, 1e-320),))
    candidate = composite_renyi_score_candidate(problem, order=0.5)
    assert all(math.isfinite(score) for score in candidate.score_blocks[0])
    assert candidate.uniform_moment_bounds_verified


def test_verified_projected_score_is_strictly_suboptimal_to_unrestricted_minimax():
    # Rational n=1 categorical example found with seed 382927, then rounded
    # and rechecked. The score's uniform moments pass; finite minimax still
    # chooses a better decision than its calibrated upper-score thresholds.
    problem = _problem(
        ((0.606, 0.010, 0.384), (0.425, 0.171, 0.404)),
        ((0.948, 0.050, 0.002), (0.517, 0.419, 0.064)),
    )
    candidate = verified_composite_renyi_score(problem, order=0.5)
    projected = calibrate_composite_score_test(candidate, epsilon=0.2)
    exact = exact_finite_composite_minimax(problem, epsilon=0.2)
    assert projected.worst_type_i_error == pytest.approx(0.2, abs=2e-14)
    assert projected.worst_type_ii_error == pytest.approx(0.8853129411764706, abs=2e-13)
    assert exact.minimax_type_ii_error == pytest.approx(0.6778133486027607, abs=2e-10)
    assert projected.worst_type_ii_error - exact.minimax_type_ii_error > 0.2
    # Independently map each n=1 count outcome to its categorical coordinate.
    for family, reported, reject in (
        (problem.null, projected.null_type_i_errors, True),
        (problem.alternative, projected.alternative_type_ii_errors, False),
    ):
        direct = tuple(math.fsum(
            law.blocks[0].probabilities[outcome[0].index(1)] * (phi if reject else 1 - phi)
            for outcome, phi in zip(projected.outcomes, projected.rejection_probabilities, strict=True)
        ) for law in family.members)
        assert reported == pytest.approx(direct, abs=2e-14)
    assert not candidate.finite_n_least_favourable_claimed
    assert not candidate.joint_convex_projection_claimed


def test_product_zero_and_infinite_moments_do_not_mask_undefined_supported_scores():
    def product_law(first, second):
        return IndependentMIDProductLaw(blocks=(
            MultinomialMIDLaw(1, first), MultinomialMIDLaw(1, second),
        ))
    p = product_law((1.0, 0.0), (1.0, 0.0))
    cross = product_law((1.0, 0.0), (0.0, 1.0))
    q = product_law((0.0, 1.0), (0.0, 1.0))
    problem = CompositeBinaryTestingProblem(
        null=CompositeMIDLawFamily(members=(p, cross)),
        alternative=CompositeMIDLawFamily(members=(q,)),
    )
    candidate = composite_renyi_score_candidate(problem, order=0.5)
    # The first selected pair has h=-infinity+infinity on the cross law.
    # A different tied candidate may be well-defined; whichever is returned,
    # certification must imply its enumerated score really is defined.
    if candidate.uniform_moment_bounds_verified:
        evaluated = evaluate_composite_score_test(composite_score_bound_at_order(candidate, epsilon=0.1))
        assert evaluated.worst_type_i_error <= 0.1
        assert evaluated.worst_type_ii_error == 0
    else:
        assert any("support" in message or "undefined" in message for message in candidate.verification_failures)


def test_failed_moment_conditions_still_allow_direct_score_calibration():
    problem = _problem(((0.9, 0.1), (0.1, 0.9)), ((0.5, 0.5),))
    candidate = composite_renyi_score_candidate(problem, order=0.5)
    assert not candidate.uniform_moment_bounds_verified
    with pytest.raises(CompositeScoreVerificationError):
        composite_score_bound_at_order(candidate, epsilon=0.2)
    calibrated = calibrate_composite_score_test(candidate, epsilon=0.2)
    assert calibrated.worst_type_i_error == pytest.approx(0.2, abs=2e-14)
    assert calibrated.worst_type_ii_error == pytest.approx(8 / 9, abs=2e-14)
    assert not calibrated.candidate.uniform_moment_bounds_verified


def test_positive_achieved_error_below_float_resolution_is_explicitly_refused():
    from fluxemu.testing import NumericalLimitError
    problem = _problem(((1.0, 0.0),), ((1e-320, 1.0),))
    candidate = verified_composite_renyi_score(problem, order=0.5)
    # The exact achieved beta is about 1e-330. Returning zero would claim
    # perfect separation of laws that share positive support.
    with pytest.raises(NumericalLimitError, match='underflow|resolution'):
        calibrate_composite_score_test(candidate, epsilon=1.0 - 1e-10)
