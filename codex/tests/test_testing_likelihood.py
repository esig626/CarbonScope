"""Independent likelihood-ratio checks, including joint support and cancellation."""

from dataclasses import replace
import math

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import MIDCountObservation, MultinomialMIDLaw
from fluxemu.testing import (
    SimpleBinaryLawPair,
    UndefinedLikelihoodRatioError,
    log_likelihood_ratio,
)


def _pair(n=4):
    return SimpleBinaryLawPair(
        null=MultinomialMIDLaw(n, (0.25, 0.75)),
        alternative=MultinomialMIDLaw(n, (0.5, 0.5)),
    )


@pytest.mark.parametrize("counts", [(4, 0), (3, 1), (1, 3), (0, 4)])
def test_one_block_llr_has_log_p1_minus_log_p0_direction(counts):
    pair = _pair()
    coefficient = math.factorial(4) / math.prod(math.factorial(k) for k in counts)
    p0 = coefficient * 0.25 ** counts[0] * 0.75 ** counts[1]
    p1 = coefficient * 0.5 ** sum(counts)
    expected = math.log(p1 / p0)
    assert log_likelihood_ratio(pair, counts) == pytest.approx(expected, abs=1e-15)
    assert log_likelihood_ratio(pair, MIDCountObservation(counts, 4)) == pytest.approx(expected, abs=1e-15)
    assert log_likelihood_ratio(pair, (value for value in counts)) == pytest.approx(expected, abs=1e-15)
    swapped = SimpleBinaryLawPair(null=pair.alternative, alternative=pair.null)
    assert log_likelihood_ratio(swapped, counts) == pytest.approx(-expected, abs=1e-15)


def test_identical_laws_have_zero_llr_including_exact_zero_cells():
    law = MultinomialMIDLaw(7, (0.25, 0.0, 0.75))
    pair = SimpleBinaryLawPair(null=law, alternative=law)
    assert log_likelihood_ratio(pair, (2, 0, 5)) == 0.0


@pytest.mark.parametrize("null,alternative,counts,expected", [
    ((1.0, 0.0), (0.5, 0.5), (0, 2), math.inf),
    ((0.5, 0.5), (1.0, 0.0), (0, 2), -math.inf),
    ((1.0, 0.0), (0.0, 1.0), (2, 0), -math.inf),
    ((1.0, 0.0), (0.0, 1.0), (0, 2), math.inf),
])
def test_support_mismatch_retains_correct_infinite_llr_without_theorem_gate(null, alternative, counts, expected):
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(2, null), alternative=MultinomialMIDLaw(2, alternative),
    )
    assert not pair.mutually_absolutely_continuous
    assert log_likelihood_ratio(pair, counts) == expected
    assert log_likelihood_ratio(pair, MIDCountObservation(counts, 2)) == expected


def test_support_mismatch_can_still_have_a_finite_realised_llr():
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(2, (1.0, 0.0)),
        alternative=MultinomialMIDLaw(2, (0.5, 0.5)),
    )
    assert log_likelihood_ratio(pair, (2, 0)) == pytest.approx(math.log(0.25))


@pytest.mark.parametrize("counts", [(1, 0, 1), (1, 0, 0), (0, 0, 0)])
def test_common_zero_probability_or_wrong_total_is_precisely_undefined(counts):
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(2, (0.25, 0.75, 0.0)),
        alternative=MultinomialMIDLaw(2, (0.5, 0.5, 0.0)),
    )
    with pytest.raises(UndefinedLikelihoodRatioError, match="probability zero under both null P0 and alternative P1"):
        log_likelihood_ratio(pair, counts)


def test_independent_products_preserve_distinct_totals_and_declared_observation_order():
    pair = SimpleBinaryLawPair(
        null=(MultinomialMIDLaw(2, (0.25, 0.75)), MultinomialMIDLaw(3, (0.75, 0.25))),
        alternative=(MultinomialMIDLaw(2, (0.5, 0.5)), MultinomialMIDLaw(3, (0.25, 0.75))),
        independent=True,
    )
    samples = (MIDCountObservation((1, 1), 2), MIDCountObservation((1, 2), 3))
    null_probability = (2 * 0.25 * 0.75) * (3 * 0.75 * 0.25 ** 2)
    alternative_probability = (2 * 0.5 ** 2) * (3 * 0.25 * 0.75 ** 2)
    expected = math.log(alternative_probability / null_probability)
    assert log_likelihood_ratio(pair, samples) == pytest.approx(expected, abs=1e-15)
    assert log_likelihood_ratio(pair, tuple(sample.counts for sample in samples)) == pytest.approx(expected, abs=1e-15)
    with pytest.raises(UndefinedLikelihoodRatioError):
        log_likelihood_ratio(pair, samples[::-1])


def test_product_joint_zero_zero_is_undefined_even_when_per_block_llrs_have_opposite_infinities():
    left, right = MultinomialMIDLaw(1, (1.0, 0.0)), MultinomialMIDLaw(1, (0.0, 1.0))
    pair = SimpleBinaryLawPair(null=(left, right), alternative=(right, left), independent=True)
    with pytest.raises(UndefinedLikelihoodRatioError, match="complete observation laws"):
        log_likelihood_ratio(pair, ((1, 0), (1, 0)))
    assert log_likelihood_ratio(pair, ((0, 1), (1, 0))) == math.inf
    assert log_likelihood_ratio(pair, ((1, 0), (0, 1))) == -math.inf


def test_count_weighted_llr_retains_small_signal_lost_in_large_log_pmf_subtraction():
    n = 10**15
    delta = math.nextafter(0.5, math.inf) - 0.5
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(n, (0.5, 0.5)),
        alternative=MultinomialMIDLaw(n, (0.5 + delta, 0.5 - delta)),
    )
    counts = (n // 2, n // 2)
    # Independently cancel the symmetric probability product analytically.
    expected = 0.5 * n * math.log1p(-4 * delta**2)
    assert expected < 0
    assert pair.alternative.log_pmf(counts) - pair.null.log_pmf(counts) == 0.0
    assert log_likelihood_ratio(pair, counts) == pytest.approx(expected, rel=2e-14, abs=0.0)


@pytest.mark.parametrize("counts", [(1.0, 3.0), (True, 3), (-1, 5), (4,), {0: 1, 1: 3}, {1, 3}, None])
def test_llr_does_not_convert_or_repair_malformed_counts(counts):
    with pytest.raises(InputValidationError):
        log_likelihood_ratio(_pair(), counts)


@pytest.mark.parametrize("samples", [((1, 0),), [(1, 0), (0, 1)], {(1, 0), (0, 1)}, None])
def test_product_requires_exact_declared_block_cardinality_and_order(samples):
    pair = SimpleBinaryLawPair(
        null=(MultinomialMIDLaw(1, (0.25, 0.75)),) * 2,
        alternative=(MultinomialMIDLaw(1, (0.5, 0.5)),) * 2,
        independent=True,
    )
    with pytest.raises(InputValidationError, match="immutable tuple"):
        log_likelihood_ratio(pair, samples)


def test_a_later_malformed_product_block_is_validated_after_an_impossible_earlier_block():
    pair = SimpleBinaryLawPair(
        null=(MultinomialMIDLaw(1, (1.0, 0.0)),) * 2,
        alternative=(MultinomialMIDLaw(1, (0.0, 1.0)),) * 2,
        independent=True,
    )
    with pytest.raises(InputValidationError, match="nonnegative integer"):
        log_likelihood_ratio(pair, ((1, 0), (0.0, 1.0)))


def test_invalid_law_holder_does_not_reach_count_evaluation():
    with pytest.raises(InputValidationError, match="SimpleBinaryLawPair"):
        log_likelihood_ratio(object(), (1, 3))


def test_explicit_one_block_product_retains_product_observation_shape():
    single = _pair()
    product = replace(single, null=(single.null,), alternative=(single.alternative,), independent=True)
    assert log_likelihood_ratio(product, ((1, 3),)) == log_likelihood_ratio(single, (1, 3))
    with pytest.raises(InputValidationError, match="block order"):
        log_likelihood_ratio(product, (1, 3))
