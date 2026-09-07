"""Exact simple-null likelihood-ratio p-value tests."""

import math

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    ExactPValueEnumerationLimitError,
    SimpleBinaryLawPair,
    UndefinedLikelihoodRatioError,
    likelihood_ratio_p_value,
    log_likelihood_ratio,
)


def _pair(n=4):
    return SimpleBinaryLawPair(
        null=MultinomialMIDLaw(n, (0.25, 0.75)),
        alternative=MultinomialMIDLaw(n, (0.5, 0.5)),
    )


@pytest.mark.parametrize("counts, expected, tail_outcomes", [
    ((4, 0), 0.00390625, 1),
    ((3, 1), 0.05078125, 2),
    ((1, 3), 0.68359375, 4),
    ((0, 4), 1.0, 5),
])
def test_exact_simple_null_llr_p_value(counts, expected, tail_outcomes):
    pair = _pair()
    result = likelihood_ratio_p_value(pair, counts)
    assert result.p_value == pytest.approx(expected, abs=2e-15)
    assert result.log_p_value == pytest.approx(math.log(expected), abs=2e-15)
    assert result.observed_log_likelihood_ratio == pytest.approx(
        log_likelihood_ratio(pair, counts), abs=1e-15,
    )
    assert result.null_outcomes == 5
    assert result.tail_outcomes == tail_outcomes
    assert not result.underflowed


def test_identical_laws_give_exact_p_value_one():
    law = MultinomialMIDLaw(4, (0.25, 0.75))
    result = likelihood_ratio_p_value(SimpleBinaryLawPair(null=law, alternative=law), (3, 1))
    assert result.observed_log_likelihood_ratio == 0.0
    assert result.p_value == 1.0
    assert result.log_p_value == 0.0
    assert result.null_outcomes == result.tail_outcomes == 5


def test_support_boundaries_are_retained_without_smoothing():
    null_impossible = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(2, (1.0, 0.0)),
        alternative=MultinomialMIDLaw(2, (0.5, 0.5)),
    )
    result = likelihood_ratio_p_value(null_impossible, (0, 2))
    assert result.observed_log_likelihood_ratio == math.inf
    assert result.p_value == 0.0 and result.log_p_value == -math.inf

    alternative_impossible = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(2, (0.5, 0.5)),
        alternative=MultinomialMIDLaw(2, (1.0, 0.0)),
    )
    result = likelihood_ratio_p_value(alternative_impossible, (0, 2))
    assert result.observed_log_likelihood_ratio == -math.inf
    assert result.p_value == 1.0 and result.log_p_value == 0.0

    disjoint = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(2, (1.0, 0.0, 0.0)),
        alternative=MultinomialMIDLaw(2, (0.0, 1.0, 0.0)),
    )
    with pytest.raises(UndefinedLikelihoodRatioError):
        likelihood_ratio_p_value(disjoint, (0, 0, 2))


def test_exact_enumeration_limit_fails_without_approximation():
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(100, (1 / 3, 1 / 3, 1 / 3)),
        alternative=MultinomialMIDLaw(100, (0.2, 0.3, 0.5)),
    )
    with pytest.raises(ExactPValueEnumerationLimitError, match="no asymptotic approximation"):
        likelihood_ratio_p_value(pair, (34, 33, 33), max_outcomes=100)


@pytest.mark.parametrize("max_outcomes", [0, -1, True, 1.5, None])
def test_enumeration_limit_must_be_a_positive_integer(max_outcomes):
    with pytest.raises(InputValidationError):
        likelihood_ratio_p_value(_pair(), (3, 1), max_outcomes=max_outcomes)


def test_exact_p_value_supports_independent_product_blocks():
    pair = SimpleBinaryLawPair(
        null=(MultinomialMIDLaw(1, (0.25, 0.75)), MultinomialMIDLaw(1, (0.75, 0.25))),
        alternative=(MultinomialMIDLaw(1, (0.5, 0.5)), MultinomialMIDLaw(1, (0.25, 0.75))),
        independent=True,
    )
    observed = ((1, 0), (0, 1))
    result = likelihood_ratio_p_value(pair, observed)
    assert result.p_value == pytest.approx(0.0625, abs=1e-15)
    assert result.null_outcomes == 4
    assert result.tail_outcomes == 1
