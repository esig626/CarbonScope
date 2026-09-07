"""Independent finite-support probability oracles for the count-law layer.

Expected probability masses use exact rational arithmetic and integer
factorials. Expected divergences use those masses and Decimal logarithms,
without consulting the production PMF or MID-divergence implementation.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import math

import pytest

from fluxemu.mfa import kl_divergence
from fluxemu.observation import (
    MIDCountObservation,
    MultinomialMIDLaw,
    independent_product_kl,
    independent_product_renyi,
    kl_multinomial,
    multinomial_count_constant,
    renyi_multinomial,
)


ORDERS = (0.5, 0.73, 1.0, 1.3, 2.0)
PROBABILITIES = (
    (1.0,),
    (0.5, 0.5),
    (0.25, 0.75),
    (0.0, 1.0),
    (0.25, 0.25, 0.5),
    (0.0, 0.5, 0.5),
    (0.125, 0.125, 0.25, 0.5),
    (0.0, 0.0, 0.25, 0.75),
)
PAIRS = (
    ((0.25, 0.75), (0.5, 0.5)),
    ((0.5, 0.5), (0.25, 0.75)),
    ((0.0, 0.25, 0.75), (0.0, 0.25, 0.75)),
    ((0.5, 0.5, 0.0), (0.0, 0.5, 0.5)),
    ((1.0, 0.0), (0.25, 0.75)),
    ((0.25, 0.75), (1.0, 0.0)),
    ((1.0, 0.0), (0.0, 1.0)),
    ((0.125, 0.375, 0.0, 0.5), (0.25, 0.25, 0.25, 0.25)),
)


def _count_space(total, dimension):
    """Enumerate every ordered nonnegative composition, including zeros."""
    if dimension == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remaining in _count_space(total - first, dimension - 1):
            yield (first, *remaining)


def _exact_distribution(total, probabilities):
    fractions = tuple(Fraction(value) for value in probabilities)
    assert sum(fractions) == 1
    masses = {}
    for counts in _count_space(total, len(fractions)):
        coefficient = Fraction(
            math.factorial(total),
            math.prod(math.factorial(count) for count in counts),
        )
        masses[counts] = coefficient * math.prod(
            probability ** count
            for probability, count in zip(fractions, counts, strict=True)
        )
    assert sum(masses.values()) == 1
    return masses


def _decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _oracle_divergence(p, q, alpha):
    """Evaluate definitions on complete finite laws, preserving exact zeros."""
    assert p.keys() == q.keys()
    assert sum(p.values()) == sum(q.values()) == 1
    if alpha >= 1 and any(p[key] > 0 and q[key] == 0 for key in p):
        return math.inf
    with localcontext() as context:
        context.prec = 80
        if alpha == 1:
            return float(sum(
                _decimal(p[key]) * (_decimal(p[key]) / _decimal(q[key])).ln()
                for key in p if p[key] > 0
            ))
        order = Decimal.from_float(alpha)
        mass = sum(
            (order * _decimal(p[key]).ln()
             + (1 - order) * _decimal(q[key]).ln()).exp()
            for key in p if p[key] > 0 and q[key] > 0
        )
        return math.inf if mass == 0 else float(mass.ln() / (order - 1))


def _assert_close_or_infinite(actual, expected):
    if expected == math.inf:
        assert actual == math.inf
    else:
        assert math.isfinite(actual)
        assert actual == pytest.approx(expected, rel=3e-13, abs=3e-14)


@pytest.mark.parametrize("total", range(1, 7))
@pytest.mark.parametrize("probabilities", PROBABILITIES)
def test_every_small_support_mass_matches_exact_factorials_and_sums_to_one(total, probabilities):
    law = MultinomialMIDLaw(total, probabilities)
    exact = _exact_distribution(total, probabilities)
    actual_masses = []
    for counts, expected in exact.items():
        log_mass = law.log_pmf(counts)
        if expected == 0:
            assert log_mass == -math.inf
            actual_masses.append(0.0)
        else:
            with localcontext() as context:
                context.prec = 80
                expected_log_mass = float(_decimal(expected).ln())
            assert log_mass == pytest.approx(expected_log_mass, rel=3e-13, abs=3e-14)
            actual_masses.append(math.exp(log_mass))
    assert len(exact) == math.comb(total + len(probabilities) - 1, len(probabilities) - 1)
    assert math.fsum(actual_masses) == pytest.approx(1.0, abs=3e-14)


@pytest.mark.parametrize("counts", [(0, 0), (1, 1), (3, 3), (20, 0)])
def test_well_formed_counts_outside_the_fixed_total_have_exact_zero_probability(counts):
    assert MultinomialMIDLaw(5, (0.25, 0.75)).log_pmf(counts) == -math.inf


@pytest.mark.parametrize("total", (1, 2, 4))
@pytest.mark.parametrize("p,q", PAIRS)
@pytest.mark.parametrize("alpha", ORDERS)
def test_law_divergences_match_independent_complete_count_support(total, p, q, alpha):
    observed = MultinomialMIDLaw(total, p)
    predicted = MultinomialMIDLaw(total, q)
    p_law, q_law = _exact_distribution(total, p), _exact_distribution(total, q)
    expected = _oracle_divergence(p_law, q_law, alpha)
    actual = renyi_multinomial(observed, predicted, alpha)
    _assert_close_or_infinite(actual, expected)
    # This also verifies the mathematical n-scaling independently of production
    # evaluation: the one-trial oracle is a categorical law on a different space.
    categorical = _oracle_divergence(_exact_distribution(1, p), _exact_distribution(1, q), alpha)
    _assert_close_or_infinite(actual, total * categorical)
    if alpha == 1:
        assert actual == kl_multinomial(observed, predicted)


@pytest.mark.parametrize("alpha", ORDERS)
@pytest.mark.parametrize("p2,q2", (
    ((0.25, 0.25, 0.5), (0.5, 0.25, 0.25)),
    ((0.5, 0.5, 0.0), (0.0, 0.5, 0.5)),
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
))
def test_independent_product_with_different_block_totals_matches_full_product_law(alpha, p2, q2):
    p1, q1 = (0.25, 0.75), (0.5, 0.5)
    p_blocks = (_exact_distribution(2, p1), _exact_distribution(3, p2))
    q_blocks = (_exact_distribution(2, q1), _exact_distribution(3, q2))
    support = tuple(product(p_blocks[0], p_blocks[1]))
    joint_p = {(k1, k2): p_blocks[0][k1] * p_blocks[1][k2] for k1, k2 in support}
    joint_q = {(k1, k2): q_blocks[0][k1] * q_blocks[1][k2] for k1, k2 in support}
    expected = _oracle_divergence(joint_p, joint_q, alpha)
    p_laws = (MultinomialMIDLaw(2, p1), MultinomialMIDLaw(3, p2))
    q_laws = (MultinomialMIDLaw(2, q1), MultinomialMIDLaw(3, q2))
    actual = independent_product_renyi(p_laws, q_laws, alpha)
    _assert_close_or_infinite(actual, expected)
    _assert_close_or_infinite(actual, math.fsum(
        _oracle_divergence(left, right, alpha)
        for left, right in zip(p_blocks, q_blocks, strict=True)
    ))
    if alpha == 1:
        assert actual == independent_product_kl(p_laws, q_laws)


@pytest.mark.parametrize("counts,probabilities", (
    ((2, 3), (0.25, 0.75)),
    ((0, 4), (0.25, 0.75)),
    ((1, 0, 4), (0.25, 0.0, 0.75)),
    ((2, 3, 1), (0.25, 0.25, 0.5)),
    ((0, 0, 7), (0.0, 0.0, 1.0)),
    ((2, 0, 3), (0.0, 0.25, 0.75)),
))
def test_count_likelihood_kl_identity_with_data_only_constant(counts, probabilities):
    total = sum(counts)
    observation = MIDCountObservation(counts, total)
    law = MultinomialMIDLaw(total, probabilities)
    with localcontext() as context:
        context.prec = 80
        expected_constant = float(
            -Decimal(math.factorial(total)).ln()
            + sum(Decimal(math.factorial(count)).ln() for count in counts)
            - sum(Decimal(count) * (Decimal(count) / Decimal(total)).ln()
                  for count in counts if count > 0)
        )
    assert multinomial_count_constant(observation) == pytest.approx(expected_constant, abs=3e-14)
    empirical = tuple(count / total for count in counts)
    assert observation.empirical_mid == empirical
    rhs = total * kl_divergence(empirical, probabilities) + expected_constant
    _assert_close_or_infinite(-law.log_pmf(observation), rhs)


def test_joint_count_likelihood_weights_each_empirical_kl_by_its_observed_total():
    counts = ((1, 1), (2, 4, 1))
    probabilities = ((0.25, 0.75), (0.25, 0.25, 0.5))
    observations = tuple(MIDCountObservation(values, sum(values)) for values in counts)
    laws = tuple(MultinomialMIDLaw(sum(values), p) for values, p in zip(counts, probabilities, strict=True))
    divergences = tuple(kl_divergence(item.empirical_mid, law.probabilities)
                        for item, law in zip(observations, laws, strict=True))
    negative_log_likelihood = -math.fsum(law.log_pmf(item)
                                       for item, law in zip(observations, laws, strict=True))
    data_constant = math.fsum(multinomial_count_constant(item) for item in observations)
    weighted = math.fsum(item.n * divergence
                         for item, divergence in zip(observations, divergences, strict=True))
    assert negative_log_likelihood == pytest.approx(weighted + data_constant, abs=3e-14)
    assert negative_log_likelihood - data_constant != pytest.approx(math.fsum(divergences))
