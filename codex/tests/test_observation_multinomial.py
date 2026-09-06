"""Focused numerical, support and explicit-RNG multinomial contract checks."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

from fluxemu.exceptions import InputValidationError, ValidationError
from fluxemu.mfa.divergence import kl_divergence, renyi_divergence
from fluxemu.observation import MIDCountObservation, MultinomialMIDLaw
import fluxemu.observation.multinomial as multinomial


def test_direct_log_pmf_and_immutable_probability_order():
    law = MultinomialMIDLaw(5, (0.25, 0.0, 0.75))
    expected = math.log(10 * 0.25**2 * 0.75**3)
    observation = MIDCountObservation((2, 0, 3), 5)
    assert law.log_pmf(observation) == pytest.approx(expected, abs=5e-15)
    assert law.log_prob([2, 0, 3]) == law.log_pmf(observation)
    assert law.probabilities == (0.25, 0.0, 0.75)
    assert law.mass_classes == (0, 1, 2)
    assert law.fingerprint == MultinomialMIDLaw(np.int64(5), (0.25, 0, 0.75)).fingerprint
    assert law.fingerprint != MultinomialMIDLaw(6, law.probabilities).fingerprint
    with pytest.raises(FrozenInstanceError):
        law.n = 6


def test_exact_zero_and_off_total_support():
    law = MultinomialMIDLaw(3, (0, 1, 0))
    assert law.log_pmf((0, 3, 0)) == 0.0
    assert law.log_pmf((1, 2, 0)) == -math.inf
    assert law.log_pmf((0, 2, 1)) == -math.inf
    assert law.log_pmf((0, 0, 0)) == -math.inf
    assert law.log_pmf((0, 4, 0)) == -math.inf
    assert law.log_pmf(MIDCountObservation((0, 4, 0), 4)) == -math.inf


@pytest.mark.parametrize("counts", [
    (), (1,), (1, 1, 1), (-1, 4), (1.0, 2), (True, 2),
    (False, 3), (math.nan, 3), (math.inf, 3), ("1", 2),
    ((1,), (2,)), 3, None,
])
def test_malformed_count_arguments_fail(counts):
    with pytest.raises(InputValidationError):
        MultinomialMIDLaw(3, (0.25, 0.75)).log_pmf(counts)


@pytest.mark.parametrize("total", [0, -1, True, False, 3.0, math.nan, math.inf, "3", 2**63])
def test_invalid_total_and_explicit_representability_limit(total):
    with pytest.raises(InputValidationError):
        MultinomialMIDLaw(total, (0.5, 0.5))


@pytest.mark.parametrize("probabilities", [
    (), (0, 0), (0.2, 0.2), (math.nan, 1), (math.inf, 0),
    (True, False), (-1e-16, 1), (1 + math.ulp(1), 0),
    (Fraction(-1, 10**400), 1), (Fraction(10**400 + 1, 10**400), 0),
])
def test_probability_validation_never_repairs_source_values(probabilities):
    with pytest.raises(InputValidationError):
        MultinomialMIDLaw(3, probabilities)


def test_positive_support_cannot_disappear_in_float_conversion():
    with pytest.raises(ValidationError, match="erase positive support"):
        MultinomialMIDLaw(3, (Fraction(1, 10**400), 1))
    tiny = np.longdouble("1e-400")
    if tiny > 0 and float(tiny) == 0:
        with pytest.raises(ValidationError, match="erase positive support"):
            MultinomialMIDLaw(3, (tiny, np.longdouble(1)))
        with pytest.raises(InputValidationError):
            MultinomialMIDLaw(3, (-tiny, np.longdouble(1)))


def test_subnormal_probability_support_stays_positive_in_log_space():
    tiny = math.ulp(0.0)
    law = MultinomialMIDLaw(2, (tiny, 1.0))
    assert law.probabilities[0] == tiny
    assert law.log_pmf((1, 1)) == pytest.approx(math.log(2) + math.log(tiny), abs=2e-13)
    assert math.isfinite(law.log_pmf((2, 0)))


@pytest.mark.parametrize("probabilities", [(0.2, 0.8), (0.5, 0.5 + math.ulp(0.5))])
def test_huge_count_amplification_of_represented_mass_is_rejected(probabilities):
    assert MultinomialMIDLaw(100, probabilities).probabilities == probabilities
    with pytest.raises(InputValidationError, match="amplifies.*mass residual"):
        MultinomialMIDLaw(10**18, probabilities)


@pytest.mark.parametrize("total", [10**6, 10**12, 10**18, 2**63 - 2])
def test_large_count_mode_avoids_log_factorial_cancellation(total):
    law = MultinomialMIDLaw(total, (0.5, 0.5))
    counts = MIDCountObservation((total // 2, total // 2), total)
    # Central-binomial asymptotic expansion has O(n**-3) error here, an
    # independent oracle that never subtracts large log-factorials.
    expected = -0.5 * math.log(math.pi * total / 2) - 1 / (4 * total)
    assert law.log_pmf(counts) < 0
    assert law.log_pmf(counts) == pytest.approx(expected, abs=8e-15)
    assert multinomial.multinomial_count_constant(counts) == pytest.approx(-expected, abs=8e-15)


def test_extreme_off_mode_large_counts_remain_finite():
    total = 10**18
    law = MultinomialMIDLaw(total, (0.25, 0.75))
    assert law.log_pmf((total, 0)) == pytest.approx(total * math.log(0.25), rel=2e-16)


@pytest.mark.parametrize("total,counts", [
    (32, (2, 25, 5, 0)), (33, (0, 32, 1, 0)),
    (64, (32, 31, 1, 0)), (100, (33, 34, 33, 0)),
])
def test_log_factorial_switch_against_independent_exact_factorials(total, counts):
    probabilities = (0.125, 0.5, 0.375, 0.0)
    law = MultinomialMIDLaw(total, probabilities)
    observation = MIDCountObservation(counts, total)
    # Moderate exact integer factorials and 90-digit logarithms independently
    # cover both sides of the production Stirling switch at 32.
    with localcontext() as context:
        context.prec = 90
        coefficient = Decimal(math.factorial(total)).ln() - sum(
            (Decimal(math.factorial(k)).ln() for k in counts), Decimal(0)
        )
        log_pmf = coefficient + sum(
            (Decimal(k) * Decimal.from_float(p).ln()
             for k, p in zip(counts, probabilities, strict=True) if k > 0),
            Decimal(0),
        )
        constant = -coefficient - sum(
            (Decimal(k) * (Decimal(k) / total).ln() for k in counts if k > 0),
            Decimal(0),
        )
    assert law.log_pmf(counts) == pytest.approx(float(log_pmf), abs=1e-14)
    assert multinomial.multinomial_count_constant(observation) == pytest.approx(float(constant), abs=1e-14)


def test_seeded_sampling_is_reproducible_and_retains_exact_counts():
    law = MultinomialMIDLaw(1000, (0.125, 0, 0.375, 0.5, 0))
    fingerprint = law.fingerprint
    first = law.sample(seed=42)
    assert first == law.sample(seed=np.int64(42))
    assert first.n == 1000 and sum(first.counts) == 1000
    assert first.counts[1] == first.counts[4] == 0
    assert all(type(value) is int for value in first.counts)
    assert law.fingerprint == fingerprint
    left, right = np.random.default_rng(27), np.random.default_rng(27)
    assert [law.sample(rng=left) for _ in range(3)] == [law.sample(rng=right) for _ in range(3)]


def test_large_n_sampling_cannot_fill_exact_zero_support():
    law = MultinomialMIDLaw(10**18, (0.0, 0.5, 0.0, 0.5, 0.0))
    for seed in range(5):
        counts = law.sample(seed=seed).counts
        assert counts[0] == counts[2] == counts[4] == 0
        assert sum(counts) == law.n
    # Even the greatest supported total is exact for a degenerate law.
    point = MultinomialMIDLaw(2**63 - 1, (0.0, 1.0, 0.0))
    assert point.sample(seed=0).counts == (0, 2**63 - 1, 0)
    assert point.log_pmf((0, 2**63 - 1, 0)) == 0


@pytest.mark.parametrize("kwargs", [
    {}, {"seed": True}, {"seed": -1}, {"seed": 1.0},
    {"rng": 0}, {"rng": np.random.RandomState(1)},
    {"seed": 1, "rng": np.random.default_rng(1)},
])
def test_sampling_requires_one_explicit_valid_rng_boundary(kwargs):
    with pytest.raises(InputValidationError):
        MultinomialMIDLaw(10, (0.5, 0.5)).sample(**kwargs)


def test_law_divergences_reuse_existing_kernels_and_order_one_dispatch(monkeypatch):
    p, q = MultinomialMIDLaw(3, (0.25, 0.75)), MultinomialMIDLaw(3, (0.5, 0.5))
    assert multinomial.kl_multinomial(p, q) == 3 * kl_divergence(p.probabilities, q.probabilities)
    assert multinomial.renyi_multinomial(p, q, 0.73) == 3 * renyi_divergence(p.probabilities, q.probabilities, 0.73)
    monkeypatch.setattr(multinomial._divergence, "kl_divergence", lambda *args, **kwargs: 17.0)
    assert multinomial.renyi_multinomial(p, q, 1) == 51
    assert multinomial.independent_product_renyi((p, p), (q, q), 1) == 102


@pytest.mark.parametrize("alpha", [False, True, 0, -1, math.nan, math.inf, "0.5"])
def test_invalid_law_renyi_orders_fail(alpha):
    law = MultinomialMIDLaw(3, (0.5, 0.5))
    with pytest.raises(InputValidationError):
        multinomial.renyi_multinomial(law, law, alpha)
    with pytest.raises(InputValidationError):
        multinomial.independent_product_renyi((law,), (law,), alpha)


def test_law_comparisons_require_matching_spaces_and_declared_blocks():
    law = MultinomialMIDLaw(3, (0.5, 0.5))
    for other in (MultinomialMIDLaw(4, (0.5, 0.5)), MultinomialMIDLaw(3, (0.5, 0.5, 0)), object()):
        with pytest.raises(InputValidationError):
            multinomial.kl_multinomial(law, other)
        with pytest.raises(InputValidationError):
            multinomial.renyi_multinomial(law, other, 0.5)
    for left, right in (((), ()), ((law,), (law, law)), (law, (law,))):
        with pytest.raises(InputValidationError):
            multinomial.independent_product_kl(left, right)
    with pytest.raises(InputValidationError):
        multinomial.multinomial_count_constant((1, 2))


def test_unordered_containers_cannot_supply_positional_scientific_data():
    law = MultinomialMIDLaw(3, (0.25, 0.75))
    for probabilities in ({0.25, 0.75}, frozenset((0.25, 0.75)), {0.25: 0, 0.75: 1}):
        with pytest.raises(InputValidationError, match="positional order"):
            MultinomialMIDLaw(3, probabilities)
    for counts in ({1, 2}, frozenset((1, 2)), {1: 0, 2: 1}):
        with pytest.raises(InputValidationError, match="positional order"):
            law.log_pmf(counts)
    for blocks in ({law}, frozenset((law,)), {law: 0}):
        with pytest.raises(InputValidationError, match="block order"):
            multinomial.independent_product_kl(blocks, (law,))
        with pytest.raises(InputValidationError, match="block order"):
            multinomial.independent_product_renyi((law,), blocks, 0.5)
