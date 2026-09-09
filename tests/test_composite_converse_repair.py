"""Independent Hölder/NP controls for the untrusted converse audit.

These tests derive masses, Rényi sums and randomized Neyman–Pearson tests
without calling production PMF, divergence or finite-LP helpers. The audited
reverse-only composite inequality was not found to require an algebraic fix.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import math
import sys

import pytest

from fluxemu.exceptions import ValidationError
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    CompositeBinaryTestingProblem,
    CompositeMIDLawFamily,
    IndependentMIDProductLaw,
    SimpleBinaryLawPair,
    bruno_converse_at_order,
    composite_renyi_converse_at_order,
)


def _compositions(total, size):
    if size == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in _compositions(total - first, size - 1):
                yield (first, *rest)


def _decimal_masses(blocks):
    """Explicit multinomial/product identity using integer factorials."""
    block_masses = []
    for block in blocks:
        p = tuple(Decimal.from_float(value) for value in block.probabilities)
        masses = []
        for counts in _compositions(block.n, len(p)):
            coefficient = Decimal(math.factorial(block.n))
            for count in counts:
                coefficient /= math.factorial(count)
            masses.append(coefficient * math.prod(
                probability**count if count else Decimal(1)
                for probability, count in zip(p, counts, strict=True)
            ))
        block_masses.append(masses)
    return tuple(math.prod(values) for values in product(*block_masses))


def _decimal_renyi(p, q, order):
    delta = Decimal.from_float(order) - 1
    if any(left and not right for left, right in zip(p, q, strict=True)):
        return Decimal("Infinity")
    log_terms = tuple(
        left.ln() + delta * (left.ln() - right.ln())
        for left, right in zip(p, q, strict=True) if left and right
    )
    largest = max(log_terms)
    return (largest + sum(((term - largest).exp() for term in log_terms), Decimal(0)).ln()) / delta


def _decimal_np_beta(p, q, epsilon):
    budget = Decimal.from_float(epsilon)
    beta = Decimal(0)
    ranked = sorted(zip(p, q, strict=True), key=lambda pair:
                    pair[1] / pair[0] if pair[0] else Decimal("Infinity"), reverse=True)
    for null_mass, alternative_mass in ranked:
        rejection = min(Decimal(1), budget / null_mass) if null_mass else Decimal(1)
        beta += alternative_mass * (1 - rejection)
        budget -= null_mass * rejection
    return beta


def _family(*laws):
    return CompositeMIDLawFamily(members=tuple(
        IndependentMIDProductLaw(blocks=blocks) for blocks in laws
    ))


@pytest.mark.parametrize("order", [math.nextafter(1.0, math.inf), 1.0001, 2.0, 16.0, sys.float_info.max])
@pytest.mark.parametrize("null,alternative", [
    (((1, (0.5, 0.5)),), ((1, (0.25, 0.75)),)),
    (((3, (0.25, 0.75)),), ((3, (0.5, 0.5)),)),
    (((2, (0.25, 0.25, 0.5)),), ((2, (0.125, 0.625, 0.25)),)),
    (((2, (0.5, 0.0, 0.5)),), ((2, (0.25, 0.0, 0.75)),)),
    (((1, (0.5, 0.5)), (2, (0.125, 0.875))),
     ((1, (0.25, 0.75)), (2, (0.5, 0.5)))),
])
def test_both_directions_and_composite_converse_against_decimal_np(order, null, alternative):
    null_blocks = tuple(MultinomialMIDLaw(n, p) for n, p in null)
    alternative_blocks = tuple(MultinomialMIDLaw(n, p) for n, p in alternative)
    problem = CompositeBinaryTestingProblem(null=_family(null_blocks), alternative=_family(alternative_blocks))
    pair = SimpleBinaryLawPair(null=null_blocks, alternative=alternative_blocks, independent=True)
    epsilon = 0.05
    simple = bruno_converse_at_order(pair, epsilon=epsilon, order=order)
    composite = composite_renyi_converse_at_order(problem, epsilon=epsilon, order=order)
    with localcontext() as context:
        context.prec = 100
        p, q = _decimal_masses(null_blocks), _decimal_masses(alternative_blocks)
        reverse = _decimal_renyi(q, p, order)
        forward = _decimal_renyi(p, q, order)
        np_beta = float(_decimal_np_beta(p, q, epsilon))
    assert simple.order == composite.order == order
    assert simple.reverse_renyi == pytest.approx(float(reverse), rel=3e-13, abs=3e-13)
    assert simple.forward_renyi == pytest.approx(float(forward), rel=3e-13, abs=3e-13)
    assert composite.reverse_renyi == pytest.approx(float(reverse), rel=3e-13, abs=3e-13)
    assert composite.type_ii_lower_bound <= np_beta + 3e-13
    assert simple.type_ii_lower_bound <= np_beta + 3e-13


def test_reverse_pair_minimum_is_not_the_forward_pair_minimum():
    nulls = ((MultinomialMIDLaw(1, (0.0625, 0.9375)),),
             (MultinomialMIDLaw(1, (0.5, 0.5)),))
    alternatives = ((MultinomialMIDLaw(1, (0.25, 0.75)),),
                    (MultinomialMIDLaw(1, (0.75, 0.25)),))
    problem = CompositeBinaryTestingProblem(null=_family(*nulls), alternative=_family(*alternatives))
    with localcontext() as context:
        context.prec = 100
        directed_pairs = [
            ((i, j), _decimal_renyi(_decimal_masses(q), _decimal_masses(p), 2.0),
             _decimal_renyi(_decimal_masses(p), _decimal_masses(q), 2.0))
            for i, p in enumerate(nulls) for j, q in enumerate(alternatives)
        ]
    reverse_pair = min(directed_pairs, key=lambda value: value[1])
    forward_pair = min(directed_pairs, key=lambda value: value[2])
    assert reverse_pair[0] == (1, 0)
    assert forward_pair[0] == (0, 0)
    result = composite_renyi_converse_at_order(problem, epsilon=0.05, order=2)
    assert (result.null_member_index, result.alternative_member_index) == reverse_pair[0]
    assert result.reverse_renyi == pytest.approx(float(reverse_pair[1]), abs=2e-15)


@pytest.mark.parametrize("null,alternative", [
    ((0.0, 1.0), (1.0, 0.0)),
    ((0.0, 1.0), (0.25, 0.75)),
    ((0.25, 0.75), (0.0, 1.0)),
])
def test_structural_support_composite_bound_matches_extended_renyi_semantics(null, alternative):
    p, q = (MultinomialMIDLaw(1, law) for law in (null, alternative))
    result = composite_renyi_converse_at_order(
        CompositeBinaryTestingProblem(null=_family((p,)), alternative=_family((q,))),
        epsilon=0.1, order=2,
    )
    with localcontext() as context:
        context.prec = 100
        pm, qm = _decimal_masses((p,)), _decimal_masses((q,))
        divergence = float(_decimal_renyi(qm, pm, 2.0))
        beta = float(_decimal_np_beta(pm, qm, 0.1))
    assert result.reverse_renyi == pytest.approx(divergence)
    assert result.type_ii_lower_bound <= beta + 2e-15


def test_smallest_positive_probability_retained_and_unrepresentable_support_refused():
    tiny = math.ulp(0.0)
    law = MultinomialMIDLaw(1, (tiny, 1.0))
    assert law.probabilities[0] == tiny
    pair = SimpleBinaryLawPair(null=law, alternative=MultinomialMIDLaw(1, (0.25, 0.75)))
    result = bruno_converse_at_order(pair, epsilon=0.5, order=2)
    assert math.isfinite(result.reverse_renyi)
    assert result.forward_lower_bound == pytest.approx(0.1875, abs=2e-15)
    with pytest.raises(ValidationError, match="erase positive support"):
        MultinomialMIDLaw(1, (Fraction(1, 10**1000), 1))
