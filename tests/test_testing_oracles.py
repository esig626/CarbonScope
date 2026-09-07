"""Independent finite-count and deterministic-test oracles for Bruno bounds."""

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import math

import pytest

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import SimpleBinaryLawPair, bruno_converse_at_order


PRECISION = 100
BOUND_TOLERANCE = 1e-12


def _count_space(total: int, dimension: int):
    if dimension == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _count_space(total - first, dimension - 1):
            yield (first, *rest)


def _distribution(total: int, probabilities: tuple[float, ...]):
    p = tuple(Fraction(value) for value in probabilities)
    assert sum(p) == 1
    masses = {}
    for counts in _count_space(total, len(p)):
        coefficient = Fraction(
            math.factorial(total),
            math.prod(math.factorial(count) for count in counts),
        )
        masses[counts] = coefficient * math.prod(
            probability**count
            for probability, count in zip(p, counts, strict=True)
        )
    assert sum(masses.values()) == 1
    return masses


def _product_distribution(blocks):
    marginals = tuple(_distribution(total, probabilities) for total, probabilities in blocks)
    masses = {
        outcome: math.prod(
            marginal[counts]
            for marginal, counts in zip(marginals, outcome, strict=True)
        )
        for outcome in product(*marginals)
    }
    assert sum(masses.values()) == 1
    return masses


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _renyi(left, right, order: float) -> float:
    left_values = tuple(left.values())
    right_values = tuple(right.values())
    assert tuple(left) == tuple(right)
    with localcontext() as context:
        context.prec = PRECISION
        alpha = Decimal.from_float(order)
        terms = []
        for p, q in zip(left_values, right_values, strict=True):
            if p == 0:
                continue
            if q == 0:
                return math.inf
            terms.append(alpha * _decimal(p).ln() + (1 - alpha) * _decimal(q).ln())
        pivot = max(terms)
        mass = sum(((term - pivot).exp() for term in terms), Decimal(0))
        return float((pivot + mass.ln()) / (alpha - 1))


def _minimum_deterministic_beta(null, alternative, epsilon: float) -> Fraction:
    assert tuple(null) == tuple(alternative)
    outcomes = tuple(null)
    threshold = Fraction(epsilon)
    best = Fraction(1)
    for mask in range(1 << len(outcomes)):
        alpha = Fraction(0)
        power = Fraction(0)
        for index, outcome in enumerate(outcomes):
            if mask & (1 << index):
                alpha += null[outcome]
                power += alternative[outcome]
        if alpha <= threshold:
            best = min(best, 1 - power)
    return best


def _expected_components(reverse: float, forward: float, epsilon: float, order: float):
    reverse_component = 1 - math.exp(
        ((order - 1) / order) * (math.log(epsilon) + reverse)
    )
    forward_component = math.exp(
        (order / (order - 1)) * math.log1p(-epsilon) - forward
    )
    return reverse_component, forward_component, max(reverse_component, forward_component)


SINGLE_CASES = (
    (1, (0.25, 0.75), (0.5, 0.5)),
    (2, (0.125, 0.875), (0.875, 0.125)),
    (3, (0.0, 0.25, 0.75), (0.0, 0.5, 0.5)),
    (2, (0.5, 0.5), (0.5, 0.5)),
)


@pytest.mark.parametrize("total,p0,p1", SINGLE_CASES)
@pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.25, 0.5])
@pytest.mark.parametrize("order", [1.1, 1.7, 2.0, 5.0])
def test_order_bound_matches_independent_complete_law_and_deterministic_optimum(
    total, p0, p1, epsilon, order,
):
    null = _distribution(total, p0)
    alternative = _distribution(total, p1)
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(total, p0),
        alternative=MultinomialMIDLaw(total, p1),
    )
    reverse = _renyi(alternative, null, order)
    forward = _renyi(null, alternative, order)
    expected_reverse, expected_forward, expected_bound = _expected_components(
        reverse, forward, epsilon, order
    )
    bound = bruno_converse_at_order(pair, epsilon=epsilon, order=order)
    assert bound.reverse_renyi == pytest.approx(reverse, rel=3e-13, abs=3e-14)
    assert bound.forward_renyi == pytest.approx(forward, rel=3e-13, abs=3e-14)
    assert bound.reverse_lower_bound == pytest.approx(expected_reverse, rel=3e-13, abs=3e-14)
    assert bound.forward_lower_bound == pytest.approx(expected_forward, rel=3e-13, abs=3e-14)
    assert bound.type_ii_lower_bound == pytest.approx(expected_bound, rel=3e-13, abs=3e-14)
    optimum = _minimum_deterministic_beta(null, alternative, epsilon)
    assert bound.type_ii_lower_bound <= float(optimum) + BOUND_TOLERANCE


def test_independent_product_bound_matches_cartesian_complete_law():
    blocks = (
        (1, (0.25, 0.75), (0.5, 0.5)),
        (2, (0.5, 0.5), (0.125, 0.875)),
    )
    null_blocks = tuple((n, p0) for n, p0, _ in blocks)
    alternative_blocks = tuple((n, p1) for n, _, p1 in blocks)
    null = _product_distribution(null_blocks)
    alternative = _product_distribution(alternative_blocks)
    pair = SimpleBinaryLawPair(
        null=tuple(MultinomialMIDLaw(n, p0) for n, p0, _ in blocks),
        alternative=tuple(MultinomialMIDLaw(n, p1) for n, _, p1 in blocks),
        independent=True,
    )
    epsilon, order = 0.05, 2.0
    reverse = _renyi(alternative, null, order)
    forward = _renyi(null, alternative, order)
    expected = _expected_components(reverse, forward, epsilon, order)
    bound = bruno_converse_at_order(pair, epsilon=epsilon, order=order)
    assert bound.count_totals if hasattr(bound, "count_totals") else pair.count_totals == (1, 2)
    assert bound.reverse_renyi == pytest.approx(reverse, abs=3e-14)
    assert bound.forward_renyi == pytest.approx(forward, abs=3e-14)
    assert bound.type_ii_lower_bound == pytest.approx(expected[2], abs=3e-14)
    assert bound.type_ii_lower_bound <= float(
        _minimum_deterministic_beta(null, alternative, epsilon)
    ) + BOUND_TOLERANCE


def test_discrete_oracle_does_not_randomise_boundary_atoms():
    law = _distribution(1, (0.5, 0.5))
    assert _minimum_deterministic_beta(law, law, 0.25) == 1
    assert _minimum_deterministic_beta(law, law, math.nextafter(0.5, 0.0)) == 1
    assert _minimum_deterministic_beta(law, law, 0.5) == Fraction(1, 2)


def test_discrete_oracle_is_not_restricted_to_likelihood_ratio_prefixes():
    null = _distribution(1, (0.125, 0.375, 0.5))
    alternative = _distribution(1, (0.25, 0.5, 0.25))
    assert _minimum_deterministic_beta(null, alternative, 0.375) == Fraction(1, 2)
