"""Independent complete-law and deterministic-test oracles for Bruno v2.

All supplied probabilities are exact binary fractions, so Fraction(float)
represents the production law inputs without a normalization approximation.
Integer factorials enumerate every multinomial count outcome. Product laws
are enumerated on their complete Cartesian spaces, retaining zero-mass atoms.

The oracle follows Bruno Eq. (1)/(2): a deterministic test assigns each whole
atom either H0 or H1. It exhausts every H1 rejection subset R and minimizes
P1(R complement) subject to the exact constraint P0(R) <= epsilon. It neither
interpolates atoms nor substitutes a likelihood-ratio threshold search. See
docs/BRUNO_V2_THEOREM_TRANSFER.md for the finite-atomic convention caveat.

Expected divergences use Decimal summation on the complete law, independently
of production MID divergence, multinomial tensorization, and Bruno routines.
Finite order tuples below are test coverage, never a mathematical order grid.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math

import pytest

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    BrunoTheoremAssumptionError,
    SimpleBinaryLawPair,
    bruno_converse_at_order,
)


# Exact PMF normalization is checked in Fraction arithmetic. These tolerances
# cover only the comparison with float production output, not model repair.
PMF_ABSOLUTE_TOLERANCE = 3e-14
FORMULA_RELATIVE_TOLERANCE = 3e-13
FORMULA_ABSOLUTE_TOLERANCE = 3e-14
BOUND_ABSOLUTE_TOLERANCE = 1e-12
ORACLE_DECIMAL_PRECISION = 100
# This also preserves 1-epsilon for the smallest positive binary64 epsilon.
COMPONENT_DECIMAL_PRECISION = 400

TOTALS = (1, 2, 3)
EPSILONS = (
    math.nextafter(0.0, 1.0), 0.01, 0.125, 0.25, 0.5, 0.875,
    math.nextafter(1.0, 0.0),
)
ORDERS = (math.nextafter(1.0, math.inf), 1.1, 1.7, 2.0, 5.0, 31.0, 1e6)
PROBABILITY_PAIRS = (
    ((0.25, 0.75), (0.5, 0.5)),
    ((0.5, 0.5), (0.25, 0.75)),
    ((0.125, 0.875), (0.875, 0.125)),
    ((0.5, 0.5), (0.5, 0.5)),
    ((1.0, 0.0), (1.0, 0.0)),
    ((0.0, 0.25, 0.75), (0.0, 0.5, 0.5)),
    ((0.125, 0.375, 0.5), (0.25, 0.5, 0.25)),
    ((0.5, 0.5), (0.5 + 2**-20, 0.5 - 2**-20)),
)
SINGLE_CASES = tuple(
    (total, null, alternative)
    for null, alternative in PROBABILITY_PAIRS
    for total in TOTALS
)
PRODUCT_CASES = (
    (
        (1, (0.25, 0.75), (0.5, 0.5)),
        (2, (0.5, 0.5), (0.125, 0.875)),
    ),
    (
        (1, (0.0, 0.25, 0.75), (0.0, 0.5, 0.5)),
        (2, (0.125, 0.875), (0.875, 0.125)),
    ),
    (
        (1, (0.25, 0.75), (0.5, 0.5)),
        (2, (0.5, 0.5), (0.25, 0.75)),
        (1, (0.125, 0.875), (0.25, 0.75)),
    ),
)


def _count_space(total, dimension):
    """Every ordered nonnegative composition, including structural zeros."""
    if dimension == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remaining in _count_space(total - first, dimension - 1):
            yield (first, *remaining)


@lru_cache(maxsize=None)
def _exact_distribution(total, probabilities):
    rational = tuple(Fraction(value) for value in probabilities)
    assert sum(rational) == 1
    masses = {}
    for counts in _count_space(total, len(rational)):
        coefficient = Fraction(
            math.factorial(total),
            math.prod(math.factorial(count) for count in counts),
        )
        masses[counts] = coefficient * math.prod(
            probability**count
            for probability, count in zip(rational, counts, strict=True)
        )
    assert sum(masses.values()) == 1
    assert len(masses) == math.comb(total + len(rational) - 1, len(rational) - 1)
    return masses


@lru_cache(maxsize=None)
def _exact_product(blocks):
    marginals = tuple(_exact_distribution(total, p) for total, p in blocks)
    masses = {
        outcomes: math.prod(
            marginal[counts]
            for marginal, counts in zip(marginals, outcomes, strict=True)
        )
        for outcomes in product(*marginals)
    }
    assert sum(masses.values()) == 1
    return masses


def _decimal(rational):
    return Decimal(rational.numerator) / Decimal(rational.denominator)


@lru_cache(maxsize=None)
def _renyi_from_masses(p, q, order):
    """Direct complete-support definition, with exact support predicates."""
    assert sum(p) == sum(q) == 1
    if any(pi > 0 and qi == 0 for pi, qi in zip(p, q, strict=True)):
        return Decimal("Infinity")
    with localcontext() as context:
        context.prec = ORACLE_DECIMAL_PRECISION
        alpha = Decimal.from_float(order)
        terms = tuple(
            alpha * _decimal(pi).ln() + (1 - alpha) * _decimal(qi).ln()
            for pi, qi in zip(p, q, strict=True) if pi > 0
        )
        pivot = max(terms)
        mass = sum(((term - pivot).exp() for term in terms), Decimal(0))
        return (pivot + mass.ln()) / (alpha - 1)


def _oracle_renyi(p, q, order):
    assert tuple(p) == tuple(q)
    return _renyi_from_masses(tuple(p.values()), tuple(q.values()), order)


@lru_cache(maxsize=None)
def _all_deterministic_error_pairs(null_items, alternative_items):
    """Enumerate all 2**|Omega| rejection regions; no atom randomization."""
    null, alternative = dict(null_items), dict(alternative_items)
    assert tuple(null) == tuple(alternative)
    assert sum(null.values()) == sum(alternative.values()) == 1
    # Tests deliberately bound the outcome space: this is validation only.
    assert len(null) <= 12
    rejection_masses = [(Fraction(0), Fraction(0))]
    for outcome in null:
        rejection_masses += [
            (alpha + null[outcome], power + alternative[outcome])
            for alpha, power in rejection_masses
        ]
    assert len(rejection_masses) == 2 ** len(null)
    return tuple((alpha, 1 - power) for alpha, power in rejection_masses)


@lru_cache(maxsize=None)
def _minimum_deterministic_beta(null_items, alternative_items, epsilon):
    threshold = Fraction(epsilon)
    return min(
        beta
        for alpha, beta in _all_deterministic_error_pairs(null_items, alternative_items)
        if alpha <= threshold
    )


def _oracle_beta(null, alternative, epsilon):
    return _minimum_deterministic_beta(
        tuple(null.items()), tuple(alternative.items()), epsilon,
    )


@lru_cache(maxsize=None)
def _formula_components(reverse_renyi, forward_renyi, epsilon, order):
    """Published powers applied to independent full-law Decimal sums."""
    with localcontext() as context:
        context.prec = COMPONENT_DECIMAL_PRECISION
        alpha, budget = Decimal.from_float(order), Decimal.from_float(epsilon)
        reverse = 1 - (budget * reverse_renyi.exp()) ** ((alpha - 1) / alpha)
        forward = (1 - budget) ** (alpha / (alpha - 1)) * (-forward_renyi).exp()
        return float(reverse), float(forward), float(max(reverse, forward))


def _assert_formula_close(actual, expected):
    assert float(actual) == pytest.approx(
        float(expected), rel=FORMULA_RELATIVE_TOLERANCE,
        abs=FORMULA_ABSOLUTE_TOLERANCE,
    )


def _check_certificate(pair, null, alternative, epsilon, order):
    exact_reverse = _oracle_renyi(alternative, null, order)
    exact_forward = _oracle_renyi(null, alternative, order)
    exact_beta = _oracle_beta(null, alternative, epsilon)
    expected = _formula_components(exact_reverse, exact_forward, epsilon, order)
    certificate = bruno_converse_at_order(pair, epsilon=epsilon, order=order)
    assert certificate.order == order
    assert certificate.type_i_constraint == epsilon
    _assert_formula_close(certificate.reverse_renyi, exact_reverse)
    _assert_formula_close(certificate.forward_renyi, exact_forward)
    for actual, target in zip(
        (certificate.reverse_lower_bound, certificate.forward_lower_bound,
         certificate.type_ii_lower_bound), expected, strict=True,
    ):
        _assert_formula_close(actual, target)
    assert math.isfinite(certificate.type_ii_lower_bound)
    assert certificate.type_ii_lower_bound <= float(exact_beta) + BOUND_ABSOLUTE_TOLERANCE
    return certificate


@pytest.mark.parametrize("total,p0,p1", SINGLE_CASES)
def test_complete_multinomial_pmfs_match_exact_factorials_and_normalize(total, p0, p1):
    for probabilities in (p0, p1):
        law = MultinomialMIDLaw(total, probabilities)
        exact = _exact_distribution(total, probabilities)
        actual_masses = []
        for counts, mass in exact.items():
            log_mass = law.log_pmf(counts)
            if mass == 0:
                assert log_mass == -math.inf
                actual_masses.append(0.0)
            else:
                actual = math.exp(log_mass)
                assert actual == pytest.approx(float(mass), abs=PMF_ABSOLUTE_TOLERANCE)
                actual_masses.append(actual)
        assert math.fsum(actual_masses) == pytest.approx(1.0, abs=PMF_ABSOLUTE_TOLERANCE)


@pytest.mark.parametrize("total,p0,p1", SINGLE_CASES)
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("order", ORDERS)
def test_single_law_bound_matches_full_support_and_exhaustive_deterministic_optimum(
    total, p0, p1, epsilon, order,
):
    null, alternative = _exact_distribution(total, p0), _exact_distribution(total, p1)
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(total, p0), alternative=MultinomialMIDLaw(total, p1),
    )
    certificate = _check_certificate(pair, null, alternative, epsilon, order)
    # Independently verify tensorization using a separately enumerated one-trial
    # law on its own count space; no production MID divergence enters expected.
    categorical_null = _exact_distribution(1, p0)
    categorical_alternative = _exact_distribution(1, p1)
    _assert_formula_close(
        certificate.reverse_renyi,
        total * _oracle_renyi(categorical_alternative, categorical_null, order),
    )
    _assert_formula_close(
        certificate.forward_renyi,
        total * _oracle_renyi(categorical_null, categorical_alternative, order),
    )


@pytest.mark.parametrize("blocks", PRODUCT_CASES)
def test_product_complete_pmf_enumeration_retains_unequal_explicit_totals(blocks):
    assert len({total for total, _, _ in blocks}) > 1
    for side in (1, 2):
        declarations = tuple((block[0], block[side]) for block in blocks)
        laws = tuple(MultinomialMIDLaw(total, p) for total, p in declarations)
        actual_masses = []
        for outcomes, expected in _exact_product(declarations).items():
            actual = math.exp(math.fsum(
                law.log_pmf(counts) for law, counts in zip(laws, outcomes, strict=True)
            ))
            assert actual == pytest.approx(float(expected), abs=PMF_ABSOLUTE_TOLERANCE)
            actual_masses.append(actual)
        assert math.fsum(actual_masses) == pytest.approx(1.0, abs=PMF_ABSOLUTE_TOLERANCE)


@pytest.mark.parametrize("blocks", PRODUCT_CASES)
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("order", ORDERS)
def test_unequal_total_product_bound_matches_joint_support_and_deterministic_optimum(
    blocks, epsilon, order,
):
    null_blocks = tuple((total, p0) for total, p0, _ in blocks)
    alternative_blocks = tuple((total, p1) for total, _, p1 in blocks)
    pair = SimpleBinaryLawPair(
        null=tuple(MultinomialMIDLaw(total, p0) for total, p0 in null_blocks),
        alternative=tuple(MultinomialMIDLaw(total, p1) for total, p1 in alternative_blocks),
        independent=True,
    )
    assert pair.count_totals == tuple(total for total, _, _ in blocks)
    certificate = _check_certificate(
        pair, _exact_product(null_blocks), _exact_product(alternative_blocks), epsilon, order,
    )
    independent_reverse = sum(
        _oracle_renyi(_exact_distribution(total, p1), _exact_distribution(total, p0), order)
        for total, p0, p1 in blocks
    )
    independent_forward = sum(
        _oracle_renyi(_exact_distribution(total, p0), _exact_distribution(total, p1), order)
        for total, p0, p1 in blocks
    )
    _assert_formula_close(certificate.reverse_renyi, independent_reverse)
    _assert_formula_close(certificate.forward_renyi, independent_forward)


def test_deterministic_oracle_does_not_randomize_boundary_atoms():
    law = _exact_distribution(1, (0.5, 0.5))
    assert _oracle_beta(law, law, 0.25) == 1
    # The randomized value would be 3/4. At the actual atom budget, rejecting
    # one atom becomes feasible; no tolerance may admit it below that budget.
    assert _oracle_beta(law, law, math.nextafter(0.5, 0.0)) == 1
    assert _oracle_beta(law, law, 0.5) == Fraction(1, 2)
    assert _oracle_beta(law, law, math.nextafter(0.5, 1.0)) == Fraction(1, 2)


def test_deterministic_oracle_does_not_substitute_a_likelihood_ratio_prefix():
    null = _exact_distribution(1, (0.125, 0.375, 0.5))
    alternative = _exact_distribution(1, (0.25, 0.5, 0.25))
    # Rejecting the middle atom uses the entire 3/8 budget and has beta=1/2.
    # A sorted likelihood-ratio prefix can only reject the first atom here.
    assert _oracle_beta(null, alternative, 0.375) == Fraction(1, 2)
    assert null[(0, 1, 0)] == Fraction(3, 8)
    assert alternative[(0, 1, 0)] == Fraction(1, 2)


def test_swapping_roles_matches_independent_asymmetric_divergences_and_changes_bound():
    p0, p1 = (0.75, 0.25), (0.5, 0.5)
    null, alternative = _exact_distribution(1, p0), _exact_distribution(1, p1)
    pair = SimpleBinaryLawPair(null=MultinomialMIDLaw(1, p0), alternative=MultinomialMIDLaw(1, p1))
    swapped = SimpleBinaryLawPair(null=pair.alternative, alternative=pair.null)
    original = _check_certificate(pair, null, alternative, 0.25, 2.0)
    reverse = _check_certificate(swapped, alternative, null, 0.25, 2.0)
    _assert_formula_close(original.reverse_renyi, math.log(4 / 3))
    _assert_formula_close(original.forward_renyi, math.log(5 / 4))
    _assert_formula_close(original.type_ii_lower_bound, 0.45)
    assert original.reverse_renyi == reverse.forward_renyi
    assert original.forward_renyi == reverse.reverse_renyi
    assert original.type_ii_lower_bound != pytest.approx(reverse.type_ii_lower_bound)
    assert _oracle_beta(null, alternative, 0.25) == Fraction(1, 2)
    assert _oracle_beta(alternative, null, 0.25) == 1


@pytest.mark.parametrize("p0,p1", (
    ((1.0, 0.0), (0.5, 0.5)),
    ((0.5, 0.5), (1.0, 0.0)),
    ((0.25, 0.75, 0.0), (0.0, 0.5, 0.5)),
    ((1.0, 0.0), (0.0, 1.0)),
))
def test_exact_support_mismatch_rejects_theorem_even_if_one_direction_is_finite(p0, p1):
    null, alternative = _exact_distribution(2, p0), _exact_distribution(2, p1)
    divergences = (_oracle_renyi(alternative, null, 2.0), _oracle_renyi(null, alternative, 2.0))
    assert any(value.is_infinite() for value in divergences)
    pair = SimpleBinaryLawPair(null=MultinomialMIDLaw(2, p0), alternative=MultinomialMIDLaw(2, p1))
    with pytest.raises(BrunoTheoremAssumptionError, match="mutual absolute continuity"):
        bruno_converse_at_order(pair, epsilon=0.25, order=2.0)
    assert pair.null.probabilities == p0
    assert pair.alternative.probabilities == p1


def test_independent_oracle_helpers_do_not_call_production_probability_or_bound_kernels(monkeypatch):
    from fluxemu.mfa import divergence as mid_module
    from fluxemu.observation import multinomial as law_module
    from fluxemu.testing import bruno as bruno_module

    def forbidden(*args, **kwargs):
        pytest.fail("independent oracle called a production kernel")

    for name in ("kl_divergence", "renyi_divergence"):
        monkeypatch.setattr(mid_module, name, forbidden)
    for name in ("renyi_multinomial", "kl_multinomial", "independent_product_renyi"):
        monkeypatch.setattr(law_module, name, forbidden)
    monkeypatch.setattr(MultinomialMIDLaw, "log_pmf", forbidden)
    monkeypatch.setattr(bruno_module, "bruno_converse_at_order", forbidden)
    monkeypatch.setitem(globals(), "bruno_converse_at_order", forbidden)
    monkeypatch.setitem(globals(), "_formula_components", forbidden)
    # Fresh inputs prevent a cache hit from concealing a production dependency.
    null = _exact_distribution(1, (0.625, 0.375))
    alternative = _exact_distribution(1, (0.375, 0.625))
    assert sum(null.values()) == sum(alternative.values()) == 1
    assert _oracle_beta(null, alternative, 0.375) == Fraction(3, 8)
    _assert_formula_close(_oracle_renyi(alternative, null, 2.0), math.log(19 / 15))


def test_documented_oracle_matrix_is_bounded_and_exhausts_all_rejection_regions():
    pairs = [
        (_exact_distribution(total, p0), _exact_distribution(total, p1))
        for total, p0, p1 in SINGLE_CASES
    ] + [
        (_exact_product(tuple((n, p0) for n, p0, _ in blocks)),
         _exact_product(tuple((n, p1) for n, _, p1 in blocks)))
        for blocks in PRODUCT_CASES
    ]
    assert len(pairs) == 27
    assert max(len(null) for null, _ in pairs) == 12
    assert sum(
        len(_all_deterministic_error_pairs(tuple(null.items()), tuple(alternative.items())))
        for null, alternative in pairs
    ) == 7032
    assert len(pairs) * len(EPSILONS) * len(ORDERS) == 1323
