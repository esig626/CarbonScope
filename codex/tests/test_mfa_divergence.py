"""Mathematical and numerical contract of the stationary MID objective."""

from decimal import Decimal, localcontext
import math
import sys

import numpy as np
import pytest

from fluxemu.exceptions import InputValidationError
import fluxemu.mfa.divergence as divergence
from fluxemu.mfa.divergence import (
    MACHINE_SIMPLEX_TOLERANCE,
    kl_divergence,
    renyi_divergence,
    validate_alpha,
    validate_mid,
    validate_mid_tolerance,
)


P = (0.25, 0.75)
Q = (0.5, 0.5)


def _decimal_renyi(p, q, alpha):
    """Independent high-precision direct formula on exactly normalized inputs."""
    with localcontext() as context:
        context.prec = 90
        order = Decimal.from_float(alpha)
        terms = [
            (order * Decimal.from_float(pi).ln()
             + (1 - order) * Decimal.from_float(qi).ln()).exp()
            for pi, qi in zip(p, q, strict=True)
        ]
        return float(sum(terms).ln() / (order - 1))


@pytest.mark.parametrize("alpha", [0.01, 0.5, 1, 1.7, 2, 13.25, sys.float_info.max])
def test_equal_distributions_are_exactly_zero(alpha):
    assert renyi_divergence((0.0, 0.25, 0.75), (0.0, 0.25, 0.75), alpha) == 0.0
    assert kl_divergence(P, P) == 0.0


def test_known_kl_and_directionality():
    expected = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
    reverse = 0.5 * math.log(2) + 0.5 * math.log(2 / 3)
    assert kl_divergence(P, Q) == pytest.approx(expected, abs=2e-16)
    assert kl_divergence(Q, P) == pytest.approx(reverse, abs=2e-16)
    assert abs(kl_divergence(P, Q) - kl_divergence(Q, P)) > 0.01
    assert renyi_divergence(P, Q, 2) != pytest.approx(renyi_divergence(Q, P, 2))


def test_known_renyi_values_below_and_above_one():
    assert renyi_divergence(P, Q, 0.5) == pytest.approx(
        -2 * math.log(math.sqrt(0.125) + math.sqrt(0.375)), abs=3e-16
    )
    assert renyi_divergence(P, Q, 2) == pytest.approx(math.log(1.25), abs=3e-16)


def test_exact_one_dispatches_to_kl(monkeypatch):
    calls = []

    def exact_kl(observed, predicted, *, tolerance):
        calls.append((observed, predicted, tolerance))
        return 17.0

    monkeypatch.setattr(divergence, "kl_divergence", exact_kl)
    assert renyi_divergence(P, Q, 1, tolerance=1e-10) == 17.0
    assert calls == [(P, Q, 1e-10)]


@pytest.mark.parametrize("alpha", [
    0.000_003_7, 0.173_205, 0.37, 0.823, 1 - 1e-7,
    math.nextafter(1, 0), math.nextafter(1, 2), 1 + 1e-7,
    1.234_567_89, 3.141_592_65, 37.123,
])
def test_continuous_orders_match_high_precision_formula(alpha):
    assert renyi_divergence(P, Q, alpha) == pytest.approx(
        _decimal_renyi(P, Q, alpha), rel=2e-12, abs=3e-16
    )


def test_near_one_converges_without_snapping_to_kl():
    kl = kl_divergence(P, Q)
    for delta in (1e-4, 1e-7, 1e-10):
        below = renyi_divergence(P, Q, 1 - delta)
        above = renyi_divergence(P, Q, 1 + delta)
        assert below < kl < above
        assert abs(above - kl) < delta
        assert abs(below - kl) < delta


@pytest.mark.parametrize("alpha", [0.1, 0.5, math.nextafter(1, 0)])
def test_partial_support_is_finite_below_one(alpha):
    assert renyi_divergence((0.5, 0.5, 0), (0, 0.5, 0.5), alpha) == pytest.approx(
        math.log(0.5) / (alpha - 1), rel=2e-15
    )


@pytest.mark.parametrize("alpha", [1, math.nextafter(1, 2), 2, sys.float_info.max])
def test_observed_mass_outside_predicted_support_is_infinite_at_or_above_one(alpha):
    assert renyi_divergence((0.5, 0.5), (0, 1), alpha) == math.inf
    assert kl_divergence((0.5, 0.5), (0, 1)) == math.inf


@pytest.mark.parametrize("alpha", [math.ulp(0.0), 0.3, 1, 2, sys.float_info.max])
def test_disjoint_support_is_always_infinite(alpha):
    assert renyi_divergence((1, 0), (0, 1), alpha) == math.inf


@pytest.mark.parametrize("alpha", [math.ulp(0.0), 1e-14, 0.5, 1 - 1e-12, 1, 2, 1e300])
def test_zero_observed_mass_has_correct_point_mass_formula(alpha):
    assert renyi_divergence((1, 0), (0.25, 0.75), alpha) == pytest.approx(math.log(4), rel=2e-14)
    assert kl_divergence((1, 0), (0.25, 0.75)) == math.log(4)


@pytest.mark.parametrize("observed", [P, (0.4375, 0.5625)])
@pytest.mark.parametrize("alpha", [1e-14, 1e-100, 1e-300])
def test_tiny_positive_order_retains_the_reverse_kl_first_order_limit(observed, alpha):
    actual = renyi_divergence(observed, Q, alpha)
    assert actual > 0
    assert actual / alpha == pytest.approx(kl_divergence(Q, observed), rel=3e-14)
    smallest = renyi_divergence(observed, Q, math.ulp(0.0))
    assert math.isfinite(smallest) and smallest >= -math.ulp(0.0)


def test_huge_orders_and_extreme_positive_support_do_not_overflow():
    tiny = math.ulp(0.0)
    assert math.isfinite(kl_divergence(Q, (tiny, 1.0)))
    expected = math.log(0.5) - math.log(tiny)
    for alpha in (1e100, sys.float_info.max):
        assert renyi_divergence(Q, (tiny, 1.0), alpha) == pytest.approx(expected, rel=2e-15)
    assert renyi_divergence(P, Q, sys.float_info.max) == pytest.approx(math.log(1.5), abs=2e-16)
    near_zero = renyi_divergence((tiny, 1.0), Q, 1e-14)
    assert near_zero / 1e-14 == pytest.approx(kl_divergence(Q, (tiny, 1.0)), rel=3e-12)


def test_extremely_small_shared_support_remains_finite_below_one():
    tiny = math.ulp(0.0)
    actual = renyi_divergence((tiny, 1, 0), (tiny, 0, 1), 0.5)
    assert actual == pytest.approx(-2 * math.log(tiny), rel=2e-15)


def test_machine_mass_roundoff_does_not_create_a_pole_near_one():
    p = (0.25, 0.75 + math.ulp(1.0))
    assert 0 < abs(math.fsum(p) - 1) <= MACHINE_SIMPLEX_TOLERANCE
    kl = kl_divergence(p, Q)
    for alpha in (math.nextafter(1, 0), math.nextafter(1, 2)):
        assert renyi_divergence(p, Q, alpha) == pytest.approx(kl, abs=2e-15)


@pytest.mark.parametrize("alpha", [0.5, 1, 1 + 1e-12, 2])
def test_broad_data_tolerance_does_not_authorize_mass_correction(alpha):
    p = (0.25, 0.75 + 1e-10)
    assert validate_mid(p, tolerance=1e-9) == p
    with pytest.raises(InputValidationError, match="machine-simplex.*observed residual=.*predicted residual="):
        renyi_divergence(p, Q, alpha)
    with pytest.raises(InputValidationError, match="machine-simplex"):
        renyi_divergence(Q, p, alpha)


@pytest.mark.parametrize("alpha", [False, True, 0, -1, -1e-12, math.nan, math.inf, -math.inf, "0.5", 1j])
def test_invalid_orders_are_rejected(alpha):
    with pytest.raises(InputValidationError):
        renyi_divergence(P, Q, alpha)
    with pytest.raises(InputValidationError):
        validate_alpha(alpha)


@pytest.mark.parametrize("mid", [
    (), (-1e-16, 1.0), (0.1, 0.1), (math.nan, 1), (math.inf, 0),
    (True, False), ("0.5", "0.5"), ((0.5,), (0.5,)), 0.5,
    (sys.float_info.max, sys.float_info.max),
])
def test_invalid_mid_is_rejected_without_repair(mid):
    with pytest.raises(InputValidationError):
        validate_mid(mid)


@pytest.mark.parametrize("tolerance", [-1, True, math.nan, math.inf, "1e-9"])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(InputValidationError):
        validate_mid_tolerance(tolerance)
    with pytest.raises(InputValidationError):
        kl_divergence(P, Q, tolerance=tolerance)


def test_dimension_validation_and_value_order_are_preserved():
    assert validate_mid((0.75, 0, 0.25), expected_size=3) == (0.75, 0, 0.25)
    with pytest.raises(InputValidationError, match="expected 3"):
        validate_mid(P, expected_size=3)
    with pytest.raises(InputValidationError, match="expected 2"):
        kl_divergence(P, (0.25, 0.25, 0.5))
    for invalid_size in (True, 0, -1, 2.5):
        with pytest.raises(InputValidationError):
            validate_mid(P, expected_size=invalid_size)


def test_inputs_are_not_mutated_and_numpy_real_orders_are_accepted():
    p = np.array(P)
    q = np.array(Q)
    p_before, q_before = p.copy(), q.copy()
    for alpha in (np.float32(0.5), np.float64(1.3)):
        assert renyi_divergence(p, q, alpha) >= 0
    kl_divergence(p, q)
    np.testing.assert_array_equal(p, p_before)
    np.testing.assert_array_equal(q, q_before)


@pytest.mark.parametrize("alpha", [1e-10, 0.1, 0.5, 0.99, 1, 1.01, 2, 100])
def test_nearly_equal_vectors_respect_controlled_nonnegativity(alpha):
    actual = renyi_divergence((0.5 + 1e-8, 0.5 - 1e-8), Q, alpha)
    assert actual >= -64 * math.ulp(1.0)
