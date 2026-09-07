"""Published Bruno components, directions and numerical boundaries."""

from fractions import Fraction
import math
import sys

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    BrunoOrderBound,
    BrunoTheoremAssumptionError,
    NumericalLimitError,
    SimpleBinaryLawPair,
    bruno_converse_at_order,
)


def _pair(n=3):
    return SimpleBinaryLawPair(
        null=MultinomialMIDLaw(n, (0.5, 0.5)),
        alternative=MultinomialMIDLaw(n, (0.25, 0.75)),
    )


@pytest.mark.parametrize("n", [1, 7])
@pytest.mark.parametrize("order", [1.1, 1.5, 1.7, 2.0, 5.0])
@pytest.mark.parametrize("epsilon", [0.0001, 0.05, 0.3, 0.95])
def test_fixed_order_components_match_direct_multinomial_identity(n, order, epsilon):
    pair = _pair(n)
    bound = bruno_converse_at_order(pair, epsilon=epsilon, order=order)
    assert isinstance(bound, BrunoOrderBound)

    def direct_renyi(p, q):
        return n * math.log(math.fsum(
            a**order * b**(1 - order) for a, b in zip(p, q, strict=True)
        )) / (order - 1)

    reverse = direct_renyi(pair.alternative.probabilities, pair.null.probabilities)
    forward = direct_renyi(pair.null.probabilities, pair.alternative.probabilities)
    raw_reverse = 1 - (epsilon * math.exp(reverse)) ** ((order - 1) / order)
    raw_forward = (1 - epsilon) ** (order / (order - 1)) * math.exp(-forward)
    assert bound.reverse_renyi == pytest.approx(reverse, rel=3e-14, abs=3e-14)
    assert bound.forward_renyi == pytest.approx(forward, rel=3e-14, abs=3e-14)
    assert bound.reverse_lower_bound == pytest.approx(raw_reverse, rel=5e-14, abs=3e-14)
    assert bound.forward_lower_bound == pytest.approx(raw_forward, rel=5e-14, abs=3e-14)
    assert bound.type_ii_lower_bound == pytest.approx(max(raw_reverse, raw_forward), abs=3e-14)
    assert bound.global_envelope_evaluated is False


def test_direction_swap_preserves_roles_and_changes_asymmetric_bound():
    pair = _pair(1)
    original = bruno_converse_at_order(pair, epsilon=0.2, order=2)
    swapped = bruno_converse_at_order(
        SimpleBinaryLawPair(null=pair.alternative, alternative=pair.null),
        epsilon=0.2,
        order=2,
    )
    assert original.reverse_renyi == pytest.approx(math.log(1.25), abs=2e-15)
    assert original.forward_renyi == pytest.approx(math.log(4 / 3), abs=2e-15)
    assert original.reverse_renyi == swapped.forward_renyi
    assert original.forward_renyi == swapped.reverse_renyi
    assert original.type_ii_lower_bound == pytest.approx(0.5, abs=2e-15)
    assert swapped.type_ii_lower_bound == pytest.approx(0.512, abs=2e-15)


def test_support_mismatch_fails_before_bound_evaluation_without_repair():
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(3, (0.0, 1.0)),
        alternative=MultinomialMIDLaw(3, (0.25, 0.75)),
    )
    with pytest.raises(BrunoTheoremAssumptionError, match="mutual absolute continuity"):
        bruno_converse_at_order(pair, epsilon=0.05, order=2)
    assert pair.null.probabilities == (0.0, 1.0)


def test_matching_structural_zeros_are_retained():
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(3, (0.5, 0.0, 0.5)),
        alternative=MultinomialMIDLaw(3, (0.25, 0.0, 0.75)),
    )
    bound = bruno_converse_at_order(pair, epsilon=0.05, order=2)
    assert bound.type_ii_lower_bound > 0
    assert pair.null.probabilities[1] == pair.alternative.probabilities[1] == 0.0


def test_reverse_overflow_keeps_positive_forward_bound_and_log_diagnostics():
    tiny = math.ulp(0.0)
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(3, (tiny, 1.0)),
        alternative=MultinomialMIDLaw(3, (0.25, 0.75)),
    )
    bound = bruno_converse_at_order(pair, epsilon=0.5, order=2)
    assert bound.reverse_lower_bound == -math.inf
    assert math.isfinite(bound.reverse_log_power)
    assert bound.forward_lower_bound == pytest.approx(0.25 * 0.75**3)
    assert bound.type_ii_lower_bound == bound.forward_lower_bound
    assert bound.numerical_diagnostics == ("reverse_component_overflow",)


def test_large_counts_retain_log_of_underflowed_forward_component():
    bound = bruno_converse_at_order(_pair(10000), epsilon=0.05, order=2)
    assert bound.forward_lower_bound == 0
    assert bound.type_ii_lower_bound == 0
    assert math.isfinite(bound.log_forward_lower_bound)
    assert bound.log_forward_lower_bound < -745
    assert "forward_component_underflow" in bound.numerical_diagnostics


@pytest.mark.parametrize("order", [math.nextafter(1.0, math.inf), 1.1, 2, 5, sys.float_info.max])
def test_every_representable_order_reaches_public_boundary_unchanged(order):
    bound = bruno_converse_at_order(_pair(), epsilon=0.05, order=order)
    assert bound.order == order
    assert 0 <= bound.type_ii_lower_bound <= 1


def test_unrepresentable_finite_order_fails_explicitly():
    with pytest.raises(NumericalLimitError):
        bruno_converse_at_order(_pair(), epsilon=0.05, order=1 + Fraction(1, 10**100))


@pytest.mark.parametrize("value", [True, "2", None, math.nan, math.inf, 1, 0, -1])
def test_invalid_order_fails_at_public_bound_boundary(value):
    with pytest.raises(InputValidationError):
        bruno_converse_at_order(_pair(), epsilon=0.05, order=value)
