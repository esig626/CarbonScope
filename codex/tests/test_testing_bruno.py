"""Published exponents, full-law reuse, directions, and numerical boundaries.

Independent count-space and deterministic-test enumeration live separately in
test_testing_oracles.py; these checks target the production numeric contract.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math
import sys

import numpy as np
import pytest

from fluxemu.exceptions import InputValidationError, ValidationError
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    BrunoTheoremAssumptionError,
    NumericalLimitError,
    SimpleBinaryLawPair,
    bruno_converse_at_order,
)
import fluxemu.testing.bruno as bruno


def _pair(n=3):
    return SimpleBinaryLawPair(
        null=MultinomialMIDLaw(n, (0.5, 0.5)),
        alternative=MultinomialMIDLaw(n, (0.25, 0.75)),
    )


@pytest.mark.parametrize("n", [1, 7])
@pytest.mark.parametrize("order", [1.1, 1.5, 1.7, 2.0, 5.0])
@pytest.mark.parametrize("epsilon", [0.0001, 0.05, 0.3, 0.95])
def test_fixed_order_components_against_direct_multinomial_identity(n, order, epsilon):
    pair = _pair(n)
    certificate = bruno_converse_at_order(pair, epsilon=epsilon, order=order)
    # Direct categorical powers followed by the explicit multinomial identity.
    # These deliberately do not invoke any production divergence function.
    def direct_renyi(p, q):
        return n * math.log(math.fsum(
            a**order * b**(1 - order) for a, b in zip(p, q, strict=True)
        )) / (order - 1)

    reverse = direct_renyi(pair.alternative.probabilities, pair.null.probabilities)
    forward = direct_renyi(pair.null.probabilities, pair.alternative.probabilities)
    raw_reverse = 1 - (epsilon * math.exp(reverse)) ** ((order - 1) / order)
    raw_forward = (1 - epsilon) ** (order / (order - 1)) * math.exp(-forward)
    assert certificate.reverse_renyi == pytest.approx(reverse, rel=3e-14, abs=3e-14)
    assert certificate.forward_renyi == pytest.approx(forward, rel=3e-14, abs=3e-14)
    assert certificate.reverse_lower_bound == pytest.approx(raw_reverse, rel=5e-14, abs=3e-14)
    assert certificate.forward_lower_bound == pytest.approx(raw_forward, rel=5e-14, abs=3e-14)
    assert certificate.type_ii_lower_bound == pytest.approx(max(raw_reverse, raw_forward), abs=3e-14)


def test_direction_swap_preserves_roles_and_generally_changes_certificate():
    pair = _pair(1)
    original = bruno_converse_at_order(pair, epsilon=0.2, order=2)
    swapped_pair = SimpleBinaryLawPair(null=pair.alternative, alternative=pair.null)
    swapped = bruno_converse_at_order(swapped_pair, epsilon=0.2, order=2)
    assert original.reverse_renyi == pytest.approx(math.log(1.25), abs=2e-15)
    assert original.forward_renyi == pytest.approx(math.log(4 / 3), abs=2e-15)
    assert original.reverse_renyi == swapped.forward_renyi
    assert original.forward_renyi == swapped.reverse_renyi
    assert original.type_ii_lower_bound == pytest.approx(0.5, abs=2e-15)
    assert swapped.type_ii_lower_bound == pytest.approx(0.512, abs=2e-15)
    assert original.null_fingerprint == pair.null.fingerprint
    assert original.alternative_fingerprint == pair.alternative.fingerprint
    assert swapped.null_fingerprint == original.alternative_fingerprint
    assert swapped.fingerprint != original.fingerprint


def test_single_law_calls_existing_full_law_divergence_in_both_directions(monkeypatch):
    pair = _pair()
    calls = []
    existing = bruno._observation.renyi_multinomial

    def record(left, right, order):
        calls.append((left, right, order))
        return existing(left, right, order)

    monkeypatch.setattr(bruno._observation, "renyi_multinomial", record)
    certificate = bruno_converse_at_order(pair, epsilon=0.05, order=1.7)
    assert calls == [(pair.alternative, pair.null, 1.7), (pair.null, pair.alternative, 1.7)]
    assert certificate.order == 1.7


def test_unequal_total_product_reuses_declared_order_and_product_additivity(monkeypatch):
    pair = SimpleBinaryLawPair(
        null=(_pair(2).null, MultinomialMIDLaw(11, (0.0, 0.25, 0.75))),
        alternative=(_pair(2).alternative, MultinomialMIDLaw(11, (0.0, 0.5, 0.5))),
        independent=True,
        block_identities=(("z-experiment", "a-target", "rep-2"),
                          ("a-experiment", "z-target", "rep-1")),
        null_state_fingerprint="null-state",
        alternative_state_fingerprint="alternative-state",
        observation_specification_fingerprint="observation-specification",
    )
    existing = bruno._observation.independent_product_renyi
    calls = []

    def record(left, right, order):
        calls.append((left, right, order))
        return existing(left, right, order)

    monkeypatch.setattr(bruno._observation, "independent_product_renyi", record)
    certificate = bruno_converse_at_order(pair, epsilon=0.05, order=2)
    assert calls == [(pair.alternative, pair.null, 2), (pair.null, pair.alternative, 2)]
    assert certificate.reverse_renyi == pytest.approx(2 * math.log(1.25) + 11 * math.log(4 / 3))
    assert certificate.forward_renyi == pytest.approx(2 * math.log(4 / 3) + 11 * math.log(1.25))
    assert certificate.pair is pair
    assert certificate.pair.count_totals == (2, 11)
    assert certificate.pair.block_identities == pair.block_identities
    assert certificate.null_state_fingerprint == "null-state"
    assert certificate.alternative_state_fingerprint == "alternative-state"
    assert certificate.observation_specification_fingerprint == "observation-specification"


@pytest.mark.parametrize("product", [False, True])
def test_support_mismatch_fails_before_any_divergence_without_repairs(monkeypatch, product):
    null = MultinomialMIDLaw(3, (0.0, 1.0))
    alternative = MultinomialMIDLaw(3, (0.25, 0.75))
    if product:
        pair = SimpleBinaryLawPair(
            null=(_pair().null, null), alternative=(_pair().alternative, alternative),
            independent=True,
        )
    else:
        pair = SimpleBinaryLawPair(null=null, alternative=alternative)

    def unexpected(*args, **kwargs):
        pytest.fail("support mismatch must fail before either full-law divergence")

    monkeypatch.setattr(bruno._observation, "renyi_multinomial", unexpected)
    monkeypatch.setattr(bruno._observation, "independent_product_renyi", unexpected)
    with pytest.raises(BrunoTheoremAssumptionError, match="Theorem 1 requires mutual absolute continuity"):
        bruno_converse_at_order(pair, epsilon=0.05, order=2)
    assert null.probabilities == (0.0, 1.0)
    assert alternative.probabilities == (0.25, 0.75)


def test_matching_structural_zeros_are_retained_and_pass_theorem_gate():
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(3, (0.5, 0.0, 0.5)),
        alternative=MultinomialMIDLaw(3, (0.25, 0.0, 0.75)),
    )
    certificate = bruno_converse_at_order(pair, epsilon=0.05, order=2)
    control = bruno_converse_at_order(_pair(), epsilon=0.05, order=2)
    assert pair.mutually_absolutely_continuous
    assert certificate.type_ii_lower_bound == control.type_ii_lower_bound
    assert certificate.reverse_renyi == control.reverse_renyi
    assert pair.null.probabilities == (0.5, 0.0, 0.5)
    assert pair.alternative.probabilities == (0.25, 0.0, 0.75)


def test_vacuous_finite_reverse_component_is_exposed_without_clipping():
    certificate = bruno_converse_at_order(_pair(10), epsilon=0.9, order=2)
    assert math.isfinite(certificate.reverse_lower_bound)
    assert certificate.reverse_lower_bound < 0
    assert certificate.forward_lower_bound > 0
    assert certificate.type_ii_lower_bound == certificate.forward_lower_bound
    assert certificate.numerical_diagnostics == ()


def test_raw_reverse_overflow_retains_a_meaningful_positive_forward_bound():
    tiny = math.ulp(0.0)
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(3, (tiny, 1.0)),
        alternative=MultinomialMIDLaw(3, (0.25, 0.75)),
    )
    certificate = bruno_converse_at_order(pair, epsilon=0.5, order=2)
    assert pair.mutually_absolutely_continuous
    assert pair.null.probabilities == (tiny, 1.0)
    assert pair.alternative.probabilities == (0.25, 0.75)
    assert certificate.reverse_lower_bound == -math.inf
    assert math.isfinite(certificate.reverse_log_power)
    assert certificate.reverse_log_power > math.log(sys.float_info.max)
    assert certificate.forward_lower_bound == pytest.approx(0.25 * 0.75**3)
    assert certificate.type_ii_lower_bound == certificate.forward_lower_bound
    assert certificate.numerical_diagnostics == ("reverse_component_overflow",)
    assert certificate.reverse_component_overflowed
    assert not certificate.forward_component_underflowed
    assert len(certificate.fingerprint) == 64


@pytest.mark.parametrize("n", [10000, 2**63 - 1])
def test_large_supported_counts_preserve_finite_log_of_underflowed_forward(n):
    pair = _pair(n)
    certificate = bruno_converse_at_order(pair, epsilon=0.05, order=2)
    assert certificate.reverse_renyi == pytest.approx(n * math.log(1.25))
    assert certificate.forward_renyi == pytest.approx(n * math.log(4 / 3))
    assert certificate.reverse_lower_bound == -math.inf
    assert certificate.forward_lower_bound == 0
    assert certificate.type_ii_lower_bound == 0
    assert math.isfinite(certificate.log_forward_lower_bound)
    assert certificate.log_forward_lower_bound < -745
    assert certificate.numerical_diagnostics == (
        "reverse_component_overflow", "forward_component_underflow",
    )
    assert pair.count_totals == (n,)


@pytest.mark.parametrize("order", [math.nextafter(1.0, math.inf), 1.1, 1.7, 2, 5, 1e100, sys.float_info.max])
@pytest.mark.parametrize("epsilon", [math.ulp(0.0), 1e-300, 0.05, math.nextafter(1.0, 0.0)])
def test_finite_order_and_epsilon_extremes_have_finite_combined_certificates(order, epsilon):
    certificate = bruno_converse_at_order(_pair(), epsilon=epsilon, order=order)
    assert certificate.order == order
    assert certificate.type_i_constraint == epsilon
    assert math.isfinite(certificate.type_ii_lower_bound)
    assert 0 <= certificate.type_ii_lower_bound <= 1
    assert math.isfinite(certificate.reverse_log_power)
    assert math.isfinite(certificate.log_forward_lower_bound)
    assert certificate.type_ii_lower_bound == max(
        certificate.reverse_lower_bound, certificate.forward_lower_bound,
    )
    assert not certificate.global_envelope_certified


def test_order_immediately_above_one_reaches_existing_kernel_without_snapping(monkeypatch):
    order = math.nextafter(1.0, math.inf)
    existing = bruno._observation.renyi_multinomial
    seen = []

    def record(left, right, alpha):
        seen.append(alpha)
        return existing(left, right, alpha)

    monkeypatch.setattr(bruno._observation, "renyi_multinomial", record)
    certificate = bruno_converse_at_order(_pair(), epsilon=0.05, order=order)
    assert seen == [order, order]
    assert certificate.order.hex() == order.hex()
    assert certificate.reverse_lower_bound > 0
    assert certificate.reverse_lower_bound < 1e-14


def test_maximum_finite_order_uses_stable_existing_kernel():
    certificate = bruno_converse_at_order(_pair(), epsilon=0.05, order=sys.float_info.max)
    # At this representable order the O(1/lambda) terms are below float precision.
    assert certificate.reverse_renyi == pytest.approx(3 * math.log(1.5), abs=2e-15)
    assert certificate.forward_renyi == pytest.approx(3 * math.log(2), abs=2e-15)
    assert certificate.type_ii_lower_bound == pytest.approx(1 - 0.05 * 1.5**3, abs=2e-15)
    assert certificate.order == sys.float_info.max
    assert not certificate.global_envelope_certified


@pytest.mark.parametrize("order", [math.nextafter(1.0, math.inf), 1.1, 1.5, 2, 5, sys.float_info.max])
def test_identical_laws_keep_the_order_specific_formula(order):
    law = MultinomialMIDLaw(2**63 - 1, (0.5, 0.0, 0.5))
    epsilon = 0.05
    certificate = bruno_converse_at_order(SimpleBinaryLawPair(null=law, alternative=law),
                                          epsilon=epsilon, order=order)
    assert certificate.reverse_renyi == certificate.forward_renyi == 0
    expected_reverse = -math.expm1((order - 1) / order * math.log(epsilon))
    expected_forward = math.exp(order / (order - 1) * math.log1p(-epsilon))
    assert certificate.reverse_lower_bound == expected_reverse
    assert certificate.forward_lower_bound == expected_forward
    assert certificate.type_ii_lower_bound == max(expected_reverse, expected_forward)
    assert certificate.type_ii_lower_bound <= 1 - epsilon + 2e-16


@pytest.mark.parametrize("order", [math.nextafter(1.0, math.inf), 1.1, 1.7, 2, 5, sys.float_info.max])
def test_matching_support_very_close_laws_do_not_require_probability_repair(order):
    probabilities = (0.5 + 2**-52, 0.5 - 2**-52)
    pair = SimpleBinaryLawPair(null=MultinomialMIDLaw(11, (0.5, 0.5)),
                               alternative=MultinomialMIDLaw(11, probabilities))
    certificate = bruno_converse_at_order(pair, epsilon=0.05, order=order)
    assert certificate.reverse_renyi >= 0
    assert certificate.forward_renyi >= 0
    assert certificate.type_ii_lower_bound > 0
    assert pair.alternative.probabilities == probabilities


@pytest.mark.parametrize("value", [True, False, np.bool_(True), "2", None, [], math.nan, math.inf, -math.inf, 1, 0, -1])
def test_invalid_order_fails_at_public_certificate_boundary(value):
    with pytest.raises(InputValidationError):
        bruno_converse_at_order(_pair(), epsilon=0.05, order=value)


@pytest.mark.parametrize("value", [True, False, np.bool_(False), "0.05", None, [], math.nan, math.inf, -math.inf, 0, 1, -0.05, 1.05])
def test_invalid_epsilon_fails_at_public_certificate_boundary(value):
    with pytest.raises(InputValidationError):
        bruno_converse_at_order(_pair(), epsilon=value, order=2)


@pytest.mark.parametrize("laws", [None, [], (), {}, object(), MultinomialMIDLaw(1, (0.5, 0.5))])
def test_arbitrary_inputs_cannot_bypass_explicit_null_and_alternative_records(laws):
    with pytest.raises(InputValidationError, match="SimpleBinaryLawPair"):
        bruno_converse_at_order(laws, epsilon=0.05, order=2)


@pytest.mark.parametrize("epsilon,order", [
    (0.05, Fraction(10**400 + 1, 10**400)),
    (0.05, 10**400),
    (Fraction(1, 10**400), 2),
    (Fraction(10**400 - 1, 10**400), 2),
])
def test_unrepresentable_admissible_reals_fail_explicitly_without_snapping(epsilon, order):
    with pytest.raises(NumericalLimitError):
        bruno_converse_at_order(_pair(), epsilon=epsilon, order=order)


@pytest.mark.parametrize("value", [-1e-20, -math.inf, math.inf, math.nan])
@pytest.mark.parametrize("direction", ["reverse", "forward"])
def test_inherited_negative_or_nonfinite_divergence_fails_without_clipping(monkeypatch, value, direction):
    pair = _pair()
    existing = bruno._observation.renyi_multinomial

    def limit(left, right, order):
        targeted_left = pair.alternative if direction == "reverse" else pair.null
        return value if left is targeted_left else existing(left, right, order)

    monkeypatch.setattr(bruno._observation, "renyi_multinomial", limit)
    with pytest.raises(NumericalLimitError, match=f"{direction}.*inherited.*law divergence"):
        bruno_converse_at_order(pair, epsilon=0.05, order=2)


@pytest.mark.parametrize("error", [OverflowError("range"), ValidationError("roundoff boundary")])
def test_inherited_kernel_numerical_error_preserves_its_cause(monkeypatch, error):
    def limit(*args, **kwargs):
        raise error

    monkeypatch.setattr(bruno._observation, "renyi_multinomial", limit)
    with pytest.raises(NumericalLimitError, match="reverse.*existing observation divergence") as result:
        bruno_converse_at_order(_pair(), epsilon=0.05, order=2)
    assert result.value.__cause__ is error


def test_certificate_is_frozen_and_binds_order_epsilon_and_law_roles():
    pair = _pair()
    certificate = bruno_converse_at_order(pair, epsilon=0.05, order=1.7)
    assert certificate.pair is pair
    assert certificate == bruno_converse_at_order(pair, epsilon=0.05, order=1.7)
    assert certificate.fingerprint != bruno_converse_at_order(pair, epsilon=0.06, order=1.7).fingerprint
    assert certificate.fingerprint != bruno_converse_at_order(pair, epsilon=0.05, order=1.8).fingerprint
    changed = replace(pair, alternative=MultinomialMIDLaw(3, (0.375, 0.625)))
    assert certificate.fingerprint != bruno_converse_at_order(changed, epsilon=0.05, order=1.7).fingerprint
    with pytest.raises(FrozenInstanceError):
        certificate.order = 2.0
