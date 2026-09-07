"""Simple-binary role, support, scalar-domain and bound-record contracts."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    BrunoOrderBound,
    BrunoTheoremAssumptionError,
    NumericalLimitError,
    SimpleBinaryLawPair,
    SimpleBinaryTestingConstraint,
    validate_bruno_assumptions,
    validate_renyi_order,
)


def _pair():
    return SimpleBinaryLawPair(
        null=MultinomialMIDLaw(4, (0.25, 0.75)),
        alternative=MultinomialMIDLaw(4, (0.5, 0.5)),
    )


def _bound():
    law = MultinomialMIDLaw(4, (0.25, 0.75))
    epsilon = 0.25
    reverse = 1 - math.sqrt(epsilon)
    forward = (1 - epsilon) ** 2
    return BrunoOrderBound(
        pair=SimpleBinaryLawPair(null=law, alternative=law),
        constraint=SimpleBinaryTestingConstraint(epsilon=epsilon),
        order=2.0,
        reverse_renyi=0.0,
        forward_renyi=0.0,
        reverse_lower_bound=reverse,
        forward_lower_bound=forward,
        type_ii_lower_bound=max(reverse, forward),
        log_forward_lower_bound=2 * math.log1p(-epsilon),
        reverse_log_power=0.5 * math.log(epsilon),
    )


def test_null_and_alternative_roles_are_explicit_and_frozen():
    pair = _pair()
    assert pair.null.probabilities == (0.25, 0.75)
    assert pair.alternative.probabilities == (0.5, 0.5)
    with pytest.raises(TypeError):
        SimpleBinaryLawPair(pair.null, pair.alternative)
    with pytest.raises(FrozenInstanceError):
        pair.null = pair.alternative


@pytest.mark.parametrize("epsilon", [True, None, "0.05", math.nan, math.inf, -1, 0, 1, 2])
def test_type_i_constraint_rejects_invalid_values(epsilon):
    with pytest.raises(InputValidationError):
        SimpleBinaryTestingConstraint(epsilon=epsilon)


def test_type_i_constraint_retains_representable_interior_values():
    for epsilon in (math.nextafter(0.0, 1.0), 0.05, Fraction(1, 3), math.nextafter(1.0, 0.0)):
        assert SimpleBinaryTestingConstraint(epsilon=epsilon).epsilon == float(epsilon)


def test_unrepresentable_epsilon_fails_instead_of_rounding_to_endpoint():
    with pytest.raises(NumericalLimitError, match="endpoint"):
        SimpleBinaryTestingConstraint(epsilon=Fraction(1, 10**1000))


@pytest.mark.parametrize("order", [True, None, "2", math.nan, math.inf, -1, 0, 1])
def test_renyi_order_rejects_invalid_values(order):
    with pytest.raises(InputValidationError):
        validate_renyi_order(order)


def test_renyi_order_has_no_grid_or_snapping():
    for order in (math.nextafter(1.0, math.inf), 1.000000001, 1.1, Fraction(3, 2), 5.25):
        assert validate_renyi_order(order) == float(order)


def test_exact_support_mismatch_blocks_bruno_bound_without_repair():
    pair = SimpleBinaryLawPair(
        null=MultinomialMIDLaw(4, (0.0, 0.25, 0.75)),
        alternative=MultinomialMIDLaw(4, (0.25, 0.25, 0.5)),
        block_identities=(("exp", "target", "rep"),),
    )
    assert pair.mutually_absolutely_continuous is False
    with pytest.raises(BrunoTheoremAssumptionError, match="mutual absolute continuity"):
        validate_bruno_assumptions(pair)
    assert pair.null.probabilities[0] == 0.0


def test_declared_independent_products_preserve_block_totals_and_order():
    pair = SimpleBinaryLawPair(
        null=(MultinomialMIDLaw(4, (0.25, 0.75)), MultinomialMIDLaw(11, (0.5, 0.0, 0.5))),
        alternative=(MultinomialMIDLaw(4, (0.5, 0.5)), MultinomialMIDLaw(11, (0.25, 0.0, 0.75))),
        independent=True,
        block_identities=(("z", "a", "2"), ("a", "z", "1")),
    )
    assert pair.count_totals == (4, 11)
    assert pair.block_identities == (("z", "a", "2"), ("a", "z", "1"))
    with pytest.raises(InputValidationError, match="independent=True"):
        SimpleBinaryLawPair(null=pair.null, alternative=pair.alternative)


def test_bruno_order_bound_retains_roles_components_and_order_specific_provenance():
    bound = _bound()
    assert bound.type_i_constraint == 0.25
    assert bound.order == 2.0
    assert bound.reverse_lower_bound == 0.5
    assert bound.forward_lower_bound == bound.type_ii_lower_bound == 0.5625
    assert bound.null_fingerprint == bound.pair.null.fingerprint
    assert bound.alternative_fingerprint == bound.pair.alternative.fingerprint
    assert bound.global_envelope_evaluated is False
    assert bound.numerical_diagnostics == ()
    assert len(bound.fingerprint) == 64
    assert bound.fingerprint == _bound().fingerprint
    assert bound.fingerprint != replace(bound, order=1.7).fingerprint
    with pytest.raises(FrozenInstanceError):
        bound.type_ii_lower_bound = 1.0


def test_bruno_order_bound_exposes_overflow_and_underflow_without_clipping():
    bound = replace(
        _bound(),
        reverse_lower_bound=-math.inf,
        forward_lower_bound=0.0,
        type_ii_lower_bound=0.0,
        log_forward_lower_bound=-1000.0,
        reverse_log_power=1000.0,
    )
    assert bound.reverse_component_overflowed is True
    assert bound.forward_component_underflowed is True
    assert bound.numerical_diagnostics == (
        "reverse_component_overflow",
        "forward_component_underflow",
    )
