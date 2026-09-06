"""Explicit simple-test roles, scalar domains, theorem gates, and provenance."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math
import sys

import numpy as np
import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    BrunoOrderCertificate,
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


def _product():
    return SimpleBinaryLawPair(
        null=(_pair().null, MultinomialMIDLaw(11, (0.5, 0.0, 0.5))),
        alternative=(_pair().alternative, MultinomialMIDLaw(11, (0.25, 0.0, 0.75))),
        independent=True,
        block_identities=(("z-exp", "a-target", "rep-2"), ("a-exp", "z-target", "rep-1")),
        null_state_fingerprint="null-state-fingerprint",
        alternative_state_fingerprint="alternative-state-fingerprint",
        observation_specification_fingerprint="observation-specification-fingerprint",
    )


def _certificate():
    law = MultinomialMIDLaw(4, (0.25, 0.75))
    epsilon = 0.25
    reverse = 1 - math.sqrt(epsilon)
    forward = (1 - epsilon) ** 2
    return BrunoOrderCertificate(
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


def test_null_and_alternative_are_keyword_only_roles_on_frozen_records():
    pair = _pair()
    assert pair.null.probabilities == (0.25, 0.75)
    assert pair.alternative.probabilities == (0.5, 0.5)
    assert pair.null_laws == (pair.null,)
    assert pair.alternative_laws == (pair.alternative,)
    assert pair.null_fingerprint == pair.null.fingerprint
    assert pair.alternative_fingerprint == pair.alternative.fingerprint
    assert pair.count_totals == (4,)
    with pytest.raises(TypeError):
        SimpleBinaryLawPair(pair.null, pair.alternative)
    with pytest.raises(FrozenInstanceError):
        pair.null = pair.alternative
    with pytest.raises(TypeError):
        SimpleBinaryTestingConstraint(0.05)


@pytest.mark.parametrize("epsilon", [
    True, False, np.bool_(False), None, "0.05", [], (), complex(0.05, 0),
    math.nan, math.inf, -math.inf, -1.0, 0, 1, 2,
])
def test_type_i_constraint_rejects_invalid_scalars_without_coercion(epsilon):
    with pytest.raises(InputValidationError):
        SimpleBinaryTestingConstraint(epsilon=epsilon)


@pytest.mark.parametrize("epsilon", [
    math.nextafter(0.0, 1.0), 0.05, Fraction(1, 3), np.float64(0.2),
    math.nextafter(1.0, 0.0),
])
def test_every_representable_interior_type_i_constraint_is_retained(epsilon):
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    assert constraint.epsilon == float(epsilon)
    assert constraint.type_i_constraint == float(epsilon)
    assert type(constraint.epsilon) is float
    assert len(constraint.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        constraint.epsilon = 0.1


@pytest.mark.parametrize("epsilon", [Fraction(1, 10**1000), 1 - Fraction(1, 10**1000)])
def test_unrepresentable_epsilon_never_becomes_an_endpoint(epsilon):
    with pytest.raises(NumericalLimitError, match="endpoint"):
        SimpleBinaryTestingConstraint(epsilon=epsilon)


@pytest.mark.parametrize("order", [
    True, False, np.bool_(True), None, "2", [], (), complex(2, 0),
    math.nan, math.inf, -math.inf, -1.0, 0, 0.9, 1,
])
def test_renyi_order_rejects_invalid_scalars_and_lambda_at_or_below_one(order):
    with pytest.raises(InputValidationError):
        validate_renyi_order(order)


@pytest.mark.parametrize("order", [
    math.nextafter(1.0, math.inf), 1.000000001, 1.1, Fraction(3, 2),
    math.pi, np.float64(1.7), np.int64(5), 5.25, 10**100, sys.float_info.max,
])
def test_order_validation_has_no_grid_or_near_one_substitution(order):
    assert validate_renyi_order(order) == float(order)


@pytest.mark.parametrize("order", [1 + Fraction(1, 10**100), 10**1000])
def test_unrepresentable_finite_orders_fail_with_explicit_numerical_limit(order):
    with pytest.raises(NumericalLimitError, match="float|rounds to 1"):
        validate_renyi_order(order)


def test_products_preserve_count_totals_block_order_and_explicit_identities():
    pair = _product()
    assert pair.independent is True
    assert pair.count_totals == (4, 11)
    assert pair.block_identities == (("z-exp", "a-target", "rep-2"), ("a-exp", "z-target", "rep-1"))
    assert pair.null_laws[1].probabilities == (0.5, 0.0, 0.5)
    assert pair.alternative_laws[1].probabilities == (0.25, 0.0, 0.75)
    assert pair.mutually_absolutely_continuous is True
    validate_bruno_assumptions(pair)
    assert pair.require_mutual_absolute_continuity() is None
    with pytest.raises(FrozenInstanceError):
        pair.block_identities = pair.block_identities[::-1]


@pytest.mark.parametrize("independent", [False, 0, 1, None, "true", np.bool_(True)])
def test_independent_products_require_literal_boolean_opt_in(independent):
    with pytest.raises(InputValidationError, match="independent"):
        replace(_product(), independent=independent)


def test_a_single_block_tuple_also_requires_declared_product_semantics():
    pair = _pair()
    with pytest.raises(InputValidationError, match="independent=True"):
        SimpleBinaryLawPair(null=(pair.null,), alternative=(pair.alternative,))
    product = SimpleBinaryLawPair(null=(pair.null,), alternative=(pair.alternative,), independent=True)
    assert product.count_totals == pair.count_totals
    assert product.fingerprint != pair.fingerprint


@pytest.mark.parametrize("changes", [
    {"null": []}, {"alternative": []}, {"null": ()}, {"alternative": ()},
    {"null": "not a law"}, {"null": ("not a law",)},
    {"alternative": (_pair().alternative,)},
    {"null": [_pair().null, _pair().null]},
    {"null": {_pair().null, _pair().alternative}},
])
def test_products_refuse_mutable_unordered_malformed_or_unaligned_blocks(changes):
    with pytest.raises(InputValidationError):
        replace(_product(), **changes)


def test_single_and_product_representations_cannot_be_mixed():
    with pytest.raises(InputValidationError, match="both be single"):
        replace(_pair(), alternative=(_pair().alternative,), independent=True)


@pytest.mark.parametrize("alternative,message", [
    (MultinomialMIDLaw(5, (0.5, 0.5)), "count totals"),
    (MultinomialMIDLaw(4, (0.25, 0.25, 0.5)), "mass-class"),
])
def test_corresponding_laws_require_identical_declared_outcome_spaces(alternative, message):
    with pytest.raises(InputValidationError, match=message):
        replace(_pair(), alternative=alternative)


@pytest.mark.parametrize("identities", [
    [], [["e", "t", "r"], ["e", "t2", "r"]],
    (("e", "t", "r"),),
    (("e", "t"), ("e", "t2", "r")),
    (("e", "t", "r"), ("e", "t", "r")),
    (("", "t", "r"), ("e", "t2", "r")),
    (("e", [], "r"), ("e", "t2", "r")),
    (("e", "t", "r"), ("e", "t2", 1)),
])
def test_block_identity_provenance_is_complete_unique_and_immutable(identities):
    with pytest.raises(InputValidationError):
        replace(_product(), block_identities=identities)


@pytest.mark.parametrize("changes", [
    {"null_state_fingerprint": []}, {"alternative_state_fingerprint": ""},
    {"observation_specification_fingerprint": {}},
    {"null_state_fingerprint": None}, {"alternative_state_fingerprint": None},
])
def test_state_and_observation_provenance_cannot_be_mutable_or_role_incomplete(changes):
    with pytest.raises(InputValidationError):
        replace(_product(), **changes)


def test_exact_positive_support_gate_rejects_bruno_assumption_and_preserves_zeros():
    null = MultinomialMIDLaw(4, (0.0, 0.25, 0.75))
    alternative = MultinomialMIDLaw(4, (0.25, 0.25, 0.5))
    pair = SimpleBinaryLawPair(null=null, alternative=alternative,
                              block_identities=(("exp", "target", "rep"),))
    assert pair.mutually_absolutely_continuous is False
    assert pair.null is null
    assert pair.alternative is alternative
    assert pair.null.probabilities == (0.0, 0.25, 0.75)
    with pytest.raises(BrunoTheoremAssumptionError, match="Theorem 1 requires mutual absolute continuity") as error:
        validate_bruno_assumptions(pair)
    assert "block 0 ('exp', 'target', 'rep')" in str(error.value)
    assert "null positive support (1, 2)" in str(error.value)
    assert "alternative positive support (0, 1, 2)" in str(error.value)
    with pytest.raises(BrunoTheoremAssumptionError):
        pair.require_mutual_absolute_continuity()


def test_product_support_assumptions_are_checked_for_every_ordered_block():
    pair = _product()
    mismatch = replace(pair, alternative=(pair.alternative[0], MultinomialMIDLaw(11, (0.25, 0.25, 0.5))))
    with pytest.raises(BrunoTheoremAssumptionError, match="block 1.*a-exp.*z-target.*rep-1"):
        validate_bruno_assumptions(mismatch)


def test_shared_zero_classes_and_single_point_support_are_valid_without_repair():
    null = MultinomialMIDLaw(9, (0.0, 1.0, 0.0))
    pair = SimpleBinaryLawPair(null=null, alternative=null)
    assert pair.mutually_absolutely_continuous is True
    validate_bruno_assumptions(pair)
    assert pair.null.probabilities == (0.0, 1.0, 0.0)


def test_theorem_gate_rejects_nonpair_inputs():
    with pytest.raises(InputValidationError, match="SimpleBinaryLawPair"):
        validate_bruno_assumptions((_pair().null, _pair().alternative))


def test_pair_fingerprint_binds_role_laws_count_totals_order_and_all_metadata():
    pair = _product()
    changes = (
        {"null": pair.alternative, "alternative": pair.null},
        {"null": pair.null[::-1], "alternative": pair.alternative[::-1],
         "block_identities": pair.block_identities[::-1]},
        {"null": (MultinomialMIDLaw(5, pair.null[0].probabilities), pair.null[1]),
         "alternative": (MultinomialMIDLaw(5, pair.alternative[0].probabilities), pair.alternative[1])},
        {"null": (MultinomialMIDLaw(4, (0.375, 0.625)), pair.null[1])},
        {"block_identities": (("new", "a-target", "rep-2"), pair.block_identities[1])},
        {"null_state_fingerprint": "another-null"},
        {"alternative_state_fingerprint": "another-alternative"},
        {"observation_specification_fingerprint": "another-specification"},
    )
    assert len(pair.fingerprint) == 64
    assert pair.fingerprint == _product().fingerprint
    assert all(replace(pair, **change).fingerprint != pair.fingerprint for change in changes)


def test_order_certificate_keeps_roles_components_and_order_specific_provenance():
    certificate = _certificate()
    assert certificate.type_i_constraint == 0.25
    assert certificate.order == 2.0
    assert certificate.reverse_renyi == certificate.forward_renyi == 0.0
    assert certificate.reverse_lower_bound == 0.5
    assert certificate.forward_lower_bound == certificate.type_ii_lower_bound == 0.5625
    assert certificate.null_fingerprint == certificate.pair.null.fingerprint
    assert certificate.alternative_fingerprint == certificate.pair.alternative.fingerprint
    assert certificate.global_envelope_certified is False
    assert certificate.numerical_diagnostics == ()
    assert len(certificate.fingerprint) == 64
    assert certificate.fingerprint == _certificate().fingerprint
    assert certificate.fingerprint != replace(certificate, order=1.7).fingerprint
    assert certificate.fingerprint != replace(certificate, constraint=SimpleBinaryTestingConstraint(epsilon=0.1)).fingerprint
    with pytest.raises(FrozenInstanceError):
        certificate.type_ii_lower_bound = 1.0


def test_certificate_retains_state_and_observation_fingerprints():
    certificate = replace(_certificate(), pair=_product())
    assert certificate.null_state_fingerprint == "null-state-fingerprint"
    assert certificate.alternative_state_fingerprint == "alternative-state-fingerprint"
    assert certificate.observation_specification_fingerprint == "observation-specification-fingerprint"


def test_certificate_constructor_cannot_skip_failed_bruno_assumption():
    pair = SimpleBinaryLawPair(null=MultinomialMIDLaw(4, (0.0, 1.0)), alternative=_pair().alternative)
    with pytest.raises(BrunoTheoremAssumptionError):
        replace(_certificate(), pair=pair)


def test_certificate_exposes_raw_overflow_and_underflow_without_clipping():
    certificate = replace(
        _certificate(), reverse_lower_bound=-math.inf, forward_lower_bound=0.0,
        type_ii_lower_bound=0.0, log_forward_lower_bound=-1000.0, reverse_log_power=1000.0,
    )
    assert certificate.reverse_lower_bound == -math.inf
    assert math.isfinite(certificate.type_ii_lower_bound)
    assert certificate.reverse_component_overflowed is True
    assert certificate.forward_component_underflowed is True
    assert certificate.numerical_diagnostics == ("reverse_component_overflow", "forward_component_underflow")
    assert len(certificate.fingerprint) == 64


@pytest.mark.parametrize("changes", [
    {"constraint": 0.05}, {"order": True}, {"reverse_renyi": math.nan},
    {"forward_renyi": math.inf}, {"reverse_lower_bound": math.inf},
    {"forward_lower_bound": -math.inf}, {"type_ii_lower_bound": math.nan},
    {"type_ii_lower_bound": 1.0}, {"reverse_lower_bound": 1.1},
    {"forward_lower_bound": 1.1}, {"forward_lower_bound": -0.1},
    {"log_forward_lower_bound": 0.1}, {"reverse_log_power": math.inf},
    {"forward_renyi": True},
])
def test_certificate_refuses_invalid_or_inconsistent_diagnostic_records(changes):
    with pytest.raises(InputValidationError):
        replace(_certificate(), **changes)


@pytest.mark.parametrize("name", ["reverse_renyi", "forward_renyi"])
def test_certificate_does_not_clip_negative_divergence_roundoff(name):
    with pytest.raises(NumericalLimitError, match="signed roundoff is not clipped"):
        replace(_certificate(), **{name: -1e-16})
