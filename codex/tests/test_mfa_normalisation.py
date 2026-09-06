"""Explicit preprocessing of experimental MID-like measurements."""

import math
import sys

import pytest

from fluxemu.exceptions import InputValidationError, ValidationError
from fluxemu.mfa import kl_divergence, normalise_mid, renyi_divergence
from fluxemu.mfa.divergence import MACHINE_SIMPLEX_TOLERANCE


def test_rounded_fractions_are_explicitly_closed_to_the_simplex():
    raw = (0.3333, 0.3333, 0.3333)
    normalised = normalise_mid(raw)
    assert raw == (0.3333, 0.3333, 0.3333)
    assert abs(math.fsum(normalised) - 1.0) <= MACHINE_SIMPLEX_TOLERANCE
    assert kl_divergence(normalised, normalised) == 0.0
    assert renyi_divergence(normalised, normalised, 0.5) == 0.0
    assert renyi_divergence(normalised, normalised, 2.0) == 0.0


def test_percentages_and_intensities_preserve_composition_and_order():
    assert normalise_mid((30.0, 70.0)) == pytest.approx((0.3, 0.7), abs=2e-16)
    assert normalise_mid((300.0, 0.0, 700.0)) == pytest.approx(
        (0.3, 0.0, 0.7), abs=2e-16
    )


def test_large_finite_intensities_do_not_overflow_during_normalisation():
    huge = sys.float_info.max
    assert normalise_mid((huge, huge)) == (0.5, 0.5)


@pytest.mark.parametrize(
    "values",
    [
        (),
        (0.0, 0.0),
        (-1.0, 2.0),
        (math.nan, 1.0),
        (math.inf, 1.0),
        (True, False),
        ("30", "70"),
        1.0,
    ],
)
def test_invalid_measurements_are_rejected_without_repair(values):
    with pytest.raises(InputValidationError):
        normalise_mid(values)


def test_dimension_contract_is_explicit():
    assert len(normalise_mid((1.0, 2.0, 3.0), expected_size=3)) == 3
    with pytest.raises(InputValidationError, match="expected 2"):
        normalise_mid((1.0, 2.0, 3.0), expected_size=2)
    for invalid_size in (True, 0, -1, 2.5):
        with pytest.raises(InputValidationError):
            normalise_mid((1.0, 2.0), expected_size=invalid_size)


def test_support_is_not_silently_erased_by_extreme_rescaling():
    with pytest.raises(ValidationError, match="positive support"):
        normalise_mid((math.ulp(0.0), sys.float_info.max))
