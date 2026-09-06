"""Information divergences of observed and predicted stationary MID vectors.

The direction is always ``observed || predicted``.  Inputs are never smoothed,
clipped, or normalized.  The caller's MID tolerance is a data-validation
boundary; evaluation additionally requires mass error no larger than sixteen
ulps at one.  A looser data tolerance cannot authorize changing the objective.
This explicit numerical boundary prevents a tolerated nonunit observed mass
from producing a spurious ``log(sum(observed)) / (alpha - 1)`` pole near one.

Within that machine-roundoff boundary, compensated simplex identities using
``expm1`` and ``log1p`` retain the requested order, including adjacent floats to
one.  Tiny signed roundoff in the answer is retained, rather than clipped.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from numbers import Integral, Real

from ..exceptions import InputValidationError, ValidationError


DEFAULT_MID_TOLERANCE = 1e-9
MACHINE_SIMPLEX_TOLERANCE = 16 * math.ulp(1.0)
_RESULT_ROUNDOFF_TOLERANCE = 64 * math.ulp(1.0)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(f"{name} must be a finite real number, not a boolean")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise InputValidationError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise InputValidationError(f"{name} must be a finite real number")
    return result


def validate_alpha(alpha: Real) -> float:
    """Validate a finite positive real order, without an order grid."""
    result = _finite_real(alpha, "alpha")
    if result <= 0:
        raise InputValidationError("alpha must be strictly positive")
    return result


def validate_mid_tolerance(tolerance: Real) -> float:
    """Validate the explicit, nonnegative MID mass tolerance."""
    result = _finite_real(tolerance, "MID tolerance")
    if result < 0:
        raise InputValidationError("MID tolerance must be nonnegative")
    return result


def validate_mid(
    mid: Iterable[Real],
    *,
    expected_size: int | None = None,
    tolerance: float = DEFAULT_MID_TOLERANCE,
    name: str = "MID",
) -> tuple[float, ...]:
    """Return an immutable validated MID, retaining its values and order.

    Each entry must be real, finite and nonnegative.  The nonempty vector must
    have the declared dimension and sum to one within ``tolerance``.  No value
    is repaired, including a negative value within the normalization tolerance.
    Passing this validation with a broad tolerance does not guarantee that the
    more stringent numerical divergence boundary can evaluate the vector.
    """
    tolerance = validate_mid_tolerance(tolerance)
    if expected_size is not None and (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, Integral)
        or expected_size < 1
    ):
        raise InputValidationError("expected MID size must be a positive integer")
    try:
        entries = tuple(mid)
    except TypeError as exc:
        raise InputValidationError(f"{name} must be a one-dimensional MID vector") from exc
    if not entries:
        raise InputValidationError(f"{name} must contain at least one mass class")
    if expected_size is not None and len(entries) != expected_size:
        raise InputValidationError(
            f"{name} has {len(entries)} mass classes; expected {expected_size}"
        )
    values = tuple(_finite_real(value, f"{name}[{index}]") for index, value in enumerate(entries))
    if any(value < 0 for value in values):
        raise InputValidationError(f"{name} entries must be nonnegative")
    try:
        mass = math.fsum(values)
    except OverflowError as exc:
        raise InputValidationError(f"{name} mass must sum to one") from exc
    if abs(mass - 1.0) > tolerance:
        raise InputValidationError(
            f"{name} mass must sum to one within {tolerance}; got {mass}"
        )
    return values


def _validated_pair(
    observed: Iterable[Real], predicted: Iterable[Real], tolerance: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    p = validate_mid(observed, tolerance=tolerance, name="observed MID")
    q = validate_mid(predicted, expected_size=len(p), tolerance=tolerance, name="predicted MID")
    observed_error = abs(math.fsum(p) - 1.0)
    predicted_error = abs(math.fsum(q) - 1.0)
    if max(observed_error, predicted_error) > MACHINE_SIMPLEX_TOLERANCE:
        raise InputValidationError(
            "divergence requires machine-simplex normalization: "
            f"observed residual={observed_error}, predicted residual={predicted_error}; "
            f"each must be <= {MACHINE_SIMPLEX_TOLERANCE}. "
            "The declared MID tolerance does not authorize renormalization."
        )
    return p, q


def _log_ratio(p: float, q: float) -> float:
    # Avoid cancellation for close entries and overflow/underflow of p / q.
    difference = p - q
    if abs(difference) <= 0.5 * q:
        return math.log1p(difference / q)
    return math.log(p) - math.log(q)


def _checked_result(value: float) -> float:
    if value == -math.inf or math.isnan(value) or value < -_RESULT_ROUNDOFF_TOLERANCE * max(1.0, abs(value)):
        raise ValidationError("MID divergence exceeded its negative roundoff tolerance")
    return value


def kl_divergence(
    observed: Iterable[Real],
    predicted: Iterable[Real],
    *,
    tolerance: float = DEFAULT_MID_TOLERANCE,
) -> float:
    """Compute ``sum(p * log(p / q))`` in the direction observed || predicted.

    Zero observed entries contribute zero; a positive observed entry with zero
    prediction gives positive infinity.  Both vectors must satisfy the explicit
    data tolerance and the machine-simplex evaluation boundary above.
    """
    p, q = _validated_pair(observed, predicted, tolerance)
    if p == q:
        return 0.0
    if any(pi > 0 and qi == 0 for pi, qi in zip(p, q, strict=True)):
        return math.inf
    return _checked_result(math.fsum(
        pi * _log_ratio(pi, qi) for pi, qi in zip(p, q, strict=True) if pi > 0
    ))


def renyi_divergence(
    observed: Iterable[Real],
    predicted: Iterable[Real],
    alpha: Real,
    *,
    tolerance: float = DEFAULT_MID_TOLERANCE,
) -> float:
    """Compute finite-order Rényi divergence; exactly order one calls KL.

    For every finite positive floating-point order other than one this is
    ``log(sum(p**alpha * q**(1-alpha))) / (alpha-1)``.  Above one, observed
    positive mass outside predicted support gives infinity.  Below one, only
    shared support contributes: partial overlap is finite and disjoint support
    gives infinity.  No near-one order is snapped to KL.

    Shift log ratios *before* multiplying by large orders to avoid overflow.
    Near one (or zero), a compensated exponential sum around observed (or
    predicted) mass avoids subtracting nearly equal logarithms.  These simplex
    identities are evaluated only on machine-normalized vectors.
    """
    alpha = validate_alpha(alpha)
    if alpha == 1:
        return kl_divergence(observed, predicted, tolerance=tolerance)
    p, q = _validated_pair(observed, predicted, tolerance)
    if p == q:
        return 0.0
    pairs = tuple(zip(p, q, strict=True))
    if alpha > 1 and any(pi > 0 and qi == 0 for pi, qi in pairs):
        return math.inf
    overlap = tuple((pi, qi, _log_ratio(pi, qi)) for pi, qi in pairs if pi > 0 and qi > 0)
    if not overlap:
        return math.inf
    offset = alpha - 1.0

    # Use the q-centered identity first near zero, even when the p-centered
    # exponents would also be small: the latter would cancel its leading mass
    # terms and lose an arbitrarily small positive alpha.
    if alpha <= 0.5 and all(abs(alpha * ratio) <= 0.5 for _, _, ratio in overlap):
        change = math.fsum([
            *(qi * math.expm1(alpha * ratio) for _, qi, ratio in overlap),
            *(-qi for pi, qi in pairs if pi == 0),
        ])
        if change > -1.0:
            return _checked_result(math.log1p(change) / offset)

    # The omitted p mass matters below one when q has exact zeros.
    if all(abs(offset * ratio) <= 0.5 for _, _, ratio in overlap):
        change = math.fsum([
            *(pi * math.expm1(offset * ratio) for pi, _, ratio in overlap),
            *(-pi for pi, qi in pairs if qi == 0),
        ])
        if change > -1.0:
            return _checked_result(math.log1p(change) / offset)

    if alpha > 1:
        largest_ratio = max(ratio for _, _, ratio in overlap)
        # Every exponent is nonpositive.  Overflow to -inf is harmless and
        # expresses numerical underflow of a negligible term, not smoothing.
        mass = math.fsum(
            pi * math.exp(offset * (ratio - largest_ratio))
            for pi, _, ratio in overlap
        )
        return _checked_result(largest_ratio + math.log(mass) / offset)

    # Both coefficients lie in (0, 1), so the below-one log terms are finite.
    terms = tuple(alpha * math.log(pi) + (1.0 - alpha) * math.log(qi) for pi, qi, _ in overlap)
    largest = max(terms)
    log_mass = largest + math.log(math.fsum(math.exp(term - largest) for term in terms))
    return _checked_result(log_mass / offset)


__all__ = [
    "DEFAULT_MID_TOLERANCE", "MACHINE_SIMPLEX_TOLERANCE", "kl_divergence",
    "renyi_divergence", "validate_alpha", "validate_mid", "validate_mid_tolerance",
]
