"""Explicit preprocessing of non-negative MID-like measurements.

The divergence layer deliberately refuses to repair its inputs.  Experimental
fractions, percentages, or intensity vectors that require closure to the
probability simplex must therefore be normalised explicitly before they are
stored in a :class:`StationaryMIDObservation`.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from numbers import Integral, Real

from ..exceptions import InputValidationError, ValidationError
from .divergence import MACHINE_SIMPLEX_TOLERANCE


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


def normalise_mid(
    values: Iterable[Real],
    *,
    expected_size: int | None = None,
    name: str = "MID",
) -> tuple[float, ...]:
    """Explicitly close non-negative measurements to a probability MID.

    This function is intentionally separate from KL/Rényi evaluation: calling
    it is an explicit preprocessing decision, never a hidden repair performed
    by the fitting objective.  Inputs may be rounded fractions, percentages,
    or non-negative intensities; their absolute total is discarded and only
    their composition is retained.

    No pseudocounts, clipping, support filling, or sign repair are performed.
    At least one entry must be strictly positive.  The returned tuple has the
    original order and sums to one within the divergence layer's machine-simplex
    tolerance.
    """

    if expected_size is not None and (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, Integral)
        or expected_size < 1
    ):
        raise InputValidationError("expected MID size must be a positive integer")
    try:
        entries = tuple(values)
    except TypeError as exc:
        raise InputValidationError(f"{name} must be a one-dimensional MID vector") from exc
    if not entries:
        raise InputValidationError(f"{name} must contain at least one mass class")
    if expected_size is not None and len(entries) != expected_size:
        raise InputValidationError(
            f"{name} has {len(entries)} mass classes; expected {expected_size}"
        )

    raw = tuple(
        _finite_real(value, f"{name}[{index}]")
        for index, value in enumerate(entries)
    )
    if any(value < 0.0 for value in raw):
        raise InputValidationError(f"{name} entries must be nonnegative")

    scale = max(raw)
    if scale <= 0.0:
        raise InputValidationError(f"{name} must contain positive total mass")
    scaled = tuple(value / scale for value in raw)
    if any(value > 0.0 and scaled_value == 0.0 for value, scaled_value in zip(raw, scaled, strict=True)):
        raise ValidationError(
            f"{name} dynamic range is too large to normalise without losing positive support"
        )
    total = math.fsum(scaled)
    if not math.isfinite(total) or total <= 0.0:
        raise InputValidationError(f"{name} must contain finite positive total mass")

    normalised = tuple(value / total for value in scaled)
    mass = math.fsum(normalised)
    if mass != 1.0:
        normalised = tuple(value / mass for value in normalised)
        mass = math.fsum(normalised)
    if abs(mass - 1.0) > MACHINE_SIMPLEX_TOLERANCE:
        raise ValidationError(
            f"{name} could not be normalised to machine-simplex precision; residual={abs(mass - 1.0)}"
        )
    if any(value > 0.0 and normalised_value == 0.0 for value, normalised_value in zip(raw, normalised, strict=True)):
        raise ValidationError(
            f"{name} normalisation would erase positive support"
        )
    return normalised


__all__ = ["normalise_mid"]
