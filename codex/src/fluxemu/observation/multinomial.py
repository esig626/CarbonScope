"""Fixed-total multinomial laws for measurements with genuine count semantics.

Probabilities are never normalized or repaired. The existing machine-simplex
boundary is strengthened by an explicit representability check: the logarithm
of the represented PMF's total mass, ``n * log(sum(p))``, must have magnitude at
most ``MAX_MULTINOMIAL_LOG_MASS_ERROR``. This prevents large totals amplifying
otherwise harmless floating-point simplex residuals. No effective total is
inferred from a normalized composition or an intensity measurement.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
from numbers import Integral, Real

import numpy as np

from ..exceptions import InputValidationError, ValidationError
from ..mfa import divergence as _divergence
from .schema import (
    MIDCountObservation,
    _count_total,
    _count_vector,
    _observation_digest,
)


MAX_MULTINOMIAL_LOG_MASS_ERROR = _divergence.DEFAULT_MID_TOLERANCE
_DECIMAL_PRECISION = 60
_LOG_SQRT_TWO_PI = Decimal(
    "0.918938533204672741780329736405617639861397473637783412817151540483"
)
_STIRLING_COEFFICIENTS = (
    (1, 12), (-1, 360), (1, 1260), (-1, 1680), (1, 1188),
    (-691, 360360), (1, 156), (-3617, 122400),
)


def _log_factorial(value: int) -> Decimal:
    """Stable log-factorials within the caller's Decimal context.

    Only the bounded small-count branch forms an exact integer factorial.
    Above 32, the Stirling series through x**-15 has absolute remainder below
    4e-27. Sixty decimal digits prevent cancellation of O(n log n) terms over
    the explicitly supported signed-int64 total-count range.
    """
    if value <= 32:
        return Decimal(math.factorial(value)).ln()
    x = Decimal(value)
    inverse = 1 / x
    square = inverse * inverse
    power = inverse
    correction = Decimal(0)
    for numerator, denominator in _STIRLING_COEFFICIENTS:
        correction += (Decimal(numerator) / denominator) * power
        power *= square
    return (x + Decimal("0.5")) * x.ln() - x + _LOG_SQRT_TWO_PI + correction


def _probabilities(values: Iterable[Real], n: int) -> tuple[float, ...]:
    if isinstance(values, (Mapping, Set)):
        raise InputValidationError("probabilities require a declared positional order")
    try:
        source = tuple(values)
    except TypeError as error:
        raise InputValidationError("probabilities must be a probability MID") from error
    # Check source scalars before float conversion can erase a tiny negative
    # sign or round an out-of-range component to one. The shared validator
    # handles non-real, boolean, nonfinite, dimension and simplex validation.
    for value in source:
        if isinstance(value, Real) and not isinstance(value, bool):
            if value < 0 or value > 1:
                raise InputValidationError("probabilities must lie in [0, 1]")
    result = _divergence.validate_mid(
        source,
        tolerance=_divergence.MACHINE_SIMPLEX_TOLERANCE,
        name="multinomial probabilities",
    )
    if any(original > 0 and converted == 0 for original, converted in zip(source, result, strict=True)):
        raise ValidationError("probability conversion to float would erase positive support")
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        represented_mass = sum((Decimal.from_float(value) for value in result), Decimal(0))
        log_mass_error = abs(Decimal(n) * represented_mass.ln())
        if log_mass_error > Decimal.from_float(MAX_MULTINOMIAL_LOG_MASS_ERROR):
            raise InputValidationError(
                "total count amplifies the represented probability mass residual: "
                f"abs(n * log(sum(p)))={log_mass_error} exceeds "
                f"{MAX_MULTINOMIAL_LOG_MASS_ERROR}; probabilities are not repaired"
            )
    return result


def _resolve_rng(
    *, seed: int | None = None, rng: np.random.Generator | None = None,
) -> np.random.Generator:
    """Use exactly one explicit seed or caller-owned NumPy generator."""
    if (seed is None) == (rng is None):
        raise InputValidationError("provide exactly one of seed or rng")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise InputValidationError("rng must be a numpy.random.Generator")
        return rng
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise InputValidationError("seed must be a nonnegative integer, not bool")
    return np.random.default_rng(int(seed))


@dataclass(frozen=True, slots=True)
class MultinomialMIDLaw:
    """Immutable ``Multinomial(n, probabilities)`` in mass-class order.

    ``n`` is an explicitly declared genuine count total in [1, 2**63-1].
    Entries retain exact zeros and their declared order. Besides machine-simplex
    validation, an amplified mass residual above 1e-9 is rejected explicitly.
    The law does not infer count semantics from the numeric appearance of data.
    """

    n: int
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        n = _count_total(self.n)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "probabilities", _probabilities(self.probabilities, n))

    @property
    def mass_classes(self) -> tuple[int, ...]:
        return tuple(range(len(self.probabilities)))

    @property
    def fingerprint(self) -> str:
        return _observation_digest(
            ("multinomial-mid-law-v1", self.n, self.mass_classes, self.probabilities)
        )

    def log_pmf(self, counts: Iterable[int] | MIDCountObservation) -> float:
        """Evaluate log P(k); malformed counts fail, off-support counts are -inf.

        A valid nonnegative integer vector with a different total is outside
        support, including an all-zero vector. A measurement record separately
        requires a strictly positive matching declared total. Zero counts do
        not evaluate log(p), even at p=0. No floor or clipping is applied.
        """
        if isinstance(counts, MIDCountObservation):
            raw = counts.counts
        else:
            if isinstance(counts, (Mapping, Set)):
                raise InputValidationError("counts require a declared positional order")
            try:
                raw = tuple(counts)
            except TypeError as error:
                raise InputValidationError("counts must be a count vector or MIDCountObservation") from error
        values = _count_vector(raw, expected_size=len(self.probabilities))
        if sum(values) != self.n:
            return -math.inf
        if any(k > 0 and p == 0 for k, p in zip(values, self.probabilities, strict=True)):
            return -math.inf
        with localcontext() as context:
            context.prec = _DECIMAL_PRECISION
            result = _log_factorial(self.n)
            result -= sum((_log_factorial(k) for k in values), Decimal(0))
            result += sum(
                (Decimal(k) * Decimal.from_float(p).ln()
                 for k, p in zip(values, self.probabilities, strict=True) if k > 0),
                Decimal(0),
            )
            return float(result)

    def log_prob(self, counts: Iterable[int] | MIDCountObservation) -> float:
        """Alias for :meth:`log_pmf`."""
        return self.log_pmf(counts)

    def sample(
        self, *, seed: int | None = None, rng: np.random.Generator | None = None,
    ) -> MIDCountObservation:
        """Draw one raw count observation from an explicit seed or Generator.

        NumPy assigns the final cell its residual probability. Sampling only
        positive support, with the largest positive cell last, ensures this
        floating-point residual can never fill an exact-zero class. Results
        are scattered back to the original class order; stored p is unchanged.
        A supplied generator advances; the immutable law never changes.
        """
        generator = _resolve_rng(seed=seed, rng=rng)
        support = [index for index, p in enumerate(self.probabilities) if p > 0]
        residual = max(support, key=lambda index: self.probabilities[index])
        sample_order = [index for index in support if index != residual] + [residual]
        sampled = generator.multinomial(self.n, [self.probabilities[index] for index in sample_order])
        counts = [0] * len(self.probabilities)
        for index, value in zip(sample_order, sampled, strict=True):
            counts[index] = int(value)
        return MIDCountObservation(counts=tuple(counts), n=self.n)


def _comparable(p: MultinomialMIDLaw, q: MultinomialMIDLaw) -> None:
    if not isinstance(p, MultinomialMIDLaw) or not isinstance(q, MultinomialMIDLaw):
        raise InputValidationError("divergence requires two MultinomialMIDLaw records")
    if p.n != q.n or p.mass_classes != q.mass_classes:
        raise InputValidationError("multinomial divergence requires the same total and mass-class space")


def kl_multinomial(p: MultinomialMIDLaw, q: MultinomialMIDLaw) -> float:
    """Exact law identity ``D_KL(Mult(n,p) || Mult(n,q)) = n D_KL(p||q)``.

    Comparisons require a common total and ordered mass-class space. The
    existing MID kernel retains all support and numerical semantics.
    """
    _comparable(p, q)
    return p.n * _divergence.kl_divergence(p.probabilities, q.probabilities)


def renyi_multinomial(p: MultinomialMIDLaw, q: MultinomialMIDLaw, alpha: Real) -> float:
    """Return ``n D_alpha(p||q)`` for any finite real alpha>0; one is KL."""
    _comparable(p, q)
    order = _divergence.validate_alpha(alpha)
    if order == 1:
        return kl_multinomial(p, q)
    return p.n * _divergence.renyi_divergence(p.probabilities, q.probabilities, order)


def _product_pairs(
    p: Iterable[MultinomialMIDLaw], q: Iterable[MultinomialMIDLaw],
) -> tuple[tuple[MultinomialMIDLaw, MultinomialMIDLaw], ...]:
    if isinstance(p, (Mapping, Set)) or isinstance(q, (Mapping, Set)):
        raise InputValidationError("independent products require a declared block order")
    try:
        left, right = tuple(p), tuple(q)
    except TypeError as error:
        raise InputValidationError("independent products require ordered law collections") from error
    if not left or len(left) != len(right):
        raise InputValidationError("independent products require the same positive number of blocks")
    pairs = tuple(zip(left, right, strict=True))
    for a, b in pairs:
        _comparable(a, b)
    return pairs


def independent_product_kl(
    p: Iterable[MultinomialMIDLaw], q: Iterable[MultinomialMIDLaw],
) -> float:
    """Add law KL over explicitly independent corresponding ordered blocks.

    Totals can differ between blocks; each corresponding pair must share n.
    Independence is the caller's declared modeling assumption, not inferred.
    """
    return math.fsum(kl_multinomial(a, b) for a, b in _product_pairs(p, q))


def independent_product_renyi(
    p: Iterable[MultinomialMIDLaw], q: Iterable[MultinomialMIDLaw], alpha: Real,
) -> float:
    """Add finite-order law Rényi divergences under declared independence."""
    order = _divergence.validate_alpha(alpha)
    if order == 1:
        return independent_product_kl(p, q)
    return math.fsum(renyi_multinomial(a, b, order) for a, b in _product_pairs(p, q))


def multinomial_count_constant(observation: MIDCountObservation) -> float:
    """The data-only ``C(k)`` in ``-log P_p(k) = n KL(k/n||p) + C(k)``.

    This identity characterizes genuine fixed-total multinomial likelihoods;
    it does not change the existing unweighted multi-MID MFA objective.
    """
    if not isinstance(observation, MIDCountObservation):
        raise InputValidationError("count constant requires a MIDCountObservation")
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        total = Decimal(observation.n)
        result = -_log_factorial(observation.n)
        result += sum((_log_factorial(k) for k in observation.counts), Decimal(0))
        result -= sum(
            (Decimal(k) * (Decimal(k) / total).ln() for k in observation.counts if k > 0),
            Decimal(0),
        )
        return float(result)


__all__ = [
    "MAX_MULTINOMIAL_LOG_MASS_ERROR", "MultinomialMIDLaw", "kl_multinomial",
    "renyi_multinomial", "independent_product_kl", "independent_product_renyi",
    "multinomial_count_constant",
]
