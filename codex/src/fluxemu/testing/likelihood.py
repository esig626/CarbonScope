"""Log P1 - log P0 for realised genuine-count samples, preserving exact support."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from decimal import Decimal, localcontext
import math

from ..exceptions import InputValidationError
from ..observation import MIDCountObservation, MultinomialMIDLaw
from .simple import SimpleBinaryLawPair
from .stationary import StationarySimpleTestingResult


class UndefinedLikelihoodRatioError(InputValidationError):
    """The realised count sample has zero probability under both complete laws."""


def _raw_counts(observation: MIDCountObservation | Iterable[int]) -> tuple[int, ...]:
    if isinstance(observation, MIDCountObservation):
        return observation.counts
    if isinstance(observation, (Mapping, Set)):
        raise InputValidationError("counts require a declared positional order")
    try:
        return tuple(observation)
    except TypeError as error:
        raise InputValidationError("observation must be a count vector or MIDCountObservation") from error


def log_likelihood_ratio(
    laws: SimpleBinaryLawPair | StationarySimpleTestingResult,
    observations: MIDCountObservation | Iterable[int] | tuple[MIDCountObservation | Iterable[int], ...],
) -> float:
    """Return natural-log LLR(y)=log P1(y)-log P0(y), with fixed H0/H1 roles.

    A single-law pair accepts one raw count vector or ``MIDCountObservation``.
    An explicitly independent product accepts an immutable tuple of count
    vectors/records, one per law block in its declared order. All count and
    mass-class validation, including exact-zero support, uses the existing
    observation laws; floats, intensities, and percentages are not converted.

    Return +inf when only P0(y)=0, and -inf when only P1(y)=0. If both complete
    laws assign zero probability, raise ``UndefinedLikelihoodRatioError``.
    Joint support is checked before aggregation, so opposite per-block
    infinities cannot silently produce NaN. Unlike Bruno certification, this
    utility does not require mutual absolute continuity.

    When both probabilities are positive, common multinomial coefficients
    cancel exactly. Sixty-digit Decimal count-weighted log ratios avoid the
    cancellation from subtracting two rounded, very large log PMFs. No test
    threshold, decision, or observed error probability is inferred.
    """
    pair = laws.law_pair if isinstance(laws, StationarySimpleTestingResult) else laws
    if not isinstance(pair, SimpleBinaryLawPair):
        raise InputValidationError("LLR requires SimpleBinaryLawPair or StationarySimpleTestingResult")
    if isinstance(pair.null, MultinomialMIDLaw):
        samples = (_raw_counts(observations),)
    else:
        if not isinstance(observations, tuple) or len(observations) != len(pair.null_laws):
            raise InputValidationError("product observations require an immutable tuple matching the declared block order")
        samples = tuple(_raw_counts(item) for item in observations)

    null_possible = True
    alternative_possible = True
    for null, alternative, counts in zip(pair.null_laws, pair.alternative_laws, samples, strict=True):
        # Validate both roles even if an earlier block already ruled one out.
        null_log_pmf = null.log_pmf(counts)
        alternative_log_pmf = alternative.log_pmf(counts)
        null_possible = null_possible and null_log_pmf != -math.inf
        alternative_possible = alternative_possible and alternative_log_pmf != -math.inf
    if not null_possible and not alternative_possible:
        raise UndefinedLikelihoodRatioError(
            "LLR is undefined: realised count sample has probability zero under "
            "both null P0 and alternative P1 complete observation laws"
        )
    if not null_possible:
        return math.inf
    if not alternative_possible:
        return -math.inf

    with localcontext() as context:
        context.prec = 60
        total = Decimal(0)
        for null, alternative, counts in zip(pair.null_laws, pair.alternative_laws, samples, strict=True):
            for count, p0, p1 in zip(counts, null.probabilities, alternative.probabilities, strict=True):
                if count:
                    total += Decimal(int(count)) * (
                        Decimal.from_float(p1).ln() - Decimal.from_float(p0).ln()
                    )
        return float(total)


__all__ = ["UndefinedLikelihoodRatioError", "log_likelihood_ratio"]
