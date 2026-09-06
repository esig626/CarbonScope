"""Likelihood-ratio evidence and exact simple-null p-values for genuine counts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
from numbers import Integral

from ..exceptions import InputValidationError, ValidationError
from ..observation import MIDCountObservation, MultinomialMIDLaw
from .simple import NumericalLimitError, SimpleBinaryLawPair
from .stationary import StationarySimpleTestingResult


DEFAULT_EXACT_P_VALUE_MAX_OUTCOMES = 1_000_000


class UndefinedLikelihoodRatioError(InputValidationError):
    """The realised count sample has zero probability under both complete laws."""


class ExactPValueEnumerationLimitError(ValidationError):
    """The exact null sample space is larger than the caller's enumeration limit."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LikelihoodRatioPValue:
    """Exact one-sided simple-null likelihood-ratio p-value with provenance.

    ``p_value`` is P0{LLR(Y) >= LLR(y_obs)} for
    LLR=log(P1/P0). It is tied to this fixed law pair and this realised
    genuine-count observation. ``log_p_value`` is retained when exponentiation
    underflows.
    """

    pair: SimpleBinaryLawPair
    observations: tuple[tuple[int, ...], ...]
    p_value: float
    log_p_value: float
    observed_log_likelihood_ratio: float
    null_outcomes: int
    tail_outcomes: int

    def __post_init__(self) -> None:
        if not isinstance(self.pair, SimpleBinaryLawPair):
            raise InputValidationError("pair must be SimpleBinaryLawPair")
        if not isinstance(self.observations, tuple) or len(self.observations) != len(self.pair.null_laws):
            raise InputValidationError("observations must preserve the declared law-block order")
        if not math.isfinite(self.p_value) or not 0 <= self.p_value <= 1:
            raise InputValidationError("p_value must be finite and lie in [0, 1]")
        if math.isnan(self.log_p_value) or self.log_p_value > 0:
            raise InputValidationError("log_p_value must be nonpositive")
        if math.isnan(self.observed_log_likelihood_ratio):
            raise InputValidationError("observed_log_likelihood_ratio cannot be NaN")
        for name in ("null_outcomes", "tail_outcomes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise InputValidationError(f"{name} must be a nonnegative integer, not bool")
        if self.null_outcomes < 1 or self.tail_outcomes > self.null_outcomes:
            raise InputValidationError("tail_outcomes must lie between zero and null_outcomes")

    @property
    def underflowed(self) -> bool:
        return self.p_value == 0.0 and math.isfinite(self.log_p_value)

    @property
    def null_fingerprint(self) -> str:
        return self.pair.null_fingerprint

    @property
    def alternative_fingerprint(self) -> str:
        return self.pair.alternative_fingerprint


def _raw_counts(observation: MIDCountObservation | Iterable[int]) -> tuple[int, ...]:
    if isinstance(observation, MIDCountObservation):
        return observation.counts
    if isinstance(observation, (Mapping, Set)):
        raise InputValidationError("counts require a declared positional order")
    try:
        return tuple(observation)
    except TypeError as error:
        raise InputValidationError("observation must be a count vector or MIDCountObservation") from error


def _resolve_pair(
    laws: SimpleBinaryLawPair | StationarySimpleTestingResult,
) -> SimpleBinaryLawPair:
    pair = laws.law_pair if isinstance(laws, StationarySimpleTestingResult) else laws
    if not isinstance(pair, SimpleBinaryLawPair):
        raise InputValidationError(
            "simple likelihood evaluation requires SimpleBinaryLawPair or "
            "StationarySimpleTestingResult"
        )
    return pair


def _resolve_samples(
    pair: SimpleBinaryLawPair,
    observations: MIDCountObservation | Iterable[int] | tuple[MIDCountObservation | Iterable[int], ...],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(pair.null, MultinomialMIDLaw):
        return (_raw_counts(observations),)
    if not isinstance(observations, tuple) or len(observations) != len(pair.null_laws):
        raise InputValidationError(
            "product observations require an immutable tuple matching the declared block order"
        )
    return tuple(_raw_counts(item) for item in observations)


def _llr_score(
    pair: SimpleBinaryLawPair,
    samples: tuple[tuple[int, ...], ...],
) -> Decimal | float:
    null_possible = True
    alternative_possible = True
    for null, alternative, counts in zip(
        pair.null_laws, pair.alternative_laws, samples, strict=True,
    ):
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
        for null, alternative, counts in zip(
            pair.null_laws, pair.alternative_laws, samples, strict=True,
        ):
            for count, p0, p1 in zip(
                counts, null.probabilities, alternative.probabilities, strict=True,
            ):
                if count:
                    total += Decimal(int(count)) * (
                        Decimal.from_float(p1).ln() - Decimal.from_float(p0).ln()
                    )
        return total


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
    infinities cannot silently produce NaN. Bruno-bound evaluation has a
    stronger mutual-absolute-continuity requirement; this utility does not.

    When both probabilities are positive, common multinomial coefficients
    cancel exactly. Sixty-digit Decimal count-weighted log ratios avoid the
    cancellation from subtracting two rounded, very large log PMFs.
    """
    pair = _resolve_pair(laws)
    samples = _resolve_samples(pair, observations)
    return float(_llr_score(pair, samples))


def _validate_max_outcomes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise InputValidationError("max_outcomes must be a positive integer, not bool")
    return int(value)


def _null_outcome_count(pair: SimpleBinaryLawPair, max_outcomes: int) -> int:
    count = 1
    for law in pair.null_laws:
        support_size = sum(probability > 0 for probability in law.probabilities)
        count *= math.comb(law.n + support_size - 1, support_size - 1)
        if count > max_outcomes:
            raise ExactPValueEnumerationLimitError(
                "exact likelihood-ratio p-value requires enumeration of "
                f"{count} or more positive-probability null outcomes, exceeding "
                f"max_outcomes={max_outcomes}; no asymptotic approximation was substituted"
            )
    return count


def _null_support_counts(law: MultinomialMIDLaw):
    support = tuple(index for index, probability in enumerate(law.probabilities) if probability > 0)
    size = len(law.probabilities)

    def compositions(total: int, cells: int):
        if cells == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, cells - 1):
                yield (first, *rest)

    for positive_counts in compositions(law.n, len(support)):
        counts = [0] * size
        for index, count in zip(support, positive_counts, strict=True):
            counts[index] = count
        yield tuple(counts)


def _joint_null_samples(pair: SimpleBinaryLawPair):
    laws = pair.null_laws

    def recurse(index: int, prefix: tuple[tuple[int, ...], ...]):
        if index == len(laws):
            yield prefix
            return
        for counts in _null_support_counts(laws[index]):
            yield from recurse(index + 1, (*prefix, counts))

    yield from recurse(0, ())


def _joint_null_log_pmf(
    pair: SimpleBinaryLawPair, samples: tuple[tuple[int, ...], ...],
) -> float:
    return math.fsum(
        law.log_pmf(counts)
        for law, counts in zip(pair.null_laws, samples, strict=True)
    )


def _score_at_least(candidate: Decimal | float, observed: Decimal) -> bool:
    if candidate == math.inf:
        return True
    if candidate == -math.inf:
        return False
    return candidate >= observed


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def likelihood_ratio_p_value(
    laws: SimpleBinaryLawPair | StationarySimpleTestingResult,
    observations: MIDCountObservation | Iterable[int] | tuple[MIDCountObservation | Iterable[int], ...],
    *,
    max_outcomes: int = DEFAULT_EXACT_P_VALUE_MAX_OUTCOMES,
) -> LikelihoodRatioPValue:
    r"""Return the exact finite-sample p-value P0{LLR(Y) >= LLR(y_obs)}.

    The null and alternative are the fixed laws already declared by ``laws``.
    The p-value therefore answers a sample-specific question under H0 and is
    alternative-specific through the likelihood-ratio ordering. It is not a
    divergence and it is not the Bruno Type-II lower bound.

    The calculation enumerates the complete positive-probability null count
    space, including explicitly independent product blocks. It uses no
    chi-square approximation, Monte Carlo tail estimate, smoothing, pseudo-count
    conversion, or effective sample-size inference. If the exact space exceeds
    ``max_outcomes`` it fails explicitly instead of changing the statistical
    procedure.
    """
    pair = _resolve_pair(laws)
    samples = _resolve_samples(pair, observations)
    limit = _validate_max_outcomes(max_outcomes)
    null_outcomes = _null_outcome_count(pair, limit)
    observed = _llr_score(pair, samples)

    if observed == math.inf:
        return LikelihoodRatioPValue(
            pair=pair,
            observations=samples,
            p_value=0.0,
            log_p_value=-math.inf,
            observed_log_likelihood_ratio=math.inf,
            null_outcomes=null_outcomes,
            tail_outcomes=0,
        )
    if observed == -math.inf:
        return LikelihoodRatioPValue(
            pair=pair,
            observations=samples,
            p_value=1.0,
            log_p_value=0.0,
            observed_log_likelihood_ratio=-math.inf,
            null_outcomes=null_outcomes,
            tail_outcomes=null_outcomes,
        )
    assert isinstance(observed, Decimal)

    log_tail = -math.inf
    tail_outcomes = 0
    for candidate_samples in _joint_null_samples(pair):
        candidate = _llr_score(pair, candidate_samples)
        if _score_at_least(candidate, observed):
            tail_outcomes += 1
            log_tail = _logaddexp(
                log_tail, _joint_null_log_pmf(pair, candidate_samples),
            )

    if tail_outcomes == null_outcomes:
        log_tail = 0.0
        p_value = 1.0
    else:
        if log_tail > 0:
            raise NumericalLimitError(
                "exact null-tail probability exceeded one at numerical precision; "
                "the represented law was not renormalised or clipped"
            )
        p_value = math.exp(log_tail)

    return LikelihoodRatioPValue(
        pair=pair,
        observations=samples,
        p_value=p_value,
        log_p_value=log_tail,
        observed_log_likelihood_ratio=float(observed),
        null_outcomes=null_outcomes,
        tail_outcomes=tail_outcomes,
    )


__all__ = [
    "DEFAULT_EXACT_P_VALUE_MAX_OUTCOMES", "ExactPValueEnumerationLimitError",
    "LikelihoodRatioPValue", "UndefinedLikelihoodRatioError",
    "likelihood_ratio_p_value", "log_likelihood_ratio",
]
