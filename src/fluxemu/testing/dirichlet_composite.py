"""Finite represented-family testing for continuous Dirichlet MID laws.

This is a parallel continuous-observation implementation.  It deliberately
does not reuse or generalise the repaired genuine-count enumeration kernels.
The exact finite minimax LP and count-space score calibration are inapplicable
to the continuous simplex and return explicit structured-workflow refusals.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
from numbers import Real

import numpy as np

from ..exceptions import FluxEMUError, InputValidationError
from ..observation import (
    DirichletLogLikelihoodScore,
    DirichletMIDLaw,
    DirichletNumericalError,
    renyi_dirichlet,
)
from ..observation.multinomial import _resolve_rng
from .simple import (
    NumericalLimitError,
    SimpleBinaryTestingConstraint,
    _finite_real,
    _testing_digest,
    validate_renyi_order,
)


DIRICHLET_MOMENT_VERIFICATION_TOLERANCE = 1e-10


class UnsupportedContinuousObservationError(FluxEMUError):
    """A requested finite-observation-space procedure has no continuous V1 analogue."""


class DirichletTestingAssumptionError(FluxEMUError):
    """A Dirichlet testing guarantee lacks a required modelling declaration."""


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class IndependentDirichletMIDProductLaw:
    """One complete law over explicitly independent MID blocks and replicates."""

    blocks: tuple[DirichletMIDLaw, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise InputValidationError(
                "independent Dirichlet product blocks must be a nonempty immutable tuple"
            )
        if not all(isinstance(block, DirichletMIDLaw) for block in self.blocks):
            raise InputValidationError(
                "every independent Dirichlet product block must be DirichletMIDLaw"
            )
        identities = self.block_identities
        if len(set(identities)) != len(identities):
            raise InputValidationError("independent Dirichlet block identities must be unique")

    @property
    def block_identities(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(block.observation_identity for block in self.blocks)

    @property
    def block_signature(
        self,
    ) -> tuple[
        tuple[
            tuple[str, str, str], tuple[int, ...], tuple[int, ...], int, bool
        ], ...
    ]:
        return tuple(
            (
                block.observation_identity,
                block.mass_classes,
                block.active_support,
                block.replicate_count,
                block.independent_replicates,
            )
            for block in self.blocks
        )

    @property
    def replicate_counts(self) -> tuple[int, ...]:
        return tuple(block.replicate_count for block in self.blocks)

    @property
    def rigorous_testing_suitable(self) -> bool:
        return all(block.rigorous_testing_suitable for block in self.blocks)

    @property
    def fingerprint(self) -> str:
        return _testing_digest((
            "independent-dirichlet-mid-product-law-v1",
            tuple(block.fingerprint for block in self.blocks),
        ))

    def renyi_divergence(
        self, other: "IndependentDirichletMIDProductLaw", order: Real,
    ) -> float:
        _comparable_products(self, other)
        values = []
        for left, right in zip(self.blocks, other.blocks, strict=True):
            value = renyi_dirichlet(left, right, order)
            if value == math.inf:
                return math.inf
            values.append(left.replicate_count * value)
        result = math.fsum(values)
        if not math.isfinite(result) or result < 0:
            raise DirichletNumericalError(
                "Dirichlet product Rényi divergence is nonfinite or negative"
            )
        return result

    def sample(
        self, *, seed: int | None = None, rng: np.random.Generator | None = None,
    ) -> tuple[tuple[tuple[float, ...], ...], ...]:
        generator = _resolve_rng(seed=seed, rng=rng)
        return tuple(block.sample_replicates(rng=generator) for block in self.blocks)


def _comparable_products(
    left: IndependentDirichletMIDProductLaw,
    right: IndependentDirichletMIDProductLaw,
) -> None:
    if not isinstance(left, IndependentDirichletMIDProductLaw) or not isinstance(
        right, IndependentDirichletMIDProductLaw
    ):
        raise InputValidationError(
            "Dirichlet product comparison requires two IndependentDirichletMIDProductLaw records"
        )
    if left.block_signature != right.block_signature:
        raise InputValidationError(
            "Dirichlet products require the same ordered block identities, active faces, "
            "and replicate structure"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DirichletCompositeMIDLawFamily:
    """One explicit finite family of complete continuous observation laws."""

    members: tuple[IndependentDirichletMIDProductLaw, ...]
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple) or not self.members:
            raise InputValidationError(
                "Dirichlet composite members must be a nonempty immutable tuple"
            )
        if not all(
            isinstance(member, IndependentDirichletMIDProductLaw)
            for member in self.members
        ):
            raise InputValidationError(
                "every Dirichlet composite member must be an independent product law"
            )
        signature = self.members[0].block_signature
        for index, member in enumerate(self.members[1:], start=1):
            if member.block_signature != signature:
                raise InputValidationError(
                    f"Dirichlet family member {index} has state-dependent support or "
                    "a different observation/replicate structure; V1 refuses it"
                )
        if not isinstance(self.member_ids, tuple):
            raise InputValidationError("member_ids must be an immutable tuple")
        if self.member_ids and len(self.member_ids) != len(self.members):
            raise InputValidationError("member_ids must identify every family member")
        ids = (
            tuple(_identifier(value, "member_id") for value in self.member_ids)
            if self.member_ids
            else tuple(
                f"member-{index}:{member.fingerprint}"
                for index, member in enumerate(self.members)
            )
        )
        if len(set(ids)) != len(ids):
            raise InputValidationError("Dirichlet family member_ids must be unique")
        object.__setattr__(self, "member_ids", ids)

    @property
    def block_signature(self):
        return self.members[0].block_signature

    @property
    def fingerprint(self) -> str:
        return _testing_digest((
            "finite-dirichlet-mid-product-family-v1",
            self.member_ids,
            tuple(member.fingerprint for member in self.members),
        ))


@dataclass(frozen=True, slots=True, kw_only=True)
class DirichletCompositeBinaryTestingProblem:
    """Explicit finite biological H0/H1 families on a continuous simplex."""

    null: DirichletCompositeMIDLawFamily
    alternative: DirichletCompositeMIDLawFamily

    def __post_init__(self) -> None:
        if not isinstance(self.null, DirichletCompositeMIDLawFamily):
            raise InputValidationError("null must be DirichletCompositeMIDLawFamily")
        if not isinstance(self.alternative, DirichletCompositeMIDLawFamily):
            raise InputValidationError("alternative must be DirichletCompositeMIDLawFamily")
        if self.null.block_signature != self.alternative.block_signature:
            raise InputValidationError(
                "Dirichlet H0/H1 families require one common ordered active support and "
                "replicate structure; V1 does not add epsilon"
            )

    @property
    def block_signature(self):
        return self.null.block_signature

    @property
    def fingerprint(self) -> str:
        return _testing_digest((
            "finite-dirichlet-composite-testing-problem-v1",
            ("H0", self.null.fingerprint),
            ("H1", self.alternative.fingerprint),
        ))


def _require_rigorous_precision(problem: DirichletCompositeBinaryTestingProblem) -> None:
    unsuitable = [
        (role, family.member_ids[index], block.observation_identity)
        for role, family in (("H0", problem.null), ("H1", problem.alternative))
        for index, member in enumerate(family.members)
        for block in member.blocks
        if not block.rigorous_testing_suitable
    ]
    if unsuitable:
        raise DirichletTestingAssumptionError(
            "same_data_plugin Dirichlet precision is diagnostic/model-conditional and "
            "cannot be treated as a fixed known nuisance parameter for a finite-sample "
            f"testing guarantee; affected laws={unsuitable!r}"
        )


def _full_renyi(
    left: IndependentDirichletMIDProductLaw,
    right: IndependentDirichletMIDProductLaw,
    order: float,
) -> float:
    return left.renyi_divergence(right, order)


@dataclass(frozen=True, slots=True, kw_only=True)
class DirichletCompositeRenyiConverseBound:
    problem: DirichletCompositeBinaryTestingProblem
    constraint: SimpleBinaryTestingConstraint
    order: float
    reverse_renyi: float
    null_member_index: int | None
    alternative_member_index: int | None
    raw_reverse_lower_bound: float
    type_ii_lower_bound: float

    @property
    def null_member_id(self) -> str | None:
        return None if self.null_member_index is None else self.problem.null.member_ids[self.null_member_index]

    @property
    def alternative_member_id(self) -> str | None:
        return None if self.alternative_member_index is None else self.problem.alternative.member_ids[self.alternative_member_index]

    @property
    def global_order_envelope_evaluated(self) -> bool:
        return False


def composite_dirichlet_renyi_converse_at_order(
    problem: DirichletCompositeBinaryTestingProblem,
    *,
    epsilon: Real,
    order: Real,
) -> DirichletCompositeRenyiConverseBound:
    """Order-specific finite-family converse using ``min D_order(H1 || H0)``."""

    if not isinstance(problem, DirichletCompositeBinaryTestingProblem):
        raise InputValidationError("problem must be DirichletCompositeBinaryTestingProblem")
    _require_rigorous_precision(problem)
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    finite_order = validate_renyi_order(order)
    best = math.inf
    best_pair = None
    for null_index, null in enumerate(problem.null.members):
        for alternative_index, alternative in enumerate(problem.alternative.members):
            value = _full_renyi(alternative, null, finite_order)
            if value < best:
                best, best_pair = value, (null_index, alternative_index)
    if best == math.inf:
        raw = -math.inf
    else:
        log_power = (finite_order - 1.0) / finite_order * math.fsum((
            math.log(constraint.epsilon), best,
        ))
        try:
            raw = -math.expm1(log_power)
        except OverflowError:
            raw = -math.inf
    lower = max(0.0, raw)
    if not math.isfinite(lower) or not 0 <= lower <= 1:
        raise NumericalLimitError(
            "Dirichlet composite converse left the probability interval [0, 1]"
        )
    null_index, alternative_index = (
        (None, None) if best_pair is None else best_pair
    )
    return DirichletCompositeRenyiConverseBound(
        problem=problem,
        constraint=constraint,
        order=finite_order,
        reverse_renyi=best,
        null_member_index=null_index,
        alternative_member_index=alternative_index,
        raw_reverse_lower_bound=raw,
        type_ii_lower_bound=lower,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DirichletProductLogLikelihoodScore:
    """Selected-pair log likelihood score over all declared blocks/replicates."""

    null: IndependentDirichletMIDProductLaw
    alternative: IndependentDirichletMIDProductLaw

    def __post_init__(self) -> None:
        _comparable_products(self.null, self.alternative)

    @property
    def block_scores(self) -> tuple[DirichletLogLikelihoodScore, ...]:
        return tuple(
            null.pairwise_score(alternative)
            for null, alternative in zip(self.null.blocks, self.alternative.blocks, strict=True)
        )

    def evaluate(self, observations: object) -> float:
        if not isinstance(observations, tuple) or len(observations) != len(self.block_scores):
            raise InputValidationError(
                "Dirichlet product observations require one ordered replicate tuple per block"
            )
        return math.fsum(
            score.evaluate_replicates(block_observations)
            for score, block_observations in zip(
                self.block_scores, observations, strict=True,
            )
        )

    def log_moment(
        self, reference: IndependentDirichletMIDProductLaw, exponent: Real,
    ) -> float:
        _comparable_products(reference, self.null)
        t = _finite_real(exponent, "Dirichlet product score exponent")
        values = []
        for block, score in zip(reference.blocks, self.block_scores, strict=True):
            value = score.log_moment(block, t)
            if value == math.inf:
                return math.inf
            values.append(block.replicate_count * value)
        result = math.fsum(values)
        if not math.isfinite(result):
            raise DirichletNumericalError("Dirichlet product log moment is nonfinite")
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class DirichletCompositeRenyiScoreCandidate:
    """Finite-family vertex pair with analytic continuous-law moment checks."""

    problem: DirichletCompositeBinaryTestingProblem
    order: float
    null_member_index: int
    alternative_member_index: int
    renyi: float
    score: DirichletProductLogLikelihoodScore
    log_hellinger_integral: float
    maximum_log_null_moment: float
    maximum_log_alternative_moment: float
    uniform_moment_bounds_verified: bool
    verification_failures: tuple[str, ...]

    @property
    def null_member_id(self) -> str:
        return self.problem.null.member_ids[self.null_member_index]

    @property
    def alternative_member_id(self) -> str:
        return self.problem.alternative.member_ids[self.alternative_member_index]

    @property
    def finite_n_least_favourable_claimed(self) -> bool:
        return False

    @property
    def joint_convex_projection_claimed(self) -> bool:
        return False


def _validate_score_order(order: Real) -> float:
    value = _finite_real(order, "candidate-score Dirichlet Rényi order lambda")
    if not 0 < order < 1:
        raise InputValidationError(
            "candidate-score Dirichlet Rényi order must satisfy 0 < lambda < 1"
        )
    if not 0 < value < 1:
        raise NumericalLimitError(
            "candidate-score Dirichlet Rényi order rounds to an endpoint"
        )
    return value


def _candidate_for_pair(
    problem: DirichletCompositeBinaryTestingProblem,
    order: float,
    divergence: float,
    null_index: int,
    alternative_index: int,
) -> DirichletCompositeRenyiScoreCandidate:
    null = problem.null.members[null_index]
    alternative = problem.alternative.members[alternative_index]
    score = DirichletProductLogLikelihoodScore(null=null, alternative=alternative)
    raw_log_z = score.log_moment(null, order)
    expected_log_z = (order - 1.0) * divergence
    scale = max(1.0, abs(raw_log_z), abs(expected_log_z))
    failures = []
    if abs(raw_log_z - expected_log_z) > 5e-10 * scale:
        failures.append(
            "selected-pair score moment and directed Rényi identity disagree numerically"
        )
    null_moments = tuple(
        raw_log_z if member.blocks == null.blocks else score.log_moment(member, order)
        for member in problem.null.members
    )
    alternative_moments = tuple(
        raw_log_z if member.blocks == alternative.blocks else score.log_moment(member, order - 1.0)
        for member in problem.alternative.members
    )
    for side, family, selected, moments in (
        ("null", problem.null, null, null_moments),
        ("alternative", problem.alternative, alternative, alternative_moments),
    ):
        for index, (member, moment) in enumerate(zip(family.members, moments, strict=True)):
            if moment == math.inf:
                failures.append(
                    f"{side}-side score moment is nonintegrable for member {index}"
                )
                continue
            if member.blocks == selected.blocks:
                continue
            margin = raw_log_z - moment
            tolerance = DIRICHLET_MOMENT_VERIFICATION_TOLERANCE * max(
                1.0, abs(raw_log_z), abs(moment)
            )
            if margin < -tolerance:
                failures.append(
                    f"{side}-side uniform exponential-moment inequality fails for member {index}: "
                    f"log_moment={moment:.17g}, log_z={raw_log_z:.17g}"
                )
            elif margin <= tolerance:
                failures.append(
                    f"{side}-side uniform moment comparison is numerically unresolved for member {index}"
                )
    public_log_z = (
        math.nextafter(raw_log_z, math.inf) if math.isfinite(raw_log_z) else raw_log_z
    )
    return DirichletCompositeRenyiScoreCandidate(
        problem=problem,
        order=order,
        null_member_index=null_index,
        alternative_member_index=alternative_index,
        renyi=divergence,
        score=score,
        log_hellinger_integral=public_log_z,
        maximum_log_null_moment=max(null_moments),
        maximum_log_alternative_moment=max(alternative_moments),
        uniform_moment_bounds_verified=not failures,
        verification_failures=tuple(failures),
    )


def composite_dirichlet_renyi_score_candidate(
    problem: DirichletCompositeBinaryTestingProblem,
    *,
    order: Real,
    tolerance: Real = DIRICHLET_MOMENT_VERIFICATION_TOLERANCE,
) -> DirichletCompositeRenyiScoreCandidate:
    """Select a minimum directed pair and verify all analytic score moments."""

    if not isinstance(problem, DirichletCompositeBinaryTestingProblem):
        raise InputValidationError("problem must be DirichletCompositeBinaryTestingProblem")
    _require_rigorous_precision(problem)
    finite_order = _validate_score_order(order)
    tie_tolerance = _finite_real(tolerance, "Dirichlet candidate tie tolerance")
    if tie_tolerance <= 0:
        raise InputValidationError("Dirichlet candidate tie tolerance must be positive")
    pairs = tuple(
        (
            _full_renyi(alternative, null, finite_order),
            null_index,
            alternative_index,
        )
        for null_index, null in enumerate(problem.null.members)
        for alternative_index, alternative in enumerate(problem.alternative.members)
    )
    minimum = min(value for value, _, _ in pairs)
    minimisers = tuple(
        item for item in pairs
        if (
            item[0] == minimum == math.inf
            or abs(item[0] - minimum) <= tie_tolerance * max(1.0, abs(minimum))
        )
    )
    candidates = tuple(
        _candidate_for_pair(problem, finite_order, divergence, null_index, alternative_index)
        for divergence, null_index, alternative_index in minimisers
    )
    return next(
        (item for item in candidates if item.uniform_moment_bounds_verified),
        candidates[0],
    )


def verified_composite_dirichlet_renyi_score(
    problem: DirichletCompositeBinaryTestingProblem,
    *,
    order: Real,
    tolerance: Real = DIRICHLET_MOMENT_VERIFICATION_TOLERANCE,
) -> DirichletCompositeRenyiScoreCandidate:
    candidate = composite_dirichlet_renyi_score_candidate(
        problem, order=order, tolerance=tolerance,
    )
    if not candidate.uniform_moment_bounds_verified:
        raise DirichletTestingAssumptionError(
            "finite-family Dirichlet Rényi pair is only a candidate; uniform analytic "
            "moment verification failed: " + "; ".join(candidate.verification_failures)
        )
    return candidate


@dataclass(frozen=True, slots=True, kw_only=True)
class DirichletCompositeScoreBound:
    """Analytic projected-score upper guarantee on represented minimax Type II."""

    candidate: DirichletCompositeRenyiScoreCandidate
    constraint: SimpleBinaryTestingConstraint
    threshold: float
    raw_exponential_upper_bound: float
    constant_randomised_upper_bound: float
    minimax_type_ii_upper_bound: float


def composite_dirichlet_score_bound_at_order(
    candidate: DirichletCompositeRenyiScoreCandidate,
    *,
    epsilon: Real,
) -> DirichletCompositeScoreBound:
    if not isinstance(candidate, DirichletCompositeRenyiScoreCandidate):
        raise InputValidationError(
            "candidate must be DirichletCompositeRenyiScoreCandidate"
        )
    if not candidate.uniform_moment_bounds_verified:
        raise DirichletTestingAssumptionError(
            "Dirichlet candidate lacks verified uniform analytic score moments"
        )
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    with localcontext() as context:
        context.prec = 80
        lam = Decimal.from_float(candidate.order)
        log_z = Decimal.from_float(candidate.log_hellinger_integral)
        threshold_decimal = (log_z - Decimal.from_float(constraint.epsilon).ln()) / lam
        threshold = math.nextafter(float(threshold_decimal), math.inf)
        if not math.isfinite(threshold):
            raise NumericalLimitError(
                "Dirichlet analytical score threshold exceeds float representability"
            )
        log_raw = log_z + (1 - lam) * Decimal.from_float(threshold)
        if log_raw > Decimal.from_float(math.log(np.finfo(float).max)):
            raw = math.inf
        else:
            raw = float(log_raw.exp())
            if raw == 0:
                raise NumericalLimitError(
                    "positive Dirichlet analytical score bound underflowed"
                )
            raw = math.nextafter(raw, math.inf)
    constant = 1.0 - constraint.epsilon
    minimax_upper = min(constant, raw)
    if not math.isfinite(minimax_upper) or not 0 <= minimax_upper <= 1:
        raise NumericalLimitError(
            "Dirichlet score minimax upper bound is numerically invalid"
        )
    return DirichletCompositeScoreBound(
        candidate=candidate,
        constraint=constraint,
        threshold=threshold,
        raw_exponential_upper_bound=raw,
        constant_randomised_upper_bound=constant,
        minimax_type_ii_upper_bound=minimax_upper,
    )


def exact_dirichlet_composite_minimax(
    problem: DirichletCompositeBinaryTestingProblem,
    *,
    epsilon: Real,
    max_outcomes: int | None = None,
):
    """Refuse the finite count-space LP for a continuous simplex law."""

    if not isinstance(problem, DirichletCompositeBinaryTestingProblem):
        raise InputValidationError("problem must be DirichletCompositeBinaryTestingProblem")
    SimpleBinaryTestingConstraint(epsilon=epsilon)
    raise UnsupportedContinuousObservationError(
        "unsupported_for_continuous_observation_space: the exact finite minimax LP "
        "enumerates count outcomes and cannot be applied to a Dirichlet simplex law; "
        "no discretisation was substituted"
    )


def evaluate_dirichlet_composite_score_test(
    bound: DirichletCompositeScoreBound,
    *,
    max_outcomes: int | None = None,
):
    """Refuse an exact score CDF/error claim for the continuous V1 score."""

    if not isinstance(bound, DirichletCompositeScoreBound):
        raise InputValidationError("bound must be DirichletCompositeScoreBound")
    raise UnsupportedContinuousObservationError(
        "unsupported_for_continuous_observation_space: a weighted sum of log MID "
        "components has no implemented certified exact CDF; Monte Carlo is validation "
        "only and was not reported as an exact achieved error"
    )


def calibrate_dirichlet_composite_score_test(
    candidate: DirichletCompositeRenyiScoreCandidate,
    *,
    epsilon: Real,
    max_outcomes: int | None = None,
):
    """Refuse finite-outcome score calibration on the continuous simplex."""

    if not isinstance(candidate, DirichletCompositeRenyiScoreCandidate):
        raise InputValidationError(
            "candidate must be DirichletCompositeRenyiScoreCandidate"
        )
    SimpleBinaryTestingConstraint(epsilon=epsilon)
    raise UnsupportedContinuousObservationError(
        "unsupported_for_continuous_observation_space: exact finite score calibration "
        "cannot enumerate Dirichlet outcomes and no simplex discretisation or normal-CDF "
        "substitute was used"
    )


__all__ = [
    "DIRICHLET_MOMENT_VERIFICATION_TOLERANCE",
    "DirichletCompositeBinaryTestingProblem",
    "DirichletCompositeMIDLawFamily",
    "DirichletCompositeRenyiConverseBound",
    "DirichletCompositeRenyiScoreCandidate",
    "DirichletCompositeScoreBound",
    "DirichletProductLogLikelihoodScore",
    "DirichletTestingAssumptionError",
    "IndependentDirichletMIDProductLaw",
    "UnsupportedContinuousObservationError",
    "calibrate_dirichlet_composite_score_test",
    "composite_dirichlet_renyi_converse_at_order",
    "composite_dirichlet_renyi_score_candidate",
    "composite_dirichlet_score_bound_at_order",
    "evaluate_dirichlet_composite_score_test",
    "exact_dirichlet_composite_minimax",
    "verified_composite_dirichlet_renyi_score",
]
