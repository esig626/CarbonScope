"""Finite explicit composite testing for genuine-count MID laws.

This module implements settled finite-sample composite machinery without
assuming a finite-blocklength least-favourable-pair reduction. Hypothesis
classes are explicit finite collections of categorical MID laws observed
through one common genuine multinomial count total.

The arbitrary-class Rényi converse is the finite-family specialisation of the
pairwise composite converse. Exact minimax testing is solved over the complete
count space as a randomized linear program. Subcritical projected testing uses
a finite-family Rényi-minimizing pair only after directly verifying the two
uniform moment inequalities required by the composite achievability theorem.

No finite family is silently convexified. No selected Rényi pair is advertised
as finite-sample least favourable.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
from numbers import Integral, Real

import numpy as np

from ..exceptions import FluxEMUError, InputValidationError, ValidationError
from ..observation import MultinomialMIDLaw, renyi_multinomial
from .simple import (
    NumericalLimitError,
    SimpleBinaryTestingConstraint,
    _finite_real,
    _testing_digest,
    validate_renyi_order,
)


DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES = 1_000_000
COMPOSITE_NUMERICAL_TOLERANCE = 1e-9
_PROJECTION_TOLERANCE = 1e-10
_DECIMAL_PRECISION = 60


class CompositeEnumerationLimitError(ValidationError):
    """The exact composite count space exceeds the explicit enumeration cap."""


class CompositeProjectionError(FluxEMUError):
    """A finite family does not satisfy the verified projected-test contract."""


class CompositeOptimizationError(FluxEMUError):
    """The exact finite-family minimax optimization failed numerically."""


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeMIDLawFamily:
    """One explicit finite hypothesis class of genuine-count MID laws.

    ``members`` is the class itself. It is never interpreted as a convex hull.
    Every member must use the same genuine count total and ordered mass-class
    space. ``member_ids`` preserves the caller's declared scientific order and
    may identify flux states or other fixed mechanisms.
    """

    members: tuple[MultinomialMIDLaw, ...]
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple) or not self.members:
            raise InputValidationError(
                "composite family members must be a nonempty immutable tuple"
            )
        if not all(isinstance(item, MultinomialMIDLaw) for item in self.members):
            raise InputValidationError(
                "every composite family member must be MultinomialMIDLaw"
            )
        first = self.members[0]
        for index, member in enumerate(self.members[1:], start=1):
            if member.n != first.n:
                raise InputValidationError(
                    f"composite family member {index} has count total {member.n}; "
                    f"expected common total {first.n}"
                )
            if member.mass_classes != first.mass_classes:
                raise InputValidationError(
                    f"composite family member {index} has a different mass-class space"
                )
        if not isinstance(self.member_ids, tuple):
            raise InputValidationError("member_ids must be an immutable tuple")
        if self.member_ids and len(self.member_ids) != len(self.members):
            raise InputValidationError("member_ids must identify every family member")
        if self.member_ids:
            ids = tuple(_identifier(value, "member_id") for value in self.member_ids)
        else:
            ids = tuple(
                f"member-{index}:{member.fingerprint}"
                for index, member in enumerate(self.members)
            )
        if len(set(ids)) != len(ids):
            raise InputValidationError("composite family member_ids must be unique")
        object.__setattr__(self, "member_ids", ids)

    @property
    def n(self) -> int:
        return self.members[0].n

    @property
    def mass_classes(self) -> tuple[int, ...]:
        return self.members[0].mass_classes

    @property
    def dimension(self) -> int:
        return len(self.mass_classes)

    @property
    def full_support(self) -> bool:
        return all(
            all(probability > 0 for probability in member.probabilities)
            for member in self.members
        )

    @property
    def fingerprint(self) -> str:
        return _testing_digest(
            (
                "finite-composite-mid-family-v1",
                self.member_ids,
                tuple(item.fingerprint for item in self.members),
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeBinaryTestingProblem:
    """Explicit finite H0/H1 classes with common genuine-count semantics.

    H0 is ``null`` and H1 is ``alternative``. A randomized test value is the
    probability of deciding H1. Worst-case Type I and Type II errors therefore
    take the supremum over the null and alternative families respectively.
    """

    null: CompositeMIDLawFamily
    alternative: CompositeMIDLawFamily

    def __post_init__(self) -> None:
        if not isinstance(self.null, CompositeMIDLawFamily):
            raise InputValidationError("null must be CompositeMIDLawFamily")
        if not isinstance(self.alternative, CompositeMIDLawFamily):
            raise InputValidationError("alternative must be CompositeMIDLawFamily")
        if self.null.n != self.alternative.n:
            raise InputValidationError(
                "composite null and alternative must share one genuine count total"
            )
        if self.null.mass_classes != self.alternative.mass_classes:
            raise InputValidationError(
                "composite null and alternative must share the ordered mass-class space"
            )

    @property
    def n(self) -> int:
        return self.null.n

    @property
    def mass_classes(self) -> tuple[int, ...]:
        return self.null.mass_classes

    @property
    def dimension(self) -> int:
        return self.null.dimension

    @property
    def fingerprint(self) -> str:
        return _testing_digest(
            (
                "finite-composite-binary-testing-problem-v1",
                ("H0", self.null.fingerprint),
                ("H1", self.alternative.fingerprint),
            )
        )


def _validate_projection_order(order: Real) -> float:
    value = _finite_real(order, "projected Rényi order lambda")
    if not 0 < order < 1:
        raise InputValidationError(
            "projected Rényi order lambda must satisfy 0 < lambda < 1"
        )
    if not 0 < value < 1:
        raise NumericalLimitError(
            "projected Rényi order rounds to an endpoint at float precision"
        )
    return value


def _categorical_renyi(
    left: MultinomialMIDLaw,
    right: MultinomialMIDLaw,
    order: float,
) -> float:
    value = renyi_multinomial(left, right, order)
    if value == math.inf:
        return math.inf
    if not math.isfinite(value) or value < 0:
        raise NumericalLimitError(
            "composite categorical Rényi divergence is nonfinite or negative "
            "at numerical precision"
        )
    return value / left.n


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeRenyiConverseBound:
    """Order-specific lower bound on minimax Type-II error for a finite class."""

    problem: CompositeBinaryTestingProblem
    constraint: SimpleBinaryTestingConstraint
    order: float
    rate: float
    reverse_renyi: float
    full_law_reverse_renyi: float
    null_member_index: int | None
    alternative_member_index: int | None
    type_ii_lower_bound: float

    @property
    def null_member_id(self) -> str | None:
        return (
            None
            if self.null_member_index is None
            else self.problem.null.member_ids[self.null_member_index]
        )

    @property
    def alternative_member_id(self) -> str | None:
        return (
            None
            if self.alternative_member_index is None
            else self.problem.alternative.member_ids[self.alternative_member_index]
        )

    @property
    def global_order_envelope_evaluated(self) -> bool:
        return False


def composite_renyi_converse_at_order(
    problem: CompositeBinaryTestingProblem,
    *,
    epsilon: Real,
    order: Real,
) -> CompositeRenyiConverseBound:
    """Evaluate the finite-family composite reverse-Rényi converse at one order.

    This is the finite explicit-class specialisation of the pairwise composite
    converse. It makes no convexity, projection, ordering, or least-favourable
    pair assumption. The supplied finite order ``lambda>1`` is used unchanged;
    this function does not search or approximate the continuous-order envelope.

    The common genuine count total ``n`` is the i.i.d. blocklength and
    ``epsilon = exp(-n r)`` defines the corresponding Type-I rate ``r``.
    """

    if not isinstance(problem, CompositeBinaryTestingProblem):
        raise InputValidationError("problem must be CompositeBinaryTestingProblem")
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    finite_order = validate_renyi_order(order)
    best = math.inf
    best_pair: tuple[int, int] | None = None
    for null_index, null in enumerate(problem.null.members):
        for alternative_index, alternative in enumerate(problem.alternative.members):
            value = _categorical_renyi(alternative, null, finite_order)
            if value < best:
                best = value
                best_pair = (null_index, alternative_index)

    rate = -math.log(constraint.epsilon) / problem.n
    gap = 0.0 if best == math.inf else max(rate - best, 0.0)
    exponent = problem.n * (finite_order - 1.0) / finite_order * gap
    lower = -math.expm1(-exponent)
    if not math.isfinite(lower) or not 0 <= lower <= 1:
        raise NumericalLimitError(
            "composite Rényi converse left the probability interval [0, 1]"
        )
    null_index, alternative_index = (
        (None, None) if best_pair is None else best_pair
    )
    return CompositeRenyiConverseBound(
        problem=problem,
        constraint=constraint,
        order=finite_order,
        rate=rate,
        reverse_renyi=best,
        full_law_reverse_renyi=(
            math.inf if best == math.inf else problem.n * best
        ),
        null_member_index=null_index,
        alternative_member_index=alternative_index,
        type_ii_lower_bound=lower,
    )


def _validate_max_outcomes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise InputValidationError(
            "max_outcomes must be a positive integer, not bool"
        )
    return int(value)


def _outcome_count(problem: CompositeBinaryTestingProblem, limit: int) -> int:
    count = math.comb(problem.n + problem.dimension - 1, problem.dimension - 1)
    if count > limit:
        raise CompositeEnumerationLimitError(
            "exact composite testing requires enumeration of "
            f"{count} count outcomes, exceeding max_outcomes={limit}; "
            "no approximation was substituted"
        )
    return count


def _compositions(total: int, cells: int):
    if cells == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remaining in _compositions(total - first, cells - 1):
            yield (first, *remaining)


def _enumerate_outcomes(
    problem: CompositeBinaryTestingProblem,
    max_outcomes: int,
) -> tuple[tuple[int, ...], ...]:
    limit = _validate_max_outcomes(max_outcomes)
    expected = _outcome_count(problem, limit)
    outcomes = tuple(_compositions(problem.n, problem.dimension))
    if len(outcomes) != expected:  # pragma: no cover - mathematical invariant
        raise CompositeOptimizationError(
            "internal count-space enumeration size is inconsistent"
        )
    return outcomes


def _law_mass_vector(
    law: MultinomialMIDLaw,
    outcomes: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    values = np.empty(len(outcomes), dtype=float)
    for index, counts in enumerate(outcomes):
        log_mass = law.log_pmf(counts)
        if log_mass == -math.inf:
            values[index] = 0.0
            continue
        mass = math.exp(log_mass)
        if mass == 0.0:
            raise NumericalLimitError(
                "positive multinomial outcome mass underflowed during exact "
                "composite enumeration; the law was not repaired"
            )
        values[index] = mass
    total = math.fsum(float(item) for item in values)
    if not math.isfinite(total) or abs(total - 1.0) > 5e-9:
        raise NumericalLimitError(
            "enumerated multinomial probability mass does not sum to one "
            "within the finite-sample numerical contract"
        )
    return values


def _family_mass_matrix(
    family: CompositeMIDLawFamily,
    outcomes: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    return np.vstack(tuple(_law_mass_vector(law, outcomes) for law in family.members))


@dataclass(frozen=True, slots=True, kw_only=True)
class FiniteCompositeMinimaxResult:
    """Exact randomized minimax solution on the enumerated count space."""

    problem: CompositeBinaryTestingProblem
    constraint: SimpleBinaryTestingConstraint
    outcomes: tuple[tuple[int, ...], ...]
    rejection_probabilities: tuple[float, ...]
    null_type_i_errors: tuple[float, ...]
    alternative_type_ii_errors: tuple[float, ...]
    worst_type_i_error: float
    minimax_type_ii_error: float
    active_null_members: tuple[int, ...]
    active_alternative_members: tuple[int, ...]

    @property
    def randomized(self) -> bool:
        return any(0.0 < value < 1.0 for value in self.rejection_probabilities)

    @property
    def active_null_member_ids(self) -> tuple[str, ...]:
        return tuple(self.problem.null.member_ids[index] for index in self.active_null_members)

    @property
    def active_alternative_member_ids(self) -> tuple[str, ...]:
        return tuple(
            self.problem.alternative.member_ids[index]
            for index in self.active_alternative_members
        )


def exact_finite_composite_minimax(
    problem: CompositeBinaryTestingProblem,
    *,
    epsilon: Real,
    max_outcomes: int = DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES,
) -> FiniteCompositeMinimaxResult:
    """Solve the exact finite-class randomized minimax test by linear programming.

    The optimization ranges over one rejection probability in ``[0,1]`` for
    every complete count outcome. Every declared null and alternative law is a
    separate worst-case constraint. Structural zeros are retained. If the count
    space exceeds ``max_outcomes`` the function fails instead of changing the
    statistical problem.

    SciPy is imported only here and is available through the ``testing`` extra.
    """

    if not isinstance(problem, CompositeBinaryTestingProblem):
        raise InputValidationError("problem must be CompositeBinaryTestingProblem")
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    outcomes = _enumerate_outcomes(problem, max_outcomes)
    null_mass = _family_mass_matrix(problem.null, outcomes)
    alternative_mass = _family_mass_matrix(problem.alternative, outcomes)

    try:
        from scipy.optimize import linprog
    except ImportError as error:  # pragma: no cover - CI exercises installed extra
        raise CompositeOptimizationError(
            "exact composite minimax testing requires the 'testing' SciPy extra"
        ) from error

    outcome_count = len(outcomes)
    variable_count = outcome_count + 1
    objective = np.zeros(variable_count, dtype=float)
    objective[-1] = 1.0

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for probabilities in null_mass:
        row = np.zeros(variable_count, dtype=float)
        row[:outcome_count] = probabilities
        rows.append(row)
        rhs.append(constraint.epsilon)
    for probabilities in alternative_mass:
        # beta >= 1 - E_Q[phi]  <=>  -E_Q[phi] - beta <= -1
        row = np.zeros(variable_count, dtype=float)
        row[:outcome_count] = -probabilities
        row[-1] = -1.0
        rows.append(row)
        rhs.append(-1.0)

    result = linprog(
        objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs, dtype=float),
        bounds=[(0.0, 1.0)] * variable_count,
        method="highs",
    )
    if not result.success or result.x is None:
        raise CompositeOptimizationError(
            "exact finite composite minimax LP failed: "
            + (result.message or "unknown HiGHS failure")
        )

    phi = np.asarray(result.x[:outcome_count], dtype=float)
    beta_variable = float(result.x[-1])
    if not np.isfinite(phi).all() or np.any(phi < 0) or np.any(phi > 1):
        raise CompositeOptimizationError(
            "exact composite minimax LP returned invalid rejection probabilities"
        )
    if not math.isfinite(beta_variable) or not 0 <= beta_variable <= 1:
        raise CompositeOptimizationError(
            "exact composite minimax LP returned an invalid Type-II objective"
        )

    null_errors = null_mass @ phi
    alternative_errors = 1.0 - alternative_mass @ phi
    worst_alpha = float(np.max(null_errors))
    worst_beta = float(np.max(alternative_errors))
    tolerance = COMPOSITE_NUMERICAL_TOLERANCE
    if worst_alpha > constraint.epsilon + tolerance:
        raise CompositeOptimizationError(
            "exact composite minimax LP violates the declared Type-I constraint"
        )
    if abs(worst_beta - beta_variable) > tolerance:
        raise CompositeOptimizationError(
            "exact composite minimax LP objective disagrees with evaluated worst-case Type II"
        )
    active_null = tuple(
        index
        for index, value in enumerate(null_errors)
        if abs(float(value) - worst_alpha) <= tolerance
    )
    active_alternative = tuple(
        index
        for index, value in enumerate(alternative_errors)
        if abs(float(value) - worst_beta) <= tolerance
    )
    return FiniteCompositeMinimaxResult(
        problem=problem,
        constraint=constraint,
        outcomes=outcomes,
        rejection_probabilities=tuple(float(value) for value in phi),
        null_type_i_errors=tuple(float(value) for value in null_errors),
        alternative_type_ii_errors=tuple(float(value) for value in alternative_errors),
        worst_type_i_error=worst_alpha,
        minimax_type_ii_error=worst_beta,
        active_null_members=active_null,
        active_alternative_members=active_alternative,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedCompositeRenyiProjection:
    """Finite-family joint Rényi minimizer with directly verified uniform bounds.

    The selected pair is a minimizer over the explicitly declared finite class.
    Because a finite class is not silently convexified, projection optimality
    alone does not imply the uniform inequalities. They are checked directly
    over every family member before this object can be returned.
    """

    problem: CompositeBinaryTestingProblem
    order: float
    null_member_index: int
    alternative_member_index: int
    single_draw_renyi: float
    hellinger_integral: float
    log_likelihood_ratio: tuple[float, ...]
    maximum_null_moment: float
    maximum_alternative_moment: float

    @property
    def null_member_id(self) -> str:
        return self.problem.null.member_ids[self.null_member_index]

    @property
    def alternative_member_id(self) -> str:
        return self.problem.alternative.member_ids[self.alternative_member_index]

    @property
    def finite_n_least_favourable_claimed(self) -> bool:
        return False


def _projection_moments(
    problem: CompositeBinaryTestingProblem,
    null_index: int,
    alternative_index: int,
    order: float,
) -> tuple[float, float, float, tuple[float, ...]]:
    p_star = problem.null.members[null_index].probabilities
    q_star = problem.alternative.members[alternative_index].probabilities
    if any(p <= 0 for p in p_star) or any(q <= 0 for q in q_star):
        raise CompositeProjectionError(
            "verified projected testing currently requires full support in the selected pair"
        )
    log_ratio = tuple(math.log(q / p) for p, q in zip(p_star, q_star, strict=True))
    z = math.fsum(
        q**order * p ** (1.0 - order)
        for p, q in zip(p_star, q_star, strict=True)
    )
    null_moments = tuple(
        math.fsum(
            probability * math.exp(order * score)
            for probability, score in zip(member.probabilities, log_ratio, strict=True)
        )
        for member in problem.null.members
    )
    alternative_moments = tuple(
        math.fsum(
            probability * math.exp((order - 1.0) * score)
            for probability, score in zip(member.probabilities, log_ratio, strict=True)
        )
        for member in problem.alternative.members
    )
    return z, max(null_moments), max(alternative_moments), log_ratio


def verified_composite_renyi_projection(
    problem: CompositeBinaryTestingProblem,
    *,
    order: Real,
    tolerance: float = _PROJECTION_TOLERANCE,
) -> VerifiedCompositeRenyiProjection:
    """Return a finite-class Rényi minimizer only if the uniform bounds verify.

    All declared laws must have full support in this production path. The
    finite family is not convexified. Candidate joint minimizers are checked
    directly against both uniform moment inequalities from the composite
    achievability theorem. If no minimizer satisfies them, the function fails
    rather than presenting a pairwise statistic as a composite test.
    """

    if not isinstance(problem, CompositeBinaryTestingProblem):
        raise InputValidationError("problem must be CompositeBinaryTestingProblem")
    finite_order = _validate_projection_order(order)
    tolerance_value = _finite_real(tolerance, "projection tolerance")
    if tolerance_value <= 0:
        raise InputValidationError("projection tolerance must be positive")
    if not problem.null.full_support or not problem.alternative.full_support:
        raise CompositeProjectionError(
            "verified projected finite-family testing currently requires every "
            "null and alternative MID to have full support; exact minimax and "
            "the composite converse remain available with structural zeros"
        )

    candidates: list[tuple[float, int, int]] = []
    for null_index, null in enumerate(problem.null.members):
        for alternative_index, alternative in enumerate(problem.alternative.members):
            divergence = _categorical_renyi(alternative, null, finite_order)
            candidates.append((divergence, null_index, alternative_index))
    minimum = min(item[0] for item in candidates)
    minimizers = tuple(
        item for item in candidates
        if abs(item[0] - minimum)
        <= tolerance_value * max(1.0, abs(minimum))
    )
    failures: list[str] = []
    for divergence, null_index, alternative_index in minimizers:
        z, max_null, max_alternative, log_ratio = _projection_moments(
            problem, null_index, alternative_index, finite_order
        )
        scale = max(1.0, abs(z))
        if (
            max_null <= z + tolerance_value * scale
            and max_alternative <= z + tolerance_value * scale
        ):
            return VerifiedCompositeRenyiProjection(
                problem=problem,
                order=finite_order,
                null_member_index=null_index,
                alternative_member_index=alternative_index,
                single_draw_renyi=divergence,
                hellinger_integral=z,
                log_likelihood_ratio=log_ratio,
                maximum_null_moment=max_null,
                maximum_alternative_moment=max_alternative,
            )
        failures.append(
            f"({problem.null.member_ids[null_index]!r}, "
            f"{problem.alternative.member_ids[alternative_index]!r}): "
            f"max_null={max_null:g}, max_alternative={max_alternative:g}, z={z:g}"
        )
    raise CompositeProjectionError(
        "the finite-class Rényi minimizer does not satisfy the two uniform "
        "moment inequalities required for composite achievability; no composite "
        "projected test was constructed. Checked minimizer(s): "
        + "; ".join(failures)
    )


def _decimal_score(
    counts: tuple[int, ...],
    projection: VerifiedCompositeRenyiProjection,
) -> Decimal:
    null = projection.problem.null.members[projection.null_member_index]
    alternative = projection.problem.alternative.members[
        projection.alternative_member_index
    ]
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        total = Decimal(0)
        for count, p, q in zip(
            counts, null.probabilities, alternative.probabilities, strict=True
        ):
            if count:
                total += Decimal(int(count)) * (
                    Decimal.from_float(q).ln() - Decimal.from_float(p).ln()
                )
        return total


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeProjectedBound:
    """Closed-form projected test and its actual finite-family errors."""

    projection: VerifiedCompositeRenyiProjection
    constraint: SimpleBinaryTestingConstraint
    rate: float
    threshold: float
    raw_exponential_upper_bound: float
    type_ii_upper_bound: float
    actual_worst_type_i_error: float
    actual_worst_type_ii_error: float
    outcomes: int


def projected_composite_bound_at_order(
    projection: VerifiedCompositeRenyiProjection,
    *,
    epsilon: Real,
    max_outcomes: int = DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES,
) -> CompositeProjectedBound:
    """Evaluate the theorem's deterministic projected threshold at finite n."""

    if not isinstance(projection, VerifiedCompositeRenyiProjection):
        raise InputValidationError(
            "projection must be VerifiedCompositeRenyiProjection"
        )
    problem = projection.problem
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    outcomes = _enumerate_outcomes(problem, max_outcomes)
    null_mass = _family_mass_matrix(problem.null, outcomes)
    alternative_mass = _family_mass_matrix(problem.alternative, outcomes)
    rate = -math.log(constraint.epsilon) / problem.n
    lam = projection.order
    divergence = projection.single_draw_renyi
    threshold = problem.n * (rate - (1.0 - lam) * divergence) / lam
    threshold_decimal = Decimal.from_float(threshold)
    rejection = np.asarray(
        [
            1.0 if _decimal_score(counts, projection) >= threshold_decimal else 0.0
            for counts in outcomes
        ],
        dtype=float,
    )
    null_errors = null_mass @ rejection
    alternative_errors = 1.0 - alternative_mass @ rejection
    worst_alpha = float(np.max(null_errors))
    worst_beta = float(np.max(alternative_errors))
    if worst_alpha > constraint.epsilon + COMPOSITE_NUMERICAL_TOLERANCE:
        raise CompositeProjectionError(
            "verified projected threshold exceeded the declared composite "
            "Type-I constraint at numerical precision"
        )

    log_raw = -problem.n * (1.0 - lam) / lam * (divergence - rate)
    raw = math.exp(log_raw) if log_raw <= math.log(np.finfo(float).max) else math.inf
    upper = min(1.0 - constraint.epsilon, raw)
    if upper < 0 or not math.isfinite(upper):
        raise NumericalLimitError(
            "projected composite Type-II upper bound is numerically invalid"
        )
    return CompositeProjectedBound(
        projection=projection,
        constraint=constraint,
        rate=rate,
        threshold=threshold,
        raw_exponential_upper_bound=raw,
        type_ii_upper_bound=upper,
        actual_worst_type_i_error=worst_alpha,
        actual_worst_type_ii_error=worst_beta,
        outcomes=len(outcomes),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibratedCompositeProjectedTest:
    """Exact finite-family calibration within one projected-score threshold family."""

    projection: VerifiedCompositeRenyiProjection
    constraint: SimpleBinaryTestingConstraint
    threshold_score: float
    boundary_randomization: float
    outcomes: tuple[tuple[int, ...], ...]
    rejection_probabilities: tuple[float, ...]
    null_type_i_errors: tuple[float, ...]
    alternative_type_ii_errors: tuple[float, ...]
    worst_type_i_error: float
    worst_type_ii_error: float

    @property
    def exhausts_type_i_budget(self) -> bool:
        return abs(self.worst_type_i_error - self.constraint.epsilon) <= 1e-9


def calibrate_composite_projected_test(
    projection: VerifiedCompositeRenyiProjection,
    *,
    epsilon: Real,
    max_outcomes: int = DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES,
) -> CalibratedCompositeProjectedTest:
    """Calibrate the maximal admissible randomized upper projected-score test.

    For the explicitly finite family, worst-case errors are evaluated exactly
    over all declared members. The threshold family is traversed in decreasing
    projected score and boundary randomization is chosen to exhaust the Type-I
    budget. This is a restricted optimum within the selected score family, not
    a claim of unrestricted minimax optimality.
    """

    if not isinstance(projection, VerifiedCompositeRenyiProjection):
        raise InputValidationError(
            "projection must be VerifiedCompositeRenyiProjection"
        )
    problem = projection.problem
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    outcomes = _enumerate_outcomes(problem, max_outcomes)
    null_mass = _family_mass_matrix(problem.null, outcomes)
    alternative_mass = _family_mass_matrix(problem.alternative, outcomes)
    scores = tuple(_decimal_score(counts, projection) for counts in outcomes)
    groups: dict[Decimal, list[int]] = {}
    for index, score in enumerate(scores):
        groups.setdefault(score, []).append(index)
    ordered_scores = tuple(sorted(groups, reverse=True))

    rejection = np.zeros(len(outcomes), dtype=float)
    above_null = np.zeros(len(problem.null.members), dtype=float)
    above_alternative = np.zeros(len(problem.alternative.members), dtype=float)

    chosen_score: Decimal | None = None
    chosen_eta: float | None = None
    for score in ordered_scores:
        indices = groups[score]
        boundary_null = np.sum(null_mass[:, indices], axis=1)
        boundary_alternative = np.sum(alternative_mass[:, indices], axis=1)
        fully_included = above_null + boundary_null
        if float(np.max(fully_included)) <= constraint.epsilon:
            rejection[indices] = 1.0
            above_null = fully_included
            above_alternative = above_alternative + boundary_alternative
            continue

        eta_limits = []
        for current, boundary in zip(above_null, boundary_null, strict=True):
            if boundary <= 0:
                continue
            eta_limits.append(
                (constraint.epsilon - float(current)) / float(boundary)
            )
        if not eta_limits:
            raise CompositeProjectionError(
                "calibration encountered a score boundary with zero null mass "
                "under every declared member"
            )
        eta = min(eta_limits)
        if not math.isfinite(eta) or eta < 0 or eta > 1:
            raise CompositeProjectionError(
                "calibration boundary randomization left [0, 1]"
            )
        rejection[indices] = eta
        above_null = above_null + eta * boundary_null
        above_alternative = above_alternative + eta * boundary_alternative
        chosen_score = score
        chosen_eta = eta
        break

    if chosen_score is None or chosen_eta is None:
        raise CompositeProjectionError(
            "calibration failed to encounter the Type-I boundary"
        )

    null_errors = null_mass @ rejection
    alternative_errors = 1.0 - alternative_mass @ rejection
    worst_alpha = float(np.max(null_errors))
    worst_beta = float(np.max(alternative_errors))
    if worst_alpha > constraint.epsilon + COMPOSITE_NUMERICAL_TOLERANCE:
        raise CompositeProjectionError(
            "calibrated projected test violates the composite Type-I constraint"
        )
    if abs(worst_alpha - constraint.epsilon) > COMPOSITE_NUMERICAL_TOLERANCE:
        raise CompositeProjectionError(
            "calibrated projected test did not exhaust the Type-I budget"
        )
    return CalibratedCompositeProjectedTest(
        projection=projection,
        constraint=constraint,
        threshold_score=float(chosen_score),
        boundary_randomization=chosen_eta,
        outcomes=outcomes,
        rejection_probabilities=tuple(float(value) for value in rejection),
        null_type_i_errors=tuple(float(value) for value in null_errors),
        alternative_type_ii_errors=tuple(float(value) for value in alternative_errors),
        worst_type_i_error=worst_alpha,
        worst_type_ii_error=worst_beta,
    )


__all__ = [
    "DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES",
    "CalibratedCompositeProjectedTest",
    "CompositeBinaryTestingProblem",
    "CompositeEnumerationLimitError",
    "CompositeMIDLawFamily",
    "CompositeOptimizationError",
    "CompositeProjectedBound",
    "CompositeProjectionError",
    "CompositeRenyiConverseBound",
    "FiniteCompositeMinimaxResult",
    "VerifiedCompositeRenyiProjection",
    "calibrate_composite_projected_test",
    "composite_renyi_converse_at_order",
    "exact_finite_composite_minimax",
    "projected_composite_bound_at_order",
    "verified_composite_renyi_projection",
]
