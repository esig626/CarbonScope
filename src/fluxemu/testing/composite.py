"""Finite composite testing for explicit genuine-count observation-law classes.

This module deliberately separates two questions.

1. A projected Rényi score gives an explicit threshold test and finite-sample
   achievability/converse bounds.
2. When the complete observation space and both hypothesis classes are finite,
   the unrestricted minimax Type-II optimum is the solution of a linear
   programme.

The finite classes here are exactly the laws supplied by the caller. They are
not silently treated as the full continuous flux family or as their convex
closure. A sampled flux ensemble therefore remains a numerical approximation
to a larger scientific hypothesis unless the missing optimisation is separately
solved or bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from numbers import Integral, Real
from typing import Iterator

from .. import observation as _observation
from ..exceptions import InputValidationError, ValidationError
from ..model import deterministic_serialise
from .simple import (
    NumericalLimitError,
    SimpleBinaryTestingConstraint,
    validate_renyi_order,
)


DEFAULT_COMPOSITE_MAX_OUTCOMES = 100_000
COMPOSITE_SOLVER_FEASIBILITY_TOLERANCE = 1e-9
COMPOSITE_VALIDATION_TOLERANCE = 1e-7
COMPOSITE_HIGHS_SMALL_MATRIX_VALUE = 1e-12
COMPOSITE_MOMENT_LOG_TOLERANCE = 1e-10


class FiniteCompositeEnumerationLimitError(ValidationError):
    """The complete finite observation space exceeds the declared limit."""


class CompositeMinimaxSolverError(ValidationError):
    """The finite minimax LP could not be certified numerically."""


class CompositeAchievabilityConditionError(InputValidationError):
    """A projected score is undefined on support used by the declared classes."""


def _digest(value: object) -> str:
    return sha256(deterministic_serialise(value).encode("utf-8")).hexdigest()


def _validate_label(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise InputValidationError(f"{name} must be None or a nonempty string")


def _validate_max_outcomes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise InputValidationError("max_outcomes must be a positive integer, not bool")
    return int(value)


def _projection_order(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError("projected Rényi order must be a finite real scalar")
    try:
        order = float(value)
    except (OverflowError, ValueError) as error:
        raise NumericalLimitError("projected Rényi order is outside float representability") from error
    if not math.isfinite(order) or not 0 < value < 1:
        raise InputValidationError("projected Rényi order must satisfy 0 < lambda < 1")
    if not 0 < order < 1:
        raise NumericalLimitError("projected Rényi order rounds to an endpoint at float precision")
    return order


@dataclass(frozen=True, slots=True, kw_only=True)
class FiniteObservationLaw:
    """One complete genuine-count observation law.

    Multiple blocks are a product only when ``independent=True`` is explicitly
    declared. The block order is scientific input and is preserved exactly.
    """

    blocks: tuple[_observation.MultinomialMIDLaw, ...]
    independent: bool = False
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise InputValidationError("blocks must be a nonempty immutable tuple")
        if not all(isinstance(law, _observation.MultinomialMIDLaw) for law in self.blocks):
            raise InputValidationError("every observation block must be MultinomialMIDLaw")
        if type(self.independent) is not bool:
            raise InputValidationError("independent must be an explicit bool")
        if len(self.blocks) > 1 and not self.independent:
            raise InputValidationError("multiple observation blocks require independent=True")
        if len(self.blocks) == 1 and self.independent:
            raise InputValidationError("a single observation block must use independent=False")
        _validate_label(self.label, "law label")

    @classmethod
    def single(
        cls, law: _observation.MultinomialMIDLaw, *, label: str | None = None,
    ) -> "FiniteObservationLaw":
        return cls(blocks=(law,), independent=False, label=label)

    @property
    def count_totals(self) -> tuple[int, ...]:
        return tuple(law.n for law in self.blocks)

    @property
    def mass_class_spaces(self) -> tuple[tuple[int, ...], ...]:
        return tuple(law.mass_classes for law in self.blocks)

    @property
    def geometry(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        return tuple((law.n, law.mass_classes) for law in self.blocks)

    @property
    def fingerprint(self) -> str:
        return _digest((
            "finite-observation-law-v1",
            self.independent,
            self.label,
            tuple(law.fingerprint for law in self.blocks),
        ))

    def log_pmf(self, outcome: tuple[tuple[int, ...], ...]) -> float:
        if not isinstance(outcome, tuple) or len(outcome) != len(self.blocks):
            raise InputValidationError("outcome must preserve the declared observation-block order")
        values = tuple(
            law.log_pmf(counts)
            for law, counts in zip(self.blocks, outcome, strict=True)
        )
        if any(value == -math.inf for value in values):
            return -math.inf
        result = math.fsum(values)
        if not math.isfinite(result):
            raise NumericalLimitError("complete observation log PMF is not finite")
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class FiniteCompositeHypotheses:
    """Explicit finite H0 and H1 classes on one common observation geometry."""

    null: tuple[FiniteObservationLaw, ...]
    alternative: tuple[FiniteObservationLaw, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.null, tuple) or not self.null:
            raise InputValidationError("null class must be a nonempty immutable tuple")
        if not isinstance(self.alternative, tuple) or not self.alternative:
            raise InputValidationError("alternative class must be a nonempty immutable tuple")
        if not all(isinstance(law, FiniteObservationLaw) for law in (*self.null, *self.alternative)):
            raise InputValidationError("both hypothesis classes must contain FiniteObservationLaw records")
        geometry = self.null[0].geometry
        for role, laws in (("null", self.null), ("alternative", self.alternative)):
            for index, law in enumerate(laws):
                if law.geometry != geometry:
                    raise InputValidationError(
                        f"{role} law {index} has observation geometry different from the declared common space"
                    )

    @property
    def geometry(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        return self.null[0].geometry

    @property
    def fingerprint(self) -> str:
        return _digest((
            "finite-composite-hypotheses-v1",
            tuple(law.fingerprint for law in self.null),
            tuple(law.fingerprint for law in self.alternative),
        ))


def _renyi(left: FiniteObservationLaw, right: FiniteObservationLaw, order: float) -> float:
    if left.geometry != right.geometry:
        raise InputValidationError("Rényi comparison requires common observation geometry")
    try:
        if len(left.blocks) == 1:
            value = _observation.renyi_multinomial(left.blocks[0], right.blocks[0], order)
        else:
            value = _observation.independent_product_renyi(left.blocks, right.blocks, order)
    except (ArithmeticError, ValidationError) as error:
        raise NumericalLimitError("full-law Rényi evaluation failed without support repair") from error
    if math.isnan(value) or value < 0:
        raise NumericalLimitError(
            f"full-law Rényi divergence returned invalid value {value!r}; signed roundoff is not clipped"
        )
    return float(value)


def _compositions(total: int, cells: int) -> Iterator[tuple[int, ...]]:
    if cells == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, cells - 1):
            yield (first, *rest)


def _block_outcomes(total: int, cell_count: int) -> Iterator[tuple[int, ...]]:
    yield from _compositions(total, cell_count)


def _full_outcome_count(
    hypotheses: FiniteCompositeHypotheses, max_outcomes: int,
) -> int:
    count = 1
    for total, mass_classes in hypotheses.geometry:
        count *= math.comb(total + len(mass_classes) - 1, len(mass_classes) - 1)
        if count > max_outcomes:
            raise FiniteCompositeEnumerationLimitError(
                "finite composite testing requires enumeration of "
                f"{count} or more outcomes, exceeding max_outcomes={max_outcomes}; "
                "no asymptotic or Monte Carlo substitute was used"
            )
    return count


def _joint_outcomes(hypotheses: FiniteCompositeHypotheses) -> Iterator[tuple[tuple[int, ...], ...]]:
    geometry = hypotheses.geometry

    def recurse(index: int, prefix: tuple[tuple[int, ...], ...]):
        if index == len(geometry):
            yield prefix
            return
        total, mass_classes = geometry[index]
        for counts in _block_outcomes(total, len(mass_classes)):
            yield from recurse(index + 1, (*prefix, counts))

    yield from recurse(0, ())


def _probability_from_log(log_probability: float) -> float:
    if log_probability == -math.inf:
        return 0.0
    if not math.isfinite(log_probability) or log_probability > 0:
        raise NumericalLimitError("enumerated log probability is outside the finite probability range")
    value = math.exp(log_probability)
    if value == 0.0:
        raise NumericalLimitError(
            "a positive enumerated probability underflowed to zero; exact support would be lost"
        )
    return value


@dataclass(frozen=True, slots=True)
class _ProbabilityTable:
    outcomes: tuple[tuple[tuple[int, ...], ...], ...]
    null: tuple[tuple[float, ...], ...]
    alternative: tuple[tuple[float, ...], ...]


def _probability_table(
    hypotheses: FiniteCompositeHypotheses, *, max_outcomes: int,
) -> _ProbabilityTable:
    limit = _validate_max_outcomes(max_outcomes)
    _full_outcome_count(hypotheses, limit)
    all_outcomes = tuple(_joint_outcomes(hypotheses))

    def evaluate(laws: tuple[FiniteObservationLaw, ...]) -> tuple[tuple[float, ...], ...]:
        rows = []
        for law in laws:
            row = tuple(_probability_from_log(law.log_pmf(outcome)) for outcome in all_outcomes)
            total = math.fsum(row)
            if not math.isfinite(total) or abs(total - 1.0) > COMPOSITE_VALIDATION_TOLERANCE:
                raise NumericalLimitError(
                    "enumerated complete-law probability mass differs materially from one; "
                    "the row was not renormalised"
                )
            rows.append(row)
        return tuple(rows)

    null = evaluate(hypotheses.null)
    alternative = evaluate(hypotheses.alternative)

    active = tuple(
        index
        for index in range(len(all_outcomes))
        if any(row[index] > 0 for row in (*null, *alternative))
    )
    outcomes = tuple(all_outcomes[index] for index in active)
    null_active = tuple(tuple(row[index] for index in active) for row in null)
    alternative_active = tuple(tuple(row[index] for index in active) for row in alternative)

    for role, rows in (("null", null_active), ("alternative", alternative_active)):
        for law_index, row in enumerate(rows):
            total = math.fsum(row)
            if abs(total - 1.0) > COMPOSITE_VALIDATION_TOLERANCE:
                raise NumericalLimitError(
                    f"{role} law {law_index} lost probability mass when inactive outcomes were removed"
                )
    return _ProbabilityTable(outcomes, null_active, alternative_active)


@dataclass(frozen=True, slots=True, kw_only=True)
class FiniteMinimaxTestResult:
    """Numerically certified solution of the exact finite minimax LP."""

    hypotheses: FiniteCompositeHypotheses
    epsilon: float
    outcomes: tuple[tuple[tuple[int, ...], ...], ...]
    rejection_probabilities: tuple[float, ...]
    beta_star: float
    null_type_i_errors: tuple[float, ...]
    alternative_type_ii_errors: tuple[float, ...]
    solver_objective: float
    solver_status: str

    @property
    def worst_null_indices(self) -> tuple[int, ...]:
        worst = max(self.null_type_i_errors)
        return tuple(
            index for index, value in enumerate(self.null_type_i_errors)
            if abs(value - worst) <= COMPOSITE_VALIDATION_TOLERANCE
        )

    @property
    def worst_alternative_indices(self) -> tuple[int, ...]:
        worst = max(self.alternative_type_ii_errors)
        return tuple(
            index for index, value in enumerate(self.alternative_type_ii_errors)
            if abs(value - worst) <= COMPOSITE_VALIDATION_TOLERANCE
        )

    @property
    def fingerprint(self) -> str:
        return _digest((
            "finite-minimax-test-result-v1",
            self.hypotheses.fingerprint,
            self.epsilon,
            self.outcomes,
            self.rejection_probabilities,
            self.beta_star,
        ))


def _highspy():
    try:
        import highspy
    except ImportError as error:
        raise CompositeMinimaxSolverError(
            "finite minimax testing requires the default highspy dependency"
        ) from error
    return highspy


def _require_solver_resolvable_probabilities(table: _ProbabilityTable) -> None:
    for role, rows in (("null", table.null), ("alternative", table.alternative)):
        for law_index, row in enumerate(rows):
            tiny = [
                (index, value)
                for index, value in enumerate(row)
                if 0 < value <= COMPOSITE_HIGHS_SMALL_MATRIX_VALUE
            ]
            if tiny:
                index, value = min(tiny, key=lambda item: item[1])
                raise CompositeMinimaxSolverError(
                    f"{role} law {law_index} has positive outcome probability {value:g} "
                    f"at outcome {index}, at or below HiGHS matrix resolution "
                    f"{COMPOSITE_HIGHS_SMALL_MATRIX_VALUE:g}; exact support is not dropped"
                )


def solve_finite_minimax_test(
    hypotheses: FiniteCompositeHypotheses,
    *,
    epsilon: Real,
    max_outcomes: int = DEFAULT_COMPOSITE_MAX_OUTCOMES,
) -> FiniteMinimaxTestResult:
    r"""Solve the unrestricted finite minimax test.

    The LP is

    minimise t

    subject to
        sum_y P(y) phi(y) <= epsilon       for every P in H0,
        sum_y Q(y) phi(y) + t >= 1         for every Q in H1,
        0 <= phi(y) <= 1,
        0 <= t <= 1.

    Thus ``phi(y)`` is the probability of deciding H1 after outcome ``y`` and
    the optimum ``t`` is beta* for the explicit finite law classes. The LP
    formulation is exact; the returned numerical optimum is independently
    checked against the declared feasibility tolerance.
    """
    if not isinstance(hypotheses, FiniteCompositeHypotheses):
        raise InputValidationError("hypotheses must be FiniteCompositeHypotheses")
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    table = _probability_table(hypotheses, max_outcomes=max_outcomes)
    _require_solver_resolvable_probabilities(table)

    highspy = _highspy()
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("threads", 1)
    solver.setOptionValue("solver", "simplex")
    solver.setOptionValue(
        "primal_feasibility_tolerance", COMPOSITE_SOLVER_FEASIBILITY_TOLERANCE
    )
    solver.setOptionValue(
        "dual_feasibility_tolerance", COMPOSITE_SOLVER_FEASIBILITY_TOLERANCE
    )
    solver.setOptionValue("small_matrix_value", COMPOSITE_HIGHS_SMALL_MATRIX_VALUE)

    outcome_count = len(table.outcomes)
    variable_count = outcome_count + 1
    costs = [0.0] * outcome_count + [1.0]
    lower_bounds = [0.0] * variable_count
    upper_bounds = [1.0] * variable_count
    solver.addCols(
        variable_count,
        costs,
        lower_bounds,
        upper_bounds,
        0,
        [0] * (variable_count + 1),
        [],
        [],
    )

    row_lower: list[float] = []
    row_upper: list[float] = []
    row_starts = [0]
    indices: list[int] = []
    values: list[float] = []

    for row in table.null:
        row_lower.append(-highspy.kHighsInf)
        row_upper.append(constraint.epsilon)
        for index, value in enumerate(row):
            if value > 0:
                indices.append(index)
                values.append(value)
        row_starts.append(len(indices))

    for row in table.alternative:
        row_lower.append(1.0)
        row_upper.append(highspy.kHighsInf)
        for index, value in enumerate(row):
            if value > 0:
                indices.append(index)
                values.append(value)
        indices.append(outcome_count)
        values.append(1.0)
        row_starts.append(len(indices))

    solver.addRows(
        len(row_lower),
        row_lower,
        row_upper,
        len(indices),
        row_starts,
        indices,
        values,
    )
    solver.setMinimize()
    solver.run()
    status = solver.getModelStatus()
    status_name = solver.modelStatusToString(status)
    if status != highspy.HighsModelStatus.kOptimal:
        raise CompositeMinimaxSolverError(
            f"finite minimax LP failed: HiGHS status {status_name}"
        )

    solution = tuple(float(value) for value in solver.getSolution().col_value)
    objective = float(solver.getObjectiveValue())
    if (
        len(solution) != variable_count
        or not all(math.isfinite(value) for value in solution)
        or not math.isfinite(objective)
    ):
        raise CompositeMinimaxSolverError("finite minimax LP returned malformed numerical output")

    phi = solution[:-1]
    raw_t = solution[-1]
    if any(
        value < -COMPOSITE_VALIDATION_TOLERANCE
        or value > 1 + COMPOSITE_VALIDATION_TOLERANCE
        for value in (*phi, raw_t)
    ):
        raise CompositeMinimaxSolverError("finite minimax LP returned a decision variable outside [0, 1]")

    null_errors = tuple(
        math.fsum(probability * decision for probability, decision in zip(row, phi, strict=True))
        for row in table.null
    )
    alternative_errors = tuple(
        math.fsum(probability * (1.0 - decision) for probability, decision in zip(row, phi, strict=True))
        for row in table.alternative
    )
    worst_null = max(null_errors)
    beta_star = max(alternative_errors)
    if worst_null > constraint.epsilon + COMPOSITE_VALIDATION_TOLERANCE:
        raise CompositeMinimaxSolverError(
            "independent minimax validation found a Type-I constraint violation"
        )
    if abs(beta_star - raw_t) > COMPOSITE_VALIDATION_TOLERANCE:
        raise CompositeMinimaxSolverError(
            "independent minimax validation disagrees with the solver epigraph variable"
        )
    if abs(objective - raw_t) > COMPOSITE_VALIDATION_TOLERANCE:
        raise CompositeMinimaxSolverError(
            "finite minimax solver objective disagrees with its epigraph variable"
        )

    return FiniteMinimaxTestResult(
        hypotheses=hypotheses,
        epsilon=constraint.epsilon,
        outcomes=table.outcomes,
        rejection_probabilities=phi,
        beta_star=beta_star,
        null_type_i_errors=null_errors,
        alternative_type_ii_errors=alternative_errors,
        solver_objective=objective,
        solver_status=status_name,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeRenyiConverse:
    """Order-specific composite lower bound on beta* for explicit finite classes."""

    hypotheses: FiniteCompositeHypotheses
    epsilon: float
    order: float
    reverse_renyi: float
    forward_renyi: float
    reverse_pair_indices: tuple[int, int]
    forward_pair_indices: tuple[int, int]
    reverse_lower_bound: float
    forward_lower_bound: float
    type_ii_lower_bound: float


def composite_renyi_converse_at_order(
    hypotheses: FiniteCompositeHypotheses,
    *,
    epsilon: Real,
    order: Real,
) -> CompositeRenyiConverse:
    """Apply pairwise Rényi converses and take the strongest class lower bound.

    The laws supplied to this function are already complete finite observation
    laws. There is therefore no additional hidden tensor power or sample-size
    multiplier here.
    """
    if not isinstance(hypotheses, FiniteCompositeHypotheses):
        raise InputValidationError("hypotheses must be FiniteCompositeHypotheses")
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    finite_order = validate_renyi_order(order)

    reverse_best = math.inf
    forward_best = math.inf
    reverse_pair = (0, 0)
    forward_pair = (0, 0)
    for null_index, null in enumerate(hypotheses.null):
        for alternative_index, alternative in enumerate(hypotheses.alternative):
            reverse = _renyi(alternative, null, finite_order)
            forward = _renyi(null, alternative, finite_order)
            if reverse < reverse_best:
                reverse_best = reverse
                reverse_pair = (null_index, alternative_index)
            if forward < forward_best:
                forward_best = forward
                forward_pair = (null_index, alternative_index)

    coefficient = (finite_order - 1.0) / finite_order
    if math.isinf(reverse_best):
        reverse_lower = 0.0
    else:
        gap = max(math.log(1.0 / constraint.epsilon) - reverse_best, 0.0)
        reverse_lower = -math.expm1(-coefficient * gap)

    if math.isinf(forward_best):
        forward_lower = 0.0
    else:
        log_forward = (
            finite_order / (finite_order - 1.0)
        ) * math.log1p(-constraint.epsilon) - forward_best
        forward_lower = math.exp(log_forward)

    combined = max(reverse_lower, forward_lower)
    if not 0 <= combined <= 1 or not math.isfinite(combined):
        raise NumericalLimitError("composite Rényi converse left the probability interval")

    return CompositeRenyiConverse(
        hypotheses=hypotheses,
        epsilon=constraint.epsilon,
        order=finite_order,
        reverse_renyi=reverse_best,
        forward_renyi=forward_best,
        reverse_pair_indices=reverse_pair,
        forward_pair_indices=forward_pair,
        reverse_lower_bound=reverse_lower,
        forward_lower_bound=forward_lower,
        type_ii_lower_bound=combined,
    )


def _logsumexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    high = max(values)
    if high == math.inf:
        return math.inf
    if high == -math.inf:
        return -math.inf
    return high + math.log(math.fsum(math.exp(value - high) for value in values))


def _score_from_projected_pair(
    null_row: tuple[float, ...], alternative_row: tuple[float, ...],
    table: _ProbabilityTable,
) -> tuple[float, ...]:
    scores = []
    for index, (p, q) in enumerate(zip(null_row, alternative_row, strict=True)):
        if p == 0 and q == 0:
            if any(row[index] > 0 for row in (*table.null, *table.alternative)):
                raise CompositeAchievabilityConditionError(
                    "projected pair is zero on an outcome used by another declared law; "
                    "the projected log-likelihood ratio is undefined on class support"
                )
            scores.append(0.0)
        elif p == 0:
            scores.append(math.inf)
        elif q == 0:
            scores.append(-math.inf)
        else:
            scores.append(math.log(q) - math.log(p))
    return tuple(scores)


def _log_moment(row: tuple[float, ...], scores: tuple[float, ...], coefficient: float) -> float:
    terms: list[float] = []
    for probability, score in zip(row, scores, strict=True):
        if probability == 0:
            continue
        weighted = coefficient * score
        if weighted == math.inf:
            return math.inf
        if weighted == -math.inf:
            continue
        terms.append(math.log(probability) + weighted)
    return _logsumexp(terms)


def _calibrate_score_test(
    table: _ProbabilityTable,
    scores: tuple[float, ...],
    epsilon: float,
) -> tuple[float, float, tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    levels = sorted(set(scores), reverse=True)
    higher: set[int] = set()

    for level in levels:
        boundary = tuple(index for index, score in enumerate(scores) if score == level)
        base_errors = tuple(
            math.fsum(row[index] for index in higher)
            for row in table.null
        )
        boundary_errors = tuple(
            math.fsum(row[index] for index in boundary)
            for row in table.null
        )
        if max(base_errors) > epsilon + COMPOSITE_VALIDATION_TOLERANCE:
            raise NumericalLimitError("score-threshold calibration lost Type-I monotonicity")

        eta = 1.0
        for base, boundary_mass in zip(base_errors, boundary_errors, strict=True):
            if boundary_mass > 0:
                allowance = (epsilon - base) / boundary_mass
                eta = min(eta, allowance)
            elif base > epsilon + COMPOSITE_VALIDATION_TOLERANCE:
                raise NumericalLimitError("score-threshold calibration is infeasible")
        if eta < 1.0 - COMPOSITE_VALIDATION_TOLERANCE:
            if eta < -COMPOSITE_VALIDATION_TOLERANCE:
                raise NumericalLimitError("score-threshold calibration produced negative randomisation")
            eta = max(0.0, eta)
            decisions = tuple(
                1.0 if index in higher else eta if index in boundary else 0.0
                for index in range(len(scores))
            )
            null_errors = tuple(
                math.fsum(p * phi for p, phi in zip(row, decisions, strict=True))
                for row in table.null
            )
            alternative_errors = tuple(
                math.fsum(q * (1.0 - phi) for q, phi in zip(row, decisions, strict=True))
                for row in table.alternative
            )
            if max(null_errors) > epsilon + COMPOSITE_VALIDATION_TOLERANCE:
                raise NumericalLimitError("calibrated projected test violates the Type-I constraint")
            return level, eta, decisions, null_errors, alternative_errors
        higher.update(boundary)

    raise NumericalLimitError("finite score-threshold calibration failed to meet an interior epsilon")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectedRenyiTestResult:
    """Explicit projected-score test and its finite-class guarantees."""

    hypotheses: FiniteCompositeHypotheses
    epsilon: float
    order: float
    projected_null_index: int
    projected_alternative_index: int
    projected_renyi: float
    scores: tuple[float, ...]
    log_null_moment_supremum: float
    log_alternative_moment_supremum: float
    projected_formula_certified: bool
    uniform_score_threshold: float
    uniform_score_type_ii_upper_bound: float
    projected_formula_type_ii_upper_bound: float | None
    calibrated_threshold: float
    calibrated_boundary_randomisation: float
    calibrated_rejection_probabilities: tuple[float, ...]
    calibrated_null_type_i_errors: tuple[float, ...]
    calibrated_alternative_type_ii_errors: tuple[float, ...]
    calibrated_type_ii_error: float
    best_achievable_upper_bound: float


def projected_renyi_test(
    hypotheses: FiniteCompositeHypotheses,
    *,
    epsilon: Real,
    order: Real,
    max_outcomes: int = DEFAULT_COMPOSITE_MAX_OUTCOMES,
) -> ProjectedRenyiTestResult:
    r"""Construct a projected Rényi score and certify what it actually guarantees.

    The pair minimising D_lambda(Q||P) over the explicitly supplied finite pair
    set defines h=log(Q*/P*). Because a finite list is not automatically a
    convex hypothesis class, CarbonScope does not assume that the projection
    theorem applies. Instead it evaluates the two uniform exponential moments
    over every supplied law.

    The calibrated threshold is the best test within this one projected-score
    threshold family. It is an achieved test, hence its worst Type-II error is
    always an upper bound on the unrestricted beta* for the explicit finite
    problem. It is not claimed to be minimax optimal.
    """
    if not isinstance(hypotheses, FiniteCompositeHypotheses):
        raise InputValidationError("hypotheses must be FiniteCompositeHypotheses")
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    finite_order = _projection_order(order)
    table = _probability_table(hypotheses, max_outcomes=max_outcomes)

    projected_value = math.inf
    projected_null = 0
    projected_alternative = 0
    for null_index, null in enumerate(hypotheses.null):
        for alternative_index, alternative in enumerate(hypotheses.alternative):
            value = _renyi(alternative, null, finite_order)
            if value < projected_value:
                projected_value = value
                projected_null = null_index
                projected_alternative = alternative_index
    if math.isinf(projected_value):
        raise CompositeAchievabilityConditionError(
            "every declared null/alternative pair has infinite projected Rényi divergence"
        )

    p_star = table.null[projected_null]
    q_star = table.alternative[projected_alternative]
    scores = _score_from_projected_pair(p_star, q_star, table)

    null_log_moments = tuple(
        _log_moment(row, scores, finite_order)
        for row in table.null
    )
    alternative_log_moments = tuple(
        _log_moment(row, scores, finite_order - 1.0)
        for row in table.alternative
    )
    log_m0 = max(null_log_moments)
    log_m1 = max(alternative_log_moments)
    if math.isinf(log_m0) or math.isinf(log_m1):
        raise CompositeAchievabilityConditionError(
            "projected score has an infinite uniform exponential moment on the declared classes"
        )

    threshold = (
        math.log(1.0 / constraint.epsilon) + log_m0
    ) / finite_order
    log_beta_score = (1.0 - finite_order) * threshold + log_m1
    score_upper = 1.0 if log_beta_score >= 0 else math.exp(log_beta_score)

    log_h = (finite_order - 1.0) * projected_value
    projected_formula_certified = (
        log_m0 <= log_h + COMPOSITE_MOMENT_LOG_TOLERANCE
        and log_m1 <= log_h + COMPOSITE_MOMENT_LOG_TOLERANCE
    )
    projected_formula_upper: float | None = None
    if projected_formula_certified:
        log_projected = -(
            (1.0 - finite_order) / finite_order
        ) * (
            projected_value - math.log(1.0 / constraint.epsilon)
        )
        projected_formula_upper = 1.0 if log_projected >= 0 else math.exp(log_projected)
        projected_formula_upper = min(1.0 - constraint.epsilon, projected_formula_upper)

    (
        calibrated_threshold,
        eta,
        decisions,
        calibrated_null_errors,
        calibrated_alternative_errors,
    ) = _calibrate_score_test(table, scores, constraint.epsilon)
    calibrated_beta = max(calibrated_alternative_errors)
    best_upper = min(1.0 - constraint.epsilon, calibrated_beta)

    return ProjectedRenyiTestResult(
        hypotheses=hypotheses,
        epsilon=constraint.epsilon,
        order=finite_order,
        projected_null_index=projected_null,
        projected_alternative_index=projected_alternative,
        projected_renyi=projected_value,
        scores=scores,
        log_null_moment_supremum=log_m0,
        log_alternative_moment_supremum=log_m1,
        projected_formula_certified=projected_formula_certified,
        uniform_score_threshold=threshold,
        uniform_score_type_ii_upper_bound=score_upper,
        projected_formula_type_ii_upper_bound=projected_formula_upper,
        calibrated_threshold=calibrated_threshold,
        calibrated_boundary_randomisation=eta,
        calibrated_rejection_probabilities=decisions,
        calibrated_null_type_i_errors=calibrated_null_errors,
        calibrated_alternative_type_ii_errors=calibrated_alternative_errors,
        calibrated_type_ii_error=calibrated_beta,
        best_achievable_upper_bound=best_upper,
    )


__all__ = [
    "DEFAULT_COMPOSITE_MAX_OUTCOMES",
    "CompositeAchievabilityConditionError",
    "CompositeMinimaxSolverError",
    "CompositeRenyiConverse",
    "FiniteCompositeEnumerationLimitError",
    "FiniteCompositeHypotheses",
    "FiniteMinimaxTestResult",
    "FiniteObservationLaw",
    "ProjectedRenyiTestResult",
    "composite_renyi_converse_at_order",
    "projected_renyi_test",
    "solve_finite_minimax_test",
]
