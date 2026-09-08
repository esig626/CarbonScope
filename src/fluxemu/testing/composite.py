"""Finite composite testing for independent products of genuine-count MID laws.

Each hypothesis member is one complete observable law: an explicitly ordered
product of fixed-total multinomial MID blocks. Composite classes are explicit
finite collections of such laws. They are never silently convexified and their
member frequencies are never interpreted as biological priors.

The module provides four deliberately separate objects:

* an order-specific pairwise composite Rényi Type-II lower bound;
* a bounded exact randomised minimax LP oracle on the complete joint count space;
* a finite-family Rényi-minimising *candidate score* for 0<lambda<1, together
  with direct uniform-moment/support verification; and
* analytical or exactly enumerated threshold/calibration results for a verified
  candidate score.

A vertex-pair Rényi minimum in a finite non-convex family is not called a joint
Rényi projection and is never advertised as a finite-n least-favourable pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import product
import math
from numbers import Integral, Real
from typing import Iterable

import numpy as np

from ..exceptions import FluxEMUError, InputValidationError, ValidationError
from ..observation import MultinomialMIDLaw, independent_product_renyi
from .simple import (
    NumericalLimitError,
    SimpleBinaryTestingConstraint,
    _finite_real,
    _testing_digest,
    validate_renyi_order,
)


DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES = 1_000_000
MIN_EXACT_COMPOSITE_EPSILON = 1e-12
COMPOSITE_NUMERICAL_TOLERANCE = 1e-10
_SCORE_VERIFICATION_TOLERANCE = 1e-10
_DECIMAL_PRECISION = 60


class CompositeEnumerationLimitError(ValidationError):
    """The complete joint count space exceeds the explicit enumeration cap."""


class CompositeScoreVerificationError(FluxEMUError):
    """A finite-family candidate score lacks verified uniform error control."""


class CompositeOptimizationError(FluxEMUError):
    """Exact finite-family minimax optimisation failed numerically."""


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{name} must be a nonempty string")
    return value


def _block_identity(value: object, index: int) -> tuple[str, str, str]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise InputValidationError(
            f"block identity {index} must be an immutable experiment/target/replicate triple"
        )
    result = tuple(_identifier(item, "block identity field") for item in value)
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, kw_only=True)
class IndependentMIDProductLaw:
    """One complete observation law over explicitly independent MID blocks."""

    blocks: tuple[MultinomialMIDLaw, ...]
    block_identities: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise InputValidationError(
                "independent MID product blocks must be a nonempty immutable tuple"
            )
        if not all(isinstance(block, MultinomialMIDLaw) for block in self.blocks):
            raise InputValidationError(
                "every independent MID product block must be MultinomialMIDLaw"
            )
        if not isinstance(self.block_identities, tuple):
            raise InputValidationError("block_identities must be an immutable tuple")
        if self.block_identities:
            if len(self.block_identities) != len(self.blocks):
                raise InputValidationError(
                    "block_identities must identify every independent MID block"
                )
            identities = tuple(
                _block_identity(value, index)
                for index, value in enumerate(self.block_identities)
            )
        else:
            identities = tuple(
                ("block", str(index), "counts") for index in range(len(self.blocks))
            )
        if len(set(identities)) != len(identities):
            raise InputValidationError("independent MID block identities must be unique")
        object.__setattr__(self, "block_identities", identities)

    @property
    def block_totals(self) -> tuple[int, ...]:
        return tuple(block.n for block in self.blocks)

    @property
    def block_mass_classes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(block.mass_classes for block in self.blocks)

    @property
    def block_signature(
        self,
    ) -> tuple[tuple[tuple[str, str, str], int, tuple[int, ...]], ...]:
        return tuple(
            (identity, block.n, block.mass_classes)
            for identity, block in zip(self.block_identities, self.blocks, strict=True)
        )

    @property
    def full_support(self) -> bool:
        return all(
            all(probability > 0 for probability in block.probabilities)
            for block in self.blocks
        )

    @property
    def fingerprint(self) -> str:
        return _testing_digest(
            (
                "independent-mid-product-law-v1",
                self.block_identities,
                tuple(block.fingerprint for block in self.blocks),
            )
        )

    def log_pmf(self, outcome: tuple[tuple[int, ...], ...]) -> float:
        if not isinstance(outcome, tuple) or len(outcome) != len(self.blocks):
            raise InputValidationError(
                "joint composite outcome must contain one count vector per MID block"
            )
        values = tuple(
            block.log_pmf(counts)
            for block, counts in zip(self.blocks, outcome, strict=True)
        )
        if any(value == -math.inf for value in values):
            return -math.inf
        return math.fsum(values)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeMIDLawFamily:
    """One explicit finite hypothesis class of complete product observation laws."""

    members: tuple[IndependentMIDProductLaw, ...]
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple) or not self.members:
            raise InputValidationError(
                "composite family members must be a nonempty immutable tuple"
            )
        if not all(isinstance(member, IndependentMIDProductLaw) for member in self.members):
            raise InputValidationError(
                "every composite family member must be IndependentMIDProductLaw"
            )
        signature = self.members[0].block_signature
        for index, member in enumerate(self.members[1:], start=1):
            if member.block_signature != signature:
                raise InputValidationError(
                    f"composite family member {index} has a different observation-block structure"
                )
        if not isinstance(self.member_ids, tuple):
            raise InputValidationError("member_ids must be an immutable tuple")
        if self.member_ids and len(self.member_ids) != len(self.members):
            raise InputValidationError("member_ids must identify every composite family member")
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
    def block_signature(
        self,
    ) -> tuple[tuple[tuple[str, str, str], int, tuple[int, ...]], ...]:
        return self.members[0].block_signature

    @property
    def block_count(self) -> int:
        return len(self.members[0].blocks)

    @property
    def fingerprint(self) -> str:
        return _testing_digest(
            (
                "finite-composite-mid-product-family-v1",
                self.member_ids,
                tuple(member.fingerprint for member in self.members),
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeBinaryTestingProblem:
    """Explicit finite H0/H1 classes over one shared product observation space."""

    null: CompositeMIDLawFamily
    alternative: CompositeMIDLawFamily

    def __post_init__(self) -> None:
        if not isinstance(self.null, CompositeMIDLawFamily):
            raise InputValidationError("null must be CompositeMIDLawFamily")
        if not isinstance(self.alternative, CompositeMIDLawFamily):
            raise InputValidationError("alternative must be CompositeMIDLawFamily")
        if self.null.block_signature != self.alternative.block_signature:
            raise InputValidationError(
                "composite null and alternative must share the complete ordered observation-block structure"
            )

    @property
    def block_signature(
        self,
    ) -> tuple[tuple[tuple[str, str, str], int, tuple[int, ...]], ...]:
        return self.null.block_signature

    @property
    def fingerprint(self) -> str:
        return _testing_digest(
            (
                "finite-composite-binary-product-testing-problem-v1",
                ("H0", self.null.fingerprint),
                ("H1", self.alternative.fingerprint),
            )
        )


def _validate_score_order(order: Real) -> float:
    value = _finite_real(order, "candidate-score Rényi order lambda")
    if not 0 < order < 1:
        raise InputValidationError(
            "candidate-score Rényi order lambda must satisfy 0 < lambda < 1"
        )
    if not 0 < value < 1:
        raise NumericalLimitError(
            "candidate-score Rényi order rounds to an endpoint at float precision"
        )
    return value


def _full_renyi(
    left: IndependentMIDProductLaw,
    right: IndependentMIDProductLaw,
    order: float,
) -> float:
    value = independent_product_renyi(left.blocks, right.blocks, order)
    if value == math.inf:
        return math.inf
    if not math.isfinite(value) or value < 0:
        raise NumericalLimitError(
            "composite product Rényi divergence is nonfinite or negative at numerical precision"
        )
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeRenyiConverseBound:
    """Order-specific pairwise composite lower bound on worst-case Type-II error."""

    problem: CompositeBinaryTestingProblem
    constraint: SimpleBinaryTestingConstraint
    order: float
    reverse_renyi: float
    null_member_index: int | None
    alternative_member_index: int | None
    raw_reverse_lower_bound: float
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
    """Evaluate the full-observation finite-family converse at one lambda>1.

    For every declared P in H0 and Q in H1, the simple reverse-Rényi inequality
    applies to the corresponding full product observation laws. Minimising the
    directed full-law divergence therefore gives a valid lower bound on the
    composite worst-case Type-II error. No convexity or least-favourable-pair
    reduction is assumed.
    """

    if not isinstance(problem, CompositeBinaryTestingProblem):
        raise InputValidationError("problem must be CompositeBinaryTestingProblem")
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    finite_order = validate_renyi_order(order)
    best = math.inf
    best_pair: tuple[int, int] | None = None
    for null_index, null in enumerate(problem.null.members):
        for alternative_index, alternative in enumerate(problem.alternative.members):
            value = _full_renyi(alternative, null, finite_order)
            if value < best:
                best = value
                best_pair = (null_index, alternative_index)

    if best == math.inf:
        raw = -math.inf
    else:
        log_power = (finite_order - 1.0) / finite_order * math.fsum(
            (math.log(constraint.epsilon), best)
        )
        try:
            raw = -math.expm1(log_power)
        except OverflowError:
            raw = -math.inf
    lower = max(0.0, raw)
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
        reverse_renyi=best,
        null_member_index=null_index,
        alternative_member_index=alternative_index,
        raw_reverse_lower_bound=raw,
        type_ii_lower_bound=lower,
    )


def _validate_max_outcomes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise InputValidationError("max_outcomes must be a positive integer, not bool")
    return int(value)


def _compositions(total: int, cells: int):
    if cells == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remaining in _compositions(total - first, cells - 1):
            yield (first, *remaining)


def _joint_outcome_count(problem: CompositeBinaryTestingProblem, limit: int) -> int:
    count = 1
    for _, total, mass_classes in problem.block_signature:
        block_count = math.comb(total + len(mass_classes) - 1, len(mass_classes) - 1)
        if count > limit // block_count:
            raise CompositeEnumerationLimitError(
                "exact composite testing joint count space exceeds "
                f"max_outcomes={limit}; no approximation was substituted"
            )
        count *= block_count
    if count > limit:
        raise CompositeEnumerationLimitError(
            "exact composite testing joint count space contains "
            f"{count} outcomes, exceeding max_outcomes={limit}; no approximation was substituted"
        )
    return count


def _enumerate_joint_outcomes(
    problem: CompositeBinaryTestingProblem,
    max_outcomes: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    limit = _validate_max_outcomes(max_outcomes)
    expected = _joint_outcome_count(problem, limit)
    blocks = tuple(
        tuple(_compositions(total, len(mass_classes)))
        for _, total, mass_classes in problem.block_signature
    )
    outcomes = tuple(product(*blocks))
    if len(outcomes) != expected:  # pragma: no cover - combinatorial invariant
        raise CompositeOptimizationError(
            "internal joint count-space enumeration size is inconsistent"
        )
    return outcomes


def _law_mass_vector(
    law: IndependentMIDProductLaw,
    outcomes: tuple[tuple[tuple[int, ...], ...], ...],
) -> np.ndarray:
    values = np.empty(len(outcomes), dtype=float)
    for index, outcome in enumerate(outcomes):
        log_mass = law.log_pmf(outcome)
        if log_mass == -math.inf:
            values[index] = 0.0
            continue
        mass = math.exp(log_mass)
        if mass == 0.0:
            raise NumericalLimitError(
                "positive joint product-law outcome mass underflowed during exact "
                "composite enumeration; the law was not repaired"
            )
        values[index] = mass
    total = math.fsum(float(item) for item in values)
    if not math.isfinite(total) or abs(total - 1.0) > 5e-9:
        raise NumericalLimitError(
            "enumerated composite product-law mass does not sum to one within the numerical contract"
        )
    return values


def _family_mass_matrix(
    family: CompositeMIDLawFamily,
    outcomes: tuple[tuple[tuple[int, ...], ...], ...],
) -> np.ndarray:
    return np.vstack(tuple(_law_mass_vector(member, outcomes) for member in family.members))


def _accurate_expectations(matrix: np.ndarray, decision: np.ndarray) -> tuple[float, ...]:
    return tuple(
        math.fsum(
            float(probability) * float(value)
            for probability, value in zip(row, decision, strict=True)
        )
        for row in matrix
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class FiniteCompositeMinimaxResult:
    """Exact represented finite-class randomised minimax solution."""

    problem: CompositeBinaryTestingProblem
    constraint: SimpleBinaryTestingConstraint
    outcomes: tuple[tuple[tuple[int, ...], ...], ...]
    rejection_probabilities: tuple[float, ...]
    null_type_i_errors: tuple[float, ...]
    alternative_type_ii_errors: tuple[float, ...]
    worst_type_i_error: float
    minimax_type_ii_error: float
    active_null_members: tuple[int, ...]
    active_alternative_members: tuple[int, ...]

    @property
    def randomised(self) -> bool:
        return any(0.0 < value < 1.0 for value in self.rejection_probabilities)

    @property
    def randomized(self) -> bool:
        return self.randomised

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
    """Solve the complete represented finite-class minimax test by LP.

    The joint count space is the Cartesian product of every declared MID-block
    count space. This is intended as a small-problem/discretised oracle; if the
    product exceeds ``max_outcomes`` the function fails rather than reducing or
    approximating the observation space.

    Null constraints are divided by epsilon before being passed to HiGHS so an
    absolute LP feasibility tolerance cannot become the statistical Type-I
    tolerance. Budgets below ``MIN_EXACT_COMPOSITE_EPSILON`` fail explicitly.
    """

    if not isinstance(problem, CompositeBinaryTestingProblem):
        raise InputValidationError("problem must be CompositeBinaryTestingProblem")
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    if constraint.epsilon < MIN_EXACT_COMPOSITE_EPSILON:
        raise NumericalLimitError(
            "exact composite minimax Type-I budget is below the supported LP "
            f"numerical floor {MIN_EXACT_COMPOSITE_EPSILON:g}; no budget substitution was made"
        )
    outcomes = _enumerate_joint_outcomes(problem, max_outcomes)
    null_mass = _family_mass_matrix(problem.null, outcomes)
    alternative_mass = _family_mass_matrix(problem.alternative, outcomes)

    try:
        from scipy.optimize import linprog
    except ImportError as error:  # pragma: no cover - full CI installs testing extra
        raise CompositeOptimizationError(
            "exact composite minimax testing requires the 'testing' SciPy extra"
        ) from error

    outcome_count = len(outcomes)
    variable_count = outcome_count + 1
    objective = np.zeros(variable_count, dtype=float)
    objective[-1] = 1.0
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    inverse_budget = 1.0 / constraint.epsilon
    if not math.isfinite(inverse_budget):
        raise NumericalLimitError(
            "exact composite Type-I budget cannot be represented in the scaled LP"
        )
    for probabilities in null_mass:
        row = np.zeros(variable_count, dtype=float)
        row[:outcome_count] = probabilities * inverse_budget
        if not np.isfinite(row).all():
            raise NumericalLimitError(
                "scaled composite Type-I constraint exceeds finite LP representability"
            )
        rows.append(row)
        rhs.append(1.0)
    for probabilities in alternative_mass:
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
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
            "ipm_optimality_tolerance": 1e-12,
        },
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

    null_errors = _accurate_expectations(null_mass, phi)
    alternative_power = _accurate_expectations(alternative_mass, phi)
    alternative_errors = tuple(1.0 - value for value in alternative_power)
    worst_alpha = max(null_errors)
    worst_beta = max(alternative_errors)
    alpha_tolerance = 5e-10 * constraint.epsilon
    if worst_alpha > constraint.epsilon + alpha_tolerance:
        raise CompositeOptimizationError(
            "exact composite minimax LP violates the declared Type-I constraint: "
            f"worst alpha={worst_alpha:.17g}, epsilon={constraint.epsilon:.17g}"
        )
    if abs(worst_beta - beta_variable) > 5e-10:
        raise CompositeOptimizationError(
            "exact composite minimax LP objective disagrees with evaluated worst-case Type II"
        )
    active_null = tuple(
        index for index, value in enumerate(null_errors)
        if abs(value - worst_alpha) <= 1e-9
    )
    active_alternative = tuple(
        index for index, value in enumerate(alternative_errors)
        if abs(value - worst_beta) <= 1e-9
    )
    return FiniteCompositeMinimaxResult(
        problem=problem,
        constraint=constraint,
        outcomes=outcomes,
        rejection_probabilities=tuple(float(value) for value in phi),
        null_type_i_errors=null_errors,
        alternative_type_ii_errors=alternative_errors,
        worst_type_i_error=worst_alpha,
        minimax_type_ii_error=worst_beta,
        active_null_members=active_null,
        active_alternative_members=active_alternative,
    )


def _score_blocks(
    null: IndependentMIDProductLaw,
    alternative: IndependentMIDProductLaw,
) -> tuple[tuple[float, ...], ...]:
    result: list[tuple[float, ...]] = []
    for p_block, q_block in zip(null.blocks, alternative.blocks, strict=True):
        values: list[float] = []
        for p, q in zip(p_block.probabilities, q_block.probabilities, strict=True):
            if p > 0 and q > 0:
                values.append(math.log(q / p))
            elif p == 0 and q > 0:
                values.append(math.inf)
            elif p > 0 and q == 0:
                values.append(-math.inf)
            else:
                # The score is irrelevant on a category absent from both selected
                # laws only if every represented class member is also zero there;
                # that support condition is checked separately below.
                values.append(0.0)
        result.append(tuple(values))
    return tuple(result)


def _logsumexp(values: Iterable[float]) -> float:
    items = tuple(values)
    if not items:
        return -math.inf
    if any(value == math.inf for value in items):
        return math.inf
    finite = tuple(value for value in items if value != -math.inf)
    if not finite:
        return -math.inf
    pivot = max(finite)
    return pivot + math.log(math.fsum(math.exp(value - pivot) for value in finite))


def _block_log_moment(
    probabilities: tuple[float, ...],
    scores: tuple[float, ...],
    exponent: float,
) -> float:
    terms: list[float] = []
    for probability, score in zip(probabilities, scores, strict=True):
        if probability == 0:
            continue
        if score == math.inf:
            terms.append(math.inf if exponent > 0 else -math.inf)
        elif score == -math.inf:
            terms.append(-math.inf if exponent > 0 else math.inf)
        else:
            terms.append(math.log(probability) + exponent * score)
    return _logsumexp(terms)


def _member_log_moment(
    member: IndependentMIDProductLaw,
    score_blocks: tuple[tuple[float, ...], ...],
    exponent: float,
) -> float:
    values = []
    for block, scores in zip(member.blocks, score_blocks, strict=True):
        log_single = _block_log_moment(block.probabilities, scores, exponent)
        if log_single == math.inf:
            return math.inf
        if log_single == -math.inf:
            return -math.inf
        values.append(block.n * log_single)
    return math.fsum(values)


def _shared_zero_support_failures(
    problem: CompositeBinaryTestingProblem,
    null_index: int,
    alternative_index: int,
) -> tuple[str, ...]:
    p_star = problem.null.members[null_index]
    q_star = problem.alternative.members[alternative_index]
    failures: list[str] = []
    for block_index, (p_block, q_block) in enumerate(
        zip(p_star.blocks, q_star.blocks, strict=True)
    ):
        for class_index, (p, q) in enumerate(
            zip(p_block.probabilities, q_block.probabilities, strict=True)
        ):
            if p != 0 or q != 0:
                continue
            positive_elsewhere = any(
                member.blocks[block_index].probabilities[class_index] > 0
                for family in (problem.null, problem.alternative)
                for member in family.members
            )
            if positive_elsewhere:
                failures.append(
                    f"selected pair has p*=q*=0 at block {block_index}, mass class {class_index}, "
                    "but another represented member assigns positive mass there"
                )
    return tuple(failures)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeRenyiScoreCandidate:
    """Finite-family vertex-pair Rényi minimum and its uniform-gate diagnostics."""

    problem: CompositeBinaryTestingProblem
    order: float
    null_member_index: int
    alternative_member_index: int
    renyi: float
    score_blocks: tuple[tuple[float, ...], ...]
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


def _candidate_for_pair(
    problem: CompositeBinaryTestingProblem,
    order: float,
    divergence: float,
    null_index: int,
    alternative_index: int,
    tolerance: float,
) -> CompositeRenyiScoreCandidate:
    score_blocks = _score_blocks(
        problem.null.members[null_index], problem.alternative.members[alternative_index]
    )
    failures = list(_shared_zero_support_failures(problem, null_index, alternative_index))
    log_z = -math.inf if divergence == math.inf else (order - 1.0) * divergence
    null_moments = tuple(
        _member_log_moment(member, score_blocks, order)
        for member in problem.null.members
    )
    alternative_moments = tuple(
        _member_log_moment(member, score_blocks, order - 1.0)
        for member in problem.alternative.members
    )
    max_null = max(null_moments)
    max_alternative = max(alternative_moments)
    if max_null > log_z + tolerance:
        failures.append(
            "null-side uniform exponential-moment inequality fails: "
            f"max_log_moment={max_null:g}, log_z={log_z:g}"
        )
    if max_alternative > log_z + tolerance:
        failures.append(
            "alternative-side uniform exponential-moment inequality fails: "
            f"max_log_moment={max_alternative:g}, log_z={log_z:g}"
        )
    return CompositeRenyiScoreCandidate(
        problem=problem,
        order=order,
        null_member_index=null_index,
        alternative_member_index=alternative_index,
        renyi=divergence,
        score_blocks=score_blocks,
        log_hellinger_integral=log_z,
        maximum_log_null_moment=max_null,
        maximum_log_alternative_moment=max_alternative,
        uniform_moment_bounds_verified=not failures,
        verification_failures=tuple(failures),
    )


def composite_renyi_score_candidate(
    problem: CompositeBinaryTestingProblem,
    *,
    order: Real,
    tolerance: float = _SCORE_VERIFICATION_TOLERANCE,
) -> CompositeRenyiScoreCandidate:
    """Return a minimum-divergence finite-family pair plus honest gate status.

    A finite represented class is generally non-convex. The selected vertex
    pair is therefore only a candidate score. Among tied minimum-divergence
    pairs, a pair satisfying the two uniform moment inequalities is preferred;
    otherwise the first declared minimiser is returned with explicit failures.
    """

    if not isinstance(problem, CompositeBinaryTestingProblem):
        raise InputValidationError("problem must be CompositeBinaryTestingProblem")
    finite_order = _validate_score_order(order)
    tolerance_value = _finite_real(tolerance, "score verification tolerance")
    if tolerance_value <= 0:
        raise InputValidationError("score verification tolerance must be positive")
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
    if minimum == math.inf:
        minimisers = tuple(item for item in pairs if item[0] == math.inf)
    else:
        minimisers = tuple(
            item for item in pairs
            if abs(item[0] - minimum) <= tolerance_value * max(1.0, abs(minimum))
        )
    evaluated = tuple(
        _candidate_for_pair(
            problem, finite_order, divergence, null_index, alternative_index,
            tolerance_value,
        )
        for divergence, null_index, alternative_index in minimisers
    )
    return next(
        (candidate for candidate in evaluated if candidate.uniform_moment_bounds_verified),
        evaluated[0],
    )


def verified_composite_renyi_score(
    problem: CompositeBinaryTestingProblem,
    *,
    order: Real,
    tolerance: float = _SCORE_VERIFICATION_TOLERANCE,
) -> CompositeRenyiScoreCandidate:
    """Return the finite-family candidate score only when both gates verify."""

    candidate = composite_renyi_score_candidate(
        problem, order=order, tolerance=tolerance
    )
    if not candidate.uniform_moment_bounds_verified:
        raise CompositeScoreVerificationError(
            "finite-family Rényi-minimising vertex pair is only a candidate score; "
            "uniform composite moment/support verification failed: "
            + "; ".join(candidate.verification_failures)
        )
    return candidate


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeScoreBound:
    """Analytical threshold and Type-II guarantees for a verified score."""

    candidate: CompositeRenyiScoreCandidate
    constraint: SimpleBinaryTestingConstraint
    threshold: float
    raw_exponential_upper_bound: float
    constant_randomised_upper_bound: float
    minimax_type_ii_upper_bound: float


def composite_score_bound_at_order(
    candidate: CompositeRenyiScoreCandidate,
    *,
    epsilon: Real,
) -> CompositeScoreBound:
    """Return the analytical score construction without outcome enumeration."""

    if not isinstance(candidate, CompositeRenyiScoreCandidate):
        raise InputValidationError("candidate must be CompositeRenyiScoreCandidate")
    if not candidate.uniform_moment_bounds_verified:
        raise CompositeScoreVerificationError(
            "candidate score lacks verified uniform composite moment bounds"
        )
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    lam = candidate.order
    divergence = candidate.renyi
    if divergence == math.inf:
        threshold = -math.inf
        raw = 0.0
    else:
        threshold = (
            -math.log(constraint.epsilon) - (1.0 - lam) * divergence
        ) / lam
        log_raw = -(1.0 - lam) / lam * math.fsum(
            (divergence, math.log(constraint.epsilon))
        )
        raw = (
            math.exp(log_raw)
            if log_raw <= math.log(np.finfo(float).max)
            else math.inf
        )
    constant = 1.0 - constraint.epsilon
    minimax_upper = min(constant, raw)
    if not math.isfinite(minimax_upper) or not 0 <= minimax_upper <= 1:
        raise NumericalLimitError(
            "composite score Type-II minimax upper bound is numerically invalid"
        )
    return CompositeScoreBound(
        candidate=candidate,
        constraint=constraint,
        threshold=threshold,
        raw_exponential_upper_bound=raw,
        constant_randomised_upper_bound=constant,
        minimax_type_ii_upper_bound=minimax_upper,
    )


def _decimal_score_for_outcome(
    outcome: tuple[tuple[int, ...], ...],
    candidate: CompositeRenyiScoreCandidate,
) -> Decimal | None:
    positive_infinity = False
    negative_infinity = False
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        total = Decimal(0)
        for counts, p_block, q_block in zip(
            outcome,
            candidate.problem.null.members[candidate.null_member_index].blocks,
            candidate.problem.alternative.members[candidate.alternative_member_index].blocks,
            strict=True,
        ):
            for count, p, q in zip(
                counts, p_block.probabilities, q_block.probabilities, strict=True
            ):
                if not count:
                    continue
                if p == 0 and q > 0:
                    positive_infinity = True
                elif p > 0 and q == 0:
                    negative_infinity = True
                elif p == 0 and q == 0:
                    # Verification ensures every represented member has zero mass
                    # here. An outcome using this coordinate is therefore outside
                    # the represented joint support and its decision is irrelevant.
                    return None
                else:
                    total += Decimal(int(count)) * (
                        Decimal.from_float(q).ln() - Decimal.from_float(p).ln()
                    )
        if positive_infinity and negative_infinity:
            return None
        if positive_infinity:
            return Decimal("Infinity")
        if negative_infinity:
            return Decimal("-Infinity")
        return total


def _decision_for_threshold(
    outcomes: tuple[tuple[tuple[int, ...], ...], ...],
    candidate: CompositeRenyiScoreCandidate,
    threshold: float,
    null_mass: np.ndarray,
    alternative_mass: np.ndarray,
) -> np.ndarray:
    threshold_decimal = Decimal.from_float(threshold)
    decision = np.zeros(len(outcomes), dtype=float)
    total_mass = np.sum(null_mass, axis=0) + np.sum(alternative_mass, axis=0)
    for index, outcome in enumerate(outcomes):
        score = _decimal_score_for_outcome(outcome, candidate)
        if score is None:
            if float(total_mass[index]) > 0:
                raise CompositeScoreVerificationError(
                    "verified candidate score is undefined on positive represented joint support"
                )
            continue
        decision[index] = 1.0 if score >= threshold_decimal else 0.0
    return decision


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeScoreTestEvaluation:
    """Exactly enumerated errors of the analytical deterministic score test."""

    bound: CompositeScoreBound
    outcomes: tuple[tuple[tuple[int, ...], ...], ...]
    rejection_probabilities: tuple[float, ...]
    null_type_i_errors: tuple[float, ...]
    alternative_type_ii_errors: tuple[float, ...]
    worst_type_i_error: float
    worst_type_ii_error: float


def evaluate_composite_score_test(
    bound: CompositeScoreBound,
    *,
    max_outcomes: int = DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES,
) -> CompositeScoreTestEvaluation:
    """Enumerate the analytical score rule only when the joint space is small."""

    if not isinstance(bound, CompositeScoreBound):
        raise InputValidationError("bound must be CompositeScoreBound")
    problem = bound.candidate.problem
    outcomes = _enumerate_joint_outcomes(problem, max_outcomes)
    null_mass = _family_mass_matrix(problem.null, outcomes)
    alternative_mass = _family_mass_matrix(problem.alternative, outcomes)
    decision = _decision_for_threshold(
        outcomes, bound.candidate, bound.threshold, null_mass, alternative_mass
    )
    null_errors = _accurate_expectations(null_mass, decision)
    alternative_power = _accurate_expectations(alternative_mass, decision)
    alternative_errors = tuple(1.0 - value for value in alternative_power)
    worst_alpha = max(null_errors)
    worst_beta = max(alternative_errors)
    tolerance = max(1e-15, 5e-10 * bound.constraint.epsilon)
    if worst_alpha > bound.constraint.epsilon + tolerance:
        raise CompositeScoreVerificationError(
            "analytical verified score rule exceeds the declared Type-I budget "
            f"after exact represented-family evaluation: alpha={worst_alpha:.17g}"
        )
    if (
        math.isfinite(bound.raw_exponential_upper_bound)
        and worst_beta > bound.raw_exponential_upper_bound + 5e-10
    ):
        raise CompositeScoreVerificationError(
            "enumerated deterministic score error exceeds its analytical exponential bound"
        )
    return CompositeScoreTestEvaluation(
        bound=bound,
        outcomes=outcomes,
        rejection_probabilities=tuple(float(value) for value in decision),
        null_type_i_errors=null_errors,
        alternative_type_ii_errors=alternative_errors,
        worst_type_i_error=worst_alpha,
        worst_type_ii_error=worst_beta,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibratedCompositeScoreTest:
    """Exact represented-family calibration within one verified score family."""

    candidate: CompositeRenyiScoreCandidate
    constraint: SimpleBinaryTestingConstraint
    threshold_score: float
    boundary_randomisation: float
    outcomes: tuple[tuple[tuple[int, ...], ...], ...]
    rejection_probabilities: tuple[float, ...]
    null_type_i_errors: tuple[float, ...]
    alternative_type_ii_errors: tuple[float, ...]
    worst_type_i_error: float
    worst_type_ii_error: float

    @property
    def boundary_randomization(self) -> float:
        return self.boundary_randomisation

    @property
    def exhausts_type_i_budget(self) -> bool:
        return abs(self.worst_type_i_error - self.constraint.epsilon) <= max(
            1e-15, 5e-10 * self.constraint.epsilon
        )


def calibrate_composite_score_test(
    candidate: CompositeRenyiScoreCandidate,
    *,
    epsilon: Real,
    max_outcomes: int = DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES,
) -> CalibratedCompositeScoreTest:
    """Exhaust Type-I budget within the fixed verified upper-score family."""

    if not isinstance(candidate, CompositeRenyiScoreCandidate):
        raise InputValidationError("candidate must be CompositeRenyiScoreCandidate")
    if not candidate.uniform_moment_bounds_verified:
        raise CompositeScoreVerificationError(
            "candidate score lacks verified uniform composite moment bounds"
        )
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    problem = candidate.problem
    outcomes = _enumerate_joint_outcomes(problem, max_outcomes)
    null_mass = _family_mass_matrix(problem.null, outcomes)
    alternative_mass = _family_mass_matrix(problem.alternative, outcomes)
    total_mass = np.sum(null_mass, axis=0) + np.sum(alternative_mass, axis=0)
    scores: list[Decimal] = []
    for index, outcome in enumerate(outcomes):
        score = _decimal_score_for_outcome(outcome, candidate)
        if score is None:
            if float(total_mass[index]) > 0:
                raise CompositeScoreVerificationError(
                    "verified candidate score is undefined on positive represented joint support"
                )
            score = Decimal("-Infinity")
        scores.append(score)
    groups: dict[Decimal, list[int]] = {}
    for index, score in enumerate(scores):
        groups.setdefault(score, []).append(index)
    ordered_scores = tuple(sorted(groups, reverse=True))

    rejection = np.zeros(len(outcomes), dtype=float)
    above_null = np.zeros(len(problem.null.members), dtype=float)
    chosen_score: Decimal | None = None
    chosen_eta: float | None = None
    for score in ordered_scores:
        indices = groups[score]
        boundary_null = np.sum(null_mass[:, indices], axis=1)
        fully_included = above_null + boundary_null
        if float(np.max(fully_included)) <= constraint.epsilon:
            rejection[indices] = 1.0
            above_null = fully_included
            continue
        limits = tuple(
            (constraint.epsilon - float(current)) / float(boundary)
            for current, boundary in zip(above_null, boundary_null, strict=True)
            if boundary > 0
        )
        if not limits:
            raise CompositeScoreVerificationError(
                "calibration encountered a positive-score boundary with no null mass"
            )
        eta = min(limits)
        tolerance = max(1e-15, 5e-10 * constraint.epsilon)
        if eta < -tolerance or eta > 1.0 + tolerance or not math.isfinite(eta):
            raise CompositeScoreVerificationError(
                "calibration boundary randomisation left [0, 1]"
            )
        eta = min(1.0, max(0.0, eta))
        rejection[indices] = eta
        above_null = above_null + eta * boundary_null
        chosen_score = score
        chosen_eta = eta
        break
    if chosen_score is None or chosen_eta is None:
        raise CompositeScoreVerificationError(
            "calibration failed to encounter the composite Type-I boundary"
        )

    null_errors = _accurate_expectations(null_mass, rejection)
    alternative_power = _accurate_expectations(alternative_mass, rejection)
    alternative_errors = tuple(1.0 - value for value in alternative_power)
    worst_alpha = max(null_errors)
    worst_beta = max(alternative_errors)
    tolerance = max(1e-15, 5e-10 * constraint.epsilon)
    if worst_alpha > constraint.epsilon + tolerance:
        raise CompositeScoreVerificationError(
            "calibrated score test violates the composite Type-I constraint"
        )
    if abs(worst_alpha - constraint.epsilon) > tolerance:
        raise CompositeScoreVerificationError(
            "calibrated score test did not exhaust the composite Type-I budget"
        )
    return CalibratedCompositeScoreTest(
        candidate=candidate,
        constraint=constraint,
        threshold_score=float(chosen_score),
        boundary_randomisation=chosen_eta,
        outcomes=outcomes,
        rejection_probabilities=tuple(float(value) for value in rejection),
        null_type_i_errors=null_errors,
        alternative_type_ii_errors=alternative_errors,
        worst_type_i_error=worst_alpha,
        worst_type_ii_error=worst_beta,
    )


__all__ = [
    "DEFAULT_EXACT_COMPOSITE_MAX_OUTCOMES",
    "MIN_EXACT_COMPOSITE_EPSILON",
    "CalibratedCompositeScoreTest",
    "CompositeBinaryTestingProblem",
    "CompositeEnumerationLimitError",
    "CompositeMIDLawFamily",
    "CompositeOptimizationError",
    "CompositeRenyiConverseBound",
    "CompositeRenyiScoreCandidate",
    "CompositeScoreBound",
    "CompositeScoreTestEvaluation",
    "CompositeScoreVerificationError",
    "FiniteCompositeMinimaxResult",
    "IndependentMIDProductLaw",
    "calibrate_composite_score_test",
    "composite_renyi_converse_at_order",
    "composite_renyi_score_candidate",
    "composite_score_bound_at_order",
    "evaluate_composite_score_test",
    "exact_finite_composite_minimax",
    "verified_composite_renyi_score",
]
