"""Native HiGHS FBA/FVA over canonical ``FluxModel``.

The reference FVA intentionally cold-starts each endpoint.  Production FVA uses
worker-local reusable HiGHS models and dynamically scheduled endpoint jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from queue import Empty, Queue
import threading
from typing import Sequence

import numpy as np
import pandas as pd

from fluxemu.exceptions import AnalysisError
from fluxemu.model.schema import FluxModel
from fluxemu.model.validation import CanonicalModelError, validate_flux_model
from .results import FBAResult, FVAResult, PrimalDiagnostics, _fva_ranges_sha256

FEASIBILITY_TOLERANCE = 1e-7
OBJECTIVE_TOLERANCE = 1e-7
SOLVER_FEASIBILITY_TOLERANCE = 1e-9
HIGHS_SMALL_MATRIX_VALUE = 1e-12
FVA_NUMERICAL_COLLAPSE_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class CompiledFluxLP:
    """Immutable row-compressed canonical LP; ordering is scientifically significant."""

    reaction_ids: tuple[str, ...]
    balanced_metabolite_ids: tuple[str, ...]
    row_starts: tuple[int, ...]
    column_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    solver_row_starts: tuple[int, ...]
    solver_column_indices: tuple[int, ...]
    solver_coefficients: tuple[float, ...]
    balance_row_space_basis: tuple[tuple[float, ...], ...]
    balance_rank: int
    affine_equality_basis: tuple[tuple[float, ...], ...]
    affine_equality_rhs: tuple[float, ...]
    affine_rank: int
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    objective_coefficients: tuple[float, ...]
    effective_objective_coefficients: tuple[float, ...]
    objective_constant: float
    objective_direction: str
    fingerprint: str

    @property
    def nonzero_count(self) -> int:
        return len(self.coefficients)


RANK_BASE_FACTOR = 8.0
AFFINE_CONDITIONING_FLOOR = math.sqrt(np.finfo(float).eps)


def _condition_affine_equalities(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return a stable equivalent ``Q v = q`` and its orthogonal null basis.

    Rows are normalized before factorization, so a uniformly rescaled
    conservation law is scientifically identical.  A nonzero singular
    direction below ``sqrt(machine epsilon)`` is rejected rather than retained:
    its row space cannot be recovered accurately enough for the public 1e-7
    feasibility contract.
    """

    if matrix.ndim != 2 or rhs.ndim != 1 or matrix.shape[0] != rhs.shape[0]:
        raise AnalysisError(f"{description} equality system is malformed")
    dimension = matrix.shape[1]
    normalized_rows: list[np.ndarray] = []
    normalized_rhs: list[float] = []
    for row, bound in zip(matrix, rhs):
        norm = math.hypot(*(float(value) for value in row))
        if not math.isfinite(norm) or not math.isfinite(float(bound)):
            raise AnalysisError(f"{description} equality system is non-finite")
        if norm == 0.0:
            if abs(float(bound)) > FEASIBILITY_TOLERANCE:
                raise AnalysisError(
                    f"{description} affine equalities are inconsistent"
                )
            continue
        normalized_rows.append(row / norm)
        normalized_rhs.append(float(bound) / norm)
    if not normalized_rows:
        return (
            np.empty((0, dimension), dtype=float),
            np.empty(0, dtype=float),
            np.eye(dimension, dtype=float),
            0,
        )

    scaled = np.vstack(normalized_rows)
    scaled_rhs = np.asarray(normalized_rhs, dtype=float)
    try:
        left, singular_values, right = np.linalg.svd(scaled, full_matrices=True)
    except np.linalg.LinAlgError as error:
        raise AnalysisError(f"could not factor {description} equality system") from error
    largest = float(singular_values[0])
    # Only singular values at the immediate machine-noise floor may represent
    # exact redundant rows.  Matrix-size-scaled rank heuristics can silently
    # erase small but real conservation laws (for example a 1e-14 secondary
    # direction in a four-column system).
    base_relative = RANK_BASE_FACTOR * np.finfo(float).eps
    relative = singular_values / largest
    ambiguous = relative[
        (relative > base_relative) & (relative <= AFFINE_CONDITIONING_FLOOR)
    ]
    if ambiguous.size:
        raise AnalysisError(
            f"{description} equality system is numerically ill-conditioned or has "
            "ambiguous rank: relative_singular_value="
            f"{float(ambiguous[0]):g}, required>{AFFINE_CONDITIONING_FLOOR:g}"
        )
    rank = int(np.count_nonzero(relative > AFFINE_CONDITIONING_FLOOR))
    exact_rank: int | None = None
    if rank < min(matrix.shape) or matrix.shape[0] > rank:
        exact_rank = _exact_matrix_rank(matrix)
        if exact_rank > rank:
            raise AnalysisError(
                f"{description} equality system has an exact independent "
                "direction below numerical resolution: "
                f"stable_rank={rank}, exact_rank={exact_rank}"
            )
    if matrix.shape[0] > rank:
        if exact_rank is None:  # pragma: no cover - guarded by the branch above
            exact_rank = _exact_matrix_rank(matrix)
        augmented_rank = _exact_matrix_rank(
            np.column_stack((matrix, rhs))
        )
        if augmented_rank > exact_rank:
            raise AnalysisError(
                f"{description} affine equalities are exactly inconsistent"
            )
    row_basis = right[:rank, :].copy()
    null_basis = right[rank:, :].copy()
    transformed_rhs = (
        (left[:, :rank].T @ scaled_rhs) / singular_values[:rank]
        if rank
        else np.empty(0, dtype=float)
    )
    reconstructed_rhs = (
        left[:, :rank] @ (singular_values[:rank] * transformed_rhs)
        if rank
        else np.zeros_like(scaled_rhs)
    )
    consistency_error = float(np.linalg.norm(scaled_rhs - reconstructed_rhs))
    consistency_scale = max(1.0, float(np.linalg.norm(scaled_rhs)))
    if consistency_error > FEASIBILITY_TOLERANCE * consistency_scale:
        raise AnalysisError(
            f"{description} affine equalities are inconsistent: normalized "
            f"residual={consistency_error:g}"
        )
    return row_basis, transformed_rhs, null_basis, rank


def _exact_matrix_rank(matrix: np.ndarray) -> int:
    """Compute binary-float-exact rank for discarded-singular-value checks."""

    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    return _exact_matrix_rank_cached(contiguous.shape, contiguous.tobytes())


@lru_cache(maxsize=64)
def _exact_matrix_rank_cached(shape: tuple[int, int], data: bytes) -> int:
    matrix = np.frombuffer(data, dtype=np.float64).reshape(shape)
    rows = [
        [Fraction.from_float(float(value)) for value in row]
        for row in matrix
        if np.any(row)
    ]
    if not rows:
        return 0
    rank = 0
    column_count = matrix.shape[1]
    for column in range(column_count):
        pivot = next(
            (row for row in range(rank, len(rows)) if rows[row][column]),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [value / divisor for value in rows[rank]]
        for row in range(rank + 1, len(rows)):
            multiplier = rows[row][column]
            if multiplier:
                rows[row] = [
                    value - multiplier * pivot_value
                    for value, pivot_value in zip(rows[row], rows[rank])
                ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _orthonormal_row_space(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a conditioned basis, failing closed on numerical rank ambiguity."""

    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    rows, rank = _orthonormal_row_space_cached(
        contiguous.shape, contiguous.tobytes()
    )
    return np.asarray(rows, dtype=float).reshape(rank, contiguous.shape[1]), rank


@lru_cache(maxsize=32)
def _orthonormal_row_space_cached(
    shape: tuple[int, int], data: bytes
) -> tuple[tuple[tuple[float, ...], ...], int]:
    matrix = np.frombuffer(data, dtype=np.float64).reshape(shape)
    basis, _, _, rank = _condition_affine_equalities(
        matrix,
        np.zeros(matrix.shape[0], dtype=float),
        description="balanced-metabolite",
    )
    return tuple(tuple(float(value) for value in row) for row in basis), rank


@dataclass(frozen=True, slots=True)
class RetainedObjectiveConstraint:
    """The biological-objective constraint shared by FVA and sampling."""

    fraction_of_optimum: float
    biological_optimum: float
    bound: float
    sense: str
    objective_direction: str
    effective_optimum: float
    effective_bound: float


@dataclass(frozen=True, slots=True)
class PreparedFluxRegion:
    """One compiled LP and its single independently validated FBA optimum."""

    lp: CompiledFluxLP
    fba: FBAResult
    retention: RetainedObjectiveConstraint


def _condition_objective(
    objective: np.ndarray,
    equality_basis: np.ndarray,
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
    canonical_balance_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Construct an exact binary-rational objective quotient.

    Exact elimination yields ``c = c_eff + alpha A`` for the supplied binary
    floats.  Results are cached by every canonical numeric input: integrity
    validation and repeated warm compilation therefore reuse the proof rather
    than repeating an expensive rational RREF.
    """

    objective_array = np.ascontiguousarray(objective, dtype=np.float64)
    basis_array = np.ascontiguousarray(equality_basis, dtype=np.float64)
    balance_array = np.ascontiguousarray(
        canonical_balance_matrix, dtype=np.float64
    )
    fixed_array = np.ascontiguousarray(fixed_values, dtype=np.float64)
    lower_array = np.ascontiguousarray(lower_bounds, dtype=np.float64)
    upper_array = np.ascontiguousarray(upper_bounds, dtype=np.float64)
    effective, constant = _condition_objective_cached(
        objective_array.shape,
        objective_array.tobytes(),
        basis_array.shape,
        basis_array.tobytes(),
        tuple(int(index) for index in fixed_indices),
        fixed_array.tobytes(),
        balance_array.shape,
        balance_array.tobytes(),
        lower_array.tobytes(),
        upper_array.tobytes(),
    )
    return np.asarray(effective, dtype=float), constant


def _objective_pivot_columns(
    equality_basis: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    fixed: set[int],
) -> tuple[int, ...]:
    """Choose stable exact-elimination pivots in flux-coordinate scale.

    A column-pivoted orthogonal selection proves numerical independence, while
    the coordinate magnitude only breaks choices between valid directions.
    This avoids eliminating a tightly bounded scientific objective coordinate
    in favour of much larger auxiliary fluxes.
    """

    rank, dimension = equality_basis.shape
    if rank == 0:
        return ()
    try:
        _, _, right = np.linalg.svd(equality_basis, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise AnalysisError(
            "could not select stable biological-objective quotient pivots"
        ) from error
    row_basis = right[:rank, :]
    selected: list[int] = []
    orthonormal: list[np.ndarray] = []
    for _ in range(rank):
        best_column: int | None = None
        best_residual: np.ndarray | None = None
        best_norm = 0.0
        best_score = -math.inf
        existing = (
            np.vstack(orthonormal)
            if orthonormal
            else np.empty((0, rank), dtype=float)
        )
        for column in range(dimension):
            if column in fixed or column in selected:
                continue
            residual = row_basis[:, column].copy()
            if len(existing):
                for _ in range(2):
                    residual -= existing.T @ (existing @ residual)
            residual_norm = math.hypot(*(float(value) for value in residual))
            if residual_norm <= RANK_BASE_FACTOR * np.finfo(float).eps:
                continue
            coordinate_scale = max(
                abs(float(lower_bounds[column])),
                abs(float(upper_bounds[column])),
                float(upper_bounds[column] - lower_bounds[column]),
            )
            if coordinate_scale <= 0.0 or not math.isfinite(coordinate_scale):
                raise AnalysisError(
                    "biological-objective quotient has an invalid free-coordinate "
                    "scale"
                )
            score = math.log(residual_norm) + math.log(coordinate_scale)
            if score > best_score:
                best_column = column
                best_residual = residual
                best_norm = residual_norm
                best_score = score
        if best_column is None or best_residual is None:
            raise AnalysisError(
                "could not select a full-rank biological-objective quotient"
            )
        selected.append(best_column)
        orthonormal.append(best_residual / best_norm)
    return tuple(selected)


@lru_cache(maxsize=32)
def _condition_objective_cached(
    objective_shape: tuple[int, ...],
    objective_data: bytes,
    basis_shape: tuple[int, int],
    basis_data: bytes,
    fixed_indices: tuple[int, ...],
    fixed_values_data: bytes,
    balance_shape: tuple[int, int],
    balance_data: bytes,
    lower_bounds_data: bytes,
    upper_bounds_data: bytes,
) -> tuple[tuple[float, ...], float]:
    """Return a binary-rational-exact affine quotient for immutable inputs."""

    objective = np.frombuffer(objective_data, dtype=np.float64).reshape(
        objective_shape
    )
    equality_basis = np.frombuffer(basis_data, dtype=np.float64).reshape(
        basis_shape
    )
    balance_matrix = np.frombuffer(balance_data, dtype=np.float64).reshape(
        balance_shape
    )
    fixed_values = np.frombuffer(fixed_values_data, dtype=np.float64)
    lower_bounds = np.frombuffer(lower_bounds_data, dtype=np.float64)
    upper_bounds = np.frombuffer(upper_bounds_data, dtype=np.float64)
    fixed = set(fixed_indices)
    fixed_value_by_index = dict(zip(fixed_indices, fixed_values))
    working = [Fraction.from_float(float(value)) for value in objective]
    objective_constant = sum(
        (
            working[index] * Fraction.from_float(float(fixed_value_by_index[index]))
            for index in fixed_indices
        ),
        Fraction(0),
    )
    for index in fixed:
        working[index] = Fraction(0)

    rows: list[list[Fraction]] = []
    for canonical_row in balance_matrix:
        row = [
            Fraction(0)
            if column in fixed
            else Fraction.from_float(float(value))
            for column, value in enumerate(canonical_row)
        ]
        if not any(row):
            continue
        bound = -sum(
            (
                Fraction.from_float(float(canonical_row[index]))
                * Fraction.from_float(float(fixed_value_by_index[index]))
                for index in fixed_indices
            ),
            Fraction(0),
        )
        rows.append(row + [bound])

    pivot_columns = _objective_pivot_columns(
        equality_basis, lower_bounds, upper_bounds, fixed
    )
    for pivot_row, column in enumerate(pivot_columns):
        pivot: int | None = None
        pivot_size = -1.0
        for candidate_row in range(pivot_row, len(rows)):
            value = rows[candidate_row][column]
            if value:
                try:
                    size = abs(float(value))
                except OverflowError:
                    size = math.inf
                if size > pivot_size:
                    pivot = candidate_row
                    pivot_size = size
        if pivot is None:
            raise AnalysisError(
                "exact and conditioned affine ranks disagree during objective "
                "quotient construction"
            )
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        divisor = rows[pivot_row][column]
        rows[pivot_row] = [value / divisor for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row:
                continue
            multiplier = rows[row][column]
            if multiplier:
                rows[row] = [
                    value - multiplier * pivot_value
                    for value, pivot_value in zip(rows[row], rows[pivot_row])
                ]
    for row, pivot_column in zip(rows, pivot_columns):
        coefficient = working[pivot_column]
        if coefficient:
            objective_constant += coefficient * row[-1]
            for column in range(objective.size):
                if column != pivot_column:
                    working[column] -= coefficient * row[column]
            working[pivot_column] = Fraction(0)
    try:
        effective = np.asarray([float(value) for value in working], dtype=float)
        constant = float(objective_constant)
    except (OverflowError, ValueError) as error:
        raise AnalysisError(
            "biological objective quotient is outside floating-point range"
        ) from error
    if not np.isfinite(effective).all() or not math.isfinite(constant):
        raise AnalysisError("biological objective conditioning produced non-finite data")
    return tuple(float(value) for value in effective), constant


def _compiled_lp_fingerprint(
    reaction_ids: Sequence[str],
    balanced_metabolite_ids: Sequence[str],
    row_starts: Sequence[int],
    column_indices: Sequence[int],
    coefficients: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    objective_coefficients: Sequence[float],
    objective_direction: str,
) -> str:
    """Hash the canonical, unconditioned LP identity."""

    identity = json.dumps(
        [
            list(reaction_ids),
            list(balanced_metabolite_ids),
            list(row_starts),
            list(column_indices),
            [float(value) for value in coefficients],
            [float(value) for value in lower_bounds],
            [float(value) for value in upper_bounds],
            [float(value) for value in objective_coefficients],
            objective_direction,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _condition_balance_after_fixed_substitution(
    balance_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """Condition mass balance after substituting explicit fixed coordinates."""

    matrix = np.ascontiguousarray(balance_matrix, dtype=np.float64)
    lower = np.ascontiguousarray(lower_bounds, dtype=np.float64)
    upper = np.ascontiguousarray(upper_bounds, dtype=np.float64)
    (
        affine_rows,
        conditioned_bounds,
        affine_rank,
        canonical_rows,
        canonical_bounds,
        fixed,
    ) = _condition_balance_after_fixed_substitution_cached(
        matrix.shape,
        matrix.tobytes(),
        lower.tobytes(),
        upper.tobytes(),
    )
    return (
        np.asarray(affine_rows, dtype=float).reshape(affine_rank, matrix.shape[1]),
        np.asarray(conditioned_bounds, dtype=float),
        affine_rank,
        np.asarray(canonical_rows, dtype=float).reshape(matrix.shape),
        np.asarray(canonical_bounds, dtype=float),
        np.asarray(fixed, dtype=int),
    )


@lru_cache(maxsize=32)
def _condition_balance_after_fixed_substitution_cached(
    shape: tuple[int, int],
    matrix_data: bytes,
    lower_data: bytes,
    upper_data: bytes,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    int,
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[int, ...],
]:
    balance_matrix = np.frombuffer(matrix_data, dtype=np.float64).reshape(shape)
    lower_bounds = np.frombuffer(lower_data, dtype=np.float64)
    upper_bounds = np.frombuffer(upper_data, dtype=np.float64)
    n = balance_matrix.shape[1]
    fixed_indices = np.flatnonzero(lower_bounds == upper_bounds)
    free_indices = np.flatnonzero(lower_bounds != upper_bounds)
    reduced_matrix = balance_matrix[:, free_indices]
    reduced_rhs = np.asarray(
        [
            -math.fsum(
                float(balance_matrix[row, index]) * float(lower_bounds[index])
                for index in fixed_indices
            )
            for row in range(balance_matrix.shape[0])
        ],
        dtype=float,
    )
    _, _, _, affine_rank = (
        _condition_affine_equalities(
            reduced_matrix,
            reduced_rhs,
            description="mass-balance after fixed-bound substitution",
        )
    )
    reduced_basis, conditioned_rhs = _select_independent_normalized_equalities(
        reduced_matrix, reduced_rhs, affine_rank
    )
    affine_basis = np.zeros((affine_rank, n), dtype=float)
    if affine_rank:
        affine_basis[:, free_indices] = reduced_basis
    canonical_affine_matrix = np.zeros_like(balance_matrix)
    canonical_affine_matrix[:, free_indices] = reduced_matrix
    return (
        tuple(tuple(float(value) for value in row) for row in affine_basis),
        tuple(float(value) for value in conditioned_rhs),
        affine_rank,
        tuple(
            tuple(float(value) for value in row)
            for row in canonical_affine_matrix
        ),
        tuple(float(value) for value in reduced_rhs),
        tuple(int(value) for value in fixed_indices),
    )


def _select_independent_normalized_equalities(
    matrix: np.ndarray, rhs: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select a stable sparse original-row basis with deterministic pivoting."""

    dimension = matrix.shape[1]
    candidates: list[tuple[np.ndarray, float]] = []
    for row, bound in zip(matrix, rhs):
        norm = math.hypot(*(float(value) for value in row))
        if norm > 0.0:
            candidates.append((row / norm, float(bound) / norm))
    selected_rows: list[np.ndarray] = []
    selected_rhs: list[float] = []
    orthonormal_rows: list[np.ndarray] = []
    remaining = list(range(len(candidates)))
    while len(selected_rows) < rank:
        best_position: int | None = None
        best_residual: np.ndarray | None = None
        best_norm = -1.0
        existing = (
            np.vstack(orthonormal_rows)
            if orthonormal_rows
            else np.empty((0, dimension), dtype=float)
        )
        for position, candidate_index in enumerate(remaining):
            residual = candidates[candidate_index][0].copy()
            if len(existing):
                for _ in range(2):
                    residual -= existing.T @ (existing @ residual)
            residual_norm = math.hypot(*(float(value) for value in residual))
            if residual_norm > best_norm:
                best_position = position
                best_residual = residual
                best_norm = residual_norm
        if (
            best_position is None
            or best_residual is None
            or best_norm <= AFFINE_CONDITIONING_FLOOR
        ):
            raise AnalysisError(
                "could not select a stable sparse mass-balance row basis"
            )
        candidate_index = remaining.pop(best_position)
        row, bound = candidates[candidate_index]
        selected_rows.append(row)
        selected_rhs.append(bound)
        orthonormal_rows.append(best_residual / best_norm)
    return (
        np.vstack(selected_rows).reshape(rank, dimension)
        if rank
        else np.empty((0, dimension), dtype=float),
        np.asarray(selected_rhs, dtype=float),
    )


def _validate_objective_solver_resolution(
    objective: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> None:
    """Reject material objective activity below HiGHS coefficient resolution."""

    scale = max((abs(float(value)) for value in objective), default=0.0)
    if scale == 0.0:
        return
    contributions = tuple(
        abs(float(coefficient)) * float(upper - lower)
        for coefficient, lower, upper in zip(
            objective, lower_bounds, upper_bounds
        )
    )
    activity = math.fsum(contributions)
    unresolved_contributions = tuple(
        contribution
        for coefficient, contribution in zip(objective, contributions)
        if 0.0
        < abs(float(coefficient)) / scale
        <= max(HIGHS_SMALL_MATRIX_VALUE, SOLVER_FEASIBILITY_TOLERANCE)
    )
    unresolved_activity = math.fsum(unresolved_contributions)
    if unresolved_activity > OBJECTIVE_TOLERANCE * activity:
        raise AnalysisError(
            "biological objective has material aggregate bounded activity below "
            "HiGHS coefficient resolution: "
            f"unresolved_activity={unresolved_activity:g}, "
            f"total_activity={activity:g}"
        )


def _validate_optimal_face_solver_resolution(lp: CompiledFluxLP) -> None:
    """Reject objectives whose exact optimal face is below solver resolution."""

    objective = np.asarray(lp.effective_objective_coefficients, dtype=float)
    scale = max((abs(float(value)) for value in objective), default=0.0)
    if scale == 0.0:
        return
    threshold = max(HIGHS_SMALL_MATRIX_VALUE, SOLVER_FEASIBILITY_TOLERANCE)
    ambiguous = tuple(
        (column, abs(float(coefficient)) / scale)
        for column, (coefficient, lower, upper) in enumerate(
            zip(objective, lp.lower_bounds, lp.upper_bounds)
        )
        if lower != upper
        and 0.0 < abs(float(coefficient)) / scale <= threshold
    )
    if ambiguous:
        column, normalized = min(ambiguous, key=lambda item: item[1])
        raise AnalysisError(
            "exact retained optimal face is ambiguous at HiGHS objective "
            "resolution: "
            f"reaction={lp.reaction_ids[column]!r}, "
            f"normalized_coefficient={normalized:g}"
        )


def _validate_mass_balance_solver_resolution(
    affine_equalities: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> None:
    """Reject equality terms that HiGHS would discard but can be material.

    The rows supplied here are the selected, normalized rows written to the
    solver. A bound magnitude is deliberately used instead of only the bound
    span: a tiny coefficient on a large-offset coordinate can materially alter
    an affine equality even when that coordinate has a narrow range.
    """

    for row_index, row in enumerate(affine_equalities):
        unresolved: list[tuple[int, float, float]] = []
        for column, coefficient in enumerate(row):
            magnitude = abs(float(coefficient))
            if not 0.0 < magnitude <= HIGHS_SMALL_MATRIX_VALUE:
                continue
            coordinate_magnitude = max(
                abs(float(lower_bounds[column])),
                abs(float(upper_bounds[column])),
            )
            contribution = magnitude * coordinate_magnitude
            unresolved.append((column, magnitude, contribution))
        try:
            unresolved_activity = math.fsum(item[2] for item in unresolved)
        except (OverflowError, ValueError):
            unresolved_activity = math.inf
        if not math.isfinite(unresolved_activity) or (
            unresolved_activity > FEASIBILITY_TOLERANCE
        ):
            largest = max(unresolved, key=lambda item: item[2])
            raise AnalysisError(
                "mass-balance equality has material aggregate bounded "
                "coefficients below HiGHS matrix resolution: "
                f"row={row_index}, largest_column={largest[0]}, "
                f"largest_normalized_coefficient={largest[1]:g}, "
                f"unresolved_maximum_contribution={unresolved_activity:g}"
            )


def compile_flux_lp(model: FluxModel) -> CompiledFluxLP:
    """Validate and compile a canonical model into deterministic sparse CSR data."""
    try:
        validate_flux_model(model)
    except (CanonicalModelError, AttributeError) as error:
        raise AnalysisError(f"invalid canonical flux model: {error}") from error
    if not model.objective.terms:
        raise AnalysisError("invalid canonical flux model: objective must be nonempty")
    reaction_ids = tuple(r.reaction_id for r in model.reactions)
    if tuple(dict.fromkeys(reaction_ids)) != reaction_ids:
        raise AnalysisError("invalid canonical flux model: nondeterministic reaction ordering")
    try:
        hash(model)
    except TypeError:
        return _compile_validated_flux_lp(model)
    return _compile_validated_flux_lp_cached(model)


@lru_cache(maxsize=16)
def _compile_validated_flux_lp_cached(model: FluxModel) -> CompiledFluxLP:
    """Cache only fully validated immutable canonical model values."""

    return _compile_validated_flux_lp(model)


def _compile_validated_flux_lp(model: FluxModel) -> CompiledFluxLP:
    reaction_ids = tuple(reaction.reaction_id for reaction in model.reactions)
    reaction_index = {rid: i for i, rid in enumerate(reaction_ids)}
    balanced = tuple(m.metabolite_id for m in model.metabolites if m.steady_state_balanced)
    terms_by_metabolite: dict[str, list[tuple[int, float]]] = {mid: [] for mid in balanced}
    for column, reaction in enumerate(model.reactions):
        accumulated: dict[str, list[float]] = {}
        for term in reaction.stoichiometric_terms:
            accumulated.setdefault(term.metabolite_id, []).append(
                float(term.coefficient)
            )
        for mid, coefficients in accumulated.items():
            try:
                value = math.fsum(coefficients)
            except (OverflowError, ValueError) as error:
                raise AnalysisError(
                    "aggregated stoichiometric coefficient is non-finite for "
                    f"reaction {reaction.reaction_id!r}, metabolite {mid!r}"
                ) from error
            if not math.isfinite(value):
                raise AnalysisError(
                    "aggregated stoichiometric coefficient is non-finite for "
                    f"reaction {reaction.reaction_id!r}, metabolite {mid!r}"
                )
            if mid in terms_by_metabolite and value != 0.0:
                terms_by_metabolite[mid].append((column, value))
    starts = [0]; indices: list[int] = []; values: list[float] = []
    for mid in balanced:
        entries = terms_by_metabolite[mid]
        if any(i < 0 or i >= len(reaction_ids) for i, _ in entries):
            raise AnalysisError("invalid canonical flux model: malformed sparse index")
        indices.extend(i for i, _ in entries); values.extend(v for _, v in entries); starts.append(len(indices))
    balance_matrix = np.zeros((len(balanced), len(reaction_ids)), dtype=float)
    for row in range(len(balanced)):
        for offset in range(starts[row], starts[row + 1]):
            balance_matrix[row, indices[offset]] = values[offset]
    row_space_basis, balance_rank = _orthonormal_row_space(balance_matrix)
    lower_bounds = np.asarray(
        [float(reaction.lower_bound) for reaction in model.reactions], dtype=float
    )
    upper_bounds = np.asarray(
        [float(reaction.upper_bound) for reaction in model.reactions], dtype=float
    )
    (
        affine_basis,
        conditioned_rhs,
        affine_rank,
        affine_matrix,
        affine_rhs,
        fixed_indices,
    ) = _condition_balance_after_fixed_substitution(
        balance_matrix, lower_bounds, upper_bounds
    )
    _validate_mass_balance_solver_resolution(
        affine_basis, lower_bounds, upper_bounds
    )
    solver_starts = [0]
    solver_indices: list[int] = []
    solver_values: list[float] = []
    for row in affine_basis:
        for column, value in enumerate(row):
            if value != 0.0:
                solver_indices.append(column)
                solver_values.append(float(value))
        solver_starts.append(len(solver_indices))
    objective_terms: list[list[float]] = [[] for _ in reaction_ids]
    for term in model.objective.terms:
        objective_terms[reaction_index[term.reaction_id]].append(
            float(term.coefficient)
        )
    objective: list[float] = []
    for reaction_id, coefficients in zip(reaction_ids, objective_terms):
        try:
            value = math.fsum(coefficients)
        except (OverflowError, ValueError) as error:
            raise AnalysisError(
                "aggregated objective coefficient is non-finite for reaction "
                f"{reaction_id!r}"
            ) from error
        if not math.isfinite(value):
            raise AnalysisError(
                "aggregated objective coefficient is non-finite for reaction "
                f"{reaction_id!r}"
            )
        objective.append(value)
    objective_array = np.asarray(objective, dtype=float)
    effective_objective, objective_constant = _condition_objective(
        objective_array,
        affine_basis,
        fixed_indices,
        lower_bounds[fixed_indices],
        balance_matrix,
        lower_bounds,
        upper_bounds,
    )
    _validate_objective_solver_resolution(
        effective_objective, lower_bounds, upper_bounds
    )
    direction = {"maximise": "max", "minimise": "min"}[model.objective.direction]
    fingerprint = _compiled_lp_fingerprint(
        reaction_ids,
        balanced,
        starts,
        indices,
        values,
        lower_bounds,
        upper_bounds,
        objective,
        direction,
    )
    return CompiledFluxLP(
        reaction_ids=reaction_ids,
        balanced_metabolite_ids=balanced,
        row_starts=tuple(starts),
        column_indices=tuple(indices),
        coefficients=tuple(values),
        solver_row_starts=tuple(solver_starts),
        solver_column_indices=tuple(solver_indices),
        solver_coefficients=tuple(solver_values),
        balance_row_space_basis=tuple(
            tuple(float(value) for value in row) for row in row_space_basis
        ),
        balance_rank=balance_rank,
        affine_equality_basis=tuple(
            tuple(float(value) for value in row) for row in affine_basis
        ),
        affine_equality_rhs=tuple(float(value) for value in conditioned_rhs),
        affine_rank=affine_rank,
        lower_bounds=tuple(float(value) for value in lower_bounds),
        upper_bounds=tuple(float(value) for value in upper_bounds),
        objective_coefficients=tuple(objective),
        effective_objective_coefficients=tuple(
            float(value) for value in effective_objective
        ),
        objective_constant=objective_constant,
        objective_direction=direction,
        fingerprint=fingerprint,
    )


def _condition_objective_after_fixed_coordinates(
    lp: CompiledFluxLP,
    fixed_indices: Sequence[int],
    fixed_values: Sequence[float],
) -> tuple[tuple[float, ...], float, float]:
    """Recompute the exact affine quotient after fixing more coordinates.

    Original fixed bounds are included automatically.  The returned tuple is
    ``(effective_coefficients, objective_constant, effective_scale)``.
    """

    if len(fixed_indices) != len(fixed_values):
        raise AnalysisError("fixed objective coordinates are malformed")
    n = len(lp.reaction_ids)
    fixed: dict[int, float] = {
        index: lower
        for index, (lower, upper) in enumerate(
            zip(lp.lower_bounds, lp.upper_bounds)
        )
        if lower == upper
    }
    supplied: set[int] = set()
    for raw_index, raw_value in zip(fixed_indices, fixed_values):
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, (int, np.integer))
            or not 0 <= int(raw_index) < n
            or int(raw_index) in supplied
        ):
            raise AnalysisError("fixed objective coordinates are malformed")
        index = int(raw_index)
        supplied.add(index)
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise AnalysisError("fixed objective coordinates are malformed") from error
        if (
            not math.isfinite(value)
            or value < lp.lower_bounds[index] - FEASIBILITY_TOLERANCE
            or value > lp.upper_bounds[index] + FEASIBILITY_TOLERANCE
            or index in fixed
            and abs(value - fixed[index]) > FEASIBILITY_TOLERANCE
        ):
            raise AnalysisError(
                "fixed objective coordinate is inconsistent with reaction bounds"
            )
        fixed[index] = value

    balance_matrix = np.zeros((len(lp.balanced_metabolite_ids), n), dtype=float)
    for row in range(len(lp.balanced_metabolite_ids)):
        for offset in range(lp.row_starts[row], lp.row_starts[row + 1]):
            balance_matrix[row, lp.column_indices[offset]] = lp.coefficients[offset]
    lower_bounds = np.asarray(lp.lower_bounds, dtype=float).copy()
    upper_bounds = np.asarray(lp.upper_bounds, dtype=float).copy()
    ordered_indices = np.asarray(sorted(fixed), dtype=int)
    ordered_values = np.asarray(
        [fixed[index] for index in ordered_indices], dtype=float
    )
    lower_bounds[ordered_indices] = ordered_values
    upper_bounds[ordered_indices] = ordered_values
    free_indices = np.asarray(
        [index for index in range(n) if index not in fixed], dtype=int
    )
    free_matrix = balance_matrix[:, free_indices]
    conditioned_rows = np.asarray(lp.balance_row_space_basis, dtype=float).reshape(
        lp.balance_rank, n
    )
    if lp.balance_rank:
        conditioned_free = conditioned_rows[:, free_indices]
        conditioned_fixed = conditioned_rows[:, ordered_indices]
        induced_rhs = -(conditioned_fixed @ ordered_values)
        try:
            least_squares = np.linalg.lstsq(
                conditioned_free,
                induced_rhs,
                rcond=AFFINE_CONDITIONING_FLOOR,
            )[0]
        except np.linalg.LinAlgError as error:
            raise AnalysisError(
                "could not validate fixed-coordinate mass-balance consistency"
            ) from error
        consistency_residual = float(
            np.linalg.norm(conditioned_free @ least_squares - induced_rhs)
        )
        if consistency_residual > FEASIBILITY_TOLERANCE:
            raise AnalysisError(
                "mass-balance after fixed-coordinate substitution is "
                "inconsistent: conditioned residual="
                f"{consistency_residual:g}"
            )
    free_row_basis, _, _, affine_rank = _condition_affine_equalities(
        free_matrix,
        np.zeros(free_matrix.shape[0], dtype=float),
        description="mass-balance after sampled-face coordinate reduction",
    )
    affine_basis = np.zeros((affine_rank, n), dtype=float)
    if affine_rank:
        affine_basis[:, free_indices] = free_row_basis
    effective, constant = _condition_objective(
        np.asarray(lp.objective_coefficients, dtype=float),
        affine_basis,
        ordered_indices,
        ordered_values,
        balance_matrix,
        lower_bounds,
        upper_bounds,
    )
    scale = max((abs(float(value)) for value in effective), default=0.0)
    return tuple(float(value) for value in effective), constant, scale


def _highspy():
    try:
        import highspy
    except ImportError as error:
        raise AnalysisError("native HiGHS analysis requires the default 'highspy' dependency") from error
    return highspy


def _objective_scale(lp: CompiledFluxLP) -> float:
    """Return a positive scale for numerically stable biological objectives."""

    return max(
        (abs(value) for value in lp.effective_objective_coefficients),
        default=0.0,
    )


def _objective_activity_scale(lp: CompiledFluxLP) -> float:
    """Bound the objective variation over the canonical coordinate box."""

    try:
        scale = math.fsum(
            abs(coefficient) * (upper - lower)
            for coefficient, lower, upper in zip(
                lp.effective_objective_coefficients,
                lp.lower_bounds,
                lp.upper_bounds,
            )
        )
    except (OverflowError, ValueError) as error:
        raise AnalysisError(
            "biological objective activity scale is outside floating-point range"
        ) from error
    if not math.isfinite(scale):
        raise AnalysisError(
            "biological objective activity scale is outside floating-point range"
        )
    return scale


def _effective_objective_value(
    lp: CompiledFluxLP, fluxes: Sequence[float]
) -> float:
    return math.fsum(
        coefficient * value
        for coefficient, value in zip(
            lp.effective_objective_coefficients, fluxes
        )
    )


def _biological_objective_value(
    lp: CompiledFluxLP, fluxes: Sequence[float]
) -> float:
    """Evaluate the declared objective through its stable affine quotient."""

    return math.fsum((lp.objective_constant, _effective_objective_value(lp, fluxes)))


def _declared_objective_value(
    lp: CompiledFluxLP, fluxes: Sequence[float]
) -> float:
    """Evaluate the literal canonical ``c^T v`` with accurate summation."""

    return math.fsum(
        coefficient * value
        for coefficient, value in zip(lp.objective_coefficients, fluxes)
    )


def _validate_declared_objective_equivalence(
    lp: CompiledFluxLP,
    fluxes: Sequence[float],
    stable_value: float,
    operation: str,
    *active_values: float,
) -> float:
    """Fail closed when an approximate primal breaks exact affine equivalence."""

    try:
        declared = _declared_objective_value(lp, fluxes)
    except (OverflowError, ValueError) as error:
        raise AnalysisError(
            f"{operation} declared biological objective is not finite"
        ) from error
    scale = max(
        _objective_activity_scale(lp),
        *(abs(float(value)) for value in active_values),
    )
    tolerance = OBJECTIVE_TOLERANCE * scale if scale > 0.0 else OBJECTIVE_TOLERANCE
    discrepancy = abs(declared - stable_value)
    if not math.isfinite(declared) or discrepancy > tolerance:
        raise AnalysisError(
            f"{operation} declared biological objective differs from its stable "
            "affine quotient: "
            f"declared={declared:g}, quotient={stable_value:g}, "
            f"discrepancy={discrepancy:g}"
        )
    return declared


def _retained_objective_violation(
    lp: CompiledFluxLP,
    value: float,
    sense: str,
    bound: float,
    *,
    effective_value: float | None = None,
    effective_bound: float | None = None,
) -> tuple[float, float]:
    """Return reported raw and stable quotient-normalized row violations."""

    raw = max(bound - value, 0.0) if sense == ">=" else max(value - bound, 0.0)
    if effective_value is None:
        effective_value = value - lp.objective_constant
    if effective_bound is None:
        effective_bound = bound - lp.objective_constant
    effective_violation = (
        max(effective_bound - effective_value, 0.0)
        if sense == ">="
        else max(effective_value - effective_bound, 0.0)
    )
    scale = max(
        _objective_activity_scale(lp),
        abs(effective_value),
        abs(effective_bound),
    )
    return (
        raw,
        effective_violation / scale if scale > 0.0 else effective_violation,
    )


def _solve(
    lp: CompiledFluxLP,
    costs: Sequence[float],
    direction: str,
    operation: str,
    retention: RetainedObjectiveConstraint | None = None,
) -> tuple[list[float], float, str]:
    highspy = _highspy(); solver = highspy.Highs()
    solver.setOptionValue("output_flag", False); solver.setOptionValue("threads", 1)
    solver.setOptionValue("solver", "simplex")
    solver.setOptionValue(
        "primal_feasibility_tolerance", SOLVER_FEASIBILITY_TOLERANCE
    )
    solver.setOptionValue(
        "dual_feasibility_tolerance", SOLVER_FEASIBILITY_TOLERANCE
    )
    solver.setOptionValue("small_matrix_value", HIGHS_SMALL_MATRIX_VALUE)
    n = len(lp.reaction_ids)
    solver.addCols(n, list(costs), list(lp.lower_bounds), list(lp.upper_bounds), 0, [0] * (n + 1), [], [])
    lower = list(lp.affine_equality_rhs)
    upper = list(lp.affine_equality_rhs)
    starts = list(lp.solver_row_starts)
    indices = list(lp.solver_column_indices)
    values = list(lp.solver_coefficients)
    if retention is not None:
        sense = retention.sense
        effective_bound = retention.effective_bound
        objective_scale = _objective_scale(lp)
        normalized_bound = (
            effective_bound / objective_scale
            if objective_scale > 0.0
            else effective_bound
        )
        optimal_face = (
            retention.fraction_of_optimum == 1.0
            or retention.bound == retention.biological_optimum
        )
        lower.append(
            normalized_bound
            if optimal_face or sense == ">="
            else -highspy.kHighsInf
        )
        upper.append(
            normalized_bound
            if optimal_face or sense == "<="
            else highspy.kHighsInf
        )
        indices.extend(
            i
            for i, value in enumerate(lp.effective_objective_coefficients)
            if value != 0.0
        )
        values.extend(
            value / objective_scale if objective_scale > 0.0 else value
            for value in lp.effective_objective_coefficients
            if value != 0.0
        )
        starts.append(len(indices))
    solver.addRows(len(lower), lower, upper, len(indices), starts, indices, values)
    solver.setMaximize() if direction == "max" else solver.setMinimize()
    solver.run(); status = solver.getModelStatus(); status_name = solver.modelStatusToString(status)
    if status != highspy.HighsModelStatus.kOptimal:
        lowered = status_name.lower()
        category = "infeasible" if "infeasible" in lowered else "unbounded" if "unbounded" in lowered else "solver failure/unknown"
        raise AnalysisError(f"{operation} failed: HiGHS status {status_name} ({category})")
    solution = [float(v) for v in solver.getSolution().col_value]
    objective = float(solver.getObjectiveValue())
    if len(solution) != n or not all(math.isfinite(v) for v in solution) or not math.isfinite(objective):
        raise AnalysisError(f"{operation} returned a malformed or non-finite solution")
    return solution, objective, status_name


@lru_cache(maxsize=32)
def _primal_validation_matrices(
    reaction_count: int,
    metabolite_count: int,
    row_starts: tuple[int, ...],
    column_indices: tuple[int, ...],
    coefficients: tuple[float, ...],
    balance_rank: int,
    row_space_basis: tuple[tuple[float, ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache dense read-only matrices used for repeated primal certification."""

    raw = np.zeros((metabolite_count, reaction_count), dtype=float)
    normalized = np.zeros_like(raw)
    for row in range(metabolite_count):
        start, stop = row_starts[row], row_starts[row + 1]
        indices = column_indices[start:stop]
        values = coefficients[start:stop]
        raw[row, list(indices)] = values
        norm = math.hypot(*(float(value) for value in values))
        if norm > 0.0:
            normalized[row, list(indices)] = np.asarray(values, dtype=float) / norm
    row_space = np.asarray(row_space_basis, dtype=float).reshape(
        balance_rank, reaction_count
    )
    for matrix in (raw, normalized, row_space):
        matrix.setflags(write=False)
    return raw, normalized, row_space


def _validate(lp: CompiledFluxLP, fluxes: Sequence[float], reported: float,
              costs: Sequence[float]) -> PrimalDiagnostics:
    reaction_count = len(lp.reaction_ids)
    try:
        vector = np.asarray(fluxes, dtype=float)
        cost_vector = np.asarray(costs, dtype=float)
        reported_value = float(reported)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError(
            "independent primal validation received malformed numeric values"
        ) from error
    if vector.shape != (reaction_count,) or cost_vector.shape != (reaction_count,):
        raise AnalysisError(
            "independent primal validation received a malformed vector length"
        )
    if not np.isfinite(vector).all():
        raise AnalysisError(
            "independent primal validation received a non-finite flux vector"
        )
    if not np.isfinite(cost_vector).all() or not math.isfinite(reported_value):
        raise AnalysisError(
            "independent primal validation received a non-finite objective value"
        )

    lower_bounds = np.asarray(lp.lower_bounds, dtype=float)
    upper_bounds = np.asarray(lp.upper_bounds, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        lower_violations = np.maximum(lower_bounds - vector, 0.0)
        upper_violations = np.maximum(vector - upper_bounds, 0.0)
    if not np.isfinite(lower_violations).all() or not np.isfinite(
        upper_violations
    ).all():
        raise AnalysisError(
            "independent primal validation produced non-finite bound arithmetic"
        )
    lower = float(np.max(lower_violations, initial=0.0))
    upper = float(np.max(upper_violations, initial=0.0))
    raw_matrix, normalized_matrix, row_space = _primal_validation_matrices(
        reaction_count,
        len(lp.balanced_metabolite_ids),
        lp.row_starts,
        lp.column_indices,
        lp.coefficients,
        lp.balance_rank,
        lp.balance_row_space_basis,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        raw_products = raw_matrix @ vector
        normalized_products = normalized_matrix @ vector
        conditioned_products = row_space @ vector
    if not np.isfinite(raw_products).all():
        raise AnalysisError(
            "independent primal validation produced non-finite raw mass-balance "
            "arithmetic"
        )
    if not np.isfinite(normalized_products).all():
        raise AnalysisError(
            "independent primal validation produced non-finite normalized "
            "mass-balance arithmetic"
        )
    if not np.isfinite(conditioned_products).all():
        raise AnalysisError(
            "independent primal validation produced non-finite conditioned "
            "mass-balance arithmetic"
        )
    raw_residual = float(np.max(np.abs(raw_products), initial=0.0))
    max_row_residual = float(
        np.max(np.abs(normalized_products), initial=0.0)
    )
    conditioned_residual = math.hypot(
        *(float(value) for value in conditioned_products)
    )
    if not all(
        math.isfinite(value)
        for value in (raw_residual, max_row_residual, conditioned_residual)
    ):
        raise AnalysisError(
            "independent primal validation produced a non-finite mass-balance "
            "residual"
        )
    try:
        recalculated = math.fsum(
            float(cost) * float(value)
            for cost, value in zip(cost_vector, vector)
        )
    except (OverflowError, ValueError) as error:
        raise AnalysisError(
            "independent primal validation produced non-finite objective "
            "arithmetic"
        ) from error
    if not math.isfinite(recalculated):
        raise AnalysisError(
            "independent primal validation produced non-finite objective "
            "arithmetic"
        )
    error = abs(recalculated - reported_value)
    if not math.isfinite(error):
        raise AnalysisError(
            "independent primal validation produced a non-finite objective "
            "recalculation error"
        )
    diagnostics = PrimalDiagnostics(
        max_lower_bound_violation=lower,
        max_upper_bound_violation=upper,
        max_mass_balance_residual=max_row_residual,
        objective_recalculation_error=error,
        max_raw_mass_balance_residual=raw_residual,
        conditioned_row_space_residual=conditioned_residual,
    )
    if (
        max(lower, upper, max_row_residual, conditioned_residual)
        > FEASIBILITY_TOLERANCE
        or error > OBJECTIVE_TOLERANCE
    ):
        raise AnalysisError(f"independent primal validation failed: {diagnostics}")
    return diagnostics


def _run_compiled_fba(lp: CompiledFluxLP) -> FBAResult:
    """Solve and independently validate the biological objective on ``lp``."""

    objective_scale = _objective_scale(lp)
    solver_costs = tuple(
        value / objective_scale if objective_scale > 0.0 else value
        for value in lp.effective_objective_coefficients
    )
    fluxes, solver_objective, status = _solve(
        lp, solver_costs, lp.objective_direction, "FBA"
    )
    try:
        effective_objective = _effective_objective_value(lp, fluxes)
        objective = math.fsum((lp.objective_constant, effective_objective))
    except (OverflowError, ValueError) as error:
        raise AnalysisError("FBA returned a non-finite biological objective") from error
    if not math.isfinite(objective):
        raise AnalysisError("FBA returned a non-finite biological objective")
    _validate_declared_objective_equivalence(
        lp, fluxes, objective, "FBA", effective_objective
    )
    diagnostics = _validate(
        lp,
        fluxes,
        solver_objective,
        solver_costs,
    )
    return FBAResult(objective, "optimal", lp.objective_direction,
                     pd.Series(fluxes, index=lp.reaction_ids, dtype=float), diagnostics)


def run_highs_fba(model: FluxModel) -> FBAResult:
    lp = compile_flux_lp(model)
    _validate_optimal_face_solver_resolution(lp)
    return _run_compiled_fba(lp)


def _fraction(value: float) -> float:
    if isinstance(value, bool):
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]")
    try: result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]") from error
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]")
    return result


def _retained_effective_bound(
    lp: CompiledFluxLP,
    biological_optimum: float,
    effective_optimum: float,
    fraction: float,
) -> tuple[float, float]:
    """Return the declared and conservatively representable quotient bounds."""

    bound = biological_optimum * fraction
    if fraction == 1.0:
        return bound, effective_optimum
    exact_target = Fraction.from_float(bound) - Fraction.from_float(
        lp.objective_constant
    )
    try:
        effective_bound = float(exact_target)
    except (OverflowError, ValueError) as error:
        raise AnalysisError(
            "retained biological objective bound is outside floating-point range"
        ) from error
    candidate = Fraction.from_float(effective_bound)
    if lp.objective_direction == "max" and candidate < exact_target:
        effective_bound = math.nextafter(effective_bound, math.inf)
    elif lp.objective_direction == "min" and candidate > exact_target:
        effective_bound = math.nextafter(effective_bound, -math.inf)
    if not math.isfinite(effective_bound):
        raise AnalysisError(
            "retained biological objective bound is outside floating-point range"
        )
    represented_bound = math.fsum((lp.objective_constant, effective_bound))
    relaxed = (
        represented_bound < bound
        if lp.objective_direction == "max"
        else represented_bound > bound
    )
    distortion_scale = max(abs(biological_optimum), abs(bound))
    distortion = abs(represented_bound - bound)
    if (
        relaxed
        or not math.isfinite(represented_bound)
        or (distortion_scale == 0.0 and distortion != 0.0)
        or (
            distortion_scale > 0.0
            and distortion > OBJECTIVE_TOLERANCE * distortion_scale
        )
    ):
        raise AnalysisError(
            "retained biological objective bound is not representable without "
            "material numerical distortion"
        )
    return bound, effective_bound


def _validate_fractional_optimum_sign(
    direction: str, optimum: float, fraction: float
) -> None:
    """Reject multiplicative retention that is stricter than the optimum."""

    if fraction == 1.0:
        return
    incompatible = (direction == "max" and optimum < 0.0) or (
        direction == "min" and optimum > 0.0
    )
    if incompatible:
        raise AnalysisError(
            "fraction-of-optimum arithmetic is infeasible for a negative "
            "maximization optimum or positive minimization optimum"
        )


def prepare_highs_flux_region(
    model: FluxModel, fraction_of_optimum: float = 1.0
) -> PreparedFluxRegion:
    """Compile once and solve the retained biological objective exactly once."""

    fraction = _fraction(fraction_of_optimum)
    lp = compile_flux_lp(model)
    if fraction == 1.0:
        _validate_optimal_face_solver_resolution(lp)
    fba = _run_compiled_fba(lp)
    _validate_fractional_optimum_sign(
        lp.objective_direction, fba.objective_value, fraction
    )
    sense = ">=" if lp.objective_direction == "max" else "<="
    effective_optimum = _effective_objective_value(lp, tuple(fba.fluxes))
    bound, effective_bound = _retained_effective_bound(
        lp, fba.objective_value, effective_optimum, fraction
    )
    if bound == fba.objective_value:
        _validate_optimal_face_solver_resolution(lp)
    retention = RetainedObjectiveConstraint(
        fraction,
        fba.objective_value,
        bound,
        sense,
        lp.objective_direction,
        effective_optimum,
        effective_bound,
    )
    return PreparedFluxRegion(lp, fba, retention)


def _validate_compiled_lp_integrity(lp: CompiledFluxLP) -> None:
    """Recompute public compiled-LP identity and all conditioned derivatives."""

    n = len(lp.reaction_ids)
    m = len(lp.balanced_metabolite_ids)
    if (
        n == 0
        or len(set(lp.reaction_ids)) != n
        or len(lp.row_starts) != m + 1
        or not lp.row_starts
        or lp.row_starts[0] != 0
        or tuple(sorted(lp.row_starts)) != tuple(lp.row_starts)
        or lp.row_starts[-1] != len(lp.column_indices)
        or len(lp.column_indices) != len(lp.coefficients)
        or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < n
            for index in lp.column_indices
        )
        or any(
            len(values) != n
            for values in (
                lp.lower_bounds,
                lp.upper_bounds,
                lp.objective_coefficients,
                lp.effective_objective_coefficients,
            )
        )
        or lp.objective_direction not in {"max", "min"}
    ):
        raise AnalysisError("prepared compiled LP has malformed canonical structure")
    numeric_values = (
        *lp.coefficients,
        *lp.lower_bounds,
        *lp.upper_bounds,
        *lp.objective_coefficients,
        *lp.effective_objective_coefficients,
        lp.objective_constant,
    )
    try:
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise AnalysisError("prepared compiled LP contains non-finite numeric data")
        expected_fingerprint = _compiled_lp_fingerprint(
            lp.reaction_ids,
            lp.balanced_metabolite_ids,
            lp.row_starts,
            lp.column_indices,
            lp.coefficients,
            lp.lower_bounds,
            lp.upper_bounds,
            lp.objective_coefficients,
            lp.objective_direction,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("prepared compiled LP contains malformed identity data") from error
    if lp.fingerprint != expected_fingerprint:
        raise AnalysisError(
            "prepared compiled LP fingerprint is stale or inconsistent with its "
            "canonical fields"
        )

    balance_matrix = np.zeros((m, n), dtype=float)
    for row in range(m):
        seen: set[int] = set()
        for offset in range(lp.row_starts[row], lp.row_starts[row + 1]):
            column = lp.column_indices[offset]
            if column in seen:
                raise AnalysisError(
                    "prepared compiled LP contains duplicate sparse coordinates"
                )
            seen.add(column)
            balance_matrix[row, column] = lp.coefficients[offset]
    expected_balance, expected_balance_rank = _orthonormal_row_space(balance_matrix)
    lower_bounds = np.asarray(lp.lower_bounds, dtype=float)
    upper_bounds = np.asarray(lp.upper_bounds, dtype=float)
    (
        expected_affine,
        expected_rhs,
        expected_affine_rank,
        affine_matrix,
        affine_rhs,
        fixed_indices,
    ) = _condition_balance_after_fixed_substitution(
        balance_matrix, lower_bounds, upper_bounds
    )
    expected_effective, expected_constant = _condition_objective(
        np.asarray(lp.objective_coefficients, dtype=float),
        expected_affine,
        fixed_indices,
        lower_bounds[fixed_indices],
        balance_matrix,
        lower_bounds,
        upper_bounds,
    )
    _validate_mass_balance_solver_resolution(
        expected_affine, lower_bounds, upper_bounds
    )
    _validate_objective_solver_resolution(
        expected_effective, lower_bounds, upper_bounds
    )

    try:
        stored_balance = np.asarray(lp.balance_row_space_basis, dtype=float).reshape(
            lp.balance_rank, n
        )
        stored_affine = np.asarray(lp.affine_equality_basis, dtype=float).reshape(
            lp.affine_rank, n
        )
        stored_rhs = np.asarray(lp.affine_equality_rhs, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("prepared compiled LP has malformed conditioned data") from error
    if (
        lp.balance_rank != expected_balance_rank
        or lp.affine_rank != expected_affine_rank
        or stored_rhs.shape != (lp.affine_rank,)
        or not np.isfinite(stored_balance).all()
        or not np.isfinite(stored_affine).all()
        or not np.isfinite(stored_rhs).all()
        or not np.allclose(stored_balance, expected_balance, rtol=1e-12, atol=1e-12)
        or not np.allclose(stored_affine, expected_affine, rtol=1e-12, atol=1e-12)
        or not np.allclose(stored_rhs, expected_rhs, rtol=1e-12, atol=1e-12)
        or not np.allclose(
            np.asarray(lp.effective_objective_coefficients, dtype=float),
            expected_effective,
            rtol=1e-12,
            atol=1e-14,
        )
        or not math.isclose(
            lp.objective_constant,
            expected_constant,
            rel_tol=1e-12,
            abs_tol=1e-14,
        )
    ):
        raise AnalysisError(
            "prepared compiled LP conditioned representation is inconsistent with "
            "its canonical fields"
        )

    expected_starts = [0]
    expected_indices: list[int] = []
    expected_values: list[float] = []
    for row in expected_affine:
        for column, value in enumerate(row):
            if value != 0.0:
                expected_indices.append(column)
                expected_values.append(float(value))
        expected_starts.append(len(expected_indices))
    if (
        lp.solver_row_starts != tuple(expected_starts)
        or lp.solver_column_indices != tuple(expected_indices)
        or lp.solver_coefficients != tuple(expected_values)
    ):
        raise AnalysisError(
            "prepared compiled LP solver matrix is inconsistent with its affine "
            "equalities"
        )


def _validate_prepared_flux_region(prepared: PreparedFluxRegion) -> None:
    """Reject forged or internally inconsistent prepared-region records."""

    if not isinstance(prepared, PreparedFluxRegion):
        raise AnalysisError("prepared analysis requires a PreparedFluxRegion")
    lp, fba, retention = prepared.lp, prepared.fba, prepared.retention
    if (
        not isinstance(lp, CompiledFluxLP)
        or not isinstance(fba, FBAResult)
        or not isinstance(retention, RetainedObjectiveConstraint)
    ):
        raise AnalysisError("prepared flux region contains malformed records")
    _validate_compiled_lp_integrity(lp)
    if not isinstance(fba.fluxes, pd.Series):
        raise AnalysisError("prepared FBA fluxes must be a pandas Series")
    if tuple(fba.fluxes.index) != lp.reaction_ids:
        raise AnalysisError("prepared FBA reaction order does not match its compiled LP")
    if fba.status != "optimal" or fba.objective_direction != lp.objective_direction:
        raise AnalysisError("prepared FBA status or objective direction is inconsistent")
    if len(fba.fluxes) != len(lp.reaction_ids):
        raise AnalysisError("prepared FBA primal has the wrong vector length")
    try:
        raw_fluxes = tuple(fba.fluxes)
        if isinstance(fba.objective_value, (bool, np.bool_)) or any(
            isinstance(value, (bool, np.bool_)) for value in raw_fluxes
        ):
            raise TypeError
        optimum = float(fba.objective_value)
        fluxes = tuple(float(value) for value in raw_fluxes)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("prepared FBA contains malformed numeric values") from error
    if not math.isfinite(optimum):
        raise AnalysisError("prepared FBA objective is non-finite")
    if not all(math.isfinite(value) for value in fluxes):
        raise AnalysisError("prepared FBA primal contains non-finite flux values")
    try:
        effective_optimum_from_fluxes = _effective_objective_value(lp, fluxes)
        stable_optimum = math.fsum(
            (lp.objective_constant, effective_optimum_from_fluxes)
        )
        if stable_optimum != optimum:
            raise AnalysisError(
                "prepared FBA objective is inconsistent with its compiled LP"
            )
        _validate_declared_objective_equivalence(
            lp,
            fluxes,
            stable_optimum,
            "prepared FBA",
            effective_optimum_from_fluxes,
        )
        _validate(
            lp,
            fluxes,
            effective_optimum_from_fluxes,
            lp.effective_objective_coefficients,
        )
    except (IndexError, TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("prepared FBA primal is malformed") from error
    fraction = _fraction(retention.fraction_of_optimum)
    try:
        if (
            isinstance(retention.biological_optimum, bool)
            or isinstance(retention.bound, bool)
            or isinstance(retention.effective_optimum, bool)
            or isinstance(retention.effective_bound, bool)
        ):
            raise TypeError
        recorded_optimum = float(retention.biological_optimum)
        recorded_bound = float(retention.bound)
        recorded_effective_optimum = float(retention.effective_optimum)
        recorded_effective_bound = float(retention.effective_bound)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("prepared retained-objective values are malformed") from error
    if not all(
        map(
            math.isfinite,
            (
                recorded_optimum,
                recorded_bound,
                recorded_effective_optimum,
                recorded_effective_bound,
            ),
        )
    ):
        raise AnalysisError("prepared retained-objective values must be finite")
    _validate_fractional_optimum_sign(
        lp.objective_direction, optimum, fraction
    )
    expected_sense = ">=" if lp.objective_direction == "max" else "<="
    expected_bound, expected_effective_bound = _retained_effective_bound(
        lp, optimum, effective_optimum_from_fluxes, fraction
    )
    if (
        retention.objective_direction != lp.objective_direction
        or retention.sense != expected_sense
        or recorded_optimum != optimum
        or recorded_bound != expected_bound
        or recorded_effective_optimum != effective_optimum_from_fluxes
        or recorded_effective_bound != expected_effective_bound
    ):
        raise AnalysisError("prepared retained-objective metadata is inconsistent")
    if recorded_bound == recorded_optimum:
        _validate_optimal_face_solver_resolution(lp)


def run_highs_fva_reference(model: FluxModel, fraction_of_optimum: float = 1.0) -> FVAResult:
    prepared = prepare_highs_flux_region(model, fraction_of_optimum)
    fraction = prepared.retention.fraction_of_optimum
    lp = prepared.lp
    fba = prepared.fba
    retention = prepared.retention
    minima: list[float] = []; maxima: list[float] = []
    for j, reaction_id in enumerate(lp.reaction_ids):
        costs = [0.0] * len(lp.reaction_ids); costs[j] = 1.0
        low_flux, low, _ = _solve(lp, costs, "min", f"FVA minimum for reaction {reaction_id!r}", retention)
        _validate(lp, low_flux, low, costs)
        high_flux, high, _ = _solve(lp, costs, "max", f"FVA maximum for reaction {reaction_id!r}", retention)
        _validate(lp, high_flux, high, costs)
        # Independently enforce the retained biological objective at every endpoint.
        for endpoint in (low_flux, high_flux):
            value = _biological_objective_value(lp, endpoint)
            declared = _validate_declared_objective_equivalence(
                lp,
                endpoint,
                value,
                f"FVA endpoint for reaction {reaction_id!r}",
                retention.effective_optimum,
                retention.effective_bound,
            )
            direct_violation = (
                max(retention.bound - declared, 0.0)
                if retention.sense == ">="
                else max(declared - retention.bound, 0.0)
            )
            direct_scale = max(
                _objective_activity_scale(lp),
                abs(retention.effective_optimum),
                abs(retention.effective_bound),
            )
            direct_tolerance = (
                OBJECTIVE_TOLERANCE * direct_scale
                if direct_scale > 0.0
                else OBJECTIVE_TOLERANCE
            )
            if direct_violation > direct_tolerance:
                raise AnalysisError(
                    f"FVA endpoint for reaction {reaction_id!r} violates the "
                    "declared biological objective retention: "
                    f"declared={declared:g}, bound={retention.bound:g}, "
                    f"violation={direct_violation:g}"
                )
            effective_value = _effective_objective_value(lp, endpoint)
            violation, normalized_violation = _retained_objective_violation(
                lp,
                value,
                prepared.retention.sense,
                prepared.retention.bound,
                effective_value=effective_value,
                effective_bound=prepared.retention.effective_bound,
            )
            if normalized_violation > OBJECTIVE_TOLERANCE:
                raise AnalysisError(f"FVA endpoint for reaction {reaction_id!r} violates objective retention")
        minima.append(low); maxima.append(high)
    _canonicalize_fva_endpoints(
        lp.reaction_ids, minima, maxima, tuple(fba.fluxes)
    )
    ranges = pd.DataFrame({"minimum": minima, "maximum": maxima}, index=lp.reaction_ids)
    return FVAResult(
        ranges,
        fraction,
        fba.objective_value,
        lp.objective_direction,
        lp.fingerprint,
        _fva_ranges_sha256(ranges, lp.fingerprint),
    )


@dataclass(frozen=True, slots=True)
class _EndpointResult:
    reaction_index: int
    direction: str
    value: float
    worker_id: int
    basis_refreshes: int


@dataclass(slots=True)
class _VFFVAWorkerTrace:
    """Actual lifecycle events recorded by one solver-owning worker thread."""

    worker_id: int
    thread_id: int | None = None
    solver_id: int | None = None
    configured_solver_threads: int | None = None
    matrix_builds: int = 0
    retention_rows: int = 0
    pass_directions: list[str] = field(default_factory=list)
    chunks_claimed: int = 0
    max_chunks_claimed: int = 0
    min_chunks_claimed: int = 0
    endpoint_attempts: int = 0
    endpoint_solves: int = 0
    max_endpoint_solves: int = 0
    min_endpoint_solves: int = 0
    objective_change_attempts: int = 0
    objective_changes: int = 0
    objective_clear_attempts: int = 0
    objective_clears: int = 0
    basis_refreshes: int = 0
    solver_released: bool = False
    exited: bool = False


def _canonicalize_fva_endpoints(
    reaction_ids: Sequence[str],
    minima: list[float],
    maxima: list[float],
    reference_fluxes: Sequence[float],
) -> None:
    """Resolve only sub-tolerance endpoint inversion; preserve positive widths."""

    for index, (minimum, maximum, reference) in enumerate(
        zip(minima, maxima, reference_fluxes)
    ):
        width = maximum - minimum
        if width >= 0.0:
            continue
        if -width <= FVA_NUMERICAL_COLLAPSE_TOLERANCE:
            if (
                reference >= min(minimum, maximum) - FEASIBILITY_TOLERANCE
                and reference <= max(minimum, maximum) + FEASIBILITY_TOLERANCE
            ):
                collapsed = float(reference)
            else:
                collapsed = 0.5 * math.fsum((minimum, maximum))
            minima[index] = collapsed
            maxima[index] = collapsed
        else:
            raise AnalysisError(
                f"FVA endpoints are numerically inconsistent for reaction "
                f"{reaction_ids[index]!r}: minimum={minimum:g}, maximum={maximum:g}"
            )


class _ReusableFVAWorker:
    """One retained LP whose simplex state is reused across endpoint solves."""

    def __init__(
        self,
        lp: CompiledFluxLP,
        retention: RetainedObjectiveConstraint,
        reference_fluxes: Sequence[float],
    ):
        highspy = _highspy()
        self.lp, self.retention, self.endpoint_count = lp, retention, 0
        self.worker_id = 0
        self.lifecycle: _VFFVAWorkerTrace | None = None
        self._pass_directions: list[str] = []
        self.reference_fluxes = tuple(float(value) for value in reference_fluxes)
        self.solver = highspy.Highs()
        self.solver.setOptionValue("output_flag", False)
        self.solver.setOptionValue("threads", 1)
        self.solver.setOptionValue("solver", "simplex")
        self.solver.setOptionValue(
            "primal_feasibility_tolerance", SOLVER_FEASIBILITY_TOLERANCE
        )
        self.solver.setOptionValue(
            "dual_feasibility_tolerance", SOLVER_FEASIBILITY_TOLERANCE
        )
        self.solver.setOptionValue(
            "small_matrix_value", HIGHS_SMALL_MATRIX_VALUE
        )
        _, configured_threads = self.solver.getOptionValue("threads")
        try:
            configured_threads = int(configured_threads)
        except (TypeError, ValueError, OverflowError) as error:
            raise AnalysisError(
                "FVA worker could not verify its HiGHS thread count"
            ) from error
        if configured_threads != 1:
            raise AnalysisError(
                "FVA worker requires exactly one HiGHS internal thread; "
                f"configured={configured_threads}"
            )
        self.configured_solver_threads = configured_threads
        n = len(lp.reaction_ids)
        self.solver.addCols(n, [0.0] * n, list(lp.lower_bounds), list(lp.upper_bounds),
                            0, [0] * (n + 1), [], [])
        lower = list(lp.affine_equality_rhs)
        upper = list(lp.affine_equality_rhs)
        starts = list(lp.solver_row_starts)
        indices = list(lp.solver_column_indices)
        values = list(lp.solver_coefficients)
        sense = retention.sense
        effective_bound = retention.effective_bound
        objective_scale = _objective_scale(lp)
        normalized_bound = (
            effective_bound / objective_scale
            if objective_scale > 0.0
            else effective_bound
        )
        optimal_face = (
            retention.fraction_of_optimum == 1.0
            or retention.bound == retention.biological_optimum
        )
        lower.append(
            normalized_bound
            if optimal_face or sense == ">="
            else -highspy.kHighsInf
        )
        upper.append(
            normalized_bound
            if optimal_face or sense == "<="
            else highspy.kHighsInf
        )
        indices.extend(
            i
            for i, value in enumerate(lp.effective_objective_coefficients)
            if value
        )
        values.extend(
            value / objective_scale if objective_scale > 0.0 else value
            for value in lp.effective_objective_coefficients
            if value
        )
        starts.append(len(indices))
        self.solver.addRows(len(lower), lower, upper, len(indices), starts, indices, values)

    def begin_pass(self, direction: str) -> None:
        """Set objective sense once for the next global VFFVA pass."""

        expected = ("max", "min")
        offset = len(self._pass_directions)
        if offset >= len(expected) or direction != expected[offset]:
            raise AnalysisError(
                "FVA worker objective passes must be exactly max then min"
            )
        if direction == "max":
            self.solver.setMaximize()
        else:
            self.solver.setMinimize()
        self._pass_directions.append(direction)
        if self.lifecycle is not None:
            self.lifecycle.pass_directions.append(direction)

    def solve(self, task: tuple[int, str]) -> _EndpointResult:
        j, direction = task
        reaction_id = self.lp.reaction_ids[j]
        operation = f"FVA {direction}imum for reaction {reaction_id!r}"
        if not hasattr(self, "_pass_directions"):
            # Backward-compatible support for narrowly constructed private test
            # doubles; production workers always enter through ``begin_pass``.
            self._pass_directions = [direction]
        if not self._pass_directions or direction != self._pass_directions[-1]:
            raise AnalysisError(
                f"{operation} failed: worker objective sense is not active"
            )
        lifecycle = getattr(self, "lifecycle", None)
        if lifecycle is not None:
            lifecycle.endpoint_attempts += 1
        basis_refreshes = 0
        value: float | None = None
        failure: BaseException | None = None
        status_name = "not available"
        try:
            # changeColCost invalidates the objective but retains the model and the
            # incumbent simplex basis. HiGHS therefore reoptimizes on the next run.
            if lifecycle is not None:
                lifecycle.objective_change_attempts += 1
            self.solver.changeColCost(j, 1.0)
            if lifecycle is not None:
                lifecycle.objective_changes += 1
            _, time_limit = self.solver.getOptionValue("time_limit")
            if float(time_limit) <= 0.0:
                status_name = "Time limit reached"
                raise AnalysisError(f"HiGHS status {status_name}")
            highspy = _highspy()
            costs = [0.0] * len(self.lp.reaction_ids)
            costs[j] = 1.0
            last_error: AnalysisError | None = None
            for attempt in range(2):
                if attempt:
                    self.solver.clearSolver()
                    basis_refreshes += 1
                    if lifecycle is not None:
                        lifecycle.basis_refreshes += 1
                try:
                    self.solver.run()
                    status = self.solver.getModelStatus()
                    status_name = self.solver.modelStatusToString(status)
                    if status != highspy.HighsModelStatus.kOptimal:
                        raise AnalysisError(f"HiGHS status {status_name}")
                    fluxes = [
                        float(value)
                        for value in self.solver.getSolution().col_value
                    ]
                    value = float(self.solver.getObjectiveValue())
                    if (
                        len(fluxes) != len(self.lp.reaction_ids)
                        or not all(map(math.isfinite, fluxes))
                        or not math.isfinite(value)
                    ):
                        raise AnalysisError("malformed or non-finite complete primal")
                    _validate(self.lp, fluxes, value, costs)
                    reference = self.reference_fluxes[j]
                    excludes_reference = (
                        direction == "min"
                        and value
                        > reference + FVA_NUMERICAL_COLLAPSE_TOLERANCE
                    ) or (
                        direction == "max"
                        and value
                        < reference - FVA_NUMERICAL_COLLAPSE_TOLERANCE
                    )
                    if excludes_reference:
                        raise AnalysisError(
                            "endpoint optimum excludes the independently validated "
                            f"FBA witness {reference:g}"
                        )
                    biological = _biological_objective_value(self.lp, fluxes)
                    declared = _validate_declared_objective_equivalence(
                        self.lp,
                        fluxes,
                        biological,
                        operation,
                        self.retention.effective_optimum,
                        self.retention.effective_bound,
                    )
                    direct_violation = (
                        max(self.retention.bound - declared, 0.0)
                        if self.retention.sense == ">="
                        else max(declared - self.retention.bound, 0.0)
                    )
                    direct_scale = max(
                        _objective_activity_scale(self.lp),
                        abs(self.retention.effective_optimum),
                        abs(self.retention.effective_bound),
                    )
                    direct_tolerance = (
                        OBJECTIVE_TOLERANCE * direct_scale
                        if direct_scale > 0.0
                        else OBJECTIVE_TOLERANCE
                    )
                    if direct_violation > direct_tolerance:
                        raise AnalysisError(
                            "declared biological objective retention violation: "
                            f"declared={declared:g}, "
                            f"bound={self.retention.bound:g}, "
                            f"violation={direct_violation:g}"
                        )
                    effective = _effective_objective_value(self.lp, fluxes)
                    violation, normalized_violation = _retained_objective_violation(
                        self.lp,
                        biological,
                        self.retention.sense,
                        self.retention.bound,
                        effective_value=effective,
                        effective_bound=self.retention.effective_bound,
                    )
                    if normalized_violation > OBJECTIVE_TOLERANCE:
                        raise AnalysisError(
                            "retained biological objective violation "
                            f"{violation:g}"
                        )
                    last_error = None
                    break
                except AnalysisError as error:
                    last_error = error
            if last_error is not None:
                raise last_error
        except BaseException as error:
            failure = error
        finally:
            if lifecycle is not None:
                lifecycle.objective_clear_attempts += 1
            try:
                # VFFVA clears the active endpoint objective after every solve.
                # Keeping this in ``finally`` also restores the retained LP after
                # a failed run or validation attempt.
                self.solver.changeColCost(j, 0.0)
                if lifecycle is not None:
                    lifecycle.objective_clears += 1
            except BaseException as clear_error:
                if failure is None:
                    failure = clear_error
                else:
                    failure = AnalysisError(
                        f"{failure}; additionally could not clear endpoint "
                        f"objective: {clear_error}"
                    )
        if failure is not None:
            message = str(failure)
            if isinstance(failure, AnalysisError) and message.startswith(operation):
                raise AnalysisError(message) from None
            raise AnalysisError(
                f"{operation} failed: solver status {status_name}; {message}"
            ) from None
        if value is None:  # pragma: no cover - guarded by the solve checks above
            raise AnalysisError(f"{operation} failed: missing endpoint objective")
        self.endpoint_count += 1
        if lifecycle is not None:
            lifecycle.endpoint_solves += 1
            if direction == "max":
                lifecycle.max_endpoint_solves += 1
            else:
                lifecycle.min_endpoint_solves += 1
        return _EndpointResult(
            j, direction, value, getattr(self, "worker_id", 0), basis_refreshes
        )


def _populate_vffva_instrumentation(
    instrumentation: dict[str, object] | None,
    traces: Sequence[_VFFVAWorkerTrace],
    threads: Sequence[threading.Thread],
    *,
    configured_workers: int,
    chunk_size: int,
    max_queue_accounted: bool,
    min_released_after_max: bool,
) -> None:
    """Expose measurements tied to the real owning-thread lifecycle."""

    if instrumentation is None:
        return
    passes = tuple(tuple(trace.pass_directions) for trace in traces)
    instrumentation.update(
        configured_workers=configured_workers,
        chunk_size=chunk_size,
        worker_threads=sum(trace.thread_id is not None for trace in traces),
        solver_instances=sum(trace.solver_id is not None for trace in traces),
        matrix_builds=sum(trace.matrix_builds for trace in traces),
        retention_rows=sum(trace.retention_rows for trace in traces),
        highs_single_thread_workers=sum(
            trace.configured_solver_threads == 1 for trace in traces
        ),
        max_pass_workers=sum("max" in directions for directions in passes),
        min_pass_workers=sum("min" in directions for directions in passes),
        workers_surviving_pass_transition=sum(
            directions == ("max", "min") for directions in passes
        ),
        endpoint_attempts=sum(trace.endpoint_attempts for trace in traces),
        endpoint_solves=sum(trace.endpoint_solves for trace in traces),
        max_endpoint_solves=sum(trace.max_endpoint_solves for trace in traces),
        min_endpoint_solves=sum(trace.min_endpoint_solves for trace in traces),
        objective_change_attempts=sum(
            trace.objective_change_attempts for trace in traces
        ),
        objective_changes=sum(trace.objective_changes for trace in traces),
        objective_clear_attempts=sum(
            trace.objective_clear_attempts for trace in traces
        ),
        objective_clears=sum(trace.objective_clears for trace in traces),
        chunks_claimed=sum(trace.chunks_claimed for trace in traces),
        workers_claiming_multiple_chunks=sum(
            trace.max_chunks_claimed > 1 or trace.min_chunks_claimed > 1
            for trace in traces
        ),
        basis_refreshes=sum(trace.basis_refreshes for trace in traces),
        joined_workers=sum(not thread.is_alive() for thread in threads),
        solver_releases=sum(trace.solver_released for trace in traces),
        max_queue_accounted=int(max_queue_accounted),
        min_released_after_max=int(min_released_after_max),
        worker_thread_ids=tuple(
            -1 if trace.thread_id is None else trace.thread_id for trace in traces
        ),
        worker_solver_ids=tuple(
            -1 if trace.solver_id is None else trace.solver_id for trace in traces
        ),
        worker_passes=passes,
        worker_chunk_counts=tuple(trace.chunks_claimed for trace in traces),
        worker_max_chunk_counts=tuple(
            trace.max_chunks_claimed for trace in traces
        ),
        worker_min_chunk_counts=tuple(
            trace.min_chunks_claimed for trace in traces
        ),
        worker_endpoint_counts=tuple(trace.endpoint_solves for trace in traces),
    )


def run_prepared_highs_vffva(
    prepared: PreparedFluxRegion,
    *,
    workers: int | None = None,
    chunk_size: int = 50,
    instrumentation: dict[str, object] | None = None,
) -> FVAResult:
    """Run a shared-memory VFFVA-style pass over a prepared flux region.

    Every owning thread builds one HiGHS LP, reuses its simplex state across
    dynamically claimed reaction chunks, and survives the completed maximum
    pass before the minimum pass starts.  HiGHS itself uses one thread per
    worker.  ``chunk_size=50`` preserves VFFVA's baseline dynamic schedule.
    """
    _validate_prepared_flux_region(prepared)
    lp = prepared.lp
    retention = prepared.retention
    if workers is None:
        workers = 1
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise AnalysisError("workers must be a positive integer")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise AnalysisError("chunk_size must be a positive integer")
    reaction_count = len(lp.reaction_ids)
    workers = min(workers, reaction_count)
    reference_fluxes = tuple(float(value) for value in prepared.fba.fluxes)
    if workers == 0:  # pragma: no cover - canonical models require reactions
        raise AnalysisError("FVA requires at least one reaction")

    def chunk_queue() -> Queue[tuple[int, ...]]:
        work: Queue[tuple[int, ...]] = Queue()
        for start in range(0, reaction_count, chunk_size):
            work.put(tuple(range(start, min(start + chunk_size, reaction_count))))
        return work

    max_work = chunk_queue()
    min_work = chunk_queue()
    start_max = threading.Event()
    start_min = threading.Event()
    cancelled = threading.Event()
    ready_reports: Queue[int] = Queue()
    max_done_reports: Queue[int] = Queue()
    min_done_reports: Queue[int] = Queue()
    traces = tuple(_VFFVAWorkerTrace(worker_id) for worker_id in range(workers))
    results: list[_EndpointResult] = []
    results_lock = threading.Lock()
    failure_lock = threading.Lock()
    failures: list[AnalysisError] = []

    def record_failure(error: BaseException, context: str) -> None:
        raw_message = str(error)
        if isinstance(error, AnalysisError) and raw_message.startswith("FVA "):
            message = raw_message
        else:
            detail = raw_message or type(error).__name__
            message = f"{context}: {detail}"
        # Store no worker traceback: traceback frames can retain ``self.solver``
        # and defer destruction beyond the owning thread's lifetime.
        clean_error = AnalysisError(message).with_traceback(None)
        with failure_lock:
            if not failures:
                failures.append(clean_error)
        cancelled.set()

    def consume_chunks(
        direction: str,
        work: Queue[tuple[int, ...]],
        worker: _ReusableFVAWorker | None,
        trace: _VFFVAWorkerTrace,
    ) -> None:
        if worker is not None and not cancelled.is_set():
            try:
                worker.begin_pass(direction)
            except BaseException as error:
                record_failure(
                    error,
                    f"FVA {direction} pass worker {trace.worker_id} failed",
                )
        while True:
            try:
                chunk = work.get_nowait()
            except Empty:
                return
            trace.chunks_claimed += 1
            if direction == "max":
                trace.max_chunks_claimed += 1
            else:
                trace.min_chunks_claimed += 1
            try:
                if worker is None or cancelled.is_set():
                    continue
                for reaction_index in chunk:
                    if cancelled.is_set():
                        break
                    try:
                        result = worker.solve((reaction_index, direction))
                    except BaseException as error:
                        reaction_id = lp.reaction_ids[reaction_index]
                        record_failure(
                            error,
                            f"FVA {direction}imum for reaction "
                            f"{reaction_id!r} failed",
                        )
                        break
                    with results_lock:
                        results.append(result)
            finally:
                work.task_done()

    def worker_target(trace: _VFFVAWorkerTrace) -> None:
        worker: _ReusableFVAWorker | None = None
        ready_reported = False
        max_reported = False
        min_reported = False
        trace.thread_id = threading.get_ident()
        try:
            try:
                # Construction happens here, never in the coordinating thread.
                worker = _ReusableFVAWorker(
                    lp,
                    retention,
                    reference_fluxes,
                )
                worker.worker_id = trace.worker_id
                worker.lifecycle = trace
                trace.solver_id = id(worker.solver)
                trace.configured_solver_threads = (
                    worker.configured_solver_threads
                )
                trace.matrix_builds += 1
                trace.retention_rows += 1
            except BaseException as error:
                record_failure(
                    error, f"FVA worker {trace.worker_id} initialization failed"
                )
            finally:
                ready_reports.put(trace.worker_id)
                ready_reported = True

            # No endpoint may run until every started worker has reported either
            # a fully configured threads=1 solver or an initialization failure.
            start_max.wait()
            consume_chunks("max", max_work, worker, trace)
            max_done_reports.put(trace.worker_id)
            max_reported = True

            # The coordinator releases this only after every maximum chunk has
            # been accounted.  Cancellation drains MIN without solving it.
            start_min.wait()
            consume_chunks("min", min_work, worker, trace)
            min_done_reports.put(trace.worker_id)
            min_reported = True
        except BaseException as error:  # pragma: no cover - defensive lifecycle
            record_failure(
                error, f"FVA worker {trace.worker_id} terminated unexpectedly"
            )
        finally:
            if not ready_reported:
                ready_reports.put(trace.worker_id)
            if not max_reported:
                start_max.wait()
                consume_chunks("max", max_work, None, trace)
                max_done_reports.put(trace.worker_id)
            if not min_reported:
                start_min.wait()
                consume_chunks("min", min_work, None, trace)
                min_done_reports.put(trace.worker_id)
            if worker is not None:
                # Drop the final Python reference in the same thread that owns
                # and used the Highs object.  Stored errors were sanitized above.
                solver = worker.solver
                del worker.solver
                del worker
                del solver
                trace.solver_released = True
            trace.exited = True

    threads = tuple(
        threading.Thread(
            target=worker_target,
            args=(trace,),
            name=f"fluxemu-vffva-{trace.worker_id}",
        )
        for trace in traces
    )
    started_threads: list[threading.Thread] = []
    max_queue_accounted = False
    min_released_after_max = False
    coordination_failed = False
    try:
        for thread in threads:
            try:
                thread.start()
                started_threads.append(thread)
            except BaseException as error:  # pragma: no cover - platform failure
                record_failure(error, f"could not start FVA worker {thread.name}")
        for _ in started_threads:
            ready_reports.get()
        start_max.set()
        for _ in started_threads:
            max_done_reports.get()
        if not started_threads:
            while True:
                try:
                    max_work.get_nowait()
                except Empty:
                    break
                else:
                    max_work.task_done()
        max_work.join()
        max_queue_accounted = True
        start_min.set()
        min_released_after_max = True
        for _ in started_threads:
            min_done_reports.get()
        if not started_threads:
            while True:
                try:
                    min_work.get_nowait()
                except Empty:
                    break
                else:
                    min_work.task_done()
        min_work.join()
    except BaseException as error:  # pragma: no cover - coordinating interruption
        coordination_failed = True
        record_failure(error, "FVA worker coordination failed")
    finally:
        if coordination_failed:
            cancelled.set()
        start_max.set()
        start_min.set()
        for thread in started_threads:
            thread.join()
        # Every taken chunk is balanced with task_done, including cancellation.
        max_work.join()
        min_work.join()
        _populate_vffva_instrumentation(
            instrumentation,
            traces,
            started_threads,
            configured_workers=workers,
            chunk_size=chunk_size,
            max_queue_accounted=max_queue_accounted,
            min_released_after_max=min_released_after_max,
        )

    if failures:
        raise AnalysisError(str(failures[0])) from None

    tasks = {
        (reaction_index, direction)
        for direction in ("max", "min")
        for reaction_index in range(reaction_count)
    }
    minima = [math.nan] * reaction_count
    maxima = [math.nan] * reaction_count
    seen: set[tuple[int, str]] = set()
    for result in results:
        if (
            not isinstance(result, _EndpointResult)
            or not 0 <= result.reaction_index < len(lp.reaction_ids)
            or result.direction not in {"min", "max"}
        ):
            raise AnalysisError("FVA worker returned a malformed endpoint result")
        key = (result.reaction_index, result.direction)
        if key in seen:
            raise AnalysisError(
                f"FVA worker returned duplicate {result.direction}imum for reaction "
                f"{lp.reaction_ids[result.reaction_index]!r}"
            )
        seen.add(key)
        (minima if result.direction == "min" else maxima)[result.reaction_index] = result.value
    if seen != set(tasks) or not all(math.isfinite(v) for v in minima + maxima):
        raise AnalysisError("FVA failed to return every endpoint")
    _canonicalize_fva_endpoints(
        lp.reaction_ids, minima, maxima, tuple(prepared.fba.fluxes)
    )
    ranges = pd.DataFrame({"minimum": minima, "maximum": maxima}, index=lp.reaction_ids)
    return FVAResult(
        ranges,
        prepared.retention.fraction_of_optimum,
        prepared.retention.biological_optimum,
        lp.objective_direction,
        lp.fingerprint,
        _fva_ranges_sha256(ranges, lp.fingerprint),
    )


def run_highs_vffva(model: FluxModel, fraction_of_optimum: float = 1.0, *,
                     workers: int | None = None,
                     chunk_size: int = 50,
                     instrumentation: dict[str, object] | None = None) -> FVAResult:
    """Run reusable native FVA with VFFVA-style shared-memory workers."""

    prepared = prepare_highs_flux_region(model, fraction_of_optimum)
    return run_prepared_highs_vffva(
        prepared,
        workers=workers,
        chunk_size=chunk_size,
        instrumentation=instrumentation,
    )
