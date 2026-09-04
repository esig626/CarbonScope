"""Native complete feasible-state sampling for canonical flux models.

The sampler uses classic hit-and-run in an explicitly constructed affine hull.
It returns complete flux vectors; reaction-wise FVA extrema are used only to
identify numerical geometry and are never assembled or sampled independently.

Hit-and-run has a uniform stationary law on a bounded convex body under its
standard assumptions.  A finite run is not evidence of convergence, so this
module records burn-in, thinning, retries, tolerances, and implementation
versions without claiming that returned draws are mixed or representative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from fluxemu.exceptions import AnalysisError
from fluxemu.execution import CanonicalFluxState
from fluxemu.model.schema import FluxModel
from fluxemu.model.validation import CanonicalModelError, validate_flux_model

from .highs import (
    FEASIBILITY_TOLERANCE,
    OBJECTIVE_TOLERANCE,
    PreparedFluxRegion,
    _highspy,
    _validate_prepared_flux_region,
    prepare_highs_flux_region,
    run_prepared_highs_vffva,
)
from .results import FVAResult


ALGORITHM_NAME = "affine_hit_and_run"
ALGORITHM_VERSION = "1"
RNG_NAME = "numpy.PCG64"
DEFAULT_BURN_IN = 100
DEFAULT_THINNING = 10
DEFAULT_MAX_DIRECTION_ATTEMPTS = 100
RANK_TOLERANCE = 1e-12
DIRECTION_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class NativeFluxStateDiagnostics:
    """Independent validation diagnostics for one complete flux state."""

    sample_id: Any
    valid: bool
    reaction_membership_valid: bool
    reaction_order_valid: bool
    finite_values_valid: bool
    bounds_valid: bool
    mass_balance_valid: bool
    retained_objective_valid: bool
    optimal_face_valid: bool
    duplicate_reactions: tuple[str, ...]
    missing_reactions: tuple[str, ...]
    unexpected_reactions: tuple[str, ...]
    nonfinite_reactions: tuple[str, ...]
    max_lower_bound_violation: float | None
    max_lower_bound_reaction: str | None
    max_upper_bound_violation: float | None
    max_upper_bound_reaction: str | None
    max_mass_balance_residual: float | None
    max_mass_balance_metabolite: str | None
    objective_value: float | None
    retained_objective_violation: float | None
    optimal_face_error: float | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeFluxValidationReport:
    """Aggregate independent validation for an ordered batch of flux states."""

    valid: bool
    sample_count: int
    unique_sample_ids_valid: bool
    reaction_membership_valid: bool
    reaction_order_valid: bool
    finite_values_valid: bool
    bounds_valid: bool
    mass_balance_valid: bool
    retained_objective_valid: bool
    optimal_face_valid: bool
    max_lower_bound_violation: float
    max_upper_bound_violation: float
    max_mass_balance_residual: float
    max_retained_objective_violation: float
    max_optimal_face_error: float
    retained_objective_bound: float
    objective_direction: str
    optimal_face_value: float | None
    bounds_tolerance: float
    mass_balance_tolerance: float
    objective_tolerance: float
    details: tuple[NativeFluxStateDiagnostics, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeFluxSamplingProvenance:
    """Reproducibility and geometry metadata for one native sampling run."""

    algorithm: str
    algorithm_version: str
    seed: int
    rng: str
    fraction_of_optimum: float
    biological_optimum: float
    retained_objective_bound: float
    retained_objective_sense: str
    objective_direction: str
    compiled_lp_fingerprint: str
    flux_model_fingerprint: str
    fva_ranges_fingerprint: str
    reaction_ids: tuple[str, ...]
    burn_in: int
    thinning: int
    total_steps: int
    direction_retries: int
    max_direction_attempts: int
    affine_dimension: int
    numerically_fixed_reactions: tuple[str, ...]
    objective_face: bool
    fixed_range_tolerance: float
    rank_tolerance: float
    direction_tolerance: float
    bounds_tolerance: float
    mass_balance_tolerance: float
    objective_tolerance: float
    center_slack: float
    numpy_version: str
    highs_version: str

    @property
    def model_fingerprint(self) -> str:
        """Return the verified full source-model fingerprint."""

        return self.flux_model_fingerprint


@dataclass(frozen=True, slots=True)
class NativeFluxSamplingResult:
    """Complete canonical draws, validation, and explicit sampler provenance."""

    states: tuple[CanonicalFluxState, ...]
    provenance: NativeFluxSamplingProvenance
    validation: NativeFluxValidationReport

    @property
    def sample_count(self) -> int:
        return len(self.states)

    @property
    def algorithm(self) -> str:
        return self.provenance.algorithm

    @property
    def seed(self) -> int:
        return self.provenance.seed

    @property
    def fraction_of_optimum(self) -> float:
        return self.provenance.fraction_of_optimum

    @property
    def biological_optimum(self) -> float:
        return self.provenance.biological_optimum

    @property
    def retained_objective_bound(self) -> float:
        return self.provenance.retained_objective_bound

    @property
    def objective_direction(self) -> str:
        return self.provenance.objective_direction

    @property
    def reaction_ids(self) -> tuple[str, ...]:
        return self.provenance.reaction_ids

    @property
    def model_fingerprint(self) -> str:
        return self.provenance.flux_model_fingerprint


@dataclass(frozen=True, slots=True)
class _SamplingGeometry:
    center: np.ndarray
    basis: np.ndarray
    fixed_mask: np.ndarray
    objective_face: bool
    center_slack: float


def _validated_nonnegative_tolerance(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise AnalysisError(f"{name} must be a finite nonnegative number") from error
    if not math.isfinite(result) or result < 0.0:
        raise AnalysisError(f"{name} must be a finite nonnegative number")
    return result


def _validated_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise AnalysisError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise AnalysisError(f"{name} must be a finite number")
    return result


def _independent_aggregate(values: Sequence[float], error_message: str) -> float:
    """Stably sum direct model terms without relying on solver helpers."""

    try:
        result = math.fsum(float(value) for value in values)
    except (OverflowError, TypeError, ValueError) as error:
        raise AnalysisError(error_message) from error
    if not math.isfinite(result):
        raise AnalysisError(error_message)
    return result


def _independent_finite_dot(
    coefficients: Sequence[float],
    values: Sequence[float],
    context: str,
) -> float:
    """Stably evaluate sampler geometry without sharing solver validation."""

    if len(coefficients) != len(values):
        raise AnalysisError(f"{context} has inconsistent vector lengths")
    products: list[float] = []
    for coefficient, value in zip(coefficients, values):
        product = float(coefficient) * float(value)
        if not math.isfinite(product):
            raise AnalysisError(f"{context} produced a non-finite product")
        products.append(product)
    try:
        result = math.fsum(products)
    except OverflowError as error:
        raise AnalysisError(f"{context} overflowed during summation") from error
    if not math.isfinite(result):
        raise AnalysisError(f"{context} produced a non-finite result")
    return result


def _validate_model_for_states(
    model: FluxModel,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    try:
        validate_flux_model(model)
    except (CanonicalModelError, AttributeError, TypeError) as error:
        raise AnalysisError(f"invalid canonical flux model: {error}") from error
    if not model.objective.terms:
        raise AnalysisError("invalid canonical flux model: objective must be nonempty")
    stoichiometry: dict[tuple[str, str], float] = {}
    for reaction in model.reactions:
        grouped: dict[str, list[float]] = {}
        for term in reaction.stoichiometric_terms:
            grouped.setdefault(term.metabolite_id, []).append(float(term.coefficient))
        for metabolite_id, coefficients in grouped.items():
            stoichiometry[(reaction.reaction_id, metabolite_id)] = (
                _independent_aggregate(
                    coefficients,
                    "invalid canonical flux model: aggregated stoichiometric "
                    f"coefficient is non-finite for reaction {reaction.reaction_id!r}, "
                    f"metabolite {metabolite_id!r}",
                )
            )
    grouped_objective = {reaction.reaction_id: [] for reaction in model.reactions}
    for term in model.objective.terms:
        grouped_objective[term.reaction_id].append(float(term.coefficient))
    objective = {
        reaction_id: _independent_aggregate(
            coefficients,
            (
                "invalid canonical flux model: aggregated objective coefficient "
                f"is non-finite for reaction {reaction_id!r}"
            ),
        )
        for reaction_id, coefficients in grouped_objective.items()
    }
    return objective, stoichiometry


def _ordered_duplicates(identifiers: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for identifier in identifiers:
        if identifier in seen and identifier not in duplicates:
            duplicates.append(identifier)
        seen.add(identifier)
    return tuple(duplicates)


def _maximum_optional(
    diagnostics: Sequence[NativeFluxStateDiagnostics], attribute: str
) -> float:
    values = [
        float(value)
        for diagnostic in diagnostics
        if (value := getattr(diagnostic, attribute)) is not None
    ]
    return max(values, default=0.0)


def _sample_ids_are_unique(states: Sequence[CanonicalFluxState]) -> bool:
    try:
        return len({state.sample_id for state in states}) == len(states)
    except (TypeError, ValueError):
        return False


def validate_flux_states(
    model: FluxModel,
    states: Sequence[CanonicalFluxState],
    *,
    retained_objective_bound: float,
    objective_direction: str,
    optimal_face_value: float | None = None,
    bounds_tolerance: float = FEASIBILITY_TOLERANCE,
    mass_balance_tolerance: float = FEASIBILITY_TOLERANCE,
    objective_tolerance: float = OBJECTIVE_TOLERANCE,
) -> NativeFluxValidationReport:
    """Independently validate complete states directly against ``FluxModel``.

    This validator intentionally does not call sampler geometry helpers and
    does not canonicalise inputs.  A state with correct membership but the
    wrong declared order is reported invalid in that supplied order.
    """

    objective_coefficients, stoichiometric_coefficients = _validate_model_for_states(
        model
    )
    expected_direction = {
        "maximise": "max",
        "minimise": "min",
    }[model.objective.direction]
    if objective_direction not in {"max", "min"}:
        raise AnalysisError("objective_direction must be exactly 'max' or 'min'")
    if objective_direction != expected_direction:
        raise AnalysisError(
            "objective_direction is inconsistent with the canonical model objective"
        )
    retained_bound = _validated_finite_number(
        retained_objective_bound, "retained_objective_bound"
    )
    face_value = (
        None
        if optimal_face_value is None
        else _validated_finite_number(optimal_face_value, "optimal_face_value")
    )
    bound_tolerance = _validated_nonnegative_tolerance(
        bounds_tolerance, "bounds_tolerance"
    )
    balance_tolerance = _validated_nonnegative_tolerance(
        mass_balance_tolerance, "mass_balance_tolerance"
    )
    retained_tolerance = _validated_nonnegative_tolerance(
        objective_tolerance, "objective_tolerance"
    )
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise AnalysisError("states must be a nonempty sequence of CanonicalFluxState records")
    state_tuple = tuple(states)
    if not state_tuple or not all(
        isinstance(state, CanonicalFluxState) for state in state_tuple
    ):
        raise AnalysisError("states must be a nonempty sequence of CanonicalFluxState records")

    expected_ids = tuple(reaction.reaction_id for reaction in model.reactions)
    expected_set = set(expected_ids)
    balanced_ids = tuple(
        metabolite.metabolite_id
        for metabolite in model.metabolites
        if metabolite.steady_state_balanced
    )
    details: list[NativeFluxStateDiagnostics] = []
    for state in state_tuple:
        errors: list[str] = []
        duplicate_reactions: tuple[str, ...] = ()
        missing_reactions: tuple[str, ...] = expected_ids
        unexpected_reactions: tuple[str, ...] = ()
        observed_ids: tuple[str, ...] = ()
        raw_pairs_valid = isinstance(state.values, tuple)
        pairs: tuple[tuple[str, object], ...] = ()
        if not raw_pairs_valid:
            errors.append("values must be a tuple of (reaction_id, value) pairs")
        else:
            parsed: list[tuple[str, object]] = []
            malformed = False
            nonstring_identifiers: list[str] = []
            for item in state.values:
                if not isinstance(item, tuple) or len(item) != 2:
                    malformed = True
                    continue
                identifier, value = item
                if not isinstance(identifier, str):
                    malformed = True
                    nonstring_identifiers.append(repr(identifier))
                    continue
                parsed.append((identifier, value))
            if malformed:
                errors.append("values contain malformed reaction/value pairs")
            pairs = tuple(parsed)
            observed_ids = tuple(identifier for identifier, _ in pairs)
            duplicate_reactions = _ordered_duplicates(observed_ids)
            observed_set = set(observed_ids)
            missing_reactions = tuple(
                identifier for identifier in expected_ids if identifier not in observed_set
            )
            unexpected_reactions = tuple(
                identifier for identifier in observed_ids if identifier not in expected_set
            ) + tuple(nonstring_identifiers)

        membership_valid = (
            raw_pairs_valid
            and len(pairs) == len(expected_ids)
            and not duplicate_reactions
            and not missing_reactions
            and not unexpected_reactions
        )
        order_valid = membership_valid and observed_ids == expected_ids
        if duplicate_reactions:
            errors.append(
                "duplicate reaction ID(s): " + ", ".join(duplicate_reactions)
            )
        if missing_reactions:
            errors.append("missing reaction ID(s): " + ", ".join(missing_reactions))
        if unexpected_reactions:
            errors.append(
                "unexpected reaction ID(s): " + ", ".join(unexpected_reactions)
            )
        if membership_valid and not order_valid:
            errors.append("reaction order does not match canonical declared order")

        finite_values_valid = False
        bounds_valid = False
        mass_balance_valid = False
        retained_objective_valid = False
        optimal_face_valid = face_value is None
        nonfinite_reactions: tuple[str, ...] = ()
        max_lower_bound_violation: float | None = None
        max_lower_bound_reaction: str | None = None
        max_upper_bound_violation: float | None = None
        max_upper_bound_reaction: str | None = None
        max_mass_balance_residual: float | None = None
        max_mass_balance_metabolite: str | None = None
        objective_value: float | None = None
        retained_objective_violation: float | None = None
        optimal_face_error: float | None = None

        if membership_valid:
            supplied = {identifier: value for identifier, value in pairs}
            numeric: dict[str, float] = {}
            bad_numeric: list[str] = []
            for reaction_id in expected_ids:
                value = supplied[reaction_id]
                if isinstance(value, bool) or not isinstance(value, Real):
                    bad_numeric.append(reaction_id)
                    continue
                converted = float(value)
                if not math.isfinite(converted):
                    bad_numeric.append(reaction_id)
                    continue
                numeric[reaction_id] = converted
            nonfinite_reactions = tuple(bad_numeric)
            finite_values_valid = not bad_numeric
            if bad_numeric:
                errors.append(
                    "non-finite or non-numeric flux value(s): "
                    + ", ".join(bad_numeric)
                )

            if finite_values_valid:
                lower_values: list[tuple[float, str]] = []
                upper_values: list[tuple[float, str]] = []
                for reaction in model.reactions:
                    value = numeric[reaction.reaction_id]
                    lower_values.append(
                        (max(float(reaction.lower_bound) - value, 0.0), reaction.reaction_id)
                    )
                    upper_values.append(
                        (max(value - float(reaction.upper_bound), 0.0), reaction.reaction_id)
                    )
                max_lower_bound_violation, max_lower_bound_reaction = max(
                    lower_values, key=lambda item: item[0]
                )
                max_upper_bound_violation, max_upper_bound_reaction = max(
                    upper_values, key=lambda item: item[0]
                )
                if max_lower_bound_violation == 0.0:
                    max_lower_bound_reaction = None
                if max_upper_bound_violation == 0.0:
                    max_upper_bound_reaction = None
                bounds_valid = (
                    max_lower_bound_violation <= bound_tolerance
                    and max_upper_bound_violation <= bound_tolerance
                )
                if not bounds_valid:
                    errors.append("reaction-bound violation exceeds tolerance")

                residuals: list[tuple[float, str]] = []
                nonfinite_residuals: list[str] = []
                for metabolite_id in balanced_ids:
                    products: list[float] = []
                    derived_finite = True
                    for reaction in model.reactions:
                        coefficient = stoichiometric_coefficients.get(
                            (reaction.reaction_id, metabolite_id), 0.0
                        )
                        contribution = coefficient * numeric[reaction.reaction_id]
                        if not math.isfinite(contribution):
                            derived_finite = False
                        else:
                            products.append(contribution)
                    try:
                        residual = math.fsum(products) if derived_finite else math.inf
                    except OverflowError:
                        residual = math.inf
                    if math.isfinite(residual):
                        residuals.append((abs(residual), metabolite_id))
                    else:
                        nonfinite_residuals.append(metabolite_id)
                        residuals.append((math.inf, metabolite_id))
                if residuals:
                    max_mass_balance_residual, max_mass_balance_metabolite = max(
                        residuals, key=lambda item: item[0]
                    )
                    if max_mass_balance_residual == 0.0:
                        max_mass_balance_metabolite = None
                else:
                    max_mass_balance_residual = 0.0
                mass_balance_valid = max_mass_balance_residual <= balance_tolerance
                if nonfinite_residuals:
                    errors.append(
                        "derived mass-balance residual is non-finite for metabolite(s): "
                        + ", ".join(nonfinite_residuals)
                    )
                elif not mass_balance_valid:
                    errors.append("steady-state mass-balance residual exceeds tolerance")

                objective_products = [
                    objective_coefficients[reaction_id] * numeric[reaction_id]
                    for reaction_id in expected_ids
                ]
                if all(math.isfinite(value) for value in objective_products):
                    try:
                        objective_value = math.fsum(objective_products)
                    except OverflowError:
                        objective_value = math.inf
                else:
                    objective_value = math.inf
                if not math.isfinite(objective_value):
                    retained_objective_violation = math.inf
                    retained_objective_valid = False
                    if face_value is not None:
                        optimal_face_error = math.inf
                        optimal_face_valid = False
                    errors.append("derived biological objective is non-finite")
                else:
                    retained_objective_violation = (
                        max(retained_bound - objective_value, 0.0)
                        if objective_direction == "max"
                        else max(objective_value - retained_bound, 0.0)
                    )
                    retained_objective_valid = (
                        retained_objective_violation <= retained_tolerance
                    )
                    if not retained_objective_valid:
                        errors.append(
                            "retained biological objective violation exceeds tolerance"
                        )
                    if face_value is not None:
                        optimal_face_error = abs(objective_value - face_value)
                        optimal_face_valid = optimal_face_error <= retained_tolerance
                        if not optimal_face_valid:
                            errors.append("optimal-face objective error exceeds tolerance")

        state_valid = all(
            (
                membership_valid,
                order_valid,
                finite_values_valid,
                bounds_valid,
                mass_balance_valid,
                retained_objective_valid,
                optimal_face_valid,
            )
        )
        details.append(
            NativeFluxStateDiagnostics(
                sample_id=state.sample_id,
                valid=state_valid,
                reaction_membership_valid=membership_valid,
                reaction_order_valid=order_valid,
                finite_values_valid=finite_values_valid,
                bounds_valid=bounds_valid,
                mass_balance_valid=mass_balance_valid,
                retained_objective_valid=retained_objective_valid,
                optimal_face_valid=optimal_face_valid,
                duplicate_reactions=duplicate_reactions,
                missing_reactions=missing_reactions,
                unexpected_reactions=unexpected_reactions,
                nonfinite_reactions=nonfinite_reactions,
                max_lower_bound_violation=max_lower_bound_violation,
                max_lower_bound_reaction=max_lower_bound_reaction,
                max_upper_bound_violation=max_upper_bound_violation,
                max_upper_bound_reaction=max_upper_bound_reaction,
                max_mass_balance_residual=max_mass_balance_residual,
                max_mass_balance_metabolite=max_mass_balance_metabolite,
                objective_value=objective_value,
                retained_objective_violation=retained_objective_violation,
                optimal_face_error=optimal_face_error,
                errors=tuple(errors),
            )
        )

    detail_tuple = tuple(details)
    unique_sample_ids_valid = _sample_ids_are_unique(state_tuple)
    aggregate_errors: list[str] = []
    if not unique_sample_ids_valid:
        aggregate_errors.append("canonical flux-state sample IDs must be unique and hashable")
    invalid_ids = [repr(item.sample_id) for item in detail_tuple if not item.valid]
    if invalid_ids:
        aggregate_errors.append("invalid flux state(s): " + ", ".join(invalid_ids))
    membership_valid = all(item.reaction_membership_valid for item in detail_tuple)
    order_valid = all(item.reaction_order_valid for item in detail_tuple)
    finite_valid = all(item.finite_values_valid for item in detail_tuple)
    bounds_valid = all(item.bounds_valid for item in detail_tuple)
    balance_valid = all(item.mass_balance_valid for item in detail_tuple)
    objective_valid = all(item.retained_objective_valid for item in detail_tuple)
    face_valid = all(item.optimal_face_valid for item in detail_tuple)
    return NativeFluxValidationReport(
        valid=(
            unique_sample_ids_valid
            and membership_valid
            and order_valid
            and finite_valid
            and bounds_valid
            and balance_valid
            and objective_valid
            and face_valid
        ),
        sample_count=len(detail_tuple),
        unique_sample_ids_valid=unique_sample_ids_valid,
        reaction_membership_valid=membership_valid,
        reaction_order_valid=order_valid,
        finite_values_valid=finite_valid,
        bounds_valid=bounds_valid,
        mass_balance_valid=balance_valid,
        retained_objective_valid=objective_valid,
        optimal_face_valid=face_valid,
        max_lower_bound_violation=_maximum_optional(
            detail_tuple, "max_lower_bound_violation"
        ),
        max_upper_bound_violation=_maximum_optional(
            detail_tuple, "max_upper_bound_violation"
        ),
        max_mass_balance_residual=_maximum_optional(
            detail_tuple, "max_mass_balance_residual"
        ),
        max_retained_objective_violation=_maximum_optional(
            detail_tuple, "retained_objective_violation"
        ),
        max_optimal_face_error=_maximum_optional(detail_tuple, "optimal_face_error"),
        retained_objective_bound=retained_bound,
        objective_direction=objective_direction,
        optimal_face_value=face_value,
        bounds_tolerance=bound_tolerance,
        mass_balance_tolerance=balance_tolerance,
        objective_tolerance=retained_tolerance,
        details=detail_tuple,
        errors=tuple(aggregate_errors),
    )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AnalysisError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalysisError(f"{name} must be a nonnegative integer")
    return value


def _validate_fva_result(
    prepared: PreparedFluxRegion, fva: FVAResult
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fva, FVAResult) or not isinstance(fva.ranges, pd.DataFrame):
        raise AnalysisError("prepared sampling requires a native FVAResult")
    if tuple(fva.ranges.index) != prepared.lp.reaction_ids:
        raise AnalysisError("FVA reaction order does not match the prepared compiled LP")
    if tuple(fva.ranges.columns) != ("minimum", "maximum"):
        raise AnalysisError("FVA ranges must contain exactly minimum and maximum columns")
    try:
        fva_fraction = _validated_finite_number(
            fva.fraction_of_optimum, "FVA fraction_of_optimum"
        )
        fva_optimum = _validated_finite_number(
            fva.objective_value, "FVA objective_value"
        )
    except AnalysisError as error:
        raise AnalysisError(
            "FVA retained-objective metadata contains malformed numbers"
        ) from error
    if (
        fva_fraction != prepared.retention.fraction_of_optimum
        or fva_optimum != prepared.retention.biological_optimum
        or fva.objective_direction != prepared.lp.objective_direction
    ):
        raise AnalysisError("FVA retained-objective metadata is inconsistent")
    minima: list[float] = []
    maxima: list[float] = []
    for index, reaction_id in enumerate(prepared.lp.reaction_ids):
        try:
            raw_minimum = fva.ranges.iat[index, 0]
            raw_maximum = fva.ranges.iat[index, 1]
            if isinstance(raw_minimum, bool) or isinstance(raw_maximum, bool):
                raise TypeError
            minimum = float(raw_minimum)
            maximum = float(raw_maximum)
        except (TypeError, ValueError, OverflowError) as error:
            raise AnalysisError(
                f"FVA range for reaction {reaction_id!r} is malformed"
            ) from error
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise AnalysisError(
                f"FVA range for reaction {reaction_id!r} is non-finite or unbounded"
            )
        if minimum > maximum + FEASIBILITY_TOLERANCE:
            raise AnalysisError(
                f"FVA minimum exceeds maximum for reaction {reaction_id!r}"
            )
        width = maximum - minimum
        if not math.isfinite(width):
            raise AnalysisError(
                f"FVA range width for reaction {reaction_id!r} is non-finite "
                "or numerically unrepresentable"
            )
        lower = prepared.lp.lower_bounds[index]
        upper = prepared.lp.upper_bounds[index]
        if (
            minimum < lower - FEASIBILITY_TOLERANCE
            or maximum > upper + FEASIBILITY_TOLERANCE
        ):
            raise AnalysisError(
                f"FVA range for reaction {reaction_id!r} violates canonical bounds"
            )
        fba_value = float(prepared.fba.fluxes.iloc[index])
        if (
            fba_value < minimum - FEASIBILITY_TOLERANCE
            or fba_value > maximum + FEASIBILITY_TOLERANCE
        ):
            raise AnalysisError(
                f"FVA range for reaction {reaction_id!r} does not bracket the "
                "prepared feasible FBA primal"
            )
        minima.append(minimum)
        maxima.append(maximum)
    return np.asarray(minima, dtype=float), np.asarray(maxima, dtype=float)


def _fva_ranges_fingerprint(
    prepared: PreparedFluxRegion,
    fva: FVAResult,
    minima: np.ndarray,
    maxima: np.ndarray,
) -> str:
    """Bind provenance to the exact ordered FVA geometry used by sampling."""

    identity = json.dumps(
        [
            prepared.lp.reaction_ids,
            tuple(
                (float(minimum).hex(), float(maximum).hex())
                for minimum, maximum in zip(minima, maxima, strict=True)
            ),
            float(fva.fraction_of_optimum).hex(),
            float(fva.objective_value).hex(),
            fva.objective_direction,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _dense_stoichiometry(prepared: PreparedFluxRegion) -> np.ndarray:
    lp = prepared.lp
    matrix = np.zeros(
        (len(lp.balanced_metabolite_ids), len(lp.reaction_ids)), dtype=float
    )
    for row in range(len(lp.balanced_metabolite_ids)):
        for offset in range(lp.row_starts[row], lp.row_starts[row + 1]):
            matrix[row, lp.column_indices[offset]] += lp.coefficients[offset]
    return matrix


def _objective_is_face(prepared: PreparedFluxRegion) -> bool:
    retention = prepared.retention
    margin = (
        retention.biological_optimum - retention.bound
        if retention.objective_direction == "max"
        else retention.bound - retention.biological_optimum
    )
    if margin < 0.0:
        raise AnalysisError("prepared retained-objective region is empty")
    # Objective retention is an equality only by recorded algebra, never merely
    # because the objective scale is small relative to a solver tolerance.
    return (
        retention.fraction_of_optimum == 1.0
        or retention.bound == retention.biological_optimum
    )


def _numerically_fixed_mask(minima: np.ndarray, maxima: np.ndarray) -> np.ndarray:
    """Identify only machine-scale FVA widths as fixed directions.

    An absolute feasibility tolerance would incorrectly erase legitimate
    low-scale dimensions (for example a width of ``5e-9``).  The floor here is
    the recorded direction tolerance, augmented only by roundoff at the scale
    of each endpoint.
    """

    scales = np.maximum(1.0, np.maximum(np.abs(minima), np.abs(maxima)))
    thresholds = np.maximum(
        DIRECTION_TOLERANCE,
        64.0 * np.finfo(float).eps * scales,
    )
    return maxima - minima <= thresholds


def _affine_basis(
    prepared: PreparedFluxRegion,
    fixed_mask: np.ndarray,
    objective_face: bool,
) -> np.ndarray:
    n = len(prepared.lp.reaction_ids)
    rows: list[np.ndarray] = []
    stoichiometry = _dense_stoichiometry(prepared)
    rows.extend(stoichiometry[row].copy() for row in range(stoichiometry.shape[0]))
    for index in np.flatnonzero(fixed_mask):
        row = np.zeros(n, dtype=float)
        row[int(index)] = 1.0
        rows.append(row)
    if objective_face:
        rows.append(np.asarray(prepared.lp.objective_coefficients, dtype=float))
    normalized: list[np.ndarray] = []
    for row in rows:
        norm = float(np.linalg.norm(row))
        if not math.isfinite(norm):
            raise AnalysisError("affine-hull equality matrix contains non-finite values")
        if norm > RANK_TOLERANCE:
            normalized.append(row / norm)
    if not normalized:
        return np.eye(n, dtype=float)
    equality_matrix = np.vstack(normalized)
    try:
        _, singular_values, right_vectors = np.linalg.svd(
            equality_matrix, full_matrices=True
        )
    except np.linalg.LinAlgError as error:
        raise AnalysisError("failed to factor native sampling affine hull") from error
    leading = float(singular_values[0]) if singular_values.size else 1.0
    threshold = RANK_TOLERANCE * max(equality_matrix.shape) * max(leading, 1.0)
    rank = int(np.count_nonzero(singular_values > threshold))
    basis = np.asarray(right_vectors[rank:, :].T, dtype=float)
    if basis.shape != (n, n - rank) or not np.all(np.isfinite(basis)):
        raise AnalysisError("native sampling affine basis is malformed")
    if basis.size:
        residual = float(np.max(np.abs(equality_matrix @ basis)))
        if not math.isfinite(residual) or residual > FEASIBILITY_TOLERANCE:
            raise AnalysisError(
                "native sampling affine basis failed independent equality validation"
            )
    return basis


def _append_sparse_row(
    starts: list[int],
    indices: list[int],
    values: list[float],
    coefficients: Sequence[float],
) -> None:
    for index, value in enumerate(coefficients):
        numeric = float(value)
        if numeric != 0.0:
            indices.append(index)
            values.append(numeric)
    starts.append(len(indices))


def _interior_center(
    prepared: PreparedFluxRegion,
    minima: np.ndarray,
    maxima: np.ndarray,
    fixed_mask: np.ndarray,
    objective_face: bool,
) -> tuple[np.ndarray, float]:
    """Find a deterministic relative-interior seed without altering it afterward."""

    highspy = _highspy()
    lp = prepared.lp
    n = len(lp.reaction_ids)
    slack_index = n
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("threads", 1)
    solver.setOptionValue("solver", "simplex")
    costs = [0.0] * n + [1.0]
    lower_bounds = list(lp.lower_bounds) + [0.0]
    upper_bounds = list(lp.upper_bounds) + [1.0]
    solver.addCols(
        n + 1,
        costs,
        lower_bounds,
        upper_bounds,
        0,
        [0] * (n + 2),
        [],
        [],
    )

    row_lower: list[float] = []
    row_upper: list[float] = []
    starts: list[int] = [0]
    indices: list[int] = []
    values: list[float] = []
    stoichiometry = _dense_stoichiometry(prepared)
    for row in range(stoichiometry.shape[0]):
        row_lower.append(0.0)
        row_upper.append(0.0)
        _append_sparse_row(starts, indices, values, stoichiometry[row])

    fba_values = np.asarray(tuple(float(value) for value in prepared.fba.fluxes), dtype=float)
    for index in np.flatnonzero(fixed_mask):
        coefficients = np.zeros(n + 1, dtype=float)
        coefficients[int(index)] = 1.0
        value = float(fba_values[int(index)])
        row_lower.append(value)
        row_upper.append(value)
        _append_sparse_row(starts, indices, values, coefficients)

    objective = np.asarray(lp.objective_coefficients, dtype=float)
    if objective_face:
        coefficients = np.zeros(n + 1, dtype=float)
        coefficients[:n] = objective
        optimum = prepared.retention.biological_optimum
        row_lower.append(optimum)
        row_upper.append(optimum)
        _append_sparse_row(starts, indices, values, coefficients)

    for index in np.flatnonzero(~fixed_mask):
        column = int(index)
        width = float(maxima[column] - minima[column])
        lower_row = np.zeros(n + 1, dtype=float)
        lower_row[column] = 1.0
        lower_row[slack_index] = -width
        row_lower.append(float(minima[column]))
        row_upper.append(highspy.kHighsInf)
        _append_sparse_row(starts, indices, values, lower_row)
        upper_row = np.zeros(n + 1, dtype=float)
        upper_row[column] = 1.0
        upper_row[slack_index] = width
        row_lower.append(-highspy.kHighsInf)
        row_upper.append(float(maxima[column]))
        _append_sparse_row(starts, indices, values, upper_row)

    if not objective_face:
        margin = (
            prepared.retention.biological_optimum - prepared.retention.bound
            if lp.objective_direction == "max"
            else prepared.retention.bound - prepared.retention.biological_optimum
        )
        coefficients = np.zeros(n + 1, dtype=float)
        coefficients[:n] = objective
        if lp.objective_direction == "max":
            coefficients[slack_index] = -margin
            row_lower.append(prepared.retention.bound)
            row_upper.append(highspy.kHighsInf)
        else:
            coefficients[slack_index] = margin
            row_lower.append(-highspy.kHighsInf)
            row_upper.append(prepared.retention.bound)
        _append_sparse_row(starts, indices, values, coefficients)

    solver.addRows(
        len(row_lower),
        row_lower,
        row_upper,
        len(indices),
        starts,
        indices,
        values,
    )
    solver.setMaximize()
    solver.run()
    status = solver.getModelStatus()
    status_name = solver.modelStatusToString(status)
    if status != highspy.HighsModelStatus.kOptimal:
        lowered = status_name.lower()
        category = (
            "infeasible"
            if "infeasible" in lowered
            else "unbounded"
            if "unbounded" in lowered
            else "solver failure/unknown"
        )
        raise AnalysisError(
            "native sampling center solve failed: "
            f"HiGHS status {status_name} ({category})"
        )
    solution = tuple(float(value) for value in solver.getSolution().col_value)
    if len(solution) != n + 1 or not all(math.isfinite(value) for value in solution):
        raise AnalysisError("native sampling center solve returned a malformed solution")
    center = np.asarray(solution[:n], dtype=float)
    center_slack = solution[slack_index]
    if center_slack <= DIRECTION_TOLERANCE:
        raise AnalysisError(
            "native sampling region has no numerically resolvable relative interior"
        )
    center_state = CanonicalFluxState(
        "native-center",
        tuple(zip(lp.reaction_ids, (float(value) for value in center), strict=True)),
    )
    center_report = validate_flux_states(
        prepared.flux_model,
        (center_state,),
        retained_objective_bound=prepared.retention.bound,
        objective_direction=lp.objective_direction,
        optimal_face_value=(
            prepared.retention.biological_optimum if objective_face else None
        ),
    )
    if not center_report.valid:
        raise AnalysisError(
            "native sampling center failed independent validation: "
            + "; ".join(center_report.details[0].errors)
        )
    return center, float(center_slack)


def _build_geometry(
    prepared: PreparedFluxRegion,
    minima: np.ndarray,
    maxima: np.ndarray,
) -> _SamplingGeometry:
    fixed_mask = _numerically_fixed_mask(minima, maxima)
    objective_face = _objective_is_face(prepared)
    basis = _affine_basis(prepared, fixed_mask, objective_face)
    if basis.shape[1] == 0:
        center = np.asarray(
            tuple(float(value) for value in prepared.fba.fluxes), dtype=float
        )
        center_slack = 0.0
        center_state = CanonicalFluxState(
            "native-center",
            tuple(
                zip(
                    prepared.lp.reaction_ids,
                    (float(value) for value in center),
                    strict=True,
                )
            ),
        )
        report = validate_flux_states(
            prepared.flux_model,
            (center_state,),
            retained_objective_bound=prepared.retention.bound,
            objective_direction=prepared.lp.objective_direction,
            optimal_face_value=(
                prepared.retention.biological_optimum if objective_face else None
            ),
        )
        if not report.valid:
            raise AnalysisError(
                "zero-dimensional native sampling state failed independent validation: "
                + "; ".join(report.details[0].errors)
            )
    else:
        center, center_slack = _interior_center(
            prepared, minima, maxima, fixed_mask, objective_face
        )
    return _SamplingGeometry(
        center=center,
        basis=basis,
        fixed_mask=fixed_mask,
        objective_face=objective_face,
        center_slack=center_slack,
    )


def _draw_affine_direction(
    rng: np.random.Generator, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    coefficients = np.asarray(rng.standard_normal(basis.shape[1]), dtype=float)
    norm = float(np.linalg.norm(coefficients))
    if not math.isfinite(norm) or norm <= DIRECTION_TOLERANCE:
        return None
    coefficients /= norm
    direction = basis @ coefficients
    direction_norm = float(np.linalg.norm(direction))
    if not math.isfinite(direction_norm) or direction_norm <= DIRECTION_TOLERANCE:
        return None
    return coefficients / direction_norm, direction / direction_norm


def _line_chord(
    prepared: PreparedFluxRegion,
    point: np.ndarray,
    direction: np.ndarray,
    *,
    objective_face: bool,
) -> tuple[float, float]:
    lower_step = -math.inf
    upper_step = math.inf
    for value, component, lower, upper in zip(
        point,
        direction,
        prepared.lp.lower_bounds,
        prepared.lp.upper_bounds,
        strict=True,
    ):
        if abs(float(component)) <= DIRECTION_TOLERANCE:
            if (
                value < lower - FEASIBILITY_TOLERANCE
                or value > upper + FEASIBILITY_TOLERANCE
            ):
                raise AnalysisError("current hit-and-run point violates a reaction bound")
            continue
        first = (lower - float(value)) / float(component)
        second = (upper - float(value)) / float(component)
        lower_step = max(lower_step, min(first, second))
        upper_step = min(upper_step, max(first, second))

    if not objective_face:
        objective = prepared.lp.objective_coefficients
        value = _independent_finite_dot(
            objective,
            point,
            "hit-and-run retained objective at current point",
        )
        slope = _independent_finite_dot(
            objective,
            direction,
            "hit-and-run retained-objective direction",
        )
        bound = prepared.retention.bound
        if abs(slope) <= DIRECTION_TOLERANCE:
            violates = (
                value < bound - OBJECTIVE_TOLERANCE
                if prepared.lp.objective_direction == "max"
                else value > bound + OBJECTIVE_TOLERANCE
            )
            if violates:
                raise AnalysisError(
                    "current hit-and-run point violates the retained objective"
                )
        else:
            crossing = (bound - value) / slope
            if prepared.lp.objective_direction == "max":
                if slope > 0.0:
                    lower_step = max(lower_step, crossing)
                else:
                    upper_step = min(upper_step, crossing)
            elif slope > 0.0:
                upper_step = min(upper_step, crossing)
            else:
                lower_step = max(lower_step, crossing)

    if not math.isfinite(lower_step) or not math.isfinite(upper_step):
        raise AnalysisError(
            "native sampling encountered an unbounded hit-and-run direction; "
            "arbitrary truncation is not permitted"
        )
    return lower_step, upper_step


def _highs_version() -> str:
    highspy = _highspy()
    parts = (
        getattr(highspy, "HIGHS_VERSION_MAJOR", None),
        getattr(highspy, "HIGHS_VERSION_MINOR", None),
        getattr(highspy, "HIGHS_VERSION_PATCH", None),
    )
    if all(isinstance(part, int) for part in parts):
        return ".".join(str(part) for part in parts)
    return str(getattr(highspy, "__version__", "unknown"))


def sample_prepared_highs_flux_states(
    prepared: PreparedFluxRegion,
    fva: FVAResult,
    count: int,
    *,
    seed: int = 0,
    burn_in: int = DEFAULT_BURN_IN,
    thinning: int = DEFAULT_THINNING,
    max_direction_attempts: int = DEFAULT_MAX_DIRECTION_ATTEMPTS,
) -> NativeFluxSamplingResult:
    """Sample one prepared bounded retained-objective region.

    Directions are uniform on the numerical affine-hull unit sphere and each
    accepted move is uniform on its exact bounded chord.  Burn-in and thinning
    are operational parameters only; no finite-run mixing guarantee is made.
    """

    sample_count = _positive_integer(count, "count")
    random_seed = _nonnegative_integer(seed, "seed")
    burn = _nonnegative_integer(burn_in, "burn_in")
    thin = _positive_integer(thinning, "thinning")
    attempt_limit = _positive_integer(
        max_direction_attempts, "max_direction_attempts"
    )
    _validate_prepared_flux_region(prepared)
    minima, maxima = _validate_fva_result(prepared, fva)
    fva_fingerprint = _fva_ranges_fingerprint(prepared, fva, minima, maxima)
    geometry = _build_geometry(prepared, minima, maxima)
    affine_dimension = int(geometry.basis.shape[1])

    states: list[CanonicalFluxState] = []
    direction_retries = 0
    total_steps = 0
    if affine_dimension == 0:
        values = tuple(
            zip(
                prepared.lp.reaction_ids,
                (float(value) for value in geometry.center),
                strict=True,
            )
        )
        states = [
            CanonicalFluxState(f"native-sample-{index:06d}", values)
            for index in range(sample_count)
        ]
    else:
        rng = np.random.Generator(np.random.PCG64(random_seed))
        coordinates = np.zeros(affine_dimension, dtype=float)
        required_steps = burn + sample_count * thin
        for step in range(1, required_steps + 1):
            point = geometry.center + geometry.basis @ coordinates
            accepted: tuple[np.ndarray, np.ndarray, float, float] | None = None
            for attempt in range(attempt_limit):
                drawn = _draw_affine_direction(rng, geometry.basis)
                if drawn is None:
                    direction_retries += 1
                    continue
                affine_direction, flux_direction = drawn
                lower_step, upper_step = _line_chord(
                    prepared,
                    point,
                    flux_direction,
                    objective_face=geometry.objective_face,
                )
                chord_width = upper_step - lower_step
                if not math.isfinite(chord_width):
                    raise AnalysisError(
                        "native affine hit-and-run encountered a non-finite or "
                        f"numerically unrepresentable chord span at step {step}"
                    )
                if chord_width <= DIRECTION_TOLERANCE:
                    direction_retries += 1
                    continue
                accepted = (
                    affine_direction,
                    flux_direction,
                    lower_step,
                    upper_step,
                )
                break
            if accepted is None:
                raise AnalysisError(
                    "native affine hit-and-run could not generate a valid next state "
                    f"at step {step} after {attempt_limit} direction attempts; "
                    "all candidate chords were numerically degenerate"
                )
            affine_direction, _, lower_step, upper_step = accepted
            try:
                distance = float(rng.uniform(lower_step, upper_step))
            except (FloatingPointError, OverflowError, ValueError) as error:
                raise AnalysisError(
                    "native affine hit-and-run could not draw a finite chord step "
                    f"at step {step}: {error}"
                ) from error
            if not math.isfinite(distance):
                raise AnalysisError(
                    f"native affine hit-and-run produced a non-finite step at step {step}"
                )
            coordinates = coordinates + distance * affine_direction
            point = geometry.center + geometry.basis @ coordinates
            step_state = CanonicalFluxState(
                f"native-step-{step}",
                tuple(
                    zip(
                        prepared.lp.reaction_ids,
                        (float(value) for value in point),
                        strict=True,
                    )
                ),
            )
            step_report = validate_flux_states(
                prepared.flux_model,
                (step_state,),
                retained_objective_bound=prepared.retention.bound,
                objective_direction=prepared.lp.objective_direction,
                optimal_face_value=(
                    prepared.retention.biological_optimum
                    if geometry.objective_face
                    else None
                ),
            )
            if not step_report.valid:
                raise AnalysisError(
                    "native affine hit-and-run generated an invalid state "
                    f"at step {step}: "
                    + "; ".join(step_report.details[0].errors)
                )
            if step > burn and (step - burn) % thin == 0:
                states.append(
                    CanonicalFluxState(
                        f"native-sample-{len(states):06d}", step_state.values
                    )
                )
            total_steps = step

    state_tuple = tuple(states)
    validation = validate_flux_states(
        prepared.flux_model,
        state_tuple,
        retained_objective_bound=prepared.retention.bound,
        objective_direction=prepared.lp.objective_direction,
        optimal_face_value=(
            prepared.retention.biological_optimum if geometry.objective_face else None
        ),
    )
    if not validation.valid:
        raise AnalysisError(
            "native sampled flux batch failed independent validation: "
            + "; ".join(validation.errors)
        )
    fixed_reactions = tuple(
        reaction_id
        for reaction_id, fixed in zip(
            prepared.lp.reaction_ids, geometry.fixed_mask, strict=True
        )
        if bool(fixed)
    )
    provenance = NativeFluxSamplingProvenance(
        algorithm=ALGORITHM_NAME,
        algorithm_version=ALGORITHM_VERSION,
        seed=random_seed,
        rng=RNG_NAME,
        fraction_of_optimum=prepared.retention.fraction_of_optimum,
        biological_optimum=prepared.retention.biological_optimum,
        retained_objective_bound=prepared.retention.bound,
        retained_objective_sense=prepared.retention.sense,
        objective_direction=prepared.lp.objective_direction,
        compiled_lp_fingerprint=prepared.lp.fingerprint,
        flux_model_fingerprint=prepared.flux_model_fingerprint,
        fva_ranges_fingerprint=fva_fingerprint,
        reaction_ids=prepared.lp.reaction_ids,
        burn_in=burn,
        thinning=thin,
        total_steps=total_steps,
        direction_retries=direction_retries,
        max_direction_attempts=attempt_limit,
        affine_dimension=affine_dimension,
        numerically_fixed_reactions=fixed_reactions,
        objective_face=geometry.objective_face,
        fixed_range_tolerance=DIRECTION_TOLERANCE,
        rank_tolerance=RANK_TOLERANCE,
        direction_tolerance=DIRECTION_TOLERANCE,
        bounds_tolerance=validation.bounds_tolerance,
        mass_balance_tolerance=validation.mass_balance_tolerance,
        objective_tolerance=validation.objective_tolerance,
        center_slack=geometry.center_slack,
        numpy_version=np.__version__,
        highs_version=_highs_version(),
    )
    return NativeFluxSamplingResult(state_tuple, provenance, validation)


def _reject_nonfinite_sampling_bounds(model: FluxModel) -> None:
    if not isinstance(model, FluxModel):
        return
    for reaction in model.reactions:
        for name, value in (
            ("lower", reaction.lower_bound),
            ("upper", reaction.upper_bound),
        ):
            if isinstance(value, Real) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    raise AnalysisError(
                        "native sampling requires finite reaction bounds; "
                        f"reaction {reaction.reaction_id!r} has a non-finite {name} "
                        "bound and may make the sampled region unbounded"
                    )


def sample_highs_flux_states(
    model: FluxModel,
    count: int,
    fraction_of_optimum: float = 1.0,
    *,
    seed: int = 0,
    burn_in: int = DEFAULT_BURN_IN,
    thinning: int = DEFAULT_THINNING,
    workers: int | None = 1,
    max_direction_attempts: int = DEFAULT_MAX_DIRECTION_ATTEMPTS,
) -> NativeFluxSamplingResult:
    """Prepare, characterize, and sample complete jointly feasible states."""

    _positive_integer(count, "count")
    _nonnegative_integer(seed, "seed")
    _nonnegative_integer(burn_in, "burn_in")
    _positive_integer(thinning, "thinning")
    _positive_integer(max_direction_attempts, "max_direction_attempts")
    if workers is not None:
        _positive_integer(workers, "workers")
    _reject_nonfinite_sampling_bounds(model)
    prepared = prepare_highs_flux_region(model, fraction_of_optimum)
    fva = run_prepared_highs_vffva(prepared, workers=workers)
    return sample_prepared_highs_flux_states(
        prepared,
        fva,
        count,
        seed=seed,
        burn_in=burn_in,
        thinning=thinning,
        max_direction_attempts=max_direction_attempts,
    )


__all__ = [
    "ALGORITHM_NAME",
    "DEFAULT_BURN_IN",
    "DEFAULT_MAX_DIRECTION_ATTEMPTS",
    "DEFAULT_THINNING",
    "NativeFluxSamplingProvenance",
    "NativeFluxSamplingResult",
    "NativeFluxStateDiagnostics",
    "NativeFluxValidationReport",
    "sample_highs_flux_states",
    "sample_prepared_highs_flux_states",
    "validate_flux_states",
]
