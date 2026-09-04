"""Native complete feasible-state sampling over a prepared HiGHS flux region.

The sampler works in an orthonormal basis for the affine feasible hull and
uses random-direction hit-and-run.  FVA ranges are used only to identify
numerically collapsed coordinates and to certify the prepared geometry; they
are never combined or sampled as if they were a complete flux vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from fluxemu.exceptions import AnalysisError
from fluxemu.execution import CanonicalFluxState
from fluxemu.model.schema import FluxModel

from .highs import (
    CompiledFluxLP,
    HIGHS_SMALL_MATRIX_VALUE,
    PreparedFluxRegion,
    _biological_objective_value,
    _condition_affine_equalities,
    _condition_objective_after_fixed_coordinates,
    _declared_objective_value,
    _effective_objective_value,
    _highspy,
    _objective_scale,
    _objective_activity_scale,
    _retained_objective_violation,
    _validate_prepared_flux_region,
    prepare_highs_flux_region,
    run_prepared_highs_vffva,
)
from .results import FVAResult, _fva_ranges_sha256


ALGORITHM_NAME = "random-direction-hit-and-run"
ALGORITHM_VERSION = "1"
RANDOM_BIT_GENERATOR = "PCG64"

BOUND_TOLERANCE = 1e-7
MASS_BALANCE_TOLERANCE = 1e-7
RETAINED_OBJECTIVE_TOLERANCE = 1e-7
FVA_COLLAPSE_TOLERANCE = 1e-10
FVA_DEGENERACY_TOLERANCE = 1e-8
AFFINE_RANK_TOLERANCE = math.sqrt(np.finfo(float).eps)
REDUCED_DIRECTION_TOLERANCE = 1e-14
INTERVAL_TOLERANCE = 1e-12
CENTER_RADIUS_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class FluxStateValidationDiagnostics:
    """Independent numerical diagnostics for one complete canonical state."""

    sample_id: Any
    valid: bool
    errors: tuple[str, ...]
    reaction_membership_valid: bool
    reaction_order_valid: bool
    finite_values_valid: bool
    objective_value: float | None
    stable_objective_value: float | None
    effective_objective_value: float | None
    objective_conditioning_discrepancy: float
    normalized_objective_conditioning_discrepancy: float
    max_lower_bound_violation: float
    lower_bound_reaction_id: str | None
    max_upper_bound_violation: float
    upper_bound_reaction_id: str | None
    max_raw_mass_balance_residual: float
    max_mass_balance_residual: float
    conditioned_row_space_residual: float
    mass_balance_metabolite_id: str | None
    raw_retained_objective_violation: float
    retained_objective_violation: float
    stable_retained_objective_violation: float


@dataclass(frozen=True, slots=True)
class FluxSampleValidationReport:
    """Evidence that a complete batch satisfies the original flux model."""

    valid: bool
    sample_count: int
    model_fingerprint: str
    reaction_order: tuple[str, ...]
    sample_ids_valid: bool
    reaction_membership_valid: bool
    reaction_order_valid: bool
    finite_values_valid: bool
    bounds_valid: bool
    mass_balance_valid: bool
    retained_objective_valid: bool
    max_lower_bound_violation: float
    max_upper_bound_violation: float
    max_raw_mass_balance_residual: float
    max_mass_balance_residual: float
    max_conditioned_row_space_residual: float
    max_objective_conditioning_discrepancy: float
    max_normalized_objective_conditioning_discrepancy: float
    max_raw_retained_objective_violation: float
    max_retained_objective_violation: float
    max_stable_retained_objective_violation: float
    diagnostics: tuple[FluxStateValidationDiagnostics, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible validation summary."""

        return {
            "valid": self.valid,
            "sample_count": self.sample_count,
            "model_fingerprint": self.model_fingerprint,
            "reaction_order": list(self.reaction_order),
            "sample_ids_valid": self.sample_ids_valid,
            "reaction_membership_valid": self.reaction_membership_valid,
            "reaction_order_valid": self.reaction_order_valid,
            "finite_values_valid": self.finite_values_valid,
            "bounds_valid": self.bounds_valid,
            "mass_balance_valid": self.mass_balance_valid,
            "retained_objective_valid": self.retained_objective_valid,
            "max_lower_bound_violation": self.max_lower_bound_violation,
            "max_upper_bound_violation": self.max_upper_bound_violation,
            "max_raw_mass_balance_residual": self.max_raw_mass_balance_residual,
            "max_mass_balance_residual": self.max_mass_balance_residual,
            "max_conditioned_row_space_residual": self.max_conditioned_row_space_residual,
            "max_objective_conditioning_discrepancy": self.max_objective_conditioning_discrepancy,
            "max_normalized_objective_conditioning_discrepancy": (
                self.max_normalized_objective_conditioning_discrepancy
            ),
            "max_raw_retained_objective_violation": self.max_raw_retained_objective_violation,
            "max_retained_objective_violation": self.max_retained_objective_violation,
            "max_stable_retained_objective_violation": (
                self.max_stable_retained_objective_violation
            ),
            "diagnostics": [
                {
                    "sample_id": item.sample_id,
                    "valid": item.valid,
                    "errors": list(item.errors),
                    "reaction_membership_valid": item.reaction_membership_valid,
                    "reaction_order_valid": item.reaction_order_valid,
                    "finite_values_valid": item.finite_values_valid,
                    "objective_value": item.objective_value,
                    "stable_objective_value": item.stable_objective_value,
                    "effective_objective_value": item.effective_objective_value,
                    "objective_conditioning_discrepancy": (
                        item.objective_conditioning_discrepancy
                    ),
                    "normalized_objective_conditioning_discrepancy": (
                        item.normalized_objective_conditioning_discrepancy
                    ),
                    "max_lower_bound_violation": item.max_lower_bound_violation,
                    "lower_bound_reaction_id": item.lower_bound_reaction_id,
                    "max_upper_bound_violation": item.max_upper_bound_violation,
                    "upper_bound_reaction_id": item.upper_bound_reaction_id,
                    "max_raw_mass_balance_residual": item.max_raw_mass_balance_residual,
                    "max_mass_balance_residual": item.max_mass_balance_residual,
                    "conditioned_row_space_residual": item.conditioned_row_space_residual,
                    "mass_balance_metabolite_id": item.mass_balance_metabolite_id,
                    "raw_retained_objective_violation": item.raw_retained_objective_violation,
                    "retained_objective_violation": item.retained_objective_violation,
                    "stable_retained_objective_violation": (
                        item.stable_retained_objective_violation
                    ),
                }
                for item in self.diagnostics
            ],
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class FluxSamplingProvenance:
    """Reproducibility and geometry metadata for one native sampling run."""

    algorithm: str
    algorithm_version: str
    random_bit_generator: str
    seed: int
    sample_count: int
    fraction_of_optimum: float
    biological_optimum: float
    retained_objective_bound: float
    retained_objective_sense: str
    objective_direction: str
    declared_objective_scale: float
    effective_objective_scale: float
    objective_constant: float
    effective_objective_bound: float
    reduced_objective_scale: float
    reduced_objective_constant: float
    reduced_effective_objective_bound: float
    model_fingerprint: str
    fva_ranges_sha256: str
    reaction_order: tuple[str, ...]
    burn_in: int
    thinning: int
    max_direction_attempts: int
    equality_rank: int
    affine_dimension: int
    fixed_reaction_count: int
    collapsed_reaction_ids: tuple[str, ...]
    optimal_face: bool
    center_method: str
    center_radius: float
    accepted_chain_steps: int
    self_loop_steps: int
    rejected_directions: int
    bound_tolerance: float
    mass_balance_tolerance: float
    retained_objective_tolerance: float
    fva_collapse_tolerance: float
    fva_degeneracy_tolerance: float
    affine_rank_tolerance: float
    reduced_direction_tolerance: float
    interval_tolerance: float
    center_radius_tolerance: float
    numpy_version: str
    highs_version: str
    stationary_target: str
    finite_chain_claim: str


@dataclass(frozen=True, slots=True)
class NativeFluxSamplingResult:
    """Ordered complete flux states plus validation and provenance."""

    states: tuple[CanonicalFluxState, ...]
    sample_count: int
    provenance: FluxSamplingProvenance
    validation: FluxSampleValidationReport

    def to_frame(self) -> pd.DataFrame:
        """Return samples with canonical reaction columns and sample IDs as rows."""

        frame = pd.DataFrame(
            [[value for _, value in state.values] for state in self.states],
            index=pd.Index(
                [state.sample_id for state in self.states], name="sample_id"
            ),
            columns=list(self.provenance.reaction_order),
            dtype=float,
        )
        return frame


@dataclass(frozen=True, slots=True)
class _ReducedFluxGeometry:
    particular: np.ndarray
    basis: np.ndarray
    inequality_rows: np.ndarray
    inequality_bounds: np.ndarray
    equality_rank: int
    collapsed_reaction_ids: tuple[str, ...]
    fixed_reaction_count: int
    optimal_face: bool
    objective_scale: float
    objective_constant: float
    effective_objective_bound: float


@dataclass(frozen=True, slots=True)
class _ObjectiveValidationMetrics:
    declared_value: float
    stable_value: float
    effective_value: float
    discrepancy: float
    normalized_discrepancy: float
    direct_violation: float
    normalized_direct_violation: float
    stable_violation: float


def _positive_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AnalysisError(f"{description} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise AnalysisError(f"{description} must be a positive integer")
    return result


def _nonnegative_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AnalysisError(f"{description} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise AnalysisError(f"{description} must be a nonnegative integer")
    return result


def _inspect_state_values(
    lp: CompiledFluxLP, state: object
) -> tuple[Any, tuple[float, ...] | None, bool, bool, bool, tuple[str, ...]]:
    """Inspect raw state structure without reordering or repairing it."""

    if not isinstance(state, CanonicalFluxState):
        error = "flux-state batch must contain CanonicalFluxState records"
        return None, None, False, False, False, (error,)
    sample_id = state.sample_id
    if not isinstance(state.values, tuple):
        error = f"flux state {sample_id!r} values must be a tuple in canonical order"
        return sample_id, None, False, False, False, (error,)

    observed: list[str] = []
    raw_values: list[object] = []
    errors: list[str] = []
    for position, item in enumerate(state.values):
        if not isinstance(item, tuple) or len(item) != 2:
            errors.append(
                f"flux state {sample_id!r} has malformed entry at position {position}"
            )
            continue
        reaction_id, value = item
        if not isinstance(reaction_id, str):
            errors.append(
                f"flux state {sample_id!r} has a non-string reaction ID at "
                f"position {position}"
            )
            continue
        observed.append(reaction_id)
        raw_values.append(value)

    duplicate_ids = tuple(
        reaction_id
        for index, reaction_id in enumerate(observed)
        if reaction_id in observed[:index]
    )
    missing = tuple(
        reaction_id for reaction_id in lp.reaction_ids if reaction_id not in observed
    )
    unexpected = tuple(
        reaction_id for reaction_id in observed if reaction_id not in lp.reaction_ids
    )
    membership_valid = not errors and not duplicate_ids and not missing and not unexpected
    if duplicate_ids or missing or unexpected:
        errors.append(
            f"flux state {sample_id!r} has invalid reaction membership: "
            f"duplicates={duplicate_ids}, missing={missing}, unexpected={unexpected}"
        )
    order_valid = membership_valid and tuple(observed) == lp.reaction_ids
    if membership_valid and not order_valid:
        errors.append(
            f"flux state {sample_id!r} does not preserve exact canonical reaction order"
        )

    values: list[float] = []
    finite_values_valid = True
    for reaction_id, value in zip(observed, raw_values):
        if isinstance(value, bool) or not isinstance(value, Real):
            finite_values_valid = False
            errors.append(
                f"flux state {sample_id!r} has nonnumeric flux for reaction "
                f"{reaction_id!r}"
            )
            continue
        try:
            numeric = float(value)
        except (OverflowError, ValueError):
            finite_values_valid = False
            errors.append(
                f"flux state {sample_id!r} has non-finite flux for reaction "
                f"{reaction_id!r}"
            )
            continue
        if not math.isfinite(numeric):
            finite_values_valid = False
            errors.append(
                f"flux state {sample_id!r} has non-finite flux for reaction "
                f"{reaction_id!r}"
            )
            continue
        values.append(numeric)
    if not membership_valid or not order_valid or not finite_values_valid:
        return (
            sample_id,
            None,
            membership_valid,
            order_valid,
            finite_values_valid,
            tuple(errors),
        )
    return sample_id, tuple(values), True, True, True, tuple(errors)


def _objective_validation_metrics(
    prepared: PreparedFluxRegion, values: Sequence[float]
) -> _ObjectiveValidationMetrics:
    """Evaluate both literal and conditioned objective semantics without repair."""

    lp = prepared.lp
    try:
        effective = _effective_objective_value(lp, values)
        stable = _biological_objective_value(lp, values)
        declared = _declared_objective_value(lp, values)
    except (OverflowError, ValueError):
        return _ObjectiveValidationMetrics(
            math.nan,
            math.nan,
            math.nan,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
        )
    if not all(math.isfinite(value) for value in (declared, stable, effective)):
        return _ObjectiveValidationMetrics(
            declared,
            stable,
            effective,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
        )
    discrepancy = abs(declared - stable)
    scale = max(
        _objective_activity_scale(lp),
        abs(prepared.retention.effective_optimum),
        abs(prepared.retention.effective_bound),
    )
    normalized_discrepancy = discrepancy / scale if scale > 0.0 else discrepancy
    direct_violation = (
        max(prepared.retention.bound - declared, 0.0)
        if prepared.retention.sense == ">="
        else max(declared - prepared.retention.bound, 0.0)
    )
    normalized_direct_violation = (
        direct_violation / scale if scale > 0.0 else direct_violation
    )
    _, stable_violation = _retained_objective_violation(
        lp,
        stable,
        prepared.retention.sense,
        prepared.retention.bound,
        effective_value=effective,
        effective_bound=prepared.retention.effective_bound,
    )
    return _ObjectiveValidationMetrics(
        declared,
        stable,
        effective,
        discrepancy,
        normalized_discrepancy,
        direct_violation,
        normalized_direct_violation,
        stable_violation,
    )


def validate_flux_states(
    prepared: PreparedFluxRegion,
    states: Sequence[CanonicalFluxState],
) -> FluxSampleValidationReport:
    """Strictly validate complete states against the original prepared LP.

    This validator does not use the sampler's reduced geometry and does not
    reorder, clip, or otherwise repair supplied values.  Invalid state content
    produces a report with ``valid=False`` and contextual errors.  Malformed
    validator inputs or a forged prepared region raise :class:`AnalysisError`.
    """

    _validate_prepared_flux_region(prepared)
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise AnalysisError("flux states must be a nonempty sequence")
    ordered_states = tuple(states)
    if not ordered_states:
        raise AnalysisError("flux states must be a nonempty sequence")

    seen_sample_ids: set[Any] = set()
    diagnostics: list[FluxStateValidationDiagnostics] = []
    errors: list[str] = []
    lp = prepared.lp
    sense = prepared.retention.sense
    retained_bound = prepared.retention.bound
    sample_ids_valid = True
    membership_valid = True
    order_valid = True
    finite_valid = True
    bounds_valid = True
    mass_balance_valid = True
    retained_valid = True

    for state in ordered_states:
        (
            sample_id,
            values,
            state_membership_valid,
            state_order_valid,
            state_finite_valid,
            inspection_errors,
        ) = _inspect_state_values(lp, state)
        state_errors = list(inspection_errors)
        try:
            hash(sample_id)
        except TypeError:
            sample_ids_valid = False
            state_errors.append(
                f"flux-state sample ID {sample_id!r} must be hashable"
            )
        else:
            if sample_id in seen_sample_ids:
                sample_ids_valid = False
                state_errors.append(f"duplicate flux-state sample ID {sample_id!r}")
            seen_sample_ids.add(sample_id)
        membership_valid &= state_membership_valid
        order_valid &= state_order_valid
        finite_valid &= state_finite_valid

        if values is None:
            bounds_valid = False
            mass_balance_valid = False
            retained_valid = False
            diagnostics.append(
                FluxStateValidationDiagnostics(
                    sample_id=sample_id,
                    valid=False,
                    errors=tuple(state_errors),
                    reaction_membership_valid=state_membership_valid,
                    reaction_order_valid=state_order_valid,
                    finite_values_valid=state_finite_valid,
                    objective_value=None,
                    stable_objective_value=None,
                    effective_objective_value=None,
                    objective_conditioning_discrepancy=0.0,
                    normalized_objective_conditioning_discrepancy=0.0,
                    max_lower_bound_violation=0.0,
                    lower_bound_reaction_id=None,
                    max_upper_bound_violation=0.0,
                    upper_bound_reaction_id=None,
                    max_raw_mass_balance_residual=0.0,
                    max_mass_balance_residual=0.0,
                    conditioned_row_space_residual=0.0,
                    mass_balance_metabolite_id=None,
                    raw_retained_objective_violation=0.0,
                    retained_objective_violation=0.0,
                    stable_retained_objective_violation=0.0,
                )
            )
            errors.extend(state_errors)
            continue

        lower_violations = tuple(
            max(lower - value, 0.0)
            for lower, value in zip(lp.lower_bounds, values)
        )
        upper_violations = tuple(
            max(value - upper, 0.0)
            for upper, value in zip(lp.upper_bounds, values)
        )
        lower_index = max(range(len(values)), key=lower_violations.__getitem__)
        upper_index = max(range(len(values)), key=upper_violations.__getitem__)
        max_lower = lower_violations[lower_index]
        max_upper = upper_violations[upper_index]
        if max_lower > BOUND_TOLERANCE:
            bounds_valid = False
            state_errors.append(
                f"flux state {state.sample_id!r} violates lower bound for reaction "
                f"{lp.reaction_ids[lower_index]!r} by {max_lower:g}"
            )
        if max_upper > BOUND_TOLERANCE:
            bounds_valid = False
            state_errors.append(
                f"flux state {state.sample_id!r} violates upper bound for reaction "
                f"{lp.reaction_ids[upper_index]!r} by {max_upper:g}"
            )

        raw_residuals: list[float] = []
        normalized_residuals: list[float] = []
        for row in range(len(lp.balanced_metabolite_ids)):
            offsets = range(lp.row_starts[row], lp.row_starts[row + 1])
            coefficients = tuple(lp.coefficients[offset] for offset in offsets)
            try:
                raw = math.fsum(
                    coefficient * values[lp.column_indices[offset]]
                    for coefficient, offset in zip(
                        coefficients,
                        range(lp.row_starts[row], lp.row_starts[row + 1]),
                    )
                )
            except (OverflowError, ValueError):
                raw = math.inf
            row_norm = math.hypot(*coefficients)
            raw_residual = abs(raw) if math.isfinite(raw) else math.inf
            raw_residuals.append(raw_residual)
            if not math.isfinite(row_norm):
                normalized_residuals.append(math.inf)
            elif row_norm > 0.0:
                normalized = raw_residual / row_norm
                normalized_residuals.append(
                    normalized if math.isfinite(normalized) else math.inf
                )
            else:
                normalized_residuals.append(raw_residual)
        if normalized_residuals:
            mass_index = max(
                range(len(normalized_residuals)),
                key=normalized_residuals.__getitem__,
            )
            max_raw_mass = max(raw_residuals)
            mass_id: str | None = lp.balanced_metabolite_ids[mass_index]
        else:
            max_raw_mass = 0.0
            mass_id = None
        max_mass = max(normalized_residuals, default=0.0)
        row_space = _balance_row_space(lp)
        if lp.balance_rank:
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    conditioned_mass = float(
                        np.linalg.norm(row_space @ np.asarray(values, dtype=float))
                    )
            except (FloatingPointError, OverflowError, ValueError):
                conditioned_mass = math.inf
            if not math.isfinite(conditioned_mass):
                conditioned_mass = math.inf
        else:
            conditioned_mass = 0.0
        if conditioned_mass > MASS_BALANCE_TOLERANCE:
            mass_balance_valid = False
            state_errors.append(
                f"flux state {state.sample_id!r} violates steady-state mass balance for "
                f"the conditioned row space: residual={conditioned_mass:g}; closest "
                f"original metabolite={mass_id!r}, normalized_residual={max_mass:g}, "
                f"maximum_raw_residual={max_raw_mass:g}"
            )

        objective_metrics = _objective_validation_metrics(prepared, values)
        if (
            objective_metrics.normalized_discrepancy
            > RETAINED_OBJECTIVE_TOLERANCE
        ):
            retained_valid = False
            state_errors.append(
                f"flux state {state.sample_id!r} declared biological objective "
                "differs from its stable affine quotient: "
                f"declared={objective_metrics.declared_value:g}, "
                f"quotient={objective_metrics.stable_value:g}, "
                f"discrepancy={objective_metrics.discrepancy:g}, "
                "normalized_discrepancy="
                f"{objective_metrics.normalized_discrepancy:g}"
            )
        if (
            objective_metrics.normalized_direct_violation
            > RETAINED_OBJECTIVE_TOLERANCE
        ):
            retained_valid = False
            state_errors.append(
                f"flux state {state.sample_id!r} violates declared retained "
                f"objective {sense} {retained_bound:g}: "
                f"declared={objective_metrics.declared_value:g}, "
                f"raw_violation={objective_metrics.direct_violation:g}, "
                "normalized_violation="
                f"{objective_metrics.normalized_direct_violation:g}"
            )
        if objective_metrics.stable_violation > RETAINED_OBJECTIVE_TOLERANCE:
            retained_valid = False
            state_errors.append(
                f"flux state {state.sample_id!r} violates the conditioned "
                f"retained objective {sense} "
                f"{prepared.retention.effective_bound:g}: "
                f"effective={objective_metrics.effective_value:g}, "
                f"normalized_violation={objective_metrics.stable_violation:g}"
            )

        diagnostics.append(
            FluxStateValidationDiagnostics(
                sample_id=sample_id,
                valid=not state_errors,
                errors=tuple(state_errors),
                reaction_membership_valid=state_membership_valid,
                reaction_order_valid=state_order_valid,
                finite_values_valid=state_finite_valid,
                objective_value=objective_metrics.declared_value,
                stable_objective_value=objective_metrics.stable_value,
                effective_objective_value=objective_metrics.effective_value,
                objective_conditioning_discrepancy=objective_metrics.discrepancy,
                normalized_objective_conditioning_discrepancy=(
                    objective_metrics.normalized_discrepancy
                ),
                max_lower_bound_violation=max_lower,
                lower_bound_reaction_id=(
                    lp.reaction_ids[lower_index] if max_lower > 0.0 else None
                ),
                max_upper_bound_violation=max_upper,
                upper_bound_reaction_id=(
                    lp.reaction_ids[upper_index] if max_upper > 0.0 else None
                ),
                max_raw_mass_balance_residual=max_raw_mass,
                max_mass_balance_residual=max_mass,
                conditioned_row_space_residual=conditioned_mass,
                mass_balance_metabolite_id=mass_id if max_mass > 0.0 else None,
                raw_retained_objective_violation=(
                    objective_metrics.direct_violation
                ),
                retained_objective_violation=(
                    objective_metrics.normalized_direct_violation
                ),
                stable_retained_objective_violation=(
                    objective_metrics.stable_violation
                ),
            )
        )
        errors.extend(state_errors)

    return FluxSampleValidationReport(
        valid=not errors,
        sample_count=len(ordered_states),
        model_fingerprint=lp.fingerprint,
        reaction_order=lp.reaction_ids,
        sample_ids_valid=sample_ids_valid,
        reaction_membership_valid=membership_valid,
        reaction_order_valid=order_valid,
        finite_values_valid=finite_valid,
        bounds_valid=bounds_valid,
        mass_balance_valid=mass_balance_valid,
        retained_objective_valid=retained_valid,
        max_lower_bound_violation=max(
            item.max_lower_bound_violation for item in diagnostics
        ),
        max_upper_bound_violation=max(
            item.max_upper_bound_violation for item in diagnostics
        ),
        max_raw_mass_balance_residual=max(
            item.max_raw_mass_balance_residual for item in diagnostics
        ),
        max_mass_balance_residual=max(
            item.max_mass_balance_residual for item in diagnostics
        ),
        max_conditioned_row_space_residual=max(
            item.conditioned_row_space_residual for item in diagnostics
        ),
        max_objective_conditioning_discrepancy=max(
            item.objective_conditioning_discrepancy for item in diagnostics
        ),
        max_normalized_objective_conditioning_discrepancy=max(
            item.normalized_objective_conditioning_discrepancy
            for item in diagnostics
        ),
        max_raw_retained_objective_violation=max(
            item.raw_retained_objective_violation for item in diagnostics
        ),
        max_retained_objective_violation=max(
            item.retained_objective_violation for item in diagnostics
        ),
        max_stable_retained_objective_violation=max(
            item.stable_retained_objective_violation for item in diagnostics
        ),
        diagnostics=tuple(diagnostics),
        errors=tuple(errors),
    )


def _require_valid_flux_states(
    report: FluxSampleValidationReport, operation: str
) -> None:
    """Turn an independent invalid report into a public sampler failure."""

    if not report.valid:
        detail = report.errors[0] if report.errors else "unknown validation failure"
        raise AnalysisError(f"{operation} failed independent validation: {detail}")


def _validate_fva_result(prepared: PreparedFluxRegion, fva: FVAResult) -> np.ndarray:
    if not isinstance(fva, FVAResult):
        raise AnalysisError("native sampling requires an FVAResult")
    lp = prepared.lp
    if tuple(fva.ranges.index) != lp.reaction_ids:
        raise AnalysisError("FVA reaction order does not match the prepared flux region")
    if tuple(fva.ranges.columns) != ("minimum", "maximum"):
        raise AnalysisError("FVA ranges must contain exactly minimum and maximum columns")
    if fva.model_fingerprint != lp.fingerprint:
        raise AnalysisError(
            "FVA model fingerprint does not match the prepared flux region"
        )
    if (
        fva.fraction_of_optimum != prepared.retention.fraction_of_optimum
        or fva.objective_value != prepared.retention.biological_optimum
        or fva.objective_direction != prepared.retention.objective_direction
    ):
        raise AnalysisError("FVA retained-objective metadata does not match the prepared region")
    try:
        ranges = fva.ranges.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("FVA ranges contain malformed numeric values") from error
    if ranges.shape != (len(lp.reaction_ids), 2):
        raise AnalysisError("FVA ranges have the wrong shape for the prepared flux region")
    if not np.isfinite(ranges).all():
        raise AnalysisError(
            "native sampling region is unbounded or has a non-finite FVA endpoint"
        )
    for index, (minimum, maximum) in enumerate(ranges):
        if minimum > maximum + FVA_COLLAPSE_TOLERANCE:
            raise AnalysisError(
                f"FVA minimum exceeds maximum for reaction {lp.reaction_ids[index]!r}"
            )
        if minimum < lp.lower_bounds[index] - BOUND_TOLERANCE:
            raise AnalysisError(
                f"FVA minimum violates the model lower bound for reaction "
                f"{lp.reaction_ids[index]!r}"
            )
        if maximum > lp.upper_bounds[index] + BOUND_TOLERANCE:
            raise AnalysisError(
                f"FVA maximum violates the model upper bound for reaction "
                f"{lp.reaction_ids[index]!r}"
            )
        fba_value = float(prepared.fba.fluxes.iloc[index])
        if (
            fba_value < minimum - BOUND_TOLERANCE
            or fba_value > maximum + BOUND_TOLERANCE
        ):
            raise AnalysisError(
                f"FVA interval for reaction {lp.reaction_ids[index]!r} does not "
                "contain the prepared feasible FBA state"
            )
    expected_digest = _fva_ranges_sha256(fva.ranges, lp.fingerprint)
    if fva.ranges_sha256 != expected_digest:
        raise AnalysisError(
            "FVA range digest is missing or inconsistent; the ordered ranges may "
            "have been mutated after native FVA"
        )
    return ranges


def _balance_row_space(lp: CompiledFluxLP) -> np.ndarray:
    """Return the conditioned row-space basis compiled for solver and validation."""

    return np.asarray(lp.balance_row_space_basis, dtype=float).reshape(
        lp.balance_rank, len(lp.reaction_ids)
    )


def _affine_equality_row_space(lp: CompiledFluxLP) -> np.ndarray:
    """Return conditioned mass-balance plus explicit-fixed equalities."""

    return np.asarray(lp.affine_equality_basis, dtype=float).reshape(
        lp.affine_rank, len(lp.reaction_ids)
    )


def _canonical_balance_matrix(lp: CompiledFluxLP) -> np.ndarray:
    """Reconstruct canonical unconditioned mass balance from immutable CSR data."""

    matrix = np.zeros(
        (len(lp.balanced_metabolite_ids), len(lp.reaction_ids)), dtype=float
    )
    for row in range(len(lp.balanced_metabolite_ids)):
        for offset in range(lp.row_starts[row], lp.row_starts[row + 1]):
            matrix[row, lp.column_indices[offset]] = lp.coefficients[offset]
    return matrix


def _sampling_affine_nullspace(
    lp: CompiledFluxLP,
    fixed_indices: set[int],
    objective: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Condition the complete canonical homogeneous sampling hull exactly once.

    The shared conditioner checks discarded numerical directions against the
    exact binary-float rank.  This prevents an FVA-collapsed coordinate or
    optimal-face equality from being silently treated as redundant.
    """

    n = len(lp.reaction_ids)
    free_indices = np.asarray(
        [index for index in range(n) if index not in fixed_indices], dtype=int
    )
    # Eliminate fixed coordinates before rank conditioning.  Combining a unit
    # fixed-coordinate row with a uniformly tiny but otherwise well-conditioned
    # conservation law would make the full matrix look spuriously ambiguous.
    matrix = _canonical_balance_matrix(lp)[:, free_indices]
    if objective is not None and np.any(objective[free_indices]):
        matrix = np.vstack((matrix, objective[free_indices]))
    _, _, null_rows, rank = _condition_affine_equalities(
        matrix,
        np.zeros(matrix.shape[0], dtype=float),
        description="sampling affine-hull",
    )
    basis = np.zeros((n, null_rows.shape[0]), dtype=float)
    basis[free_indices, :] = null_rows.T
    return basis, len(fixed_indices) + rank


def _condition_reduced_sampling_objective(
    lp: CompiledFluxLP,
    fixed_indices: set[int],
    particular: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Re-quotient the declared objective after numerical facial reduction."""

    ordered = tuple(sorted(fixed_indices))
    effective, constant, _ = _condition_objective_after_fixed_coordinates(
        lp,
        ordered,
        tuple(float(particular[index]) for index in ordered),
    )
    return np.asarray(effective, dtype=float), constant


def _build_reduced_geometry(
    prepared: PreparedFluxRegion, ranges: np.ndarray
) -> _ReducedFluxGeometry:
    lp = prepared.lp
    n = len(lp.reaction_ids)
    particular = np.asarray(prepared.fba.fluxes, dtype=float).copy()
    if particular.shape != (n,) or not np.isfinite(particular).all():
        raise AnalysisError("prepared FBA primal cannot seed native sampling geometry")

    explicit_fixed_indices = {
        index
        for index, (lower, upper) in enumerate(zip(lp.lower_bounds, lp.upper_bounds))
        if lower == upper
    }
    fixed_indices = set(explicit_fixed_indices)
    collapsed_indices: set[int] = set()
    for index, (minimum, maximum) in enumerate(ranges):
        width = float(maximum - minimum)
        if width <= FVA_COLLAPSE_TOLERANCE:
            collapsed_indices.add(index)
        elif width <= FVA_DEGENERACY_TOLERANCE:
            raise AnalysisError(
                "numerically degenerate FVA interval for reaction "
                f"{lp.reaction_ids[index]!r}: width={width:g} lies between "
                f"collapse tolerance {FVA_COLLAPSE_TOLERANCE:g} and "
                f"resolved-width threshold {FVA_DEGENERACY_TOLERANCE:g}"
            )
    fixed_indices.update(collapsed_indices)
    retention = prepared.retention
    optimal_face = (
        retention.fraction_of_optimum == 1.0
        or retention.bound == retention.biological_optimum
    )
    objective, reduced_objective_constant = _condition_reduced_sampling_objective(
        lp, fixed_indices, particular
    )
    objective_scale = max((abs(float(value)) for value in objective), default=0.0)
    normalized_objective = (
        objective / objective_scale if objective_scale > 0.0 else objective
    )
    reduced_effective_at_particular = math.fsum(
        float(coefficient) * float(value)
        for coefficient, value in zip(objective, particular)
    )
    objective_value_scale = max(
        _objective_activity_scale(lp),
        abs(retention.effective_optimum),
        abs(retention.effective_bound),
    )
    expected_reduced_effective_at_particular = math.fsum(
        (
            retention.effective_optimum,
            lp.objective_constant,
            -reduced_objective_constant,
        )
    )
    reduced_objective_discrepancy = abs(
        reduced_effective_at_particular
        - expected_reduced_effective_at_particular
    )
    objective_value_tolerance = (
        RETAINED_OBJECTIVE_TOLERANCE * objective_value_scale
        if objective_value_scale > 0.0
        else RETAINED_OBJECTIVE_TOLERANCE
    )
    if reduced_objective_discrepancy > objective_value_tolerance:
        raise AnalysisError(
            "reduced sampling objective differs from the prepared biological "
            "optimum at its FBA anchor: "
            f"reduced_effective={reduced_effective_at_particular:g}, "
            f"prepared_effective={expected_reduced_effective_at_particular:g}, "
            f"discrepancy={reduced_objective_discrepancy:g}"
        )
    reduced_effective_bound = math.fsum(
        (
            reduced_effective_at_particular,
            retention.effective_bound,
            -retention.effective_optimum,
        )
    )
    effective_at_particular = (
        reduced_effective_at_particular / objective_scale
        if objective_scale > 0.0
        else reduced_effective_at_particular
    )
    normalized_retained_bound = (
        reduced_effective_bound / objective_scale
        if objective_scale > 0.0
        else reduced_effective_bound
    )
    if optimal_face:
        if np.linalg.norm(normalized_objective) > 0.0:
            if (
                abs(effective_at_particular - normalized_retained_bound)
                > RETAINED_OBJECTIVE_TOLERANCE
            ):
                raise AnalysisError(
                    "prepared FBA state does not anchor its effective optimal face"
                )
        elif abs(normalized_retained_bound) > RETAINED_OBJECTIVE_TOLERANCE:
            raise AnalysisError(
                "constant biological objective has an inconsistent retained face"
            )

    basis, equality_rank = _sampling_affine_nullspace(
        lp,
        fixed_indices,
        objective if optimal_face else None,
    )
    affine_dimension = basis.shape[1]
    inequality_rows: list[np.ndarray] = []
    inequality_bounds: list[float] = []
    for index, (lower, upper) in enumerate(zip(lp.lower_bounds, lp.upper_bounds)):
        if index in fixed_indices:
            continue
        # v_i <= upper
        inequality_rows.append(basis[index, :].copy())
        inequality_bounds.append(upper - particular[index])
        # -v_i <= -lower
        inequality_rows.append(-basis[index, :].copy())
        inequality_bounds.append(particular[index] - lower)

    if not optimal_face:
        projected_objective = normalized_objective @ basis
        if retention.sense == ">=":
            inequality_rows.append(-projected_objective)
            inequality_bounds.append(
                effective_at_particular - normalized_retained_bound
            )
        else:
            inequality_rows.append(projected_objective)
            inequality_bounds.append(
                normalized_retained_bound - effective_at_particular
            )

    if inequality_rows:
        rows_array = np.vstack(inequality_rows).reshape(len(inequality_rows), affine_dimension)
    else:  # pragma: no cover - a validated FluxModel has at least one reaction
        rows_array = np.empty((0, affine_dimension), dtype=float)
    bounds_array = np.asarray(inequality_bounds, dtype=float)
    if not np.isfinite(rows_array).all() or not np.isfinite(bounds_array).all():
        raise AnalysisError("native sampling geometry contains non-finite constraints")
    return _ReducedFluxGeometry(
        particular,
        basis,
        rows_array,
        bounds_array,
        equality_rank,
        tuple(lp.reaction_ids[index] for index in sorted(collapsed_indices)),
        len(explicit_fixed_indices),
        optimal_face,
        objective_scale,
        reduced_objective_constant,
        reduced_effective_bound,
    )


def _active_reduced_inequalities(
    geometry: _ReducedFluxGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    norms: list[float] = []
    for row, bound in zip(geometry.inequality_rows, geometry.inequality_bounds):
        norm = math.hypot(*(float(value) for value in row))
        if not math.isfinite(norm):
            raise AnalysisError(
                "reduced sampling inequality has a non-finite normal length"
            )
        if norm == 0.0:
            if bound < -BOUND_TOLERANCE:
                raise AnalysisError(
                    "prepared sampling geometry contains an inconsistent affine constraint"
                )
            continue
        rows.append(row / norm)
        bounds.append(float(bound) / norm)
        norms.append(1.0)
    dimension = geometry.basis.shape[1]
    if not rows:
        return (
            np.empty((0, dimension), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
    return np.vstack(rows), np.asarray(bounds), np.asarray(norms)


def _chebyshev_center(
    geometry: _ReducedFluxGeometry,
) -> tuple[np.ndarray, float, str]:
    dimension = geometry.basis.shape[1]
    if dimension == 0:
        return np.empty(0, dtype=float), 0.0, "affine-singleton"
    rows, bounds, norms = _active_reduced_inequalities(geometry)
    if not len(rows):
        raise AnalysisError("native sampling region is unbounded in its affine hull")

    highspy = _highspy()
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("threads", 1)
    solver.setOptionValue("solver", "simplex")
    solver.setOptionValue("small_matrix_value", HIGHS_SMALL_MATRIX_VALUE)
    variable_count = dimension + 1
    solver.addCols(
        variable_count,
        [0.0] * dimension + [1.0],
        [-highspy.kHighsInf] * dimension + [0.0],
        [highspy.kHighsInf] * variable_count,
        0,
        [0] * (variable_count + 1),
        [],
        [],
    )
    starts = [0]
    indices: list[int] = []
    coefficients: list[float] = []
    for row, norm in zip(rows, norms):
        for index, value in enumerate(row):
            if value != 0.0:
                indices.append(index)
                coefficients.append(float(value))
        indices.append(dimension)
        coefficients.append(float(norm))
        starts.append(len(indices))
    solver.addRows(
        len(rows),
        [-highspy.kHighsInf] * len(rows),
        list(bounds),
        len(indices),
        starts,
        indices,
        coefficients,
    )
    solver.setMaximize()
    solver.run()
    status = solver.getModelStatus()
    status_name = solver.modelStatusToString(status)
    if status != highspy.HighsModelStatus.kOptimal:
        lowered = status_name.lower()
        if "unbounded" in lowered:
            category = "unbounded sampling region"
        elif "infeasible" in lowered:
            category = "infeasible sampling region"
        else:
            category = "solver failure/unknown"
        raise AnalysisError(
            f"Chebyshev-center construction failed: HiGHS status {status_name} "
            f"({category})"
        )
    solution = np.asarray(solver.getSolution().col_value, dtype=float)
    if solution.shape != (variable_count,) or not np.isfinite(solution).all():
        raise AnalysisError("Chebyshev-center construction returned a malformed solution")
    center = solution[:-1]
    radius = float(solution[-1])
    if radius <= CENTER_RADIUS_TOLERANCE:
        raise AnalysisError(
            "retained sampling region has positive affine dimension but a "
            f"numerically degenerate interior radius {radius:g}"
        )
    certificate_violation = max(
        (
            float(row @ center)
            + math.hypot(*(float(value) for value in row)) * radius
            - float(bound)
            for row, bound in zip(
                geometry.inequality_rows, geometry.inequality_bounds
            )
        ),
        default=0.0,
    )
    if certificate_violation > BOUND_TOLERANCE:
        raise AnalysisError(
            "Chebyshev-center construction returned an invalid full-precision "
            f"interior certificate: violation={certificate_violation:g}"
        )
    return center, radius, "chebyshev-center"


def _line_interval(
    rows: np.ndarray,
    bounds: np.ndarray,
    point: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, float] | None:
    """Return one finite feasible chord, or ``None`` for a degenerate direction."""

    lower = -math.inf
    upper = math.inf
    for row, bound in zip(rows, bounds):
        denominator = float(row @ direction)
        slack = float(bound - row @ point)
        if slack < -BOUND_TOLERANCE:
            raise AnalysisError(
                "hit-and-run current point violates a retained inequality"
            )
        if denominator == 0.0:
            continue
        endpoint = slack / denominator
        if denominator > 0.0:
            upper = min(upper, endpoint)
        else:
            lower = max(lower, endpoint)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise AnalysisError("hit-and-run encountered an unbounded sampling direction")
    if lower > upper + INTERVAL_TOLERANCE:
        raise AnalysisError("hit-and-run feasible chord is numerically inconsistent")
    if lower > INTERVAL_TOLERANCE or upper < -INTERVAL_TOLERANCE:
        raise AnalysisError(
            "hit-and-run feasible chord does not contain the current point"
        )
    if upper - lower <= INTERVAL_TOLERANCE:
        return None
    return lower, upper


def _max_reduced_inequality_violation(
    rows: np.ndarray, bounds: np.ndarray, point: np.ndarray
) -> float:
    """Check a proposed chain point without clipping or projecting it."""

    if not len(rows):
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = rows @ point - bounds
    if not np.isfinite(residuals).all():
        return math.inf
    return float(np.max(residuals, initial=0.0))


def _validate_chain_candidate(
    prepared: PreparedFluxRegion,
    values: np.ndarray,
    accepted_step: int,
) -> None:
    """Fail on the first invalid internal transition without repair or redraw."""

    lp = prepared.lp
    if values.shape != (len(lp.reaction_ids),) or not np.isfinite(values).all():
        raise AnalysisError(
            "native hit-and-run candidate contains non-finite ambient fluxes at "
            f"accepted step {accepted_step}; no repair or replacement was attempted"
        )
    lower_violation = max(
        (max(lower - value, 0.0) for lower, value in zip(lp.lower_bounds, values)),
        default=0.0,
    )
    upper_violation = max(
        (max(value - upper, 0.0) for upper, value in zip(lp.upper_bounds, values)),
        default=0.0,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        row_space_residuals = _balance_row_space(lp) @ values
    try:
        conditioned_mass = math.hypot(
            *(float(value) for value in row_space_residuals)
        )
    except (OverflowError, ValueError):
        conditioned_mass = math.inf
    if not math.isfinite(conditioned_mass):
        conditioned_mass = math.inf
    objective = _objective_validation_metrics(prepared, values)
    if (
        max(lower_violation, upper_violation) > BOUND_TOLERANCE
        or conditioned_mass > MASS_BALANCE_TOLERANCE
        or objective.normalized_discrepancy > RETAINED_OBJECTIVE_TOLERANCE
        or objective.normalized_direct_violation
        > RETAINED_OBJECTIVE_TOLERANCE
        or objective.stable_violation > RETAINED_OBJECTIVE_TOLERANCE
    ):
        raise AnalysisError(
            "native hit-and-run candidate failed independent ambient-space "
            f"validation at accepted step {accepted_step}: "
            f"lower_violation={lower_violation:g}, "
            f"upper_violation={upper_violation:g}, "
            f"conditioned_mass_residual={conditioned_mass:g}, "
            f"declared_objective={objective.declared_value:g}, "
            f"stable_quotient={objective.stable_value:g}, "
            f"objective_discrepancy={objective.discrepancy:g}, "
            f"direct_retention_violation={objective.direct_violation:g}; "
            "no repair or replacement was attempted"
        )


def _states_from_points(
    lp: CompiledFluxLP, points: Sequence[np.ndarray]
) -> tuple[CanonicalFluxState, ...]:
    states: list[CanonicalFluxState] = []
    for sample_index, point in enumerate(points):
        if point.shape != (len(lp.reaction_ids),) or not np.isfinite(point).all():
            raise AnalysisError("native sampler generated a malformed or non-finite state")
        states.append(
            CanonicalFluxState(
                f"sample_{sample_index:04d}",
                tuple(
                    (reaction_id, float(value))
                    for reaction_id, value in zip(lp.reaction_ids, point)
                ),
            )
        )
    return tuple(states)


def _highs_version() -> str:
    solver = _highspy().Highs()
    return str(solver.version())


def _validate_retained_region_at_fba(prepared: PreparedFluxRegion) -> None:
    """Reject sign-incompatible fractional arithmetic before any endpoint work."""

    if prepared.retention.fraction_of_optimum < 1.0 and (
        (
            prepared.retention.objective_direction == "max"
            and prepared.retention.biological_optimum < 0.0
        )
        or (
            prepared.retention.objective_direction == "min"
            and prepared.retention.biological_optimum > 0.0
        )
    ):
        raise AnalysisError(
            "fraction-of-optimum arithmetic produces an infeasible retained "
            f"objective bound {prepared.retention.sense} "
            f"{prepared.retention.bound:g}"
        )
    fba_values = tuple(float(value) for value in prepared.fba.fluxes)
    objective_at_fba = _biological_objective_value(prepared.lp, fba_values)
    effective_objective_at_fba = _effective_objective_value(
        prepared.lp, fba_values
    )
    _, normalized_optimum_violation = _retained_objective_violation(
        prepared.lp,
        objective_at_fba,
        prepared.retention.sense,
        prepared.retention.bound,
        effective_value=effective_objective_at_fba,
        effective_bound=prepared.retention.effective_bound,
    )
    if normalized_optimum_violation > RETAINED_OBJECTIVE_TOLERANCE:
        raise AnalysisError(
            "fraction-of-optimum arithmetic produces an infeasible retained "
            f"objective bound {prepared.retention.sense} "
            f"{prepared.retention.bound:g}"
        )


def sample_prepared_flux_states(
    prepared: PreparedFluxRegion,
    count: int,
    *,
    seed: int,
    fva: FVAResult | None = None,
    fva_workers: int | None = None,
    burn_in: int = 100,
    thinning: int = 10,
    max_direction_attempts: int = 100,
) -> NativeFluxSamplingResult:
    """Sample complete jointly feasible states from one prepared flux region."""

    _validate_prepared_flux_region(prepared)
    sample_count = _positive_integer(count, "sample count")
    seed_value = _nonnegative_integer(seed, "sampling seed")
    burn_in_value = _nonnegative_integer(burn_in, "burn-in")
    thinning_value = _positive_integer(thinning, "thinning")
    attempts_value = _positive_integer(
        max_direction_attempts, "maximum direction attempts"
    )
    _validate_retained_region_at_fba(prepared)

    if fva is not None and fva_workers is not None:
        raise AnalysisError(
            "fva_workers must be omitted when a precomputed FVAResult is supplied"
        )
    if fva is None:
        fva = run_prepared_highs_vffva(
            prepared, workers=1 if fva_workers is None else fva_workers
        )
    ranges = _validate_fva_result(prepared, fva)

    geometry = _build_reduced_geometry(prepared, ranges)
    center, center_radius, center_method = _chebyshev_center(geometry)
    active_rows, active_bounds, _ = _active_reduced_inequalities(geometry)
    affine_dimension = geometry.basis.shape[1]
    rejected_directions = 0
    accepted_steps = 0
    self_loop_steps = 0

    if affine_dimension == 0:
        points = [geometry.particular.copy() for _ in range(sample_count)]
    else:
        center_state = _states_from_points(
            prepared.lp, (geometry.particular + geometry.basis @ center,)
        )
        center_validation = validate_flux_states(prepared, center_state)
        _require_valid_flux_states(center_validation, "native sampler center")
        generator = np.random.Generator(np.random.PCG64(seed_value))
        current = center.copy()
        emitted: list[np.ndarray] = []
        required_steps = burn_in_value + sample_count * thinning_value
        while accepted_steps < required_steps:
            for _ in range(attempts_value):
                direction = generator.standard_normal(affine_dimension)
                norm = float(np.linalg.norm(direction))
                if not math.isfinite(norm) or norm <= REDUCED_DIRECTION_TOLERANCE:
                    rejected_directions += 1
                    continue
                direction /= norm
                interval = _line_interval(
                    active_rows, active_bounds, current, direction
                )
                if interval is None:
                    # A finite zero-width chord is a no-move Markov transition.
                    # Redrawing here would condition the direction law on the
                    # current point and change the intended stationary target.
                    accepted_steps += 1
                    self_loop_steps += 1
                    break
                lower, upper = interval
                step = float(generator.uniform(lower, upper))
                candidate = current + step * direction
                if not np.isfinite(candidate).all():
                    raise AnalysisError(
                        "native hit-and-run generated a non-finite candidate at "
                        f"accepted step {accepted_steps}"
                    )
                reduced_violation = _max_reduced_inequality_violation(
                    geometry.inequality_rows,
                    geometry.inequality_bounds,
                    candidate,
                )
                if reduced_violation > BOUND_TOLERANCE:
                    raise AnalysisError(
                        "native hit-and-run candidate violates the retained reduced "
                        f"polytope by {reduced_violation:g} at accepted step "
                        f"{accepted_steps}; no repair or replacement was attempted"
                    )
                ambient_candidate = (
                    geometry.particular + geometry.basis @ candidate
                )
                _validate_chain_candidate(
                    prepared, ambient_candidate, accepted_steps
                )
                current = candidate
                accepted_steps += 1
                break
            else:
                raise AnalysisError(
                    "native hit-and-run could not generate a valid next state after "
                    f"{attempts_value} direction attempts at accepted step "
                    f"{accepted_steps}"
                )
            if (
                accepted_steps > burn_in_value
                and (accepted_steps - burn_in_value) % thinning_value == 0
            ):
                emitted.append(geometry.particular + geometry.basis @ current)
        if len(emitted) != sample_count:
            raise AnalysisError(
                f"native sampler emitted {len(emitted)} states, expected {sample_count}"
            )
        points = emitted

    states = _states_from_points(prepared.lp, points)
    validation = validate_flux_states(prepared, states)
    _require_valid_flux_states(validation, "native sampled flux batch")
    provenance = FluxSamplingProvenance(
        algorithm=ALGORITHM_NAME,
        algorithm_version=ALGORITHM_VERSION,
        random_bit_generator=RANDOM_BIT_GENERATOR,
        seed=seed_value,
        sample_count=sample_count,
        fraction_of_optimum=prepared.retention.fraction_of_optimum,
        biological_optimum=prepared.retention.biological_optimum,
        retained_objective_bound=prepared.retention.bound,
        retained_objective_sense=prepared.retention.sense,
        objective_direction=prepared.retention.objective_direction,
        declared_objective_scale=max(
            (abs(value) for value in prepared.lp.objective_coefficients),
            default=0.0,
        ),
        effective_objective_scale=_objective_scale(prepared.lp),
        objective_constant=prepared.lp.objective_constant,
        effective_objective_bound=prepared.retention.effective_bound,
        reduced_objective_scale=geometry.objective_scale,
        reduced_objective_constant=geometry.objective_constant,
        reduced_effective_objective_bound=geometry.effective_objective_bound,
        model_fingerprint=prepared.lp.fingerprint,
        fva_ranges_sha256=fva.ranges_sha256,
        reaction_order=prepared.lp.reaction_ids,
        burn_in=burn_in_value,
        thinning=thinning_value,
        max_direction_attempts=attempts_value,
        equality_rank=geometry.equality_rank,
        affine_dimension=affine_dimension,
        fixed_reaction_count=geometry.fixed_reaction_count,
        collapsed_reaction_ids=geometry.collapsed_reaction_ids,
        optimal_face=geometry.optimal_face,
        center_method=center_method,
        center_radius=center_radius,
        accepted_chain_steps=accepted_steps,
        self_loop_steps=self_loop_steps,
        rejected_directions=rejected_directions,
        bound_tolerance=BOUND_TOLERANCE,
        mass_balance_tolerance=MASS_BALANCE_TOLERANCE,
        retained_objective_tolerance=RETAINED_OBJECTIVE_TOLERANCE,
        fva_collapse_tolerance=FVA_COLLAPSE_TOLERANCE,
        fva_degeneracy_tolerance=FVA_DEGENERACY_TOLERANCE,
        affine_rank_tolerance=AFFINE_RANK_TOLERANCE,
        reduced_direction_tolerance=REDUCED_DIRECTION_TOLERANCE,
        interval_tolerance=INTERVAL_TOLERANCE,
        center_radius_tolerance=CENTER_RADIUS_TOLERANCE,
        numpy_version=np.__version__,
        highs_version=_highs_version(),
        stationary_target=(
            "relative-volume uniform stationary target on the numerically reduced "
            "retained affine polytope"
        ),
        finite_chain_claim=(
            "finite correlated Markov-chain states; no mixing or independence guarantee"
        ),
    )
    return NativeFluxSamplingResult(states, sample_count, provenance, validation)


def sample_highs_flux_states(
    model: FluxModel,
    count: int,
    *,
    seed: int,
    fraction_of_optimum: float = 1.0,
    fva_workers: int | None = 1,
    burn_in: int = 100,
    thinning: int = 10,
    max_direction_attempts: int = 100,
) -> NativeFluxSamplingResult:
    """Prepare FBA/FVA once and sample complete native flux states."""

    prepared = prepare_highs_flux_region(model, fraction_of_optimum)
    _validate_retained_region_at_fba(prepared)
    fva = run_prepared_highs_vffva(prepared, workers=fva_workers)
    return sample_prepared_flux_states(
        prepared,
        count,
        seed=seed,
        fva=fva,
        burn_in=burn_in,
        thinning=thinning,
        max_direction_attempts=max_direction_attempts,
    )


__all__ = [
    "ALGORITHM_NAME",
    "ALGORITHM_VERSION",
    "FluxSampleValidationReport",
    "FluxSamplingProvenance",
    "FluxStateValidationDiagnostics",
    "NativeFluxSamplingResult",
    "sample_highs_flux_states",
    "sample_prepared_flux_states",
    "validate_flux_states",
]
