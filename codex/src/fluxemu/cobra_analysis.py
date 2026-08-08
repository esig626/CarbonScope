"""Constraint-based flux analysis using the installed stable COBRApy.

The functions in this module deliberately keep COBRA reaction identifiers and
reaction order intact.  In particular, flux samples are complete feasible
vectors; FVA ranges are never sampled independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from cobra import Model
from cobra.flux_analysis import flux_variability_analysis
from cobra.sampling import ACHRSampler, OptGPSampler
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import fix_objective_as_constraint, linear_reaction_coefficients
from optlang.interface import OPTIMAL

from fluxemu.exceptions import AnalysisError as FluxAnalysisError


OBJECTIVE_FLOOR_CONSTRAINT = "fluxemu_objective_floor"


@dataclass(frozen=True)
class FBAResult:
    """An optimal FBA solution in the model's exact reaction order."""

    objective_value: float
    status: str
    objective_direction: str
    fluxes: pd.Series

    def to_frame(self) -> pd.DataFrame:
        """Return an output-oriented FBA table."""

        frame = self.fluxes.rename("flux").to_frame()
        frame.index.name = "reaction_id"
        frame["objective_value"] = self.objective_value
        frame["solver_status"] = self.status
        return frame


@dataclass(frozen=True)
class FVAResult:
    """Flux-variability ranges calculated at an objective fraction."""

    ranges: pd.DataFrame
    fraction_of_optimum: float
    objective_value: float
    objective_direction: str

    def to_frame(self) -> pd.DataFrame:
        """Return an output-oriented FVA table."""

        frame = self.ranges.copy()
        frame.index.name = "reaction_id"
        frame["fraction_of_optimum"] = self.fraction_of_optimum
        return frame


@dataclass(frozen=True)
class FluxSampleValidationReport:
    """Independent feasibility checks for a table of complete flux samples."""

    valid: bool
    sample_count: int
    reaction_columns_valid: bool
    finite_values_valid: bool
    bounds_valid: bool
    mass_balance_valid: bool
    objective_floor_valid: bool
    duplicate_columns: tuple[str, ...]
    missing_reactions: tuple[str, ...]
    unexpected_reactions: tuple[str, ...]
    column_order_valid: bool
    max_lower_bound_violation: float | None
    max_upper_bound_violation: float | None
    max_mass_balance_residual: float | None
    max_objective_floor_violation: float | None
    details: pd.DataFrame
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary, including per-sample details."""

        detail_records = (
            self.details.reset_index(names="sample_id")
            .replace({np.nan: None})
            .to_dict(orient="records")
        )
        return {
            "valid": self.valid,
            "sample_count": self.sample_count,
            "reaction_columns_valid": self.reaction_columns_valid,
            "finite_values_valid": self.finite_values_valid,
            "bounds_valid": self.bounds_valid,
            "mass_balance_valid": self.mass_balance_valid,
            "objective_floor_valid": self.objective_floor_valid,
            "duplicate_columns": list(self.duplicate_columns),
            "missing_reactions": list(self.missing_reactions),
            "unexpected_reactions": list(self.unexpected_reactions),
            "column_order_valid": self.column_order_valid,
            "max_lower_bound_violation": self.max_lower_bound_violation,
            "max_upper_bound_violation": self.max_upper_bound_violation,
            "max_mass_balance_residual": self.max_mass_balance_residual,
            "max_objective_floor_violation": self.max_objective_floor_violation,
            "details": detail_records,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class FluxSamplingResult:
    """Complete constrained flux samples and their validation evidence."""

    samples: pd.DataFrame
    sampler: str
    seed: int
    fraction_of_optimum: float
    objective_value: float
    objective_floor: float
    objective_direction: str
    validation: FluxSampleValidationReport


def _reaction_ids(model: Model) -> list[str]:
    reaction_ids = [reaction.id for reaction in model.reactions]
    if len(reaction_ids) != len(set(reaction_ids)):
        raise FluxAnalysisError("COBRA model contains duplicate reaction identifiers")
    return reaction_ids


def _objective_coefficients(model: Model) -> np.ndarray:
    """Return coefficients for a finite linear reaction-flux objective."""

    if not getattr(model.objective, "is_Linear", False):
        raise FluxAnalysisError("COBRA objective must be linear")

    coefficients = linear_reaction_coefficients(model)
    if not coefficients:
        raise FluxAnalysisError(
            "COBRA model must define a non-empty linear reaction objective"
        )

    ordered = np.asarray(
        [float(coefficients.get(reaction, 0.0)) for reaction in model.reactions],
        dtype=float,
    )
    if not np.isfinite(ordered).all():
        raise FluxAnalysisError("COBRA objective contains a non-finite coefficient")
    return ordered


def _objective_direction(model: Model) -> str:
    direction = str(model.objective.direction).lower()
    if direction not in {"max", "min"}:
        raise FluxAnalysisError(
            f"Unsupported COBRA objective direction {model.objective.direction!r}"
        )
    return direction


def _validate_fraction(fraction: float) -> float:
    if isinstance(fraction, bool):
        raise FluxAnalysisError("fraction of optimum must be a finite number in (0, 1]")
    try:
        value = float(fraction)
    except (TypeError, ValueError) as error:
        raise FluxAnalysisError(
            "fraction of optimum must be a finite number in (0, 1]"
        ) from error
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise FluxAnalysisError("fraction of optimum must be a finite number in (0, 1]")
    return value


def _read_tolerance(
    tolerances: object,
    names: Sequence[str],
    description: str,
) -> float:
    """Read a nonnegative tolerance by attribute without coupling to config."""

    for name in names:
        if hasattr(tolerances, name):
            value = getattr(tolerances, name)
            break
    else:
        joined = ", ".join(names)
        raise FluxAnalysisError(
            f"Tolerance object lacks a {description} attribute ({joined})"
        )

    if isinstance(value, bool):
        raise FluxAnalysisError(f"{description} tolerance must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FluxAnalysisError(
            f"{description} tolerance must be finite and nonnegative"
        ) from error
    if not np.isfinite(result) or result < 0.0:
        raise FluxAnalysisError(f"{description} tolerance must be finite and nonnegative")
    return result


def _validation_tolerances(tolerances: object) -> tuple[float, float, float]:
    if tolerances is None:
        raise FluxAnalysisError("A numerical tolerance object is required")
    bounds = _read_tolerance(
        tolerances,
        ("bounds", "bound", "bound_tolerance", "feasibility"),
        "reaction-bound",
    )
    mass_balance = _read_tolerance(
        tolerances,
        ("mass_balance", "mass_balance_tolerance", "feasibility"),
        "mass-balance",
    )
    objective = _read_tolerance(
        tolerances,
        ("objective", "objective_floor", "objective_tolerance", "feasibility"),
        "objective-floor",
    )
    return bounds, mass_balance, objective


def run_fba(model: Model) -> FBAResult:
    """Run FBA and require a complete, finite, optimal linear solution."""

    if not isinstance(model, Model):
        raise FluxAnalysisError("run_fba requires a cobra.Model")

    reaction_ids = _reaction_ids(model)
    _objective_coefficients(model)
    direction = _objective_direction(model)

    try:
        solution = model.optimize(raise_error=False)
    except Exception as error:
        raise FluxAnalysisError(f"COBRA FBA failed: {error}") from error

    if solution.status != OPTIMAL:
        raise FluxAnalysisError(
            f"COBRA FBA did not return an optimal solution (status={solution.status!r})"
        )

    try:
        objective_value = float(solution.objective_value)
    except (TypeError, ValueError) as error:
        raise FluxAnalysisError("COBRA FBA returned an invalid objective value") from error
    if not np.isfinite(objective_value):
        raise FluxAnalysisError("COBRA FBA returned a non-finite objective value")

    fluxes = solution.fluxes.reindex(reaction_ids).astype(float)
    if list(solution.fluxes.index) != reaction_ids:
        missing = [rid for rid in reaction_ids if rid not in solution.fluxes.index]
        unexpected = [rid for rid in solution.fluxes.index if rid not in reaction_ids]
        if missing or unexpected:
            raise FluxAnalysisError(
                f"COBRA FBA returned incompatible reaction fluxes; "
                f"missing={missing}, unexpected={unexpected}"
            )
    if not np.isfinite(fluxes.to_numpy()).all():
        raise FluxAnalysisError("COBRA FBA returned non-finite reaction fluxes")
    fluxes.index.name = "reaction_id"
    fluxes.name = "flux"

    return FBAResult(
        objective_value=objective_value,
        status=solution.status,
        objective_direction=direction,
        fluxes=fluxes,
    )


def run_fva(model: Model, fraction: float) -> FVAResult:
    """Run FVA using COBRApy's objective-fraction constraint."""

    fraction_value = _validate_fraction(fraction)
    fba = run_fba(model)
    reaction_ids = _reaction_ids(model)

    try:
        ranges = flux_variability_analysis(
            model,
            reaction_list=list(model.reactions),
            fraction_of_optimum=fraction_value,
            processes=1,
        )
    except Exception as error:
        raise FluxAnalysisError(f"COBRA FVA failed: {error}") from error

    if list(ranges.index) != reaction_ids or list(ranges.columns) != [
        "minimum",
        "maximum",
    ]:
        raise FluxAnalysisError(
            "COBRA FVA returned reactions or columns in an incompatible order"
        )
    ranges = ranges.astype(float)
    values = ranges.to_numpy()
    if not np.isfinite(values).all():
        raise FluxAnalysisError("COBRA FVA returned non-finite ranges")
    if np.any(values[:, 0] > values[:, 1] + 1e-9):
        raise FluxAnalysisError("COBRA FVA returned a minimum above its maximum")
    ranges.index.name = "reaction_id"

    return FVAResult(
        ranges=ranges,
        fraction_of_optimum=fraction_value,
        objective_value=fba.objective_value,
        objective_direction=fba.objective_direction,
    )


def _structural_validation_report(
    samples: pd.DataFrame,
    reaction_columns_valid: bool,
    duplicate_columns: tuple[str, ...],
    missing_reactions: tuple[str, ...],
    unexpected_reactions: tuple[str, ...],
    column_order_valid: bool,
    errors: Sequence[str],
) -> FluxSampleValidationReport:
    details = pd.DataFrame(index=samples.index.copy())
    details.index.name = "sample_id"
    return FluxSampleValidationReport(
        valid=False,
        sample_count=len(samples),
        reaction_columns_valid=reaction_columns_valid,
        finite_values_valid=False,
        bounds_valid=False,
        mass_balance_valid=False,
        objective_floor_valid=False,
        duplicate_columns=duplicate_columns,
        missing_reactions=missing_reactions,
        unexpected_reactions=unexpected_reactions,
        column_order_valid=column_order_valid,
        max_lower_bound_violation=None,
        max_upper_bound_violation=None,
        max_mass_balance_residual=None,
        max_objective_floor_violation=None,
        details=details,
        errors=tuple(errors),
    )


def validate_flux_samples(
    model: Model,
    samples: pd.DataFrame,
    objective_floor: float,
    tolerances: object,
) -> FluxSampleValidationReport:
    """Validate complete flux vectors independently of COBRApy's sampler.

    Checks include exact reaction columns/order, finite values, reaction
    bounds, ``S v = 0``, and the maximize/minimize objective threshold.
    """

    if not isinstance(model, Model):
        raise FluxAnalysisError("validate_flux_samples requires a cobra.Model")
    if not isinstance(samples, pd.DataFrame):
        raise FluxAnalysisError("Flux samples must be provided as a pandas DataFrame")
    if samples.empty:
        raise FluxAnalysisError("Flux sample table must contain at least one sample")

    expected = _reaction_ids(model)
    observed = list(samples.columns)
    duplicate_columns = tuple(
        dict.fromkeys(str(column) for column in samples.columns[samples.columns.duplicated()])
    )
    missing = tuple(rid for rid in expected if rid not in observed)
    unexpected = tuple(str(column) for column in observed if column not in expected)
    column_order_valid = observed == expected
    reaction_columns_valid = (
        not duplicate_columns and not missing and not unexpected and column_order_valid
    )
    structural_errors: list[str] = []
    if duplicate_columns:
        structural_errors.append(f"duplicate reaction columns: {list(duplicate_columns)}")
    if missing:
        structural_errors.append(f"missing reaction columns: {list(missing)}")
    if unexpected:
        structural_errors.append(f"unexpected reaction columns: {list(unexpected)}")
    if not column_order_valid and not (missing or unexpected or duplicate_columns):
        structural_errors.append("reaction columns are not in exact COBRA model order")
    if not reaction_columns_valid:
        return _structural_validation_report(
            samples,
            reaction_columns_valid,
            duplicate_columns,
            missing,
            unexpected,
            column_order_valid,
            structural_errors,
        )

    bound_tolerance, mass_tolerance, objective_tolerance = _validation_tolerances(
        tolerances
    )
    try:
        floor = float(objective_floor)
    except (TypeError, ValueError) as error:
        raise FluxAnalysisError("Objective floor must be finite") from error
    if not np.isfinite(floor):
        raise FluxAnalysisError("Objective floor must be finite")

    try:
        values = samples.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise FluxAnalysisError("Flux samples must contain only numeric values") from error

    finite_by_sample = np.isfinite(values).all(axis=1)
    safe_values = values.copy()
    safe_values[~np.isfinite(safe_values)] = np.nan

    lower_bounds = np.asarray(
        [reaction.lower_bound for reaction in model.reactions], dtype=float
    )
    upper_bounds = np.asarray(
        [reaction.upper_bound for reaction in model.reactions], dtype=float
    )
    lower_violation = np.maximum(lower_bounds[None, :] - safe_values, 0.0)
    upper_violation = np.maximum(safe_values - upper_bounds[None, :], 0.0)
    max_lower = _row_nanmax(lower_violation)
    max_upper = _row_nanmax(upper_violation)
    bounds_by_sample = (
        finite_by_sample
        & (max_lower <= bound_tolerance)
        & (max_upper <= bound_tolerance)
    )

    stoichiometry = np.asarray(create_stoichiometric_matrix(model), dtype=float)
    residuals = safe_values @ stoichiometry.T
    max_mass = _row_nanmax(np.abs(residuals))
    mass_by_sample = finite_by_sample & (max_mass <= mass_tolerance)

    objective_coefficients = _objective_coefficients(model)
    objective_values = safe_values @ objective_coefficients
    direction = _objective_direction(model)
    if direction == "max":
        objective_violation = np.maximum(floor - objective_values, 0.0)
    else:
        objective_violation = np.maximum(objective_values - floor, 0.0)
    objective_by_sample = (
        finite_by_sample & (objective_violation <= objective_tolerance)
    )

    valid_by_sample = (
        finite_by_sample & bounds_by_sample & mass_by_sample & objective_by_sample
    )
    details = pd.DataFrame(
        {
            "finite_values": finite_by_sample,
            "max_lower_bound_violation": max_lower,
            "max_upper_bound_violation": max_upper,
            "bounds_valid": bounds_by_sample,
            "max_mass_balance_residual": max_mass,
            "mass_balance_valid": mass_by_sample,
            "objective_value": objective_values,
            "objective_floor_violation": objective_violation,
            "objective_floor_valid": objective_by_sample,
            "valid": valid_by_sample,
        },
        index=samples.index.copy(),
    )
    details.index.name = "sample_id"

    errors = []
    if not finite_by_sample.all():
        errors.append("one or more samples contain non-finite values")
    if not bounds_by_sample.all():
        errors.append("one or more samples violate reaction bounds")
    if not mass_by_sample.all():
        errors.append("one or more samples violate steady-state mass balance")
    if not objective_by_sample.all():
        errors.append("one or more samples violate the objective floor")

    return FluxSampleValidationReport(
        valid=bool(valid_by_sample.all()),
        sample_count=len(samples),
        reaction_columns_valid=True,
        finite_values_valid=bool(finite_by_sample.all()),
        bounds_valid=bool(bounds_by_sample.all()),
        mass_balance_valid=bool(mass_by_sample.all()),
        objective_floor_valid=bool(objective_by_sample.all()),
        duplicate_columns=(),
        missing_reactions=(),
        unexpected_reactions=(),
        column_order_valid=True,
        max_lower_bound_violation=_finite_max_or_none(max_lower),
        max_upper_bound_violation=_finite_max_or_none(max_upper),
        max_mass_balance_residual=_finite_max_or_none(max_mass),
        max_objective_floor_violation=_finite_max_or_none(objective_violation),
        details=details,
        errors=tuple(errors),
    )


def _row_nanmax(values: np.ndarray) -> np.ndarray:
    """Calculate per-row maxima without warnings for all-NaN rows."""

    result = np.full(values.shape[0], np.nan, dtype=float)
    finite_rows = np.isfinite(values).any(axis=1)
    if finite_rows.any():
        result[finite_rows] = np.nanmax(values[finite_rows], axis=1)
    return result


def _finite_max_or_none(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(finite.max())


def sample_fluxes(
    model: Model,
    count: int,
    fraction: float,
    seed: int,
    sampler: str,
    tolerances: object,
) -> FluxSamplingResult:
    """Sample complete flux vectors under a persistent objective floor."""

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise FluxAnalysisError("Flux sample count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise FluxAnalysisError("Flux-sampling seed must be a nonnegative integer")
    sampler_name = str(sampler).lower()
    if sampler_name not in {"achr", "optgp"}:
        raise FluxAnalysisError("Sampler must be either 'achr' or 'optgp'")

    fraction_value = _validate_fraction(fraction)
    _validation_tolerances(tolerances)
    fba = run_fba(model)
    objective_floor = fba.objective_value * fraction_value
    if not np.isfinite(objective_floor):
        raise FluxAnalysisError("Calculated objective floor is non-finite")

    try:
        fixed_bound = fix_objective_as_constraint(
            model,
            bound=objective_floor,
            name=OBJECTIVE_FLOOR_CONSTRAINT,
        )
        model.solver.update()
    except Exception as error:
        raise FluxAnalysisError(f"Could not create objective-floor constraint: {error}") from error
    if fixed_bound != objective_floor or OBJECTIVE_FLOOR_CONSTRAINT not in model.constraints:
        raise FluxAnalysisError("COBRApy did not preserve the exact objective-floor constraint")

    try:
        if sampler_name == "achr":
            sampler_instance = ACHRSampler(model, seed=int(seed))
        else:
            sampler_instance = OptGPSampler(model, processes=1, seed=int(seed))
        samples = sampler_instance.sample(int(count), fluxes=True)
    except Exception as error:
        raise FluxAnalysisError(
            f"COBRA {sampler_name.upper()} sampling failed: {error}"
        ) from error

    samples = pd.DataFrame(samples)
    expected = _reaction_ids(model)
    if list(samples.columns) == expected:
        samples = samples.astype(float)
    samples.index = pd.Index(
        [f"sample_{index:04d}" for index in range(len(samples))],
        name="sample_id",
    )
    if len(samples) != count:
        raise FluxAnalysisError(
            f"COBRA sampler returned {len(samples)} samples, expected exactly {count}"
        )

    validation = validate_flux_samples(
        model,
        samples,
        objective_floor=objective_floor,
        tolerances=tolerances,
    )
    if not validation.valid:
        raise FluxAnalysisError(
            "COBRA sampler returned invalid flux vectors: " + "; ".join(validation.errors)
        )

    return FluxSamplingResult(
        samples=samples,
        sampler=sampler_name,
        seed=int(seed),
        fraction_of_optimum=fraction_value,
        objective_value=fba.objective_value,
        objective_floor=objective_floor,
        objective_direction=fba.objective_direction,
        validation=validation,
    )


__all__ = [
    "FBAResult",
    "FVAResult",
    "FluxSampleValidationReport",
    "FluxSamplingResult",
    "OBJECTIVE_FLOOR_CONSTRAINT",
    "run_fba",
    "run_fva",
    "sample_fluxes",
    "validate_flux_samples",
]
