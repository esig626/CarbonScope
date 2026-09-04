"""Native HiGHS FBA/FVA over canonical ``FluxModel``.

The reference FVA intentionally cold-starts each endpoint.  Production FVA uses
worker-local reusable HiGHS models and dynamically scheduled endpoint jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing
import os
from numbers import Real
from typing import Sequence

import pandas as pd

from fluxemu.exceptions import AnalysisError
from fluxemu.model.schema import FluxModel
from fluxemu.model.serialisation import deterministic_serialise
from fluxemu.model.validation import CanonicalModelError, validate_flux_model
from .results import FBAResult, FVAResult, PrimalDiagnostics

FEASIBILITY_TOLERANCE = 1e-7
OBJECTIVE_TOLERANCE = 1e-7


@dataclass(frozen=True, slots=True)
class CompiledFluxLP:
    """Immutable row-compressed canonical LP; ordering is scientifically significant."""

    reaction_ids: tuple[str, ...]
    balanced_metabolite_ids: tuple[str, ...]
    row_starts: tuple[int, ...]
    column_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    objective_coefficients: tuple[float, ...]
    objective_direction: str
    fingerprint: str

    @property
    def nonzero_count(self) -> int:
        return len(self.coefficients)


@dataclass(frozen=True, slots=True)
class RetainedObjectiveConstraint:
    """The biological-objective constraint shared by FVA and sampling."""

    fraction_of_optimum: float
    biological_optimum: float
    bound: float
    sense: str
    objective_direction: str


@dataclass(frozen=True, slots=True)
class PreparedFluxRegion:
    """One compiled LP and its single independently validated FBA optimum."""

    lp: CompiledFluxLP
    fba: FBAResult
    retention: RetainedObjectiveConstraint
    flux_model: FluxModel
    flux_model_fingerprint: str


def _flux_model_fingerprint(model: FluxModel) -> str:
    """Hash every ordered source-model field without rebuilding the compiled LP."""

    return hashlib.sha256(deterministic_serialise(model).encode("utf-8")).hexdigest()


def _compiled_lp_fingerprint(lp: CompiledFluxLP) -> str:
    """Recompute the identity carried by immutable compiled-LP fields."""

    identity = json.dumps(
        [
            lp.reaction_ids,
            lp.balanced_metabolite_ids,
            lp.row_starts,
            lp.column_indices,
            lp.coefficients,
            lp.lower_bounds,
            lp.upper_bounds,
            lp.objective_coefficients,
            lp.objective_direction,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _validate_compiled_lp_structure(lp: CompiledFluxLP) -> None:
    """Fail closed on forged CSR records before indexing or fingerprint use."""

    tuple_fields = (
        lp.reaction_ids,
        lp.balanced_metabolite_ids,
        lp.row_starts,
        lp.column_indices,
        lp.coefficients,
        lp.lower_bounds,
        lp.upper_bounds,
        lp.objective_coefficients,
    )
    if not all(isinstance(field, tuple) for field in tuple_fields):
        raise AnalysisError("prepared compiled LP contains non-tuple fields")
    if (
        not lp.reaction_ids
        or any(not isinstance(identifier, str) or not identifier for identifier in lp.reaction_ids)
        or len(set(lp.reaction_ids)) != len(lp.reaction_ids)
        or any(
            not isinstance(identifier, str) or not identifier
            for identifier in lp.balanced_metabolite_ids
        )
        or len(set(lp.balanced_metabolite_ids)) != len(lp.balanced_metabolite_ids)
    ):
        raise AnalysisError("prepared compiled LP contains malformed identifiers")
    n = len(lp.reaction_ids)
    m = len(lp.balanced_metabolite_ids)
    if (
        len(lp.row_starts) != m + 1
        or len(lp.column_indices) != len(lp.coefficients)
        or len(lp.lower_bounds) != n
        or len(lp.upper_bounds) != n
        or len(lp.objective_coefficients) != n
    ):
        raise AnalysisError("prepared compiled LP contains inconsistent vector lengths")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in lp.row_starts)
        or lp.row_starts[0] != 0
        or lp.row_starts[-1] != len(lp.column_indices)
        or any(left > right for left, right in zip(lp.row_starts, lp.row_starts[1:]))
    ):
        raise AnalysisError("prepared compiled LP contains malformed CSR row starts")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n
        for index in lp.column_indices
    ):
        raise AnalysisError("prepared compiled LP contains an out-of-range column index")
    numeric_fields = (
        lp.coefficients,
        lp.lower_bounds,
        lp.upper_bounds,
        lp.objective_coefficients,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for field in numeric_fields
        for value in field
    ):
        raise AnalysisError("prepared compiled LP contains malformed or non-finite numbers")
    if any(lower > upper for lower, upper in zip(lp.lower_bounds, lp.upper_bounds)):
        raise AnalysisError("prepared compiled LP contains inconsistent reaction bounds")
    if lp.objective_direction not in {"max", "min"}:
        raise AnalysisError("prepared compiled LP contains an invalid objective direction")


def _finite_aggregate(values: Sequence[float], error_message: str) -> float:
    """Stably aggregate duplicate linear terms and reject overflow."""

    try:
        result = math.fsum(float(value) for value in values)
    except (OverflowError, TypeError, ValueError) as error:
        raise AnalysisError(error_message) from error
    if not math.isfinite(result):
        raise AnalysisError(error_message)
    return result


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
    reaction_index = {rid: i for i, rid in enumerate(reaction_ids)}
    balanced = tuple(m.metabolite_id for m in model.metabolites if m.steady_state_balanced)
    terms_by_metabolite: dict[str, list[tuple[int, float]]] = {mid: [] for mid in balanced}
    for column, reaction in enumerate(model.reactions):
        grouped: dict[str, list[float]] = {}
        for term in reaction.stoichiometric_terms:
            grouped.setdefault(term.metabolite_id, []).append(float(term.coefficient))
        accumulated = {
            metabolite_id: _finite_aggregate(
                coefficients,
                "invalid canonical flux model: aggregated stoichiometric "
                f"coefficient is non-finite for reaction {reaction.reaction_id!r}, "
                f"metabolite {metabolite_id!r}",
            )
            for metabolite_id, coefficients in grouped.items()
        }
        for mid, value in accumulated.items():
            if mid in terms_by_metabolite and value != 0.0:
                terms_by_metabolite[mid].append((column, value))
    starts = [0]; indices: list[int] = []; values: list[float] = []
    for mid in balanced:
        entries = terms_by_metabolite[mid]
        if any(i < 0 or i >= len(reaction_ids) for i, _ in entries):
            raise AnalysisError("invalid canonical flux model: malformed sparse index")
        indices.extend(i for i, _ in entries); values.extend(v for _, v in entries); starts.append(len(indices))
    grouped_objective: list[list[float]] = [[] for _ in reaction_ids]
    for term in model.objective.terms:
        index = reaction_index[term.reaction_id]
        grouped_objective[index].append(float(term.coefficient))
    objective = [
        _finite_aggregate(
            coefficients,
            (
                "invalid canonical flux model: aggregated objective coefficient "
                f"is non-finite for reaction {reaction_ids[index]!r}"
            ),
        )
        for index, coefficients in enumerate(grouped_objective)
    ]
    direction = {"maximise": "max", "minimise": "min"}[model.objective.direction]
    compiled = CompiledFluxLP(
        reaction_ids,
        balanced,
        tuple(starts),
        tuple(indices),
        tuple(values),
        tuple(float(r.lower_bound) for r in model.reactions),
        tuple(float(r.upper_bound) for r in model.reactions),
        tuple(objective),
        direction,
        "",
    )
    return CompiledFluxLP(
        reaction_ids=compiled.reaction_ids,
        balanced_metabolite_ids=compiled.balanced_metabolite_ids,
        row_starts=compiled.row_starts,
        column_indices=compiled.column_indices,
        coefficients=compiled.coefficients,
        lower_bounds=compiled.lower_bounds,
        upper_bounds=compiled.upper_bounds,
        objective_coefficients=compiled.objective_coefficients,
        objective_direction=compiled.objective_direction,
        fingerprint=_compiled_lp_fingerprint(compiled),
    )


def _highspy():
    try:
        import highspy
    except ImportError as error:
        raise AnalysisError("native HiGHS analysis requires the default 'highspy' dependency") from error
    return highspy


def _finite_dot(
    coefficients: Sequence[float],
    values: Sequence[float],
    context: str,
    *,
    inputs_prevalidated: bool = False,
) -> float:
    """Evaluate one linear form and reject every non-finite intermediate."""

    if len(coefficients) != len(values):
        raise AnalysisError(f"{context} has inconsistent vector lengths")

    if inputs_prevalidated:
        def products():
            for coefficient, value in zip(coefficients, values):
                product = coefficient * value
                if not math.isfinite(product):
                    raise AnalysisError(f"{context} produced a non-finite product")
                yield product
    else:
        def products():
            for coefficient, value in zip(coefficients, values):
                if (
                    isinstance(coefficient, bool)
                    or isinstance(value, bool)
                    or not isinstance(coefficient, Real)
                    or not isinstance(value, Real)
                ):
                    raise AnalysisError(f"{context} contains a non-numeric value")
                try:
                    left = float(coefficient)
                    right = float(value)
                    product = left * right
                except (OverflowError, TypeError, ValueError) as error:
                    raise AnalysisError(
                        f"{context} overflowed while evaluating a product"
                    ) from error
                if not (
                    math.isfinite(left)
                    and math.isfinite(right)
                    and math.isfinite(product)
                ):
                    raise AnalysisError(f"{context} produced a non-finite product")
                yield product

    try:
        result = math.fsum(products())
    except OverflowError as error:
        raise AnalysisError(f"{context} overflowed during summation") from error
    if not math.isfinite(result):
        raise AnalysisError(f"{context} produced a non-finite result")
    return result


def _finite_sparse_row_value(
    lp: CompiledFluxLP,
    fluxes: Sequence[float],
    row: int,
) -> float:
    """Evaluate one prevalidated CSR balance row without slice allocations."""

    context = f"mass balance for metabolite {lp.balanced_metabolite_ids[row]!r}"

    def products():
        for offset in range(lp.row_starts[row], lp.row_starts[row + 1]):
            coefficient = lp.coefficients[offset]
            value = fluxes[lp.column_indices[offset]]
            product = coefficient * value
            if not math.isfinite(product):
                raise AnalysisError(f"{context} produced a non-finite product")
            yield product

    try:
        result = math.fsum(products())
    except OverflowError as error:
        raise AnalysisError(f"{context} overflowed during summation") from error
    if not math.isfinite(result):
        raise AnalysisError(f"{context} produced a non-finite result")
    return result


def _solve(lp: CompiledFluxLP, costs: Sequence[float], direction: str, operation: str,
           retention: tuple[str, float] | None = None) -> tuple[list[float], float, str]:
    highspy = _highspy(); solver = highspy.Highs()
    solver.setOptionValue("output_flag", False); solver.setOptionValue("threads", 1)
    solver.setOptionValue("solver", "simplex")
    n = len(lp.reaction_ids)
    solver.addCols(n, list(costs), list(lp.lower_bounds), list(lp.upper_bounds), 0, [0] * (n + 1), [], [])
    lower = [0.0] * len(lp.balanced_metabolite_ids); upper = [0.0] * len(lower)
    starts = list(lp.row_starts); indices = list(lp.column_indices); values = list(lp.coefficients)
    if retention is not None:
        sense, bound = retention
        lower.append(bound if sense == ">=" else -highspy.kHighsInf)
        upper.append(bound if sense == "<=" else highspy.kHighsInf)
        indices.extend(i for i, value in enumerate(lp.objective_coefficients) if value != 0.0)
        values.extend(value for value in lp.objective_coefficients if value != 0.0)
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


def _validate(
    lp: CompiledFluxLP,
    fluxes: Sequence[float],
    reported: float,
    costs: Sequence[float],
    *,
    lp_prevalidated: bool = False,
) -> PrimalDiagnostics:
    if not lp_prevalidated:
        _validate_compiled_lp_structure(lp)
    try:
        if isinstance(reported, bool) or not isinstance(reported, Real):
            raise TypeError
        reported_value = float(reported)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("independent primal validation received malformed numbers") from error
    if len(fluxes) != len(lp.reaction_ids) or len(costs) != len(lp.reaction_ids):
        raise AnalysisError("independent primal validation received wrong vector lengths")
    if not math.isfinite(reported_value) or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for values in (fluxes, costs)
        for value in values
    ):
        raise AnalysisError("independent primal validation received non-finite numbers")
    lower = max(
        (
            max(lb - value, 0.0)
            for lb, value in zip(lp.lower_bounds, fluxes)
        ),
        default=0.0,
    )
    upper = max(
        (
            max(value - ub, 0.0)
            for ub, value in zip(lp.upper_bounds, fluxes)
        ),
        default=0.0,
    )
    residual = 0.0
    for row in range(len(lp.balanced_metabolite_ids)):
        value = _finite_sparse_row_value(lp, fluxes, row)
        residual = max(residual, abs(value))
    recalculated = _finite_dot(
        costs,
        fluxes,
        "objective recalculation",
        inputs_prevalidated=True,
    )
    error = abs(recalculated - reported_value)
    if not all(math.isfinite(value) for value in (lower, upper, residual, error)):
        raise AnalysisError("independent primal validation produced non-finite diagnostics")
    diagnostics = PrimalDiagnostics(lower, upper, residual, error)
    if max(lower, upper, residual) > FEASIBILITY_TOLERANCE or error > OBJECTIVE_TOLERANCE:
        raise AnalysisError(f"independent primal validation failed: {diagnostics}")
    return diagnostics


def _run_compiled_fba(lp: CompiledFluxLP) -> FBAResult:
    """Solve and independently validate the biological objective on ``lp``."""

    fluxes, objective, status = _solve(lp, lp.objective_coefficients, lp.objective_direction, "FBA")
    diagnostics = _validate(
        lp,
        fluxes,
        objective,
        lp.objective_coefficients,
        lp_prevalidated=True,
    )
    return FBAResult(objective, "optimal", lp.objective_direction,
                     pd.Series(fluxes, index=lp.reaction_ids, dtype=float), diagnostics)


def run_highs_fba(model: FluxModel) -> FBAResult:
    return _run_compiled_fba(compile_flux_lp(model))


def _fraction(value: float) -> float:
    if isinstance(value, bool):
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]")
    try: result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]") from error
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]")
    return result


def _retained_objective_constraint(
    lp: CompiledFluxLP,
    fba: FBAResult,
    fraction_of_optimum: float,
) -> RetainedObjectiveConstraint:
    """Build the one canonical multiplicative objective-retention constraint.

    The formula is deliberately literal: for maximisation the bound is
    ``c @ v >= fraction * z_star`` and for minimisation it is the reversed
    inequality.  Consequently, fractions below one make the requested region
    empty for a negative maximum or a positive minimum.  That case is rejected
    here, before any endpoint or sampling solve can obscure the reason.
    """

    fraction = _fraction(fraction_of_optimum)
    try:
        if isinstance(fba.objective_value, bool):
            raise TypeError
        optimum = float(fba.objective_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("biological optimum must be a finite number") from error
    if not math.isfinite(optimum):
        raise AnalysisError("biological optimum must be a finite number")
    retained_bound = optimum * fraction
    tightens_past_optimum = (
        retained_bound > optimum
        if lp.objective_direction == "max"
        else retained_bound < optimum
    )
    if tightens_past_optimum:
        raise AnalysisError(
            "retained objective region is empty: "
            f"direction={lp.objective_direction}, "
            f"biological_optimum={optimum:g}, "
            f"fraction_of_optimum={fraction:g}, "
            f"retained_bound={retained_bound:g}"
        )
    return RetainedObjectiveConstraint(
        fraction_of_optimum=fraction,
        biological_optimum=optimum,
        bound=retained_bound,
        sense=">=" if lp.objective_direction == "max" else "<=",
        objective_direction=lp.objective_direction,
    )


def prepare_highs_flux_region(
    model: FluxModel, fraction_of_optimum: float = 1.0
) -> PreparedFluxRegion:
    """Compile once and solve the retained biological objective exactly once."""

    fraction = _fraction(fraction_of_optimum)
    lp = compile_flux_lp(model)
    fba = _run_compiled_fba(lp)
    retention = _retained_objective_constraint(lp, fba, fraction)
    return PreparedFluxRegion(
        lp=lp,
        fba=fba,
        retention=retention,
        flux_model=model,
        flux_model_fingerprint=_flux_model_fingerprint(model),
    )


def _validate_prepared_flux_region(prepared: PreparedFluxRegion) -> None:
    """Reject forged or internally inconsistent prepared-region records."""

    if not isinstance(prepared, PreparedFluxRegion):
        raise AnalysisError("prepared analysis requires a PreparedFluxRegion")
    lp, fba, retention, model, source_fingerprint = (
        prepared.lp,
        prepared.fba,
        prepared.retention,
        prepared.flux_model,
        prepared.flux_model_fingerprint,
    )
    if (
        not isinstance(lp, CompiledFluxLP)
        or not isinstance(fba, FBAResult)
        or not isinstance(retention, RetainedObjectiveConstraint)
        or not isinstance(model, FluxModel)
        or not isinstance(source_fingerprint, str)
    ):
        raise AnalysisError("prepared flux region contains malformed records")
    try:
        validate_flux_model(model)
    except (CanonicalModelError, AttributeError) as error:
        raise AnalysisError(f"prepared flux region contains an invalid model: {error}") from error
    _validate_compiled_lp_structure(lp)
    if source_fingerprint != _flux_model_fingerprint(model):
        raise AnalysisError("prepared canonical model fingerprint is inconsistent")
    if lp.fingerprint != _compiled_lp_fingerprint(lp):
        raise AnalysisError("prepared compiled-LP fingerprint is inconsistent")
    model_reaction_ids = tuple(reaction.reaction_id for reaction in model.reactions)
    model_balanced_ids = tuple(
        metabolite.metabolite_id
        for metabolite in model.metabolites
        if metabolite.steady_state_balanced
    )
    if (
        model_reaction_ids != lp.reaction_ids
        or model_balanced_ids != lp.balanced_metabolite_ids
        or tuple(float(reaction.lower_bound) for reaction in model.reactions)
        != lp.lower_bounds
        or tuple(float(reaction.upper_bound) for reaction in model.reactions)
        != lp.upper_bounds
    ):
        raise AnalysisError("prepared canonical model does not match its compiled LP")
    reaction_index = {
        reaction_id: index for index, reaction_id in enumerate(model_reaction_ids)
    }
    grouped_objective: list[list[float]] = [[] for _ in model_reaction_ids]
    for term in model.objective.terms:
        index = reaction_index[term.reaction_id]
        grouped_objective[index].append(float(term.coefficient))
    model_objective = [
        _finite_aggregate(
            coefficients,
            (
                "prepared canonical model has a non-finite aggregated objective "
                f"coefficient for reaction {model_reaction_ids[index]!r}"
            ),
        )
        for index, coefficients in enumerate(grouped_objective)
    ]
    model_direction = {
        "maximise": "max",
        "minimise": "min",
    }[model.objective.direction]
    if (
        tuple(model_objective) != lp.objective_coefficients
        or model_direction != lp.objective_direction
    ):
        raise AnalysisError("prepared canonical model objective does not match its compiled LP")
    for row, metabolite_id in enumerate(model_balanced_ids):
        expected_entries: list[tuple[int, float]] = []
        for column, reaction in enumerate(model.reactions):
            coefficient = _finite_aggregate(
                tuple(
                    float(term.coefficient)
                    for term in reaction.stoichiometric_terms
                    if term.metabolite_id == metabolite_id
                ),
                "prepared canonical model has a non-finite aggregated "
                f"stoichiometric coefficient for reaction {reaction.reaction_id!r}, "
                f"metabolite {metabolite_id!r}",
            )
            if coefficient != 0.0:
                expected_entries.append((column, coefficient))
        actual_entries = list(
            zip(
                lp.column_indices[lp.row_starts[row] : lp.row_starts[row + 1]],
                lp.coefficients[lp.row_starts[row] : lp.row_starts[row + 1]],
            )
        )
        if expected_entries != actual_entries:
            raise AnalysisError(
                "prepared canonical model stoichiometry does not match its compiled LP"
            )
    if not isinstance(fba.fluxes, pd.Series):
        raise AnalysisError("prepared FBA fluxes must be a pandas Series")
    if tuple(fba.fluxes.index) != lp.reaction_ids:
        raise AnalysisError("prepared FBA reaction order does not match its compiled LP")
    if fba.status != "optimal" or fba.objective_direction != lp.objective_direction:
        raise AnalysisError("prepared FBA status or objective direction is inconsistent")
    if len(fba.fluxes) != len(lp.reaction_ids):
        raise AnalysisError("prepared FBA primal has the wrong vector length")
    try:
        if isinstance(fba.objective_value, bool):
            raise TypeError
        optimum = float(fba.objective_value)
        fluxes = tuple(float(value) for value in fba.fluxes)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("prepared FBA contains malformed numeric values") from error
    if not math.isfinite(optimum):
        raise AnalysisError("prepared FBA objective is non-finite")
    if not all(math.isfinite(value) for value in fluxes):
        raise AnalysisError("prepared FBA primal contains non-finite values")
    try:
        _validate(
            lp,
            fluxes,
            optimum,
            lp.objective_coefficients,
            lp_prevalidated=True,
        )
    except (IndexError, TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("prepared FBA primal is malformed") from error
    fraction = _fraction(retention.fraction_of_optimum)
    try:
        if isinstance(retention.biological_optimum, bool) or isinstance(
            retention.bound, bool
        ):
            raise TypeError
        recorded_optimum = float(retention.biological_optimum)
        recorded_bound = float(retention.bound)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("prepared retained-objective values are malformed") from error
    if not math.isfinite(recorded_optimum) or not math.isfinite(recorded_bound):
        raise AnalysisError("prepared retained-objective values must be finite")
    expected = _retained_objective_constraint(lp, fba, fraction)
    if (
        retention.objective_direction != expected.objective_direction
        or retention.sense != expected.sense
        or recorded_optimum != expected.biological_optimum
        or recorded_bound != expected.bound
    ):
        raise AnalysisError("prepared retained-objective metadata is inconsistent")


def run_highs_fva_reference(model: FluxModel, fraction_of_optimum: float = 1.0) -> FVAResult:
    fraction = _fraction(fraction_of_optimum); lp = compile_flux_lp(model); fba = _run_compiled_fba(lp)
    retained = _retained_objective_constraint(lp, fba, fraction)
    retention = (retained.sense, retained.bound)
    minima: list[float] = []; maxima: list[float] = []
    for j, reaction_id in enumerate(lp.reaction_ids):
        costs = [0.0] * len(lp.reaction_ids); costs[j] = 1.0
        low_flux, low, _ = _solve(lp, costs, "min", f"FVA minimum for reaction {reaction_id!r}", retention)
        _validate(lp, low_flux, low, costs, lp_prevalidated=True)
        high_flux, high, _ = _solve(lp, costs, "max", f"FVA maximum for reaction {reaction_id!r}", retention)
        _validate(lp, high_flux, high, costs, lp_prevalidated=True)
        # Independently enforce the retained biological objective at every endpoint.
        for endpoint in (low_flux, high_flux):
            value = _finite_dot(
                lp.objective_coefficients,
                endpoint,
                f"FVA endpoint for reaction {reaction_id!r} retained objective",
                inputs_prevalidated=True,
            )
            violation = max(retention[1] - value, 0.0) if retention[0] == ">=" else max(value - retention[1], 0.0)
            if violation > OBJECTIVE_TOLERANCE:
                raise AnalysisError(f"FVA endpoint for reaction {reaction_id!r} violates objective retention")
        minima.append(low); maxima.append(high)
    ranges = pd.DataFrame({"minimum": minima, "maximum": maxima}, index=lp.reaction_ids)
    return FVAResult(ranges, fraction, fba.objective_value, lp.objective_direction)


@dataclass(frozen=True, slots=True)
class _EndpointResult:
    reaction_index: int
    direction: str
    value: float
    worker_pid: int


class _ReusableFVAWorker:
    """One retained LP whose simplex state is reused across endpoint solves."""

    def __init__(self, lp: CompiledFluxLP, retention: tuple[str, float]):
        highspy = _highspy()
        _validate_compiled_lp_structure(lp)
        self.lp, self.retention, self.endpoint_count = lp, retention, 0
        self.solver = highspy.Highs()
        self.solver.setOptionValue("output_flag", False)
        self.solver.setOptionValue("threads", 1)
        self.solver.setOptionValue("solver", "simplex")
        n = len(lp.reaction_ids)
        self.solver.addCols(n, [0.0] * n, list(lp.lower_bounds), list(lp.upper_bounds),
                            0, [0] * (n + 1), [], [])
        lower = [0.0] * len(lp.balanced_metabolite_ids)
        upper = [0.0] * len(lower)
        starts = list(lp.row_starts)
        indices = list(lp.column_indices)
        values = list(lp.coefficients)
        sense, bound = retention
        lower.append(bound if sense == ">=" else -highspy.kHighsInf)
        upper.append(bound if sense == "<=" else highspy.kHighsInf)
        indices.extend(i for i, value in enumerate(lp.objective_coefficients) if value)
        values.extend(value for value in lp.objective_coefficients if value)
        starts.append(len(indices))
        self.solver.addRows(len(lower), lower, upper, len(indices), starts, indices, values)

    def solve(self, task: tuple[int, str]) -> _EndpointResult:
        j, direction = task
        reaction_id = self.lp.reaction_ids[j]
        operation = f"FVA {direction}imum for reaction {reaction_id!r}"
        try:
            # changeColCost invalidates the objective but retains the model and the
            # incumbent simplex basis. HiGHS therefore reoptimizes on the next run.
            if self.endpoint_count:
                previous = getattr(self, "_previous_index")
                self.solver.changeColCost(previous, 0.0)
            self.solver.changeColCost(j, 1.0)
            self._previous_index = j
            self.solver.setMaximize() if direction == "max" else self.solver.setMinimize()
            self.solver.run()
            highspy = _highspy()
            status = self.solver.getModelStatus()
            status_name = self.solver.modelStatusToString(status)
            if status != highspy.HighsModelStatus.kOptimal:
                raise AnalysisError(f"HiGHS status {status_name}")
            fluxes = [float(v) for v in self.solver.getSolution().col_value]
            value = float(self.solver.getObjectiveValue())
            if len(fluxes) != len(self.lp.reaction_ids) or not all(map(math.isfinite, fluxes)) or not math.isfinite(value):
                raise AnalysisError("malformed or non-finite complete primal")
            costs = [0.0] * len(fluxes); costs[j] = 1.0
            _validate(
                self.lp,
                fluxes,
                value,
                costs,
                lp_prevalidated=True,
            )
            biological = _finite_dot(
                self.lp.objective_coefficients,
                fluxes,
                f"{operation} retained biological objective",
                inputs_prevalidated=True,
            )
            sense, bound = self.retention
            violation = max(bound - biological, 0.0) if sense == ">=" else max(biological - bound, 0.0)
            if violation > OBJECTIVE_TOLERANCE:
                raise AnalysisError(f"retained biological objective violation {violation:g}")
        except Exception as error:
            if isinstance(error, AnalysisError) and str(error).startswith(operation):
                raise
            status_name = locals().get("status_name", "not available")
            raise AnalysisError(f"{operation} failed: solver status {status_name}; {error}") from error
        self.endpoint_count += 1
        return _EndpointResult(j, direction, value, os.getpid())


_PROCESS_WORKER: _ReusableFVAWorker | None = None


def _initialize_fva_process(lp: CompiledFluxLP, retention: tuple[str, float]) -> None:
    global _PROCESS_WORKER
    _PROCESS_WORKER = _ReusableFVAWorker(lp, retention)


def _solve_process_endpoint(task: tuple[int, str]) -> _EndpointResult:
    if _PROCESS_WORKER is None:  # pragma: no cover - defensive process contract
        raise AnalysisError("FVA worker was not initialized")
    return _PROCESS_WORKER.solve(task)


def run_prepared_highs_vffva(
    prepared: PreparedFluxRegion,
    *,
    workers: int | None = None,
    instrumentation: dict[str, int] | None = None,
) -> FVAResult:
    """Run reusable FVA over an already compiled and optimized flux region.

    Each worker creates one LP and repeatedly changes only column costs and the
    objective sense. Repeated ``Highs.run`` calls naturally retain the current
    simplex basis; no basis export/import is needed.
    """
    _validate_prepared_flux_region(prepared)
    lp = prepared.lp
    retention = (prepared.retention.sense, prepared.retention.bound)
    tasks = [(j, direction) for j in range(len(lp.reaction_ids)) for direction in ("min", "max")]
    if workers is None:
        # Serial is the portable, reusable default.  Callers opt into spawned
        # process workers explicitly, avoiding recursive top-level execution
        # on spawn-based platforms and process overhead on ordinary models.
        workers = 1
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise AnalysisError("workers must be a positive integer")
    workers = min(workers, len(tasks))
    if workers == 1:
        worker = _ReusableFVAWorker(lp, retention)
        results = [worker.solve(task) for task in tasks]
    else:
        # imap_unordered(chunksize=1) is a shared dynamic task queue: a process
        # receives its next independent endpoint only after completing one.
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers, _initialize_fva_process, (lp, retention)) as pool:
            results = list(pool.imap_unordered(_solve_process_endpoint, tasks, chunksize=1))
    minima = [math.nan] * len(lp.reaction_ids); maxima = [math.nan] * len(lp.reaction_ids)
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
    if instrumentation is not None:
        process_ids = {result.worker_pid for result in results}
        instrumentation.update(solver_instances=len(process_ids), matrix_builds=len(process_ids),
                               retention_rows=len(process_ids), endpoint_solves=len(results),
                               objective_changes=len(results))
    ranges = pd.DataFrame({"minimum": minima, "maximum": maxima}, index=lp.reaction_ids)
    return FVAResult(
        ranges,
        prepared.retention.fraction_of_optimum,
        prepared.retention.biological_optimum,
        lp.objective_direction,
    )


def run_highs_vffva(model: FluxModel, fraction_of_optimum: float = 1.0, *,
                     workers: int | None = None,
                     instrumentation: dict[str, int] | None = None) -> FVAResult:
    """Prepare and run VFFVA-style dynamically scheduled native HiGHS FVA."""

    prepared = prepare_highs_flux_region(model, fraction_of_optimum)
    return run_prepared_highs_vffva(
        prepared, workers=workers, instrumentation=instrumentation
    )
