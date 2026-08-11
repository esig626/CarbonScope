"""Cold-start native HiGHS reference FBA/FVA over canonical ``FluxModel``."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import pandas as pd

from fluxemu.exceptions import AnalysisError
from fluxemu.model.schema import FluxModel
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
        accumulated: dict[str, float] = {}
        for term in reaction.stoichiometric_terms:
            accumulated[term.metabolite_id] = accumulated.get(term.metabolite_id, 0.0) + float(term.coefficient)
        for mid, value in accumulated.items():
            if mid in terms_by_metabolite and value != 0.0:
                terms_by_metabolite[mid].append((column, value))
    starts = [0]; indices: list[int] = []; values: list[float] = []
    for mid in balanced:
        entries = terms_by_metabolite[mid]
        if any(i < 0 or i >= len(reaction_ids) for i, _ in entries):
            raise AnalysisError("invalid canonical flux model: malformed sparse index")
        indices.extend(i for i, _ in entries); values.extend(v for _, v in entries); starts.append(len(indices))
    objective = [0.0] * len(reaction_ids)
    for term in model.objective.terms:
        objective[reaction_index[term.reaction_id]] += float(term.coefficient)
    direction = {"maximise": "max", "minimise": "min"}[model.objective.direction]
    identity = json.dumps([reaction_ids, balanced, starts, indices, values,
                           [float(r.lower_bound) for r in model.reactions],
                           [float(r.upper_bound) for r in model.reactions], objective, direction],
                          separators=(",", ":"), ensure_ascii=True)
    return CompiledFluxLP(reaction_ids, balanced, tuple(starts), tuple(indices), tuple(values),
                          tuple(float(r.lower_bound) for r in model.reactions),
                          tuple(float(r.upper_bound) for r in model.reactions), tuple(objective),
                          direction, hashlib.sha256(identity.encode()).hexdigest())


def _highspy():
    try:
        import highspy
    except ImportError as error:
        raise AnalysisError("native HiGHS analysis requires the optional 'highs' extra") from error
    return highspy


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


def _validate(lp: CompiledFluxLP, fluxes: Sequence[float], reported: float,
              costs: Sequence[float]) -> PrimalDiagnostics:
    lower = max((max(lb - v, 0.0) for lb, v in zip(lp.lower_bounds, fluxes)), default=0.0)
    upper = max((max(v - ub, 0.0) for ub, v in zip(lp.upper_bounds, fluxes)), default=0.0)
    residual = 0.0
    for row in range(len(lp.balanced_metabolite_ids)):
        value = sum(lp.coefficients[k] * fluxes[lp.column_indices[k]]
                    for k in range(lp.row_starts[row], lp.row_starts[row + 1]))
        residual = max(residual, abs(value))
    recalculated = sum(c * v for c, v in zip(costs, fluxes)); error = abs(recalculated - reported)
    diagnostics = PrimalDiagnostics(lower, upper, residual, error)
    if max(lower, upper, residual) > FEASIBILITY_TOLERANCE or error > OBJECTIVE_TOLERANCE:
        raise AnalysisError(f"independent primal validation failed: {diagnostics}")
    return diagnostics


def run_highs_fba(model: FluxModel) -> FBAResult:
    lp = compile_flux_lp(model)
    fluxes, objective, status = _solve(lp, lp.objective_coefficients, lp.objective_direction, "FBA")
    diagnostics = _validate(lp, fluxes, objective, lp.objective_coefficients)
    return FBAResult(objective, "optimal", lp.objective_direction,
                     pd.Series(fluxes, index=lp.reaction_ids, dtype=float), diagnostics)


def _fraction(value: float) -> float:
    if isinstance(value, bool):
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]")
    try: result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]") from error
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise AnalysisError("fraction_of_optimum must be a finite number in (0, 1]")
    return result


def run_highs_fva_reference(model: FluxModel, fraction_of_optimum: float = 1.0) -> FVAResult:
    fraction = _fraction(fraction_of_optimum); lp = compile_flux_lp(model); fba = run_highs_fba(model)
    retention = (">=" if lp.objective_direction == "max" else "<=", fba.objective_value * fraction)
    minima: list[float] = []; maxima: list[float] = []
    for j, reaction_id in enumerate(lp.reaction_ids):
        costs = [0.0] * len(lp.reaction_ids); costs[j] = 1.0
        low_flux, low, _ = _solve(lp, costs, "min", f"FVA minimum for reaction {reaction_id!r}", retention)
        _validate(lp, low_flux, low, costs)
        high_flux, high, _ = _solve(lp, costs, "max", f"FVA maximum for reaction {reaction_id!r}", retention)
        _validate(lp, high_flux, high, costs)
        # Independently enforce the retained biological objective at every endpoint.
        for endpoint in (low_flux, high_flux):
            value = sum(c * v for c, v in zip(lp.objective_coefficients, endpoint))
            violation = max(retention[1] - value, 0.0) if retention[0] == ">=" else max(value - retention[1], 0.0)
            if violation > OBJECTIVE_TOLERANCE:
                raise AnalysisError(f"FVA endpoint for reaction {reaction_id!r} violates objective retention")
        minima.append(low); maxima.append(high)
    ranges = pd.DataFrame({"minimum": minima, "maximum": maxima}, index=lp.reaction_ids)
    return FVAResult(ranges, fraction, fba.objective_value, lp.objective_direction)
