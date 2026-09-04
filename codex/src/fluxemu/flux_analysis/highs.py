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
        raise AnalysisError("native HiGHS analysis requires the default 'highspy' dependency") from error
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


def _run_compiled_fba(lp: CompiledFluxLP) -> FBAResult:
    """Solve and independently validate the biological objective on ``lp``."""

    fluxes, objective, status = _solve(lp, lp.objective_coefficients, lp.objective_direction, "FBA")
    diagnostics = _validate(lp, fluxes, objective, lp.objective_coefficients)
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


def prepare_highs_flux_region(
    model: FluxModel, fraction_of_optimum: float = 1.0
) -> PreparedFluxRegion:
    """Compile once and solve the retained biological objective exactly once."""

    fraction = _fraction(fraction_of_optimum)
    lp = compile_flux_lp(model)
    fba = _run_compiled_fba(lp)
    sense = ">=" if lp.objective_direction == "max" else "<="
    retention = RetainedObjectiveConstraint(
        fraction,
        fba.objective_value,
        fba.objective_value * fraction,
        sense,
        lp.objective_direction,
    )
    return PreparedFluxRegion(lp, fba, retention)


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
    try:
        _validate(lp, fluxes, optimum, lp.objective_coefficients)
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
    expected_sense = ">=" if lp.objective_direction == "max" else "<="
    expected_bound = optimum * fraction
    if (
        retention.objective_direction != lp.objective_direction
        or retention.sense != expected_sense
        or recorded_optimum != optimum
        or recorded_bound != expected_bound
    ):
        raise AnalysisError("prepared retained-objective metadata is inconsistent")


def run_highs_fva_reference(model: FluxModel, fraction_of_optimum: float = 1.0) -> FVAResult:
    fraction = _fraction(fraction_of_optimum); lp = compile_flux_lp(model); fba = _run_compiled_fba(lp)
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
            _validate(self.lp, fluxes, value, costs)
            biological = sum(c * v for c, v in zip(self.lp.objective_coefficients, fluxes))
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
        workers = min(4, os.cpu_count() or 1, len(tasks))
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
