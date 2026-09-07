"""Native stationary EMU and constrained multistart MID-divergence fitting.

The initializer and optimizer share the native sampler's affine geometry.
SLSQP receives exact finite losses or positive infinity; there is no finite
bound penalty or support repair. Initial undefined/nonfinite losses reject a
start. Later such trials are recorded and let SLSQP backtrack; numerical
differences at difficult boundaries may therefore fail an individual start.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from fluxemu.emu import CompiledEMUPlan, compile_emu_plan, evaluate_stationary
from fluxemu.exceptions import (
    AnalysisError,
    ForwardEMUError,
    InputValidationError,
    MappingError,
    ValidationError,
)
from fluxemu.execution import CanonicalFluxState, _validate_states
from fluxemu.flux_analysis.highs import PreparedFluxRegion, prepare_highs_flux_region
from fluxemu.flux_analysis.sampling import (
    BOUND_TOLERANCE,
    _PreparedSamplingGeometry,
    _ReducedFluxGeometry,
    _prepare_sampling_geometry,
    _require_valid_flux_states,
    _sample_flux_states_from_geometry,
    validate_flux_states,
)

from .divergence import renyi_divergence
from .schema import (
    DivergenceObjectiveConfig,
    MFAFitError,
    MFAObjectiveEvaluation,
    MFAObservationDivergence,
    MFAOptimizationConfig,
    MFAStartDiagnostic,
    StationaryMFAProblem,
    StationaryMFAResult,
    mfa_fit_fingerprint,
    mfa_problem_fingerprint,
    validate_stationary_mfa_problem,
)


_EVALUATION_ERRORS = (
    AnalysisError, ForwardEMUError, MappingError, ValidationError, InputValidationError,
)


@dataclass(frozen=True, slots=True)
class _CompiledMFAObjective:
    problem: StationaryMFAProblem
    objective: DivergenceObjectiveConfig
    prepared: PreparedFluxRegion
    plans: tuple[CompiledEMUPlan, ...]

    def evaluate(self, state: CanonicalFluxState) -> MFAObjectiveEvaluation:
        # These are independent original-model checks, not reduced-hull checks.
        # The native execution check also enforces its stricter bound tolerance.
        report = validate_flux_states(self.prepared, (state,))
        _require_valid_flux_states(report, "stationary MFA objective state")
        _validate_states(self.problem.model, (state,))
        components: list[MFAObservationDivergence] = []
        for block, plan in zip(self.problem.experiments, self.plans, strict=True):
            try:
                result = evaluate_stationary(
                    plan, (state,), mid_tolerance=self.objective.mid_tolerance
                )
                predicted = {
                    item.target_id: item.fractions for item in result.forward.predictions
                }
                for observation in block.observations:
                    fractions = predicted[observation.target_id]
                    try:
                        divergence = renyi_divergence(
                            observation.fractions, fractions, self.objective.alpha,
                            tolerance=self.objective.mid_tolerance,
                        )
                    except (ValidationError, InputValidationError) as error:
                        raise AnalysisError(
                            f"stationary MFA experiment {block.experiment_id!r}, "
                            f"target {observation.target_id!r}, replicate "
                            f"{observation.replicate_id!r}, state {state.sample_id!r}: {error}"
                        ) from error
                    components.append(MFAObservationDivergence(
                        block.experiment_id, observation.target_id,
                        observation.replicate_id, observation.fractions,
                        fractions, divergence,
                    ))
            except (ForwardEMUError, MappingError, ValidationError, InputValidationError) as error:
                raise AnalysisError(
                    f"stationary MFA experiment {block.experiment_id!r}, "
                    f"state {state.sample_id!r}: {error}"
                ) from error
        return MFAObjectiveEvaluation(
            state, math.fsum(item.divergence for item in components), tuple(components)
        )


def _compile_objective(
    problem: StationaryMFAProblem,
    objective: DivergenceObjectiveConfig,
    prepared: PreparedFluxRegion,
) -> _CompiledMFAObjective:
    return _CompiledMFAObjective(
        problem, objective, prepared,
        tuple(compile_emu_plan(problem.model, block.experiment) for block in problem.experiments),
    )


def evaluate_stationary_mfa(
    problem: StationaryMFAProblem,
    state: CanonicalFluxState,
    *,
    objective: DivergenceObjectiveConfig = DivergenceObjectiveConfig(),
) -> MFAObjectiveEvaluation:
    """Evaluate the exact plain sum at one complete feasible canonical state.

    No optional optimizer is imported. A valid support mismatch returns an
    infinite loss. Infeasible states or undefined native predictions raise a
    contextual error. This evaluation imposes no biological retention.
    """

    validate_stationary_mfa_problem(problem, objective)
    prepared = prepare_highs_flux_region(problem.model.flux_model, None)
    return _compile_objective(problem, objective, prepared).evaluate(state)


def _load_scipy_optimizer() -> tuple[Callable, str]:
    try:
        import scipy
        from scipy.optimize import minimize
    except ImportError as error:
        raise AnalysisError(
            "stationary MFA fitting requires SciPy; install fluxemu[mfa]"
        ) from error
    return minimize, scipy.__version__


def _state_from_theta(
    shared: _PreparedSamplingGeometry, theta: np.ndarray, sample_id: object
) -> CanonicalFluxState:
    geometry = shared.geometry
    try:
        point = np.asarray(theta, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("optimizer returned malformed reduced coordinates") from error
    if point.shape != (geometry.basis.shape[1],) or not np.isfinite(point).all():
        raise AnalysisError("optimizer returned malformed or nonfinite reduced coordinates")
    values = geometry.particular + geometry.basis @ point
    if not np.isfinite(values).all():
        raise AnalysisError("optimizer reconstruction produced nonfinite complete fluxes")
    return CanonicalFluxState(
        sample_id,
        tuple((key, float(value)) for key, value in zip(shared.prepared.lp.reaction_ids, values)),
    )


def _initial_theta(geometry: _ReducedFluxGeometry, state: CanonicalFluxState) -> np.ndarray:
    values = np.asarray([value for _, value in state.values], dtype=float)
    theta = geometry.basis.T @ (values - geometry.particular)
    reconstructed = geometry.particular + geometry.basis @ theta
    if np.max(np.abs(reconstructed - values), initial=0.0) > BOUND_TOLERANCE:
        raise AnalysisError("supplied start does not belong to the prepared native affine hull")
    return theta


def _fit_one_start(
    index: int,
    initial: CanonicalFluxState,
    compiled: _CompiledMFAObjective,
    shared: _PreparedSamplingGeometry,
    optimization: MFAOptimizationConfig,
    minimize: Callable,
) -> tuple[MFAStartDiagnostic, MFAObjectiveEvaluation | None]:
    initial_loss = None
    final_loss = None
    final_state = None
    final_evaluation = None
    optimizer_success = False
    numerical_convergence_valid = True
    validated = False
    status = None
    iterations = 0
    evaluations = 0
    trial_failures: list[str] = []
    geometry = shared.geometry
    callback_error = None

    def trial(theta: np.ndarray) -> float:
        nonlocal evaluations, callback_error
        evaluations += 1
        try:
            state = _state_from_theta(shared, theta, f"mfa-start-{index:04d}")
            loss = compiled.evaluate(state).total_loss
            if not math.isfinite(loss):
                trial_failures.append(
                    f"evaluation {evaluations}: exact MID divergence is +inf (support mismatch)"
                )
            return loss
        except _EVALUATION_ERRORS as error:
            trial_failures.append(f"evaluation {evaluations}: {error}")
            return math.inf
        except Exception as error:
            # Preserve programming errors originating in our callback, even
            # when their type is also used for a backend numerical failure.
            callback_error = error
            raise

    try:
        evaluations += 1
        initial_loss = compiled.evaluate(initial).total_loss
        if not math.isfinite(initial_loss):
            raise AnalysisError("initial exact MID divergence is +inf (support mismatch)")
        theta = _initial_theta(geometry, initial)
        if geometry.basis.shape[1] == 0:
            optimizer_success, status = True, 0
            message = "affine singleton; no free coordinates to optimize"
        else:
            rows, bounds = geometry.inequality_rows, geometry.inequality_bounds
            try:
                outcome = minimize(
                    trial, theta, method="SLSQP",
                    constraints=({
                        "type": "ineq",
                        "fun": lambda point: bounds - rows @ point,
                        "jac": lambda point: -rows,
                    },),
                    options={"maxiter": optimization.maxiter, "ftol": optimization.ftol},
                )
            except (RuntimeError, ValueError, FloatingPointError) as error:
                if error is callback_error:
                    raise
                raise AnalysisError(f"SciPy SLSQP failed: {type(error).__name__}: {error}") from error
            theta = outcome.x
            optimizer_success = bool(outcome.success)
            status = int(outcome.status)
            iterations = int(outcome.nit)
            message = str(outcome.message)
            try:
                gradient = np.asarray(getattr(outcome, "jac", None), dtype=float)
            except (TypeError, ValueError, OverflowError):
                gradient = np.array([math.nan])
            if gradient.shape != (geometry.basis.shape[1],) or not np.isfinite(gradient).all():
                numerical_convergence_valid = False
                message += "; rejected numerical convergence: malformed/nonfinite SLSQP gradient"
        final_state = _state_from_theta(shared, theta, f"mfa-start-{index:04d}")
        report = validate_flux_states(compiled.prepared, (final_state,))
        _require_valid_flux_states(report, "stationary MFA final state")
        _validate_states(compiled.problem.model, (final_state,))
        validated = True
        evaluations += 1
        final_evaluation = compiled.evaluate(final_state)
        final_loss = final_evaluation.total_loss
        if not math.isfinite(final_loss):
            raise AnalysisError("final exact MID divergence is +inf (support mismatch)")
    except _EVALUATION_ERRORS as error:
        message = str(error)
        final_evaluation = None

    if trial_failures:
        message += f"; {len(trial_failures)} trial(s) returned exact +inf; see trial_failures"
        if initial_loss is not None and final_loss is not None and final_loss >= initial_loss:
            message += "; no decrease in exact MID loss"

    accepted = final_evaluation if optimizer_success and validated and numerical_convergence_valid else None
    diagnostic = MFAStartDiagnostic(
        index, initial, initial_loss, final_state, final_loss,
        optimizer_success, validated, status, message, iterations,
        evaluations, tuple(trial_failures), accepted is not None,
    )
    return diagnostic, accepted


def fit_stationary_mfa(
    problem: StationaryMFAProblem,
    *,
    objective: DivergenceObjectiveConfig = DivergenceObjectiveConfig(),
    optimization: MFAOptimizationConfig = MFAOptimizationConfig(),
    initial_states: Sequence[CanonicalFluxState] | None = None,
) -> StationaryMFAResult:
    """Fit a shared complete flux state by deterministic multistart SLSQP.

    Generated starts reuse native feasible-state hit-and-run with explicit
    ``seed``. Supplied complete states replace generation and are all attempted
    in order, independent of ``n_starts``. Native-unserializable request data
    fail before fitting; finite malformed/infeasible starts retain diagnostics.
    Every attempted objective state is independently checked against the
    original model before EMU. Only successful, independently validated finite
    final fits are eligible. All-start failure raises :class:`MFAFitError` with
    diagnostics. Multistart local optimization does not prove global optimality.
    """

    validate_stationary_mfa_problem(problem, objective)
    if not isinstance(optimization, MFAOptimizationConfig):
        raise InputValidationError("optimization must be an MFAOptimizationConfig")
    supplied = None
    if initial_states is not None:
        if not isinstance(initial_states, Sequence) or isinstance(initial_states, (str, bytes)):
            raise InputValidationError("initial_states must be a nonempty sequence of complete states")
        supplied = tuple(initial_states)
    fingerprint = mfa_fit_fingerprint(
        problem, objective=objective, optimization=optimization, initial_states=supplied
    )
    minimize, scipy_version = _load_scipy_optimizer()
    prepared = prepare_highs_flux_region(
        problem.model.flux_model, optimization.fraction_of_optimum
    )
    compiled = _compile_objective(problem, objective, prepared)
    shared = _prepare_sampling_geometry(prepared)
    starts = supplied
    if starts is None:
        starts = _sample_flux_states_from_geometry(
            shared, optimization.n_starts, seed=optimization.seed,
            burn_in=optimization.burn_in, thinning=optimization.thinning,
            max_direction_attempts=optimization.max_direction_attempts,
        ).states
    diagnostics: list[MFAStartDiagnostic] = []
    successes: list[tuple[int, MFAObjectiveEvaluation]] = []
    for index, initial in enumerate(starts):
        diagnostic, evaluated = _fit_one_start(
            index, initial, compiled, shared, optimization, minimize
        )
        diagnostics.append(diagnostic)
        if evaluated is not None:
            successes.append((index, evaluated))
    if not successes:
        raise MFAFitError(
            "no stationary MFA start produced a successful, independently validated finite fit",
            tuple(diagnostics), fit_fingerprint=fingerprint,
        )
    best_index, best = min(successes, key=lambda item: item[1].total_loss)
    # Revalidate the selected complete state and recompute the exact scientific
    # output, independently of both SLSQP's reported loss and reduced geometry.
    try:
        validation = validate_flux_states(prepared, (best.state,))
        _require_valid_flux_states(validation, "selected stationary MFA fit")
        final = compiled.evaluate(best.state)
        if not math.isfinite(final.total_loss):
            raise AnalysisError("selected exact MID divergence is +inf")
    except _EVALUATION_ERRORS as error:
        raise MFAFitError(
            f"selected stationary MFA fit failed final validation/reevaluation: {error}",
            tuple(diagnostics), fit_fingerprint=fingerprint,
        ) from error
    return StationaryMFAResult(
        final.state, final.total_loss, final.components, tuple(diagnostics),
        best_index, objective, optimization, compiled.plans[0].model_fingerprint,
        mfa_problem_fingerprint(problem, objective=objective), fingerprint, validation,
        tuple((block.experiment_id, plan.experiment_fingerprint)
              for block, plan in zip(problem.experiments, compiled.plans, strict=True)),
        optimizer_version=scipy_version, affine_dimension=shared.geometry.basis.shape[1],
    )


__all__ = ["evaluate_stationary_mfa", "fit_stationary_mfa"]
