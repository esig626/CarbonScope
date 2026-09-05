"""Immutable scientific data and provenance for stationary MID-divergence fitting.

Record construction checks structure. :func:`validate_stationary_mfa_problem`
is the mandatory contextual boundary before compilation, evaluation, or fitting:
it validates probabilities with the explicitly selected objective tolerance and
checks observations against the native model and experiment declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.model import (
    CanonicalModel,
    StationaryExperimentSemantics,
    deterministic_serialise,
    validate_canonical_model,
    validate_stationary_experiment,
)

from .divergence import validate_alpha, validate_mid, validate_mid_tolerance

if TYPE_CHECKING:
    from fluxemu.flux_analysis.sampling import FluxSampleValidationReport


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{name} must be a nonempty string")


def _nonempty_tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise InputValidationError(f"{name} must be a nonempty tuple")


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise InputValidationError(f"{name} must be an integer >= {minimum}, not bool")
    return int(value)


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(f"{name} must be a finite positive real number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise InputValidationError(f"{name} must be a finite positive real number") from error
    if not math.isfinite(number) or number <= 0.0:
        raise InputValidationError(f"{name} must be a finite positive real number")
    return number


@dataclass(frozen=True, slots=True)
class StationaryMIDObservation:
    """One complete MID in declared isotopologue order, without averaging.

    ``(target_id, replicate_id)`` uniquely identifies an observation within its
    experiment block. Fractions are never normalized, clipped, or reordered.
    Probability and target-dimension checks occur at problem validation with
    the selected :class:`DivergenceObjectiveConfig`.
    """

    target_id: str
    fractions: tuple[float, ...]
    replicate_id: str = "0"

    def __post_init__(self) -> None:
        _identifier(self.target_id, "observation target_id")
        _identifier(self.replicate_id, "observation replicate_id")
        _nonempty_tuple(self.fractions, "observation fractions")


@dataclass(frozen=True, slots=True)
class StationaryMFAExperiment:
    """An explicitly identified stationary tracer block and ordered MIDs."""

    experiment_id: str
    experiment: StationaryExperimentSemantics
    observations: tuple[StationaryMIDObservation, ...]

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        if not isinstance(self.experiment, StationaryExperimentSemantics):
            raise InputValidationError("experiment must be StationaryExperimentSemantics")
        _nonempty_tuple(self.observations, "experiment observations")
        if not all(isinstance(item, StationaryMIDObservation) for item in self.observations):
            raise InputValidationError("observations must contain StationaryMIDObservation records")


@dataclass(frozen=True, slots=True)
class StationaryMFAProblem:
    """One canonical model shared by one or more ordered experiment blocks.

    Explicitly chosen exact or interval flux measurements can be represented
    by the existing canonical reaction bounds. Soft flux residuals are outside
    this MID-only statistical contract.
    """

    model: CanonicalModel
    experiments: tuple[StationaryMFAExperiment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, CanonicalModel):
            raise InputValidationError("model must be a CanonicalModel")
        _nonempty_tuple(self.experiments, "MFA experiments")
        if not all(isinstance(item, StationaryMFAExperiment) for item in self.experiments):
            raise InputValidationError("experiments must contain StationaryMFAExperiment records")


@dataclass(frozen=True, slots=True)
class DivergenceObjectiveConfig:
    """Plain sum of ``D_alpha(observed || predicted)`` over complete MIDs.

    ``alpha == 1`` is exact KL. ``mid_tolerance`` controls schema validation;
    divergence arithmetic additionally requires unit mass within its explicit
    machine-roundoff tolerance. A broader schema tolerance does not authorize
    repairing mass or guarantee that a vector can be numerically evaluated.
    """

    alpha: float = 1.0
    mid_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_alpha(self.alpha))
        object.__setattr__(self, "mid_tolerance", validate_mid_tolerance(self.mid_tolerance))


@dataclass(frozen=True, slots=True)
class MFAOptimizationConfig:
    """Deterministic native initialization and constrained local fitting.

    ``fraction_of_optimum=None`` imposes no biological optimality restriction.
    A requested fraction adds the native retained-objective constraint, never
    a term in the MID loss. ``n_starts`` controls generated starts only; supplied
    starts are all attempted in their supplied order.
    """

    n_starts: int = 8
    seed: int = 0
    maxiter: int = 1000
    ftol: float = 1e-10
    fraction_of_optimum: float | None = None
    burn_in: int = 100
    thinning: int = 10
    max_direction_attempts: int = 100

    def __post_init__(self) -> None:
        for name in ("n_starts", "maxiter", "thinning", "max_direction_attempts"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        for name in ("seed", "burn_in"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        object.__setattr__(self, "ftol", _positive_real(self.ftol, "ftol"))
        if self.fraction_of_optimum is not None:
            fraction = _positive_real(self.fraction_of_optimum, "fraction_of_optimum")
            if fraction > 1.0:
                raise InputValidationError("fraction_of_optimum must be in (0, 1] or None")
            object.__setattr__(self, "fraction_of_optimum", fraction)


@dataclass(frozen=True, slots=True)
class MFAObservationDivergence:
    """One exact loss component with all experiment/observation identities."""

    experiment_id: str
    target_id: str
    replicate_id: str
    observed: tuple[float, ...]
    predicted: tuple[float, ...]
    divergence: float


@dataclass(frozen=True, slots=True)
class MFAObjectiveEvaluation:
    """Native predictions and their unweighted sum at one complete state."""

    state: CanonicalFluxState
    total_loss: float
    components: tuple[MFAObservationDivergence, ...]


@dataclass(frozen=True, slots=True)
class MFAStartDiagnostic:
    """One requested start, including rejected or unsuccessful attempts.

    A loss of ``None`` means it was never evaluated; ``inf`` is an exact
    divergent objective. ``validated`` records original-model feasibility,
    independently of the backend success flag. ``trial_failures`` preserves
    contextual infeasible, undefined-forward, and support-failing trials.
    ``accepted`` additionally requires finite numerical convergence and a
    finite re-evaluated loss; it identifies eligible fit candidates directly.
    """

    start_index: int
    initial_state: CanonicalFluxState | None
    initial_loss: float | None
    final_state: CanonicalFluxState | None
    final_loss: float | None
    optimizer_success: bool
    validated: bool
    status: int | None
    message: str
    iterations: int
    evaluations: int
    trial_failures: tuple[str, ...] = ()
    accepted: bool = False


@dataclass(frozen=True, slots=True)
class StationaryMFAResult:
    """Best successful independently validated fit and every start diagnostic.

    Components follow experiment order and each experiment's observation
    order. Multistart local fitting does not certify a global optimum or a
    uniquely identifiable complete flux vector.
    """

    state: CanonicalFluxState
    total_loss: float
    components: tuple[MFAObservationDivergence, ...]
    start_diagnostics: tuple[MFAStartDiagnostic, ...]
    best_start_index: int
    objective_config: DivergenceObjectiveConfig
    optimization_config: MFAOptimizationConfig
    model_fingerprint: str
    problem_fingerprint: str
    fit_fingerprint: str
    validation: FluxSampleValidationReport
    experiment_fingerprints: tuple[tuple[str, str], ...] = ()
    optimizer_backend: str = "scipy.optimize.SLSQP"
    optimizer_version: str = ""
    affine_dimension: int = 0


class MFAFitError(AnalysisError):
    """No start produced a successful validated fit; all diagnostics survive."""

    def __init__(
        self,
        message: str,
        start_diagnostics: tuple[MFAStartDiagnostic, ...],
        *,
        fit_fingerprint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.start_diagnostics = tuple(start_diagnostics)
        self.fit_fingerprint = fit_fingerprint


def validate_stationary_mfa_problem(
    problem: StationaryMFAProblem,
    objective: DivergenceObjectiveConfig = DivergenceObjectiveConfig(),
) -> None:
    """Validate native science, explicit identities, dimensions, and all MIDs.

    This function never changes its inputs. The objective/fit entry points
    always invoke it before compiling or evaluating any experiment.
    """

    if not isinstance(problem, StationaryMFAProblem):
        raise InputValidationError("problem must be a StationaryMFAProblem")
    if not isinstance(objective, DivergenceObjectiveConfig):
        raise InputValidationError("objective must be a DivergenceObjectiveConfig")
    validate_canonical_model(problem.model)
    experiment_ids: set[str] = set()
    for block in problem.experiments:
        if block.experiment_id in experiment_ids:
            raise InputValidationError(f"duplicate experiment_id: {block.experiment_id!r}")
        experiment_ids.add(block.experiment_id)
        validate_stationary_experiment(problem.model, block.experiment)
        dimensions = {
            target.target_id: len(target.atom_positions) + 1
            for target in block.experiment.targets
        }
        dimensions.update(
            (target.target_id, target.carbon_count + 1)
            for target in block.experiment.observation_targets
        )
        observation_ids: set[tuple[str, str]] = set()
        for observation in block.observations:
            identity = (observation.target_id, observation.replicate_id)
            context = f"experiment {block.experiment_id!r}, observation {identity!r}"
            if identity in observation_ids:
                raise InputValidationError(f"duplicate observation identity in {context}")
            observation_ids.add(identity)
            if observation.target_id not in dimensions:
                raise InputValidationError(f"unknown target_id in {context}")
            validate_mid(
                observation.fractions,
                expected_size=dimensions[observation.target_id],
                tolerance=objective.mid_tolerance,
                name=context,
            )


def _problem_identity(
    problem: StationaryMFAProblem, objective: DivergenceObjectiveConfig
) -> tuple[object, ...]:
    # Validated float tuples provide consistent scalar serialization even when
    # callers supply NumPy real scalars. Values and all scientific order remain
    # unchanged; the sole probability validator never normalizes.
    return (
        "stationary-mfa-problem-v1",
        problem.model,
        tuple(
            (
                block.experiment_id,
                block.experiment,
                tuple(
                    (
                        item.target_id,
                        item.replicate_id,
                        validate_mid(item.fractions, tolerance=objective.mid_tolerance),
                    )
                    for item in block.observations
                ),
            )
            for block in problem.experiments
        ),
        objective,
    )


def _digest(value: object) -> str:
    try:
        serialized = deterministic_serialise(value)
    except ValueError as error:
        raise InputValidationError(f"MFA provenance is not canonical-serializable: {error}") from error
    return sha256(serialized.encode("utf-8")).hexdigest()


def mfa_problem_fingerprint(
    problem: StationaryMFAProblem,
    *,
    objective: DivergenceObjectiveConfig = DivergenceObjectiveConfig(),
) -> str:
    """Bind the full model, ordered experiments/MIDs, and objective settings."""

    validate_stationary_mfa_problem(problem, objective)
    return _digest(_problem_identity(problem, objective))


def mfa_fit_fingerprint(
    problem: StationaryMFAProblem,
    *,
    objective: DivergenceObjectiveConfig = DivergenceObjectiveConfig(),
    optimization: MFAOptimizationConfig = MFAOptimizationConfig(),
    initial_states: tuple[CanonicalFluxState, ...] | None = None,
) -> str:
    """Also bind optimizer settings and the complete supplied-start provenance.

    ``None`` records native seeded generation. A supplied tuple records every
    state and sample ID in its given order, including finite malformed starts
    that may later be rejected by original-model validation. Nonfinite or
    otherwise nonserializable values are invalid request data.
    """

    validate_stationary_mfa_problem(problem, objective)
    if not isinstance(optimization, MFAOptimizationConfig):
        raise InputValidationError("optimization must be an MFAOptimizationConfig")
    if initial_states is not None:
        _nonempty_tuple(initial_states, "initial_states")
        if not all(isinstance(state, CanonicalFluxState) for state in initial_states):
            raise InputValidationError("initial_states must contain CanonicalFluxState records")
    return _digest(
        (
            "stationary-mfa-fit-v1",
            _problem_identity(problem, objective),
            optimization,
            "generated" if initial_states is None else "supplied",
            initial_states,
        )
    )


__all__ = [
    "DivergenceObjectiveConfig",
    "MFAFitError",
    "MFAObjectiveEvaluation",
    "MFAObservationDivergence",
    "MFAOptimizationConfig",
    "MFAStartDiagnostic",
    "StationaryMFAExperiment",
    "StationaryMFAProblem",
    "StationaryMFAResult",
    "StationaryMIDObservation",
    "mfa_fit_fingerprint",
    "mfa_problem_fingerprint",
    "validate_stationary_mfa_problem",
]
