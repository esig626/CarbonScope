"""Declarative native modelling followed by finite represented-class testing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

from fluxemu import testing
from fluxemu.exceptions import InputValidationError, ValidationError

from .schema import (
    WorkflowSpecification,
    load_hypothesis_testing_spec,
    validate_hypothesis_testing_specification,
)
from .states import HypothesisStateFamily, generate_hypothesis_state_families


TestingValue: TypeAlias = (
    testing.CompositeRenyiConverseBound
    | testing.FiniteCompositeMinimaxResult
    | testing.CompositeRenyiScoreCandidate
    | testing.CompositeScoreBound
    | testing.CompositeScoreTestEvaluation
    | testing.CalibratedCompositeScoreTest
)

# The validation campaign on the repaired primitives uses this comparison
# allowance. It does not change any primitive's acceptance or solver policy.
RELATIONSHIP_TOLERANCE = 2e-9
PROCEDURES = (
    "composite_converse", "exact_minimax", "candidate_score",
    "analytical_score_bound", "deterministic_score_error", "calibrated_score_error",
)
_REFUSALS = (
    testing.NumericalLimitError, testing.CompositeEnumerationLimitError,
    testing.CompositeOptimizationError, testing.CompositeScoreVerificationError,
)


@dataclass(frozen=True, slots=True)
class TestingRefusal:
    """An explicitly unsupported statistical evaluation of a valid problem."""

    procedure: str
    order: float | None
    category: str
    exception_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class ProcedureEvaluation:
    """One requested quantity or a transparently evaluated prerequisite."""

    procedure: str
    order: float | None
    requested: bool
    status: Literal["evaluated", "refused", "not_requested"]
    value: TestingValue | None = None
    refusal: TestingRefusal | None = None


@dataclass(frozen=True, slots=True)
class RelationshipCheck:
    lower_quantity: str
    lower_order: float | None
    upper_quantity: str
    upper_order: float | None
    lower_value: float
    upper_value: float
    tolerance: float
    passed: bool


@dataclass(frozen=True, slots=True)
class WorkflowOutputPaths:
    report: Path
    summary: Path


@dataclass(frozen=True, slots=True)
class WorkflowSoftwareProvenance:
    carbonscope_version: str
    python_version: str
    dependencies: tuple[tuple[str, str | None], ...]
    git_commit: str | None
    git_dirty: bool | None
    runtime_source_sha256: str


@dataclass(frozen=True, slots=True)
class WorkflowProvenance:
    model_fingerprint: str
    experiment_fingerprints: tuple[tuple[str, str], ...]
    specification_fingerprint: str
    null_hypothesis_fingerprint: str
    alternative_hypothesis_fingerprint: str
    null_finite_family_fingerprint: str
    alternative_finite_family_fingerprint: str
    null_observation_family_fingerprint: str
    alternative_observation_family_fingerprint: str
    observation_specification_fingerprint: str
    composite_problem_fingerprint: str
    stationary_bridge_fingerprint: str
    software: WorkflowSoftwareProvenance


@dataclass(frozen=True, slots=True)
class HypothesisTestingWorkflowResult:
    """Native feasible states, exact source laws, distinct testing results."""

    specification: WorkflowSpecification
    null_family: HypothesisStateFamily
    alternative_family: HypothesisStateFamily
    stationary: testing.StationaryCompositeTestingResult
    testing_results: tuple[ProcedureEvaluation, ...]
    relationship_checks: tuple[RelationshipCheck, ...]
    provenance: WorkflowProvenance
    output_paths: WorkflowOutputPaths | None = None

    def __post_init__(self) -> None:
        from ._validation import validate_workflow_result
        validate_workflow_result(self)

    @property
    def problem(self) -> testing.CompositeBinaryTestingProblem:
        return self.stationary.problem

    @property
    def null_observation_family(self) -> testing.CompositeMIDLawFamily:
        return self.problem.null

    @property
    def alternative_observation_family(self) -> testing.CompositeMIDLawFamily:
        return self.problem.alternative

    @property
    def refusals(self) -> tuple[TestingRefusal, ...]:
        return tuple(item.refusal for item in self.testing_results if item.refusal is not None)

    @property
    def summary(self) -> str:
        from .report import workflow_summary
        return workflow_summary(self)


def _category(error: Exception) -> str:
    if isinstance(error, testing.CompositeEnumerationLimitError):
        return "enumeration_limit"
    if isinstance(error, testing.CompositeScoreVerificationError):
        return "score_verification"
    if isinstance(error, testing.CompositeOptimizationError):
        return "optimization_or_dependency_limit"
    return "numerical_limit"


def _evaluate_testing(specification, problem) -> tuple[ProcedureEvaluation, ...]:
    settings = specification.testing
    requested = set(settings.procedures)
    records: dict[tuple[str, float | None], ProcedureEvaluation] = {}

    def evaluate(procedure, order, operation, dependency=None):
        if dependency is not None and dependency.status == "refused":
            original = dependency.refusal
            refusal = TestingRefusal(
                procedure, order, "prerequisite_refused", original.exception_type,
                f"{dependency.procedure} prerequisite refused: {original.reason}",
            )
            item = ProcedureEvaluation(procedure, order, procedure in requested, "refused", refusal=refusal)
        else:
            try:
                value = operation()
            except _REFUSALS as error:
                refusal = TestingRefusal(procedure, order, _category(error), type(error).__name__, str(error))
                item = ProcedureEvaluation(procedure, order, procedure in requested, "refused", refusal=refusal)
            else:
                item = ProcedureEvaluation(procedure, order, procedure in requested, "evaluated", value)
        records[(procedure, order)] = item
        return item

    if "composite_converse" in requested:
        for order in settings.converse_orders:
            evaluate("composite_converse", order, lambda: testing.composite_renyi_converse_at_order(
                problem, epsilon=settings.epsilon, order=order,
            ))
    if "exact_minimax" in requested:
        evaluate("exact_minimax", None, lambda: testing.exact_finite_composite_minimax(
            problem, epsilon=settings.epsilon, max_outcomes=settings.max_outcomes,
        ))
    for order in settings.score_orders:
        candidate = evaluate("candidate_score", order, lambda: testing.composite_renyi_score_candidate(problem, order=order))
        if requested & {"analytical_score_bound", "deterministic_score_error"}:
            bound = evaluate("analytical_score_bound", order, lambda: testing.composite_score_bound_at_order(
                candidate.value, epsilon=settings.epsilon,
            ), candidate)
            if "deterministic_score_error" in requested:
                evaluate("deterministic_score_error", order, lambda: testing.evaluate_composite_score_test(
                    bound.value, max_outcomes=settings.max_outcomes,
                ), bound)
        if "calibrated_score_error" in requested:
            evaluate("calibrated_score_error", order, lambda: testing.calibrate_composite_score_test(
                candidate.value, epsilon=settings.epsilon, max_outcomes=settings.max_outcomes,
            ), candidate)

    # Preserve the caller's procedure and order lists. Prerequisites follow in
    # fixed catalog order and explicitly say requested=False.
    ordered = []
    for procedure in (*settings.procedures, *(p for p in PROCEDURES if p not in requested)):
        orders = (settings.converse_orders if procedure == "composite_converse" else
                  (None,) if procedure == "exact_minimax" else settings.score_orders)
        for order in orders or (None,):
            ordered.append(records.get((procedure, order), ProcedureEvaluation(
                procedure, order, False, "not_requested",
            )))
    return tuple(ordered)


def _check_relationships(results) -> tuple[RelationshipCheck, ...]:
    accepted = tuple(item for item in results if item.status == "evaluated")
    exact = next((item for item in accepted if item.procedure == "exact_minimax"), None)
    converse = tuple(item for item in accepted if item.procedure == "composite_converse")
    achieved = tuple(item for item in accepted if item.procedure in {
        "deterministic_score_error", "calibrated_score_error",
    })
    checks = []

    def check(left, left_value, right, right_value):
        passed = left_value <= right_value + RELATIONSHIP_TOLERANCE
        checks.append(RelationshipCheck(left.procedure, left.order, right.procedure, right.order,
                                        left_value, right_value, RELATIONSHIP_TOLERANCE, passed))
        if not passed:
            raise ValidationError(
                f"workflow integration failed {left.procedure} <= {right.procedure}: "
                f"{left_value:.17g} > {right_value:.17g} within {RELATIONSHIP_TOLERANCE:g}"
            )

    for lower in converse:
        if exact is not None:
            check(lower, lower.value.type_ii_lower_bound, exact, exact.value.minimax_type_ii_error)
        for upper in achieved:
            check(lower, lower.value.type_ii_lower_bound, upper, upper.value.worst_type_ii_error)
    if exact is not None:
        for upper in achieved:
            check(exact, exact.value.minimax_type_ii_error, upper, upper.value.worst_type_ii_error)
    return tuple(checks)


def run_hypothesis_testing_workflow(
    specification: WorkflowSpecification | str | Path,
    *,
    model_path: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> HypothesisTestingWorkflowResult:
    """Run files/declarations -> complete states -> EMU -> count laws -> tests.

    Invalid science or failed construction raises a FluxEMU error. Only the
    statistical primitives' documented refusal classes become report outcomes.
    Omitting output_directory uses the specification's output setting; with
    neither setting the result is returned without writing files.
    """
    for name, value in (("model_path", model_path), ("output_directory", output_directory)):
        if value is not None and (not isinstance(value, (str, Path)) or not str(value).strip()):
            raise InputValidationError(f"{name} must be a nonempty path or None")
    if isinstance(specification, (str, Path)):
        specification = load_hypothesis_testing_spec(specification, model_path=model_path)
    elif model_path is not None:
        raise InputValidationError("model_path can override a file specification only")
    validate_hypothesis_testing_specification(specification)
    null, alternative = generate_hypothesis_state_families(specification)
    stationary = testing.evaluate_stationary_composite_hypotheses(
        specification.observation, null_states=null.states, alternative_states=alternative.states,
        independent_blocks=specification.independent_blocks,
    )
    results = _evaluate_testing(specification, stationary.problem)
    checks = _check_relationships(results)
    from .report import build_workflow_provenance, persist_workflow_report
    provenance = build_workflow_provenance(specification, null, alternative, stationary)
    result = HypothesisTestingWorkflowResult(specification, null, alternative, stationary, results, checks, provenance)
    destination = output_directory if output_directory is not None else specification.output_directory
    if destination is not None:
        paths = persist_workflow_report(result, destination)
        result = replace(result, output_paths=paths)
    return result
