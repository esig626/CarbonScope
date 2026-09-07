"""Scientific input validation and provenance for the stationary MFA API."""

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.mfa.schema import (
    DivergenceObjectiveConfig,
    MFAFitError,
    MFAOptimizationConfig,
    MFAStartDiagnostic,
    StationaryMFAExperiment,
    StationaryMFAProblem,
    StationaryMIDObservation,
    mfa_fit_fingerprint,
    mfa_problem_fingerprint,
    validate_stationary_mfa_problem,
)
from fluxemu.model import (
    CanonicalModel,
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    IsotopeMetabolite,
    IsotopeModel,
    LinearObjective,
    ObjectiveTerm,
    ObservationPrecursor,
    ObservationTarget,
    StationaryExperimentSemantics,
    StoichiometricTerm,
    Target,
    Tracer,
)


def _problem() -> StationaryMFAProblem:
    model = CanonicalModel(
        FluxModel(
            (FluxMetabolite("source", False),),
            (FluxReaction("R", (StoichiometricTerm("source", 1),), 0, 10),),
            LinearObjective("maximise", (ObjectiveTerm("R", 1),)),
        ),
        IsotopeModel((IsotopeMetabolite("source", 2, True, False),), ()),
    )
    experiment = StationaryExperimentSemantics(
        (Tracer("source", (("#10", 1.0),), "no"),),
        (
            Target("z-full", "source", (2, 1), "intermediate", "C2", "no"),
            Target("a-one", "source", (1,), "intermediate", "C1", "no"),
        ),
        (
            ObservationTarget(
                "composite",
                2,
                (ObservationPrecursor("source", (1,)), ObservationPrecursor("source", (2,))),
            ),
        ),
    )
    observations = (
        StationaryMIDObservation("a-one", (0.25, 0.75), "second"),
        StationaryMIDObservation("z-full", (0.1, 0.2, 0.7)),
        StationaryMIDObservation("a-one", (0.5, 0.5), "first"),
        StationaryMIDObservation("composite", (0.2, 0.3, 0.5)),
    )
    return StationaryMFAProblem(model, (StationaryMFAExperiment("z-exp", experiment, observations),))


def _with_observation(problem, observation):
    return replace(problem, experiments=(replace(problem.experiments[0], observations=(observation,)),))


def test_valid_full_mid_replicates_and_composite_observations_preserve_order():
    problem = _problem()
    original = problem.experiments[0]
    validate_stationary_mfa_problem(problem)
    assert problem.experiments[0] is original
    assert tuple(item.target_id for item in original.experiment.targets) == ("z-full", "a-one")
    assert original.experiment.targets[0].atom_positions == (2, 1)
    assert tuple((item.target_id, item.replicate_id) for item in original.observations) == (
        ("a-one", "second"), ("z-full", "0"), ("a-one", "first"), ("composite", "0")
    )
    assert original.observations[1].fractions == (0.1, 0.2, 0.7)
    with pytest.raises(FrozenInstanceError):
        original.observations[0].target_id = "changed"


@pytest.mark.parametrize("fractions", [(-0.1, 1.1), (math.nan, 0.5), (math.inf, 0.5),
                                         (0.3, 0.6), (True, 0), ("0.3", 0.7)])
def test_invalid_probability_vectors_are_rejected(fractions):
    problem = _with_observation(_problem(), StationaryMIDObservation("a-one", fractions))
    with pytest.raises(InputValidationError):
        validate_stationary_mfa_problem(problem)


@pytest.mark.parametrize("target,fractions", [("a-one", (0.1, 0.2, 0.7)),
                                             ("z-full", (0.5, 0.5)),
                                             ("composite", (0.5, 0.5))])
def test_target_and_composite_dimensions_are_checked(target, fractions):
    problem = _with_observation(_problem(), StationaryMIDObservation(target, fractions))
    with pytest.raises(InputValidationError, match="mass classes; expected"):
        validate_stationary_mfa_problem(problem)


def test_probability_tolerance_is_explicit_and_does_not_normalize():
    observation = StationaryMIDObservation("a-one", (0.4, 0.6000001))
    problem = _with_observation(_problem(), observation)
    with pytest.raises(InputValidationError):
        validate_stationary_mfa_problem(problem)
    validate_stationary_mfa_problem(problem, DivergenceObjectiveConfig(mid_tolerance=1e-6))
    assert observation.fractions == (0.4, 0.6000001)


def test_unknown_target_is_rejected_with_experiment_context():
    problem = _with_observation(_problem(), StationaryMIDObservation("missing", (0.5, 0.5)))
    with pytest.raises(InputValidationError, match="unknown target_id.*z-exp"):
        validate_stationary_mfa_problem(problem)


def test_duplicate_observation_identity_is_rejected_but_replicates_remain_distinct():
    problem = _problem()
    block = problem.experiments[0]
    duplicate = replace(block, observations=block.observations + (block.observations[0],))
    with pytest.raises(InputValidationError, match="duplicate observation identity"):
        validate_stationary_mfa_problem(replace(problem, experiments=(duplicate,)))


def test_multiple_experiments_keep_declared_order_and_allow_reused_target_ids():
    problem = _problem()
    second = replace(problem.experiments[0], experiment_id="a-exp")
    multiple = replace(problem, experiments=problem.experiments + (second,))
    validate_stationary_mfa_problem(multiple)
    assert tuple(item.experiment_id for item in multiple.experiments) == ("z-exp", "a-exp")
    with pytest.raises(InputValidationError, match="duplicate experiment_id"):
        validate_stationary_mfa_problem(replace(problem, experiments=problem.experiments * 2))


def test_native_model_experiment_and_duplicate_target_validation_are_reused():
    problem = _problem()
    block = problem.experiments[0]
    duplicate = replace(block.experiment, targets=block.experiment.targets * 2)
    with pytest.raises(ValueError, match="duplicate target ID"):
        validate_stationary_mfa_problem(replace(problem, experiments=(replace(block, experiment=duplicate),)))
    unknown = replace(block.experiment, tracers=(Tracer("missing", (("#10", 1.0),), "no"),))
    with pytest.raises(ValueError, match="unknown isotope metabolite"):
        validate_stationary_mfa_problem(replace(problem, experiments=(replace(block, experiment=unknown),)))
    flux = problem.model.flux_model
    malformed = replace(problem.model, flux_model=replace(flux, reactions=flux.reactions * 2))
    with pytest.raises(ValueError, match="duplicate"):
        validate_stationary_mfa_problem(replace(problem, model=malformed))


@pytest.mark.parametrize("factory", [
    lambda: StationaryMIDObservation("", (0.5, 0.5)),
    lambda: StationaryMIDObservation("a", (0.5, 0.5), ""),
    lambda: StationaryMIDObservation("a", [0.5, 0.5]),
    lambda: StationaryMIDObservation("a", ()),
    lambda: replace(_problem().experiments[0], observations=[]),
    lambda: replace(_problem().experiments[0], observations=()),
    lambda: replace(_problem().experiments[0], observations=("bad",)),
    lambda: replace(_problem().experiments[0], experiment="bad"),
    lambda: replace(_problem(), experiments=[]),
    lambda: replace(_problem(), experiments=()),
    lambda: replace(_problem(), experiments=("bad",)),
    lambda: replace(_problem(), model="bad"),
])
def test_scientific_record_structure_requires_explicit_immutable_tuples(factory):
    with pytest.raises(InputValidationError):
        factory()


@pytest.mark.parametrize("key,value", [
    ("n_starts", 0), ("n_starts", True), ("n_starts", 1.5),
    ("seed", -1), ("seed", False), ("maxiter", 0), ("maxiter", math.inf),
    ("ftol", 0), ("ftol", math.nan), ("ftol", True), ("ftol", 10 ** 1000),
    ("fraction_of_optimum", 0), ("fraction_of_optimum", 1.1),
    ("fraction_of_optimum", True), ("fraction_of_optimum", math.inf),
    ("burn_in", -1), ("burn_in", False), ("thinning", 0),
    ("max_direction_attempts", 0),
])
def test_optimizer_settings_reject_malformed_controls(key, value):
    with pytest.raises(InputValidationError):
        MFAOptimizationConfig(**{key: value})


@pytest.mark.parametrize("key,value", [("alpha", True), ("alpha", 0), ("alpha", math.inf),
                                      ("mid_tolerance", -1), ("mid_tolerance", math.nan),
                                      ("mid_tolerance", True)])
def test_objective_settings_use_shared_strict_numeric_validation(key, value):
    with pytest.raises(InputValidationError):
        DivergenceObjectiveConfig(**{key: value})


def test_default_mfa_is_unrestricted_and_controls_are_frozen():
    config = MFAOptimizationConfig()
    assert config.fraction_of_optimum is None
    assert MFAOptimizationConfig(fraction_of_optimum=1).fraction_of_optimum == 1.0
    assert DivergenceObjectiveConfig(alpha=1.237).alpha == 1.237
    with pytest.raises(FrozenInstanceError):
        config.seed = 9


def test_problem_fingerprint_binds_model_experiments_observations_and_their_order():
    problem = _problem()
    block = problem.experiments[0]
    digest = mfa_problem_fingerprint(problem)
    assert digest == mfa_problem_fingerprint(_problem())
    assert len(digest) == 64
    variants = [
        replace(problem, experiments=(replace(block, experiment_id="different"),)),
        replace(problem, experiments=(replace(block, observations=tuple(reversed(block.observations))),)),
        replace(problem, experiments=(replace(block, observations=(replace(block.observations[0], replicate_id="third"),) + block.observations[1:]),)),
        replace(problem, experiments=(replace(block, observations=(replace(block.observations[0], fractions=(0.2, 0.8)),) + block.observations[1:]),)),
        replace(problem, experiments=(replace(block, experiment=replace(block.experiment, targets=tuple(reversed(block.experiment.targets)))),)),
        replace(problem, experiments=(replace(block, experiment=replace(block.experiment, tracers=(Tracer("source", (("#01", 1.0),), "no"),))),)),
        replace(problem, model=replace(problem.model, flux_model=replace(problem.model.flux_model, reactions=(replace(problem.model.flux_model.reactions[0], upper_bound=9),)))),
    ]
    assert all(mfa_problem_fingerprint(variant) != digest for variant in variants)
    multiple = replace(problem, experiments=(block, replace(block, experiment_id="second")))
    assert mfa_problem_fingerprint(multiple) != mfa_problem_fingerprint(replace(multiple, experiments=multiple.experiments[::-1]))
    assert mfa_problem_fingerprint(problem, objective=DivergenceObjectiveConfig(alpha=0.5)) != digest
    assert mfa_problem_fingerprint(problem, objective=DivergenceObjectiveConfig(mid_tolerance=1e-8)) != digest


@pytest.mark.parametrize("key,value", [("n_starts", 9), ("seed", 10), ("maxiter", 200),
                                      ("ftol", 1e-8), ("fraction_of_optimum", 0.7),
                                      ("burn_in", 2), ("thinning", 3), ("max_direction_attempts", 7)])
def test_fit_fingerprint_binds_every_optimizer_setting(key, value):
    problem = _problem()
    changed = MFAOptimizationConfig(**{key: value})
    assert mfa_fit_fingerprint(problem, optimization=changed) != mfa_fit_fingerprint(problem)


def test_fit_fingerprint_binds_start_values_ids_order_and_generation_mode():
    problem = _problem()
    starts = (CanonicalFluxState("z", (("R", 2.0),)), CanonicalFluxState("a", (("R", 4.0),)))
    digest = mfa_fit_fingerprint(problem, initial_states=starts)
    assert digest == mfa_fit_fingerprint(problem, initial_states=starts)
    assert digest != mfa_fit_fingerprint(problem)
    assert digest != mfa_fit_fingerprint(problem, initial_states=starts[::-1])
    assert digest != mfa_fit_fingerprint(problem, initial_states=(replace(starts[0], sample_id="new"), starts[1]))
    assert digest != mfa_fit_fingerprint(problem, initial_states=(replace(starts[0], values=(("R", 3.0),)), starts[1]))
    assert digest != mfa_fit_fingerprint(problem, initial_states=starts, objective=DivergenceObjectiveConfig(alpha=2))
    # Finite malformed starts retain auditable provenance for failed attempts.
    assert mfa_fit_fingerprint(problem, initial_states=(CanonicalFluxState("bad", (("missing", -10.0),)),))


@pytest.mark.parametrize("starts", [(), [], ("bad",),
                                   (CanonicalFluxState("bad", (("R", math.nan),)),),
                                   (CanonicalFluxState(object(), (("R", 1.0),)),)])
def test_invalid_start_provenance_is_rejected_before_fitting(starts):
    with pytest.raises(InputValidationError):
        mfa_fit_fingerprint(_problem(), initial_states=starts)


def test_all_failed_error_keeps_diagnostics_and_exact_infinite_loss():
    diagnostic = MFAStartDiagnostic(
        start_index=0, initial_state=CanonicalFluxState("bad", (("R", 0.0),)),
        initial_loss=math.inf, final_state=None, final_loss=None,
        optimizer_success=False, validated=False, status=None,
        message="observed support absent from prediction", iterations=0, evaluations=1,
        trial_failures=("support mismatch",),
    )
    error = MFAFitError("no validated fit", (diagnostic,), fit_fingerprint="digest")
    assert error.start_diagnostics == (diagnostic,)
    assert error.fit_fingerprint == "digest"
    assert math.isinf(error.start_diagnostics[0].initial_loss)
    assert error.start_diagnostics[0].final_loss is None
