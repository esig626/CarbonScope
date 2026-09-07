"""Complete native flux states, declared count laws, and synthetic observations."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import numpy as np
import pytest

from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.mfa.divergence import kl_divergence
from fluxemu.model import ObservationPrecursor, ObservationTarget, Target, Tracer
from fluxemu.observation import (
    StationaryCountSpecification,
    StationaryObservationExperiment,
    StationaryObservationSpecification,
    multinomial_count_constant,
)
from fluxemu.observation import stationary as observation_module
from fluxemu.observation.stationary import (
    evaluate_stationary_observation_laws,
    sample_stationary_observations,
)
from test_native_stationary_analysis import _alternate_optimum_science


def _states():
    # A complete feasible native state in the model's nonlexical reaction order.
    return (
        CanonicalFluxState("truth", (("Z_IN", 3.0), ("A_IN", 7.0), ("M_OUT", 10.0))),
        CanonicalFluxState("alternative", (("Z_IN", 8.0), ("A_IN", 2.0), ("M_OUT", 10.0))),
    )


def _specification():
    model, experiment = _alternate_optimum_science()
    experiment = replace(
        experiment,
        targets=(Target("I-mid", "I", (1,), "intermediate", "C1", "no"),) + experiment.targets,
        observation_targets=(ObservationTarget(
            "double-I", 2,
            (ObservationPrecursor("I", (1,)), ObservationPrecursor("I", (1,))),
        ),),
    )
    alternate = replace(experiment, tracers=(
        Tracer("S0", (("#0", 0.5), ("#1", 0.5)), "no"), experiment.tracers[1],
    ))
    declarations = (
        StationaryCountSpecification("O-mid", 37, "second"),
        StationaryCountSpecification("double-I", 59, "first"),
        StationaryCountSpecification("I-mid", 11, "first"),
        StationaryCountSpecification("O-mid", 23, "first"),
    )
    return StationaryObservationSpecification(model, (
        StationaryObservationExperiment("z-exp", experiment, declarations),
        StationaryObservationExperiment("a-exp", alternate, declarations[::-1]),
    ))


def test_native_predictions_totals_declared_order_and_provenance_are_retained_exactly():
    specification, states = _specification(), _states()
    result = evaluate_stationary_observation_laws(specification, states)
    assert result.states is states
    assert result.validation.valid
    assert result.validation.max_raw_mass_balance_residual == 0.0
    expected_order = tuple(
        (state.sample_id, block.experiment_id, item.target_id, item.replicate_id)
        for state in states
        for block in specification.experiments
        for item in block.specifications
    )
    assert tuple(
        (item.sample_id, item.experiment_id, item.target_id, item.replicate_id)
        for item in result.components
    ) == expected_order
    predictions = {}
    for block in specification.experiments:
        native = evaluate_stationary(compile_emu_plan(specification.model, block.experiment), states)
        predictions[block.experiment_id] = {
            (item.sample_id, item.target_id): item.fractions for item in native.forward.predictions
        }
    declarations = {
        (block.experiment_id, item.target_id, item.replicate_id): item.total_count
        for block in specification.experiments for item in block.specifications
    }
    assert result.components[0].predicted_mid == pytest.approx((0.3, 0.7), abs=1e-15)
    for item in result.components:
        assert item.predicted_mid == predictions[item.experiment_id][(item.sample_id, item.target_id)]
        assert item.law.n == declarations[(item.experiment_id, item.target_id, item.replicate_id)]
        assert item.mass_classes == tuple(range(len(item.predicted_mid)))
        assert item.model_fingerprint == result.model_fingerprint
        assert item.experiment_fingerprint == dict(result.experiment_fingerprints)[item.experiment_id]
        assert item.specification_fingerprint == result.specification_fingerprint == specification.fingerprint
    assert tuple(key for key, _ in result.experiment_fingerprints) == ("z-exp", "a-exp")
    assert len({item.fingerprint for item in result.components}) == len(result.components)
    replay = evaluate_stationary_observation_laws(specification, states)
    assert tuple(item.fingerprint for item in replay.components) == tuple(
        item.fingerprint for item in result.components
    )
    with pytest.raises(FrozenInstanceError):
        result.components[0].law.n = 99


def test_each_experiment_compiles_and_evaluates_once_for_the_entire_state_batch(monkeypatch):
    specification, states = _specification(), _states()
    original_compile = observation_module.compile_emu_plan
    original_evaluate = observation_module.evaluate_stationary
    original_prepare = observation_module.prepare_highs_flux_region
    compiled, evaluated, prepared = [], [], []

    def record_compile(model, experiment):
        compiled.append(experiment)
        return original_compile(model, experiment)

    def record_evaluate(plan, batch):
        evaluated.append((plan.experiment, batch))
        return original_evaluate(plan, batch)

    def record_prepare(model, fraction):
        prepared.append((model, fraction))
        return original_prepare(model, fraction)

    monkeypatch.setattr(observation_module, "compile_emu_plan", record_compile)
    monkeypatch.setattr(observation_module, "evaluate_stationary", record_evaluate)
    monkeypatch.setattr(observation_module, "prepare_highs_flux_region", record_prepare)
    evaluate_stationary_observation_laws(specification, states)
    assert compiled == [block.experiment for block in specification.experiments]
    assert evaluated == [(block.experiment, states) for block in specification.experiments]
    assert prepared == [(specification.model.flux_model, None)]


def test_synthetic_count_batch_replays_and_keeps_source_identity_and_likelihood_auditable():
    result = evaluate_stationary_observation_laws(_specification(), _states())
    fingerprints = tuple(item.fingerprint for item in result.components)
    samples = sample_stationary_observations(result, seed=626)
    assert samples == sample_stationary_observations(result, seed=626)
    assert samples == sample_stationary_observations(result, rng=np.random.default_rng(626))
    assert len(samples) == len(result.components)
    for sample, component in zip(samples, result.components, strict=True):
        assert sample.component is component
        assert sample.observation.n == component.law.n
        assert sum(sample.observation.counts) == component.law.n
        assert sample.observation.mass_classes == component.mass_classes
        assert all(type(k) is int and k >= 0 for k in sample.observation.counts)
        log_probability = component.law.log_pmf(sample.observation)
        assert math.isfinite(log_probability)
        assert -log_probability == pytest.approx(
            component.law.n * kl_divergence(sample.observation.empirical_mid, component.predicted_mid)
            + multinomial_count_constant(sample.observation), abs=2e-12,
        )
    assert tuple(item.fingerprint for item in result.components) == fingerprints


def test_caller_generator_advances_in_component_order_without_mutating_laws():
    result = evaluate_stationary_observation_laws(_specification(), _states())
    rng, reference = np.random.default_rng(77), np.random.default_rng(77)
    samples = sample_stationary_observations(result, rng=rng)
    expected = tuple(item.law.sample(rng=reference) for item in result.components)
    assert tuple(sample.observation for sample in samples) == expected
    assert rng.random() == reference.random()


@pytest.mark.parametrize("state", [
    CanonicalFluxState("incomplete", (("Z_IN", 3.0), ("A_IN", 7.0))),
    CanonicalFluxState("reordered", (("A_IN", 7.0), ("Z_IN", 3.0), ("M_OUT", 10.0))),
    CanonicalFluxState("unbalanced", (("Z_IN", 3.0), ("A_IN", 7.0), ("M_OUT", 9.0))),
    CanonicalFluxState("bounds", (("Z_IN", 11.0), ("A_IN", 0.0), ("M_OUT", 11.0))),
    CanonicalFluxState("boolean", (("Z_IN", True), ("A_IN", 9.0), ("M_OUT", 10.0))),
    CanonicalFluxState("nonfinite", (("Z_IN", math.nan), ("A_IN", 7.0), ("M_OUT", 10.0))),
    CanonicalFluxState(["mutable"], (("Z_IN", 3.0), ("A_IN", 7.0), ("M_OUT", 10.0))),
])
def test_invalid_complete_state_semantics_fail_before_emu_compilation(monkeypatch, state):
    def forbidden_compile(*args, **kwargs):
        pytest.fail("invalid state reached EMU compilation")

    monkeypatch.setattr(observation_module, "compile_emu_plan", forbidden_compile)
    with pytest.raises((AnalysisError, InputValidationError)):
        evaluate_stationary_observation_laws(_specification(), (state,))


@pytest.mark.parametrize("states", [(), [], {"Z_IN": 3.0}, (object(),)])
def test_bridge_requires_an_explicit_immutable_canonical_state_batch(states):
    with pytest.raises(InputValidationError):
        evaluate_stationary_observation_laws(_specification(), states)


def test_duplicate_sample_ids_are_rejected():
    states = _states()
    with pytest.raises(AnalysisError, match="duplicate flux-state sample ID"):
        evaluate_stationary_observation_laws(
            _specification(), (states[0], replace(states[1], sample_id=states[0].sample_id))
        )


def _override_native_probability(monkeypatch, fractions):
    original_evaluate = observation_module.evaluate_stationary

    def perturbed_native(plan, states):
        native = original_evaluate(plan, states)
        predictions = tuple(
            replace(item, fractions=fractions) if item.target_id == "O-mid" else item
            for item in native.forward.predictions
        )
        return replace(native, forward=replace(native.forward, predictions=predictions))

    monkeypatch.setattr(observation_module, "evaluate_stationary", perturbed_native)


@pytest.mark.parametrize("fractions", [(-1e-12, 1.0 + 1e-12), (0.3, 0.700000000001)])
def test_native_tolerance_cannot_authorize_probability_repair(monkeypatch, fractions):
    _override_native_probability(monkeypatch, fractions)
    with pytest.raises(InputValidationError, match="experiment 'z-exp'.*target 'O-mid'.*state 'truth'"):
        evaluate_stationary_observation_laws(_specification(), _states())


def test_valid_native_machine_roundoff_is_retained_exactly_without_normalizing(monkeypatch):
    fractions = (0.30000000000000004, 0.7000000000000001)
    assert sum(Fraction.from_float(p) for p in fractions) != 1
    _override_native_probability(monkeypatch, fractions)
    result = evaluate_stationary_observation_laws(_specification(), _states())
    for component in result.components:
        if component.target_id == "O-mid":
            assert component.predicted_mid == fractions


def test_total_amplified_native_roundoff_is_rejected_with_context_without_changing_n_or_p(monkeypatch):
    _override_native_probability(monkeypatch, (0.30000000000000004, 0.7000000000000001))
    specification = _specification()
    block = specification.experiments[0]
    declaration = StationaryCountSpecification("O-mid", 10**12, "large")
    specification = replace(specification, experiments=(replace(block, specifications=(declaration,)),))
    with pytest.raises(InputValidationError, match="replicate 'large'.*amplifies"):
        evaluate_stationary_observation_laws(specification, _states()[:1])
    assert declaration.total_count == 10**12


def test_exact_native_zero_support_is_retained_in_generated_counts():
    model, experiment = _alternate_optimum_science()
    specification = StationaryObservationSpecification(model, (
        StationaryObservationExperiment("unlabelled", experiment, (
            StationaryCountSpecification("O-mid", 83, "literal-counts"),
        )),
    ))
    state = CanonicalFluxState("zero-support", (("Z_IN", 10.0), ("A_IN", 0.0), ("M_OUT", 10.0)))
    result = evaluate_stationary_observation_laws(specification, (state,))
    assert result.components[0].predicted_mid == (1.0, 0.0)
    sample = sample_stationary_observations(result, seed=6)[0]
    assert sample.observation.counts == (83, 0)
    assert sample.component.law.log_pmf(sample.observation) == 0.0
    assert sample.component.law.log_pmf((82, 1)) == -math.inf


@pytest.mark.parametrize("kwargs", [{}, {"seed": True}, {"seed": 4, "rng": np.random.default_rng(4)}])
def test_stationary_sampling_requires_exactly_one_explicit_valid_rng_boundary(kwargs):
    result = evaluate_stationary_observation_laws(_specification(), _states()[:1])
    with pytest.raises(InputValidationError):
        sample_stationary_observations(result, **kwargs)


def test_sampling_rejects_a_non_result():
    with pytest.raises(InputValidationError, match="StationaryObservationLawResult"):
        sample_stationary_observations(object(), seed=1)
