"""Original-model feasibility, native reuse, and transparent fitting failures."""

from dataclasses import replace
import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.flux_analysis import (
    prepare_highs_flux_region,
    run_highs_fva_reference,
    run_prepared_highs_vffva,
    sample_prepared_flux_states,
    validate_flux_states,
)
from fluxemu.flux_analysis import sampling as sampling_module
from fluxemu.mfa import (
    DivergenceObjectiveConfig,
    MFAFitError,
    MFAOptimizationConfig,
    StationaryMFAExperiment,
    StationaryMFAProblem,
    StationaryMIDObservation,
)
from fluxemu.mfa import stationary as mfa_module
from fluxemu.mfa.stationary import evaluate_stationary_mfa, fit_stationary_mfa
from fluxemu.model import FluxModel, FluxReaction, LinearObjective, ObjectiveTerm, Tracer
from test_native_highs_sampling import _auxiliary_bounds_model
from test_native_stationary_analysis import _alternate_optimum_science


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("highspy") is None or importlib.util.find_spec("scipy") is None,
    reason="native MFA optimization dependencies are unavailable",
)


def _problem(*, fixed_total=True, direction="maximise", coefficient=1.0, minimum_input=0.0):
    model, experiment = _alternate_optimum_science()
    reactions = list(model.flux_model.reactions)
    reactions[0] = replace(reactions[0], lower_bound=minimum_input)
    reactions[2] = replace(reactions[2], lower_bound=10.0 if fixed_total else 2.0)
    flux_model = replace(
        model.flux_model, reactions=tuple(reactions),
        objective=LinearObjective(direction, (ObjectiveTerm("Z_IN", coefficient),)),
    )
    model = replace(model, flux_model=flux_model)
    return StationaryMFAProblem(model, (
        StationaryMFAExperiment("mixture", experiment, (StationaryMIDObservation("O-mid", (0.3, 0.7)),)),
    ))


def _state(x=3.0, total=10.0, sample_id="supplied"):
    return CanonicalFluxState(sample_id, (("Z_IN", x), ("A_IN", total-x), ("M_OUT", total)))


@pytest.mark.parametrize("direction,coefficient,lower", [
    ("maximise", 1.0, 0.0), ("minimise", -1.0, 0.0),
    ("minimise", 1.0, 2.0), ("maximise", -1.0, 2.0),
])
def test_unrestricted_native_reference_reusable_and_no_retention_row(direction, coefficient, lower):
    model = FluxModel((), (FluxReaction("SIGNED", (), lower, 10.0),),
                      LinearObjective(direction, (ObjectiveTerm("SIGNED", coefficient),)))
    prepared = prepare_highs_flux_region(model, None)
    assert prepared.retention is None
    instrumentation = {}
    actual = run_prepared_highs_vffva(prepared, audit_endpoints=True, instrumentation=instrumentation)
    reference = run_highs_fva_reference(model, None)
    pd.testing.assert_frame_equal(actual.ranges, reference.ranges)
    assert actual.ranges.loc["SIGNED"].tolist() == [lower, 10.0]
    assert actual.fraction_of_optimum is None
    assert instrumentation["retention_rows"] == 0
    assert validate_flux_states(prepared, (
        CanonicalFluxState("nonoptimal", (("SIGNED", 5.0),)),
    )).valid
    # Optional retention preserves the native sign arithmetic, including its
    # explicit rejection when a fractional bound is stricter than the optimum.
    if lower > 0:
        with pytest.raises(AnalysisError, match="infeasible"):
            prepare_highs_flux_region(model, 0.5)


def test_unrestricted_shared_geometry_preserves_fixed_blocked_signed_reactions():
    model = _auxiliary_bounds_model()
    prepared = prepare_highs_flux_region(model, None)
    instrumentation = {}
    fva = run_prepared_highs_vffva(prepared, workers=2, audit_endpoints=True,
                                 instrumentation=instrumentation)
    pd.testing.assert_frame_equal(fva.ranges, run_highs_fva_reference(model, None).ranges)
    assert instrumentation["retention_rows"] == 0
    result = sample_prepared_flux_states(prepared, 12, seed=626, fva=fva, burn_in=4, thinning=1)
    assert result.validation.valid
    assert result.provenance.affine_dimension == 1
    assert not result.provenance.optimal_face
    assert result.provenance.fraction_of_optimum is None
    assert result.provenance.retained_objective_bound is None
    assert result.provenance.retained_objective_sense is None
    assert result.provenance.effective_objective_bound is None
    for state in result.states:
        assert tuple(key for key, _ in state.values) == ("FIXED", "BLOCKED", "SIGNED", "NETWORK_BLOCKED")
        values = dict(state.values)
        assert values["FIXED"] == 2.0
        assert values["BLOCKED"] == values["NETWORK_BLOCKED"] == 0.0
        assert -4.0 <= values["SIGNED"] <= 4.0
    assert min(dict(state.values)["SIGNED"] for state in result.states) < 0.0
    assert max(dict(state.values)["SIGNED"] for state in result.states) > 0.0


def test_unrestricted_fva_evidence_cannot_be_reused_for_retained_sampling():
    problem = _problem()
    unrestricted = prepare_highs_flux_region(problem.model.flux_model, None)
    fva = run_prepared_highs_vffva(unrestricted)
    retained = prepare_highs_flux_region(problem.model.flux_model, 0.8)
    with pytest.raises(AnalysisError, match="metadata"):
        sample_prepared_flux_states(retained, 2, seed=0, fva=fva)


def test_default_mfa_does_not_lock_biological_optimum():
    problem = _problem(fixed_total=False)
    result = fit_stationary_mfa(problem, initial_states=(_state(1.5, 5.0),))
    assert result.total_loss < 1e-12
    assert dict(result.state.values)["M_OUT"] == pytest.approx(5.0, abs=1e-6)
    assert dict(result.state.values)["Z_IN"] < 2.0
    assert result.affine_dimension == 2
    assert result.validation.valid
    assert result.optimization_config.fraction_of_optimum is None
    assert result.start_diagnostics[0].accepted


@pytest.mark.parametrize("direction,coefficient", [("maximise", 1.0), ("minimise", -1.0)])
def test_explicit_fraction_is_max_or_min_feasibility_constraint(direction, coefficient):
    problem = _problem(direction=direction, coefficient=coefficient)
    result = fit_stationary_mfa(
        problem, optimization=MFAOptimizationConfig(fraction_of_optimum=0.8, ftol=1e-12),
        initial_states=(_state(9.0),),
    )
    assert dict(result.state.values)["Z_IN"] == pytest.approx(8.0, abs=1e-6)
    assert result.total_loss > 0.1
    prepared = prepare_highs_flux_region(problem.model.flux_model, 0.8)
    assert validate_flux_states(prepared, (result.state,)).valid
    independently_evaluated = evaluate_stationary_mfa(problem, result.state)
    assert result.total_loss == independently_evaluated.total_loss


def test_minimization_positive_optimum_is_optional_and_singleton_retention_works():
    problem = _problem(direction="minimise", minimum_input=2.0)
    default = fit_stationary_mfa(problem, initial_states=(_state(),))
    assert default.total_loss < 1e-12
    retained = fit_stationary_mfa(
        problem, optimization=MFAOptimizationConfig(fraction_of_optimum=1.0),
        initial_states=(_state(2.0),),
    )
    assert retained.affine_dimension == 0
    assert dict(retained.state.values)["Z_IN"] == 2.0
    assert retained.start_diagnostics[0].iterations == 0
    assert retained.start_diagnostics[0].optimizer_success
    assert retained.total_loss > 0.0


def test_generated_starts_share_geometry_once_and_replay_exactly(monkeypatch):
    problem = _problem()
    calls = {"geometry": 0, "center": 0, "compile": 0, "sample": 0, "emu": 0}
    original_geometry = sampling_module._build_reduced_geometry
    original_center = sampling_module._chebyshev_center
    original_compile = mfa_module.compile_emu_plan
    original_sample = mfa_module._sample_flux_states_from_geometry
    original_emu = mfa_module.evaluate_stationary

    def geometry(*args):
        calls["geometry"] += 1
        return original_geometry(*args)

    def center(*args):
        calls["center"] += 1
        return original_center(*args)

    def compile_plan(*args):
        calls["compile"] += 1
        return original_compile(*args)

    def sample(shared, *args, **kwargs):
        calls["sample"] += 1
        return original_sample(shared, *args, **kwargs)

    def emu(plan, states, **kwargs):
        calls["emu"] += 1
        assert len(states) == 1
        assert tuple(key for key, _ in states[0].values) == ("Z_IN", "A_IN", "M_OUT")
        assert validate_flux_states(prepare_highs_flux_region(plan.model.flux_model, None), states).valid
        return original_emu(plan, states, **kwargs)

    monkeypatch.setattr(sampling_module, "_build_reduced_geometry", geometry)
    monkeypatch.setattr(sampling_module, "_chebyshev_center", center)
    monkeypatch.setattr(mfa_module, "compile_emu_plan", compile_plan)
    monkeypatch.setattr(mfa_module, "_sample_flux_states_from_geometry", sample)
    monkeypatch.setattr(mfa_module, "evaluate_stationary", emu)
    config = MFAOptimizationConfig(n_starts=3, seed=626, burn_in=4, thinning=1)
    result = fit_stationary_mfa(problem, optimization=config)
    assert {key: calls[key] for key in ("geometry", "center", "compile", "sample")} == dict.fromkeys(
        ("geometry", "center", "compile", "sample"), 1)
    assert calls["emu"] > 3
    assert len({item.initial_state.values for item in result.start_diagnostics}) == 3
    replay = fit_stationary_mfa(problem, optimization=config)
    assert result.state == replay.state
    assert result.total_loss == replay.total_loss
    assert result.start_diagnostics == replay.start_diagnostics
    assert result.fit_fingerprint == replay.fit_fingerprint


def test_multiple_experiments_replicates_plain_sum_and_compile_once(monkeypatch):
    problem = _problem()
    first = replace(problem.experiments[0], observations=(
        StationaryMIDObservation("O-mid", (0.3, 0.7), "a"),
        StationaryMIDObservation("O-mid", (0.4, 0.6), "b"),
    ))
    experiment = replace(first.experiment, tracers=(
        Tracer("S0", (("#1", 1.0),), "no"), Tracer("S1", (("#0", 1.0),), "no"),
    ))
    second = StationaryMFAExperiment("swapped", experiment, (StationaryMIDObservation("O-mid", (0.7, 0.3)),))
    problem = replace(problem, experiments=(first, second))
    original = mfa_module.compile_emu_plan
    compiled = []

    def recording(*args):
        compiled.append(args[1])
        return original(*args)

    monkeypatch.setattr(mfa_module, "compile_emu_plan", recording)
    result = fit_stationary_mfa(problem, initial_states=(_state(4.0), _state(5.0, sample_id="second")))
    assert len(compiled) == 2
    assert [(item.experiment_id, item.replicate_id) for item in result.components] == [
        ("mixture", "a"), ("mixture", "b"), ("swapped", "0"),
    ]
    assert result.total_loss == math.fsum(item.divergence for item in result.components)
    assert result.components[0].observed == (0.3, 0.7)
    assert result.components[2].predicted == pytest.approx(tuple(reversed(result.components[0].predicted)))


@pytest.mark.parametrize("malformed", [
    CanonicalFluxState("bad-order", (("M_OUT", 10.0), ("A_IN", 7.0), ("Z_IN", 3.0))),
    CanonicalFluxState("missing", (("Z_IN", 3.0), ("A_IN", 7.0))),
    _state(-1.0, sample_id="bad-bound"),
    CanonicalFluxState("mass", (("Z_IN", 3.0), ("A_IN", 7.0), ("M_OUT", 9.0))),
])
def test_malformed_finite_start_is_diagnosed_without_emu_and_good_start_survives(monkeypatch, malformed):
    problem = _problem()
    original = mfa_module.evaluate_stationary
    seen = []

    def recording(plan, states, **kwargs):
        seen.extend(state.sample_id for state in states)
        return original(plan, states, **kwargs)

    monkeypatch.setattr(mfa_module, "evaluate_stationary", recording)
    result = fit_stationary_mfa(problem, initial_states=(malformed, _state()))
    assert result.best_start_index == 1
    assert len(result.start_diagnostics) == 2
    failed = result.start_diagnostics[0]
    assert not failed.optimizer_success and not failed.validated
    assert not failed.accepted
    assert failed.initial_loss is None
    assert failed.message
    assert malformed.sample_id not in seen


@pytest.mark.parametrize("initial_states", [(), (object(),), "bad", (_state(math.nan),)])
def test_unserializable_or_malformed_start_request_is_rejected(initial_states):
    with pytest.raises(InputValidationError):
        fit_stationary_mfa(_problem(), initial_states=initial_states)


def test_initial_support_mismatch_is_exact_and_all_failed_diagnostics_are_accessible():
    problem = _problem()
    bad = (_state(0.0, sample_id="zero-first"), _state(10.0, sample_id="zero-second"))
    assert evaluate_stationary_mfa(problem, bad[0]).total_loss == math.inf
    with pytest.raises(MFAFitError) as caught:
        fit_stationary_mfa(problem, initial_states=bad)
    assert caught.value.fit_fingerprint
    assert len(caught.value.start_diagnostics) == 2
    assert all(item.initial_loss == math.inf for item in caught.value.start_diagnostics)
    assert all("support mismatch" in item.message for item in caught.value.start_diagnostics)


def test_native_undefined_initial_state_is_isolated_and_contextual():
    problem = _problem(fixed_total=False)
    reactions = list(problem.model.flux_model.reactions)
    reactions[-1] = replace(reactions[-1], lower_bound=0.0)
    problem = replace(problem, model=replace(problem.model, flux_model=replace(
        problem.model.flux_model, reactions=tuple(reactions))))
    result = fit_stationary_mfa(problem, initial_states=(_state(0.0, 0.0, "zero-flow"), _state(1.5, 5.0)))
    assert result.best_start_index == 1
    assert "mixture" in result.start_diagnostics[0].message
    assert "zero-flow" in result.start_diagnostics[0].message
    assert result.start_diagnostics[0].initial_loss is None


def _outcome(theta, *, success=True, reported_loss=-1e99):
    return SimpleNamespace(x=theta, success=success, status=0 if success else 9,
                           nit=2, nfev=3, fun=reported_loss, message="mock optimizer",
                           jac=np.zeros(np.size(theta)))


def test_trial_infeasibility_returns_inf_without_emu_and_records_failure(monkeypatch):
    calls = []
    original = mfa_module.evaluate_stationary

    def recording(plan, states, **kwargs):
        calls.extend(states)
        return original(plan, states, **kwargs)

    def optimizer(fun, theta, **kwargs):
        before = len(calls)
        assert fun(theta + 1e6) == math.inf
        assert len(calls) == before
        assert math.isfinite(fun(theta))
        return _outcome(theta)

    monkeypatch.setattr(mfa_module, "evaluate_stationary", recording)
    monkeypatch.setattr(mfa_module, "_load_scipy_optimizer", lambda: (optimizer, "test"))
    result = fit_stationary_mfa(_problem(), initial_states=(_state(),))
    assert len(result.start_diagnostics[0].trial_failures) == 1
    assert "bound" in result.start_diagnostics[0].trial_failures[0]
    assert abs(result.total_loss) < 1e-14  # optimizer's fabricated negative fun is ignored


def test_feasible_trial_support_mismatch_is_exact_infinity_and_remains_visible(monkeypatch):
    def optimizer(fun, theta, **kwargs):
        # Zero reduced coordinates reconstruct the native biological FBA
        # anchor: all input through Z_IN, hence predicted mass (1, 0).
        assert fun(np.zeros_like(theta)) == math.inf
        return _outcome(theta)

    monkeypatch.setattr(mfa_module, "_load_scipy_optimizer", lambda: (optimizer, "test"))
    result = fit_stationary_mfa(_problem(), initial_states=(_state(),))
    diagnostic = result.start_diagnostics[0]
    assert diagnostic.accepted
    assert "support mismatch" in diagnostic.trial_failures[0]
    assert abs(result.total_loss) < 1e-14


def test_feasible_but_undefined_trial_remains_visible(monkeypatch):
    problem = _problem(fixed_total=False)
    reactions = list(problem.model.flux_model.reactions)
    reactions[-1] = replace(reactions[-1], lower_bound=0.0)
    problem = replace(problem, model=replace(problem.model, flux_model=replace(
        problem.model.flux_model, reactions=tuple(reactions),
        objective=LinearObjective("minimise", (ObjectiveTerm("M_OUT", 1.0),)))))

    def optimizer(fun, theta, **kwargs):
        assert fun(np.zeros_like(theta)) == math.inf  # native zero-flux anchor
        return _outcome(theta)

    monkeypatch.setattr(mfa_module, "_load_scipy_optimizer", lambda: (optimizer, "test"))
    result = fit_stationary_mfa(problem, initial_states=(_state(1.5, 5.0),))
    assert result.start_diagnostics[0].accepted
    assert "mixture" in result.start_diagnostics[0].trial_failures[0]
    assert "support mismatch" not in result.start_diagnostics[0].trial_failures[0]


@pytest.mark.parametrize("bad_final", ["infeasible", "nonfinite", "wrong-shape", "nonnumeric", "unsuccessful"])
def test_optimizer_success_cannot_bypass_independent_validation(monkeypatch, bad_final):
    def optimizer(fun, theta, **kwargs):
        if bad_final == "infeasible":
            return _outcome(theta + 1e6)
        if bad_final == "nonfinite":
            return _outcome(np.full_like(theta, np.nan))
        if bad_final == "wrong-shape":
            return _outcome(np.zeros(len(theta) + 1))
        if bad_final == "nonnumeric":
            return _outcome(("invalid",))
        return _outcome(theta, success=False)

    monkeypatch.setattr(mfa_module, "_load_scipy_optimizer", lambda: (optimizer, "test"))
    with pytest.raises(MFAFitError) as caught:
        fit_stationary_mfa(_problem(), initial_states=(_state(),))
    failed = caught.value.start_diagnostics[0]
    assert failed.optimizer_success == (bad_final != "unsuccessful")
    assert failed.validated == (bad_final == "unsuccessful")
    assert failed.message


def test_backend_success_with_nonfinite_gradient_is_rejected_and_later_start_survives(monkeypatch):
    count = 0

    def optimizer(fun, theta, **kwargs):
        nonlocal count
        count += 1
        outcome = _outcome(theta)
        if count == 1:
            outcome.jac[:] = math.inf
        return outcome

    monkeypatch.setattr(mfa_module, "_load_scipy_optimizer", lambda: (optimizer, "test"))
    result = fit_stationary_mfa(_problem(), initial_states=(_state(), _state(4.0, sample_id="second")))
    assert result.best_start_index == 1
    rejected = result.start_diagnostics[0]
    assert rejected.optimizer_success  # Preserve the raw backend flag honestly.
    assert rejected.validated
    assert not rejected.accepted
    assert result.start_diagnostics[1].accepted
    assert "rejected numerical convergence" in rejected.message


def test_best_eligible_loss_is_recomputed_and_failed_lower_loss_cannot_win(monkeypatch):
    count = 0

    def optimizer(fun, theta, **kwargs):
        nonlocal count
        count += 1
        return _outcome(theta, success=count != 1)

    monkeypatch.setattr(mfa_module, "_load_scipy_optimizer", lambda: (optimizer, "test"))
    result = fit_stationary_mfa(_problem(), initial_states=(
        _state(), _state(4.0, sample_id="valid-second"), _state(5.0, sample_id="valid-third"),
    ))
    assert result.best_start_index == 1
    assert abs(result.start_diagnostics[0].final_loss) < 1e-14
    assert not result.start_diagnostics[0].optimizer_success
    assert result.total_loss == result.start_diagnostics[1].final_loss
    assert result.total_loss < result.start_diagnostics[2].final_loss


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, FloatingPointError])
def test_backend_failure_isolated_then_later_start_succeeds(monkeypatch, error_type):
    count = 0

    def optimizer(fun, theta, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise error_type("numerical backend failure")
        return _outcome(theta)

    monkeypatch.setattr(mfa_module, "_load_scipy_optimizer", lambda: (optimizer, "test"))
    result = fit_stationary_mfa(_problem(), initial_states=(
        _state(4.0, sample_id="fails"), _state(),
    ))
    assert result.best_start_index == 1
    assert not result.start_diagnostics[0].optimizer_success
    assert "SciPy SLSQP failed" in result.start_diagnostics[0].message
    assert error_type.__name__ in result.start_diagnostics[0].message


def test_programming_errors_in_objective_callback_are_not_swallowed(monkeypatch):
    original = mfa_module._CompiledMFAObjective.evaluate

    def broken(compiled, state):
        if state.sample_id.startswith("mfa-start"):
            raise RuntimeError("deliberate programming bug")
        return original(compiled, state)

    monkeypatch.setattr(mfa_module._CompiledMFAObjective, "evaluate", broken)
    with pytest.raises(RuntimeError, match="programming bug"):
        fit_stationary_mfa(_problem(), initial_states=(_state(),))


def test_selected_final_gate_retains_all_start_diagnostics(monkeypatch):
    original = mfa_module._CompiledMFAObjective.evaluate
    final_calls = 0

    def evaluator(compiled, state):
        nonlocal final_calls
        if state.sample_id.startswith("mfa-start"):
            final_calls += 1
            if final_calls == 2:
                raise AnalysisError("deliberate selected-state rejection")
        return original(compiled, state)

    monkeypatch.setattr(mfa_module._CompiledMFAObjective, "evaluate", evaluator)
    monkeypatch.setattr(mfa_module, "_load_scipy_optimizer", lambda: (
        lambda fun, theta, **kwargs: _outcome(theta), "test"))
    with pytest.raises(MFAFitError, match="selected-state rejection") as caught:
        fit_stationary_mfa(_problem(), initial_states=(_state(),))
    assert len(caught.value.start_diagnostics) == 1
    assert caught.value.start_diagnostics[0].optimizer_success
    assert caught.value.fit_fingerprint
