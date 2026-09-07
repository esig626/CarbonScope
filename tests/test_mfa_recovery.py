"""Noise-free acceptance: identifiable recovery versus compatible flux scales."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import runpy

import pytest

from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.flux_analysis import prepare_highs_flux_region, validate_flux_states
from fluxemu.mfa import DivergenceObjectiveConfig, MFAOptimizationConfig, renyi_divergence
from fluxemu.mfa.stationary import evaluate_stationary_mfa, fit_stationary_mfa


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("highspy") is None or importlib.util.find_spec("scipy") is None,
    reason="stationary MFA optimization dependencies are unavailable",
)

_EXAMPLE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples" / "stationary_mfa_recovery.py")
)
build_mixture_problem = _EXAMPLE["build_mixture_problem"]
identifiable_starts = _EXAMPLE["identifiable_starts"]
nonidentifiable_starts = _EXAMPLE["nonidentifiable_starts"]


def _independently_check_result(problem, result, objective):
    """Rebuild original constraints and native EMU independently of fit output."""

    prepared = prepare_highs_flux_region(problem.model.flux_model, None)
    report = validate_flux_states(prepared, (result.state,))
    assert report.valid, report.errors
    assert report.reaction_order == ("Z_IN", "A_IN", "M_OUT")
    assert report.max_raw_mass_balance_residual < 1e-10
    assert report.max_lower_bound_violation < 1e-10
    assert report.max_upper_bound_violation < 1e-10
    assert result.validation.valid

    assert tuple(key for key, _ in result.state.values) == ("Z_IN", "A_IN", "M_OUT")
    block = problem.experiments[0]
    native = evaluate_stationary(
        compile_emu_plan(problem.model, block.experiment), (result.state,),
    ).forward.predictions[0]
    component = result.components[0]
    assert (component.experiment_id, component.target_id, component.replicate_id) == (
        "labelled-mixture", "O-mid", "0",
    )
    assert component.observed == block.observations[0].fractions
    assert component.predicted == native.fractions
    exact_loss = renyi_divergence(component.observed, native.fractions, objective.alpha)
    assert component.divergence == exact_loss
    assert result.total_loss == exact_loss
    assert abs(result.total_loss) < 1e-10
    assert native.fractions == pytest.approx(component.observed, abs=1e-6)
    reevaluated = evaluate_stationary_mfa(problem, result.state, objective=objective)
    assert reevaluated.total_loss == result.total_loss
    assert reevaluated.components == result.components


@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
def test_identifiable_noise_free_truth_recovery_from_distinct_nontruth_starts(alpha):
    problem, truth = build_mixture_problem()
    starts = identifiable_starts()
    prepared = prepare_highs_flux_region(problem.model.flux_model, None)
    assert validate_flux_states(prepared, starts + (truth,)).valid
    assert len({state.values for state in starts}) == 3
    assert all(state.values != truth.values for state in starts)

    # Identifiability is algebraic: fixed measured total 10 and the two MID
    # entries determine Z_IN=10*p0 and A_IN=10*p1. Observations themselves are
    # still obtained from the native forward model, without renormalization.
    block = problem.experiments[0]
    forward_truth = evaluate_stationary(
        compile_emu_plan(problem.model, block.experiment), (truth,),
    ).forward.predictions[0].fractions
    assert block.observations[0].fractions == forward_truth
    assert forward_truth == pytest.approx((0.3, 0.7), abs=1e-14)
    assert tuple(value for _, value in truth.values) == pytest.approx(
        (10 * forward_truth[0], 10 * forward_truth[1], 10), abs=1e-13,
    )

    objective = DivergenceObjectiveConfig(alpha=alpha)
    result = fit_stationary_mfa(
        problem, objective=objective,
        optimization=MFAOptimizationConfig(seed=26, ftol=1e-13),
        initial_states=starts,
    )
    assert result.affine_dimension == 1
    assert len(result.start_diagnostics) == 3
    assert tuple(item.initial_state for item in result.start_diagnostics) == starts
    assert all(item.accepted for item in result.start_diagnostics)
    assert all(item.initial_loss > 1e-3 for item in result.start_diagnostics)
    assert all(abs(item.final_loss) < 1e-10 for item in result.start_diagnostics)
    assert tuple(value for _, value in result.state.values) == pytest.approx(
        tuple(value for _, value in truth.values), abs=5e-5,
    )
    _independently_check_result(problem, result, objective)


@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
def test_nonidentifiable_scale_control_retains_distinct_compatible_fitted_states(alpha):
    problem, _ = build_mixture_problem(fixed_throughput=False)
    starts = nonidentifiable_starts()
    prepared = prepare_highs_flux_region(problem.model.flux_model, None)
    assert validate_flux_states(prepared, starts).valid
    block = problem.experiments[0]
    plan = compile_emu_plan(problem.model, block.experiment)
    witnesses = evaluate_stationary(plan, starts[:2]).forward.predictions
    assert starts[0].values != starts[1].values
    assert witnesses[0].fractions == witnesses[1].fractions
    assert witnesses[0].fractions == block.observations[0].fractions

    objective = DivergenceObjectiveConfig(alpha=alpha)
    result = fit_stationary_mfa(
        problem, objective=objective,
        optimization=MFAOptimizationConfig(seed=26, ftol=1e-13),
        initial_states=starts,
    )
    assert result.affine_dimension == 2
    assert len(result.start_diagnostics) == len(starts)
    assert tuple(item.initial_state for item in result.start_diagnostics) == starts
    accepted = tuple(item for item in result.start_diagnostics if item.accepted)
    assert len(accepted) >= 2
    assert result.start_diagnostics[result.best_start_index].accepted
    assert all(item.initial_loss > 1e-3 for item in result.start_diagnostics[2:])
    assert all(item.accepted for item in result.start_diagnostics[2:])

    fitted_scales = []
    for diagnostic in accepted:
        assert diagnostic.optimizer_success and diagnostic.validated
        assert diagnostic.final_state is not None
        assert math.isfinite(diagnostic.final_loss)
        assert abs(diagnostic.final_loss) < 1e-10
        state = diagnostic.final_state
        assert validate_flux_states(prepared, (state,)).valid
        native_mid = evaluate_stationary(plan, (state,)).forward.predictions[0].fractions
        assert native_mid == pytest.approx(block.observations[0].fractions, abs=1e-6)
        assert renyi_divergence(block.observations[0].fractions, native_mid, alpha) == diagnostic.final_loss
        fluxes = dict(state.values)
        assert fluxes["A_IN"] / fluxes["M_OUT"] == pytest.approx(0.7, abs=1e-6)
        fitted_scales.append(fluxes["M_OUT"])
    # Different recovered scales with the same fitted MID are expected. There
    # is deliberately no assertion that the free complete state equals truth.
    assert max(fitted_scales) - min(fitted_scales) > 1.0
    _independently_check_result(problem, result, objective)


def test_identifiable_generated_multistart_replays_seeded_starts_and_fit():
    problem, _ = build_mixture_problem()
    objective = DivergenceObjectiveConfig(alpha=0.5)
    optimization = MFAOptimizationConfig(
        n_starts=4, seed=2026, ftol=1e-13, burn_in=12, thinning=3,
    )
    first = fit_stationary_mfa(problem, objective=objective, optimization=optimization)
    replay = fit_stationary_mfa(problem, objective=objective, optimization=optimization)
    starts = tuple(item.initial_state for item in first.start_diagnostics)
    assert len(starts) == 4
    assert len({state.values for state in starts}) == 4
    prepared = prepare_highs_flux_region(problem.model.flux_model, None)
    assert validate_flux_states(prepared, starts).valid
    assert first.state == replay.state
    assert first.total_loss == replay.total_loss
    assert first.components == replay.components
    assert first.start_diagnostics == replay.start_diagnostics
    assert first.fit_fingerprint == replay.fit_fingerprint
    _independently_check_result(problem, first, objective)
