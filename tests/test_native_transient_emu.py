"""Native transient EMU tests with no compatibility runtime."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluxemu.emu import compile_transient_emu_plan, evaluate_stationary, evaluate_transient
from fluxemu.exceptions import MappingError
from fluxemu.execution import CanonicalFluxState
from fluxemu.model import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    IsotopeMetabolite,
    IsotopeModel,
    IsotopeParticipant,
    IsotopeReaction,
    LinearObjective,
    MappingBranch,
    PoolQuantity,
    StationaryExperimentSemantics,
    StoichiometricTerm,
    Target,
    Tracer,
    TransientExperimentSemantics,
)


def _identity_reaction(reaction_id: str, source: str, product: str) -> IsotopeReaction:
    transition = AtomTransition(AtomPosition(source, 1), AtomPosition(product, 1))
    return IsotopeReaction(
        reaction_id,
        "forward",
        True,
        (IsotopeParticipant(source, (1,)),),
        (IsotopeParticipant(product, (1,)),),
        (MappingBranch("identity", 1.0, (transition,)),),
    )


def _model() -> CanonicalModel:
    flux = FluxModel(
        (
            FluxMetabolite("S", False),
            FluxMetabolite("A", True),
            FluxMetabolite("T", False),
        ),
        (
            FluxReaction(
                "IN",
                (StoichiometricTerm("S", -1), StoichiometricTerm("A", 1)),
                0,
                100,
            ),
            FluxReaction(
                "OUT",
                (StoichiometricTerm("A", -1), StoichiometricTerm("T", 1)),
                0,
                100,
            ),
        ),
        LinearObjective("maximise", ()),
    )
    isotope = IsotopeModel(
        (
            IsotopeMetabolite("S", 1, True, False),
            IsotopeMetabolite("A", 1, True, False),
            IsotopeMetabolite("T", 1, True, False),
        ),
        (
            _identity_reaction("IN", "S", "A"),
            _identity_reaction("OUT", "A", "T"),
        ),
    )
    return CanonicalModel(flux, isotope)


def _experiment(times=(0.0, 1.0, 10.0)) -> TransientExperimentSemantics:
    return TransientExperimentSemantics(
        tracers=(Tracer("S", (("#1", 1.0),), "no"),),
        targets=(
            Target("A", "A", (1,), "native", "C", "no"),
            Target("T", "T", (1,), "native", "C", "no"),
        ),
        time_points=tuple(times),
        pool_quantities=(PoolQuantity("A", 1.0),),
        initial_internal_mids="unlabelled",
    )


def _labelled(result, sample_id, time, target):
    prediction = next(
        item
        for item in result.forward.predictions
        if item.sample_id == sample_id and item.time == time and item.target_id == target
    )
    return prediction.fractions[1]


def test_single_pool_matches_analytic_first_order_labelling() -> None:
    pytest.importorskip("scipy")
    model = _model()
    plan = compile_transient_emu_plan(model, _experiment())
    result = evaluate_transient(plan, {"IN": 1.0, "OUT": 1.0})

    assert _labelled(result, "state-0", 0.0, "A") == 0.0
    assert _labelled(result, "state-0", 0.0, "T") == 0.0
    assert _labelled(result, "state-0", 1.0, "A") == pytest.approx(1.0 - math.exp(-1.0), abs=2e-8)
    assert _labelled(result, "state-0", 1.0, "T") == pytest.approx(1.0 - math.exp(-1.0), abs=2e-8)
    assert _labelled(result, "state-0", 10.0, "A") == pytest.approx(1.0, abs=5e-5)
    assert result.diagnostics[0].solver_success is True
    assert result.diagnostics[0].maximum_normalization_error <= 1e-8


def test_rate_changes_are_applied_to_each_complete_flux_state_in_order() -> None:
    pytest.importorskip("scipy")
    model = _model()
    plan = compile_transient_emu_plan(model, _experiment((0.0, 1.0)))
    states = (
        CanonicalFluxState.from_mapping("slow", {"IN": 1.0, "OUT": 1.0}),
        CanonicalFluxState.from_mapping("fast", {"IN": 2.0, "OUT": 2.0}),
    )
    result = evaluate_transient(plan, states)
    assert tuple(item.sample_id for item in result.diagnostics) == ("slow", "fast")
    assert _labelled(result, "slow", 1.0, "A") == pytest.approx(1.0 - math.exp(-1.0), abs=2e-8)
    assert _labelled(result, "fast", 1.0, "A") == pytest.approx(1.0 - math.exp(-2.0), abs=2e-8)


def test_terminal_target_is_instantaneous_production_mid() -> None:
    pytest.importorskip("scipy")
    result = evaluate_transient(
        compile_transient_emu_plan(_model(), _experiment((0.0, 0.25, 1.0))),
        {"IN": 1.0, "OUT": 1.0},
    )
    for time in (0.0, 0.25, 1.0):
        assert _labelled(result, "state-0", time, "T") == pytest.approx(
            _labelled(result, "state-0", time, "A"), abs=1e-12
        )


def test_long_time_transient_converges_to_native_stationary_prediction() -> None:
    pytest.importorskip("scipy")
    model = _model()
    transient = evaluate_transient(
        compile_transient_emu_plan(model, _experiment((0.0, 20.0))),
        {"IN": 1.0, "OUT": 1.0},
    )
    stationary = evaluate_stationary(
        __import__("fluxemu.emu", fromlist=["compile_emu_plan"]).compile_emu_plan(
            model,
            StationaryExperimentSemantics(_experiment().tracers, _experiment().targets),
        ),
        {"IN": 1.0, "OUT": 1.0},
    )
    expected = {item.target_id: np.asarray(item.fractions) for item in stationary.forward.predictions}
    for target in ("A", "T"):
        actual = next(
            np.asarray(item.fractions)
            for item in transient.forward.predictions
            if item.time == 20.0 and item.target_id == target
        )
        np.testing.assert_allclose(actual, expected[target], atol=5e-8, rtol=0.0)


def test_compile_requires_exact_pool_quantities_for_dynamic_metabolites() -> None:
    experiment = TransientExperimentSemantics(
        tracers=_experiment().tracers,
        targets=_experiment().targets,
        time_points=(0.0, 1.0),
        pool_quantities=(),
        initial_internal_mids="unlabelled",
    )
    with pytest.raises(MappingError, match="missing pool quantity"):
        compile_transient_emu_plan(_model(), experiment)
