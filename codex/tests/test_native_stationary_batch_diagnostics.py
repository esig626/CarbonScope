"""Failure context for ordered native stationary EMU batches."""

from __future__ import annotations

import numpy as np
import pytest

import fluxemu.emu.stationary as stationary_module
from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.exceptions import ForwardEMUError
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
    ObjectiveTerm,
    StationaryExperimentSemantics,
    StoichiometricTerm,
    Target,
    Tracer,
)


def _terminal_target_science(
) -> tuple[CanonicalModel, StationaryExperimentSemantics]:
    model = CanonicalModel(
        FluxModel(
            (
                FluxMetabolite("S", False),
                FluxMetabolite("P", False),
            ),
            (
                FluxReaction(
                    "R",
                    (
                        StoichiometricTerm("S", -1.0),
                        StoichiometricTerm("P", 1.0),
                    ),
                    0.0,
                    1.0,
                ),
            ),
            LinearObjective("maximise", (ObjectiveTerm("R", 1.0),)),
        ),
        IsotopeModel(
            (
                IsotopeMetabolite("S", 1, True, False),
                IsotopeMetabolite("P", 1, True, False),
            ),
            (
                IsotopeReaction(
                    "R",
                    "forward",
                    True,
                    (IsotopeParticipant("S", (1,)),),
                    (IsotopeParticipant("P", (1,)),),
                    (
                        MappingBranch(
                            "declared",
                            1.0,
                            (
                                AtomTransition(
                                    AtomPosition("S", 1),
                                    AtomPosition("P", 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    experiment = StationaryExperimentSemantics(
        (Tracer("S", (("#1", 1.0),), "no"),),
        (Target("P-mid", "P", (1,), "whole", "C1", "no"),),
    )
    return model, experiment


def test_batch_failure_names_the_exact_flux_state_and_preserves_cause() -> None:
    model, experiment = _terminal_target_science()
    plan = compile_emu_plan(model, experiment)
    productive = CanonicalFluxState("productive", (("R", 1.0),))
    failing = CanonicalFluxState("zero-flow", (("R", 0.0),))

    try:
        evaluate_stationary(plan, (productive, failing))
    except ForwardEMUError as error:
        assert str(error).startswith("flux state 'zero-flow': ")
        assert "terminal target 'P-mid' has zero productive flux" in str(error)
        assert isinstance(error.__cause__, ForwardEMUError)
        assert "flux state" not in str(error.__cause__)
    else:  # pragma: no cover - explicit assertion for diagnostic clarity
        raise AssertionError("zero-flow batch state unexpectedly evaluated")


def test_batch_context_wrapper_does_not_change_successful_results() -> None:
    model, experiment = _terminal_target_science()
    plan = compile_emu_plan(model, experiment)
    state = CanonicalFluxState("productive", (("R", 1.0),))

    first = evaluate_stationary(plan, (state,))
    second = evaluate_stationary(plan, (state,))

    assert first == second
    assert tuple(item.sample_id for item in first.forward.predictions) == (
        "productive",
    )
    assert first.forward.predictions[0].fractions == (0.0, 1.0)


def test_linear_solve_failure_is_fluxemu_error_with_state_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, experiment = _terminal_target_science()
    plan = compile_emu_plan(model, experiment)
    state = CanonicalFluxState("linear-failure", (("R", 1.0),))

    def fail_linear_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("deliberate singular solve")

    monkeypatch.setattr(stationary_module, "_evaluate_state", fail_linear_solve)

    with pytest.raises(ForwardEMUError) as captured:
        evaluate_stationary(plan, (state,))

    error = captured.value
    assert str(error).startswith(
        "flux state 'linear-failure': stationary EMU linear solve failed: "
    )
    assert "deliberate singular solve" in str(error)
    assert isinstance(error.__cause__, np.linalg.LinAlgError)
