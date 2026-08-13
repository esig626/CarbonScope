"""Correctness and structural controls for reusable native HiGHS FVA."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

import fluxemu.flux_analysis.highs as highs
from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis import run_highs_fva_reference, run_highs_vffva
from fluxemu.model import (FluxMetabolite, FluxModel, FluxReaction, LinearObjective,
                           ObjectiveTerm, StoichiometricTerm)

pytestmark = pytest.mark.skipif(importlib.util.find_spec("highspy") is None,
                                reason="highspy is unavailable")


def _model(*, direction="maximise", fraction_objective=(("export", 1.0),)):
    metabolites = (FluxMetabolite("A", True), FluxMetabolite("B", True))
    reactions = (
        FluxReaction("source", (StoichiometricTerm("A", 1),), 0, 10),
        FluxReaction("reversible", (StoichiometricTerm("A", -1), StoichiometricTerm("B", 1)), -4, 10),
        FluxReaction("export", (StoichiometricTerm("B", -1),), 0, 10),
        FluxReaction("fixed", (), 2, 2),
        FluxReaction("blocked", (), 0, 0),
    )
    return FluxModel(metabolites, reactions, LinearObjective(
        direction, tuple(ObjectiveTerm(*term) for term in fraction_objective)))


@pytest.mark.parametrize("fraction", [1.0, 0.9])
@pytest.mark.parametrize("workers", [1, 2])
def test_reference_parity_order_and_serial_parallel(fraction, workers):
    model = _model()
    expected = run_highs_fva_reference(model, fraction)
    actual = run_highs_vffva(model, fraction, workers=workers)
    assert tuple(actual.ranges.index) == tuple(r.reaction_id for r in model.reactions)
    pd.testing.assert_frame_equal(actual.ranges, expected.ranges, atol=1e-8, rtol=1e-8)
    assert actual.objective_value == pytest.approx(expected.objective_value)
    if workers == 2:
        pd.testing.assert_frame_equal(actual.ranges,
                                      run_highs_vffva(model, fraction, workers=1).ranges,
                                      atol=1e-8, rtol=1e-8)


def test_minimisation_and_multiterm_objectives_match_reference():
    model = _model(direction="minimise", fraction_objective=(("export", 1), ("source", 0.5)))
    expected = run_highs_fva_reference(model, 0.9)
    actual = run_highs_vffva(model, 0.9, workers=1)
    pd.testing.assert_frame_equal(actual.ranges, expected.ranges, atol=1e-8, rtol=1e-8)


def test_one_compile_one_solver_and_one_matrix_handles_all_endpoints(monkeypatch):
    compile_calls = 0
    worker_builds = 0
    original_compile, original_worker = highs.compile_flux_lp, highs._ReusableFVAWorker

    def compile_once(model):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(model)

    class RecordingWorker(original_worker):
        def __init__(self, *args):
            nonlocal worker_builds
            worker_builds += 1
            super().__init__(*args)

    monkeypatch.setattr(highs, "compile_flux_lp", compile_once)
    monkeypatch.setattr(highs, "_ReusableFVAWorker", RecordingWorker)
    metrics = {}
    run_highs_vffva(_model(), workers=1, instrumentation=metrics)
    assert compile_calls == worker_builds == 1
    assert metrics == {"solver_instances": 1, "matrix_builds": 1, "retention_rows": 1,
                       "endpoint_solves": 10, "objective_changes": 10}


def test_endpoint_solver_failure_identifies_reaction_direction_and_status(monkeypatch):
    original_run = highs._ReusableFVAWorker.solve

    def fail(self, task):
        if task == (1, "max"):
            self.solver.setOptionValue("time_limit", 0.0)
        return original_run(self, task)

    monkeypatch.setattr(highs._ReusableFVAWorker, "solve", fail)
    with pytest.raises(AnalysisError, match="FVA maximum for reaction 'reversible'.*solver status"):
        run_highs_vffva(_model(), workers=1)
