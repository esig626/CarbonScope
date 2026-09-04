"""Correctness and structural controls for reusable native HiGHS FVA."""

from __future__ import annotations

from dataclasses import replace
import importlib.util

import pandas as pd
import pytest

import fluxemu.flux_analysis.highs as highs
from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis import (
    FBAResult,
    RetainedObjectiveConstraint,
    compile_flux_lp,
    prepare_highs_flux_region,
    run_highs_fva_reference,
    run_highs_vffva,
    run_prepared_highs_vffva,
)
from fluxemu.model import (FluxMetabolite, FluxModel, FluxReaction, LinearObjective,
                           ObjectiveTerm, StoichiometricTerm)
from fluxemu.real_model import load_ecoli_core_flux_model

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


def _constant_objective_model(direction, coefficient):
    return FluxModel(
        (),
        (
            FluxReaction("Q", (), 1.0, 1.0),
            FluxReaction("free", (), -1.0, 1.0),
        ),
        LinearObjective(direction, (ObjectiveTerm("Q", coefficient),)),
    )


def _overflow_declared_objective_model():
    return FluxModel(
        (),
        (
            FluxReaction("X", (), 2.0, 2.0),
            FluxReaction("Y", (), 2.0, 2.0),
        ),
        LinearObjective(
            "maximise",
            (ObjectiveTerm("X", 1e308), ObjectiveTerm("Y", -1e308)),
        ),
    )


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(run_highs_fva_reference, id="cold"),
        pytest.param(
            lambda model, fraction: run_highs_vffva(
                model, fraction, workers=1
            ),
            id="fast",
        ),
    ],
)
@pytest.mark.parametrize(
    ("direction", "coefficient"),
    [("maximise", -1e-12), ("minimise", 1e-12)],
)
def test_sign_incompatible_small_constant_objective_fails_before_fva(
    runner, direction, coefficient
):
    with pytest.raises(
        AnalysisError, match="fraction-of-optimum arithmetic.*infeasible"
    ):
        runner(_constant_objective_model(direction, coefficient), 0.8)


@pytest.mark.parametrize(
    ("direction", "coefficient"),
    [("maximise", 1e-12), ("minimise", -1e-12)],
)
def test_sign_compatible_small_constant_objective_keeps_full_fva_region(
    direction, coefficient
):
    model = _constant_objective_model(direction, coefficient)
    cold = run_highs_fva_reference(model, 0.8)
    fast = run_highs_vffva(model, 0.8, workers=1)

    pd.testing.assert_frame_equal(cold.ranges, fast.ranges, atol=1e-12, rtol=1e-12)
    assert tuple(fast.ranges.loc["Q"]) == pytest.approx((1.0, 1.0))
    assert tuple(fast.ranges.loc["free"]) == pytest.approx((-1.0, 1.0))


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


def test_zero_activity_objective_uses_absolute_roundoff_tolerance():
    model = FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction("R0", (StoichiometricTerm("M", 4.0),), 0.0, 0.0),
            FluxReaction("R1", (StoichiometricTerm("M", 4.0),), -1.0, 1.0),
            FluxReaction("R2", (StoichiometricTerm("M", -3.0),), -1.0, 1.0),
        ),
        LinearObjective(
            "maximise",
            (
                ObjectiveTerm("R0", 4.0),
                ObjectiveTerm("R1", -4.0),
                ObjectiveTerm("R2", 3.0),
            ),
        ),
    )

    expected = run_highs_fva_reference(model, 0.3)
    actual = run_highs_vffva(model, 0.3, workers=1)
    assert expected.objective_value == pytest.approx(0.0, abs=1e-15)
    pd.testing.assert_frame_equal(actual.ranges, expected.ranges, atol=1e-8, rtol=1e-8)
    assert tuple(actual.ranges.loc["R1"]) == pytest.approx((-0.75, 0.75))
    assert tuple(actual.ranges.loc["R2"]) == pytest.approx((-1.0, 1.0))


def test_nonzero_minimisation_fraction_and_negative_signed_endpoint():
    signed = FluxModel(
        (),
        (
            FluxReaction("signed", (), -10, 2),
            FluxReaction("fixed", (), 3, 3),
            FluxReaction("blocked", (), 0, 0),
        ),
        LinearObjective("minimise", (ObjectiveTerm("signed", 1.0),)),
    )
    expected = run_highs_fva_reference(signed, 0.9)
    actual = run_highs_vffva(signed, 0.9, workers=1)
    pd.testing.assert_frame_equal(actual.ranges, expected.ranges, atol=1e-8, rtol=1e-8)
    assert actual.objective_value == pytest.approx(-10.0)
    assert actual.ranges.loc["signed"].tolist() == pytest.approx([-10.0, -9.0])
    assert actual.ranges.loc["fixed"].tolist() == pytest.approx([3.0, 3.0])
    assert actual.ranges.loc["blocked"].tolist() == pytest.approx([0.0, 0.0])


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
    assert metrics == {
        "solver_instances": 1,
        "matrix_builds": 1,
        "retention_rows": 1,
        "endpoint_solves": 10,
        "objective_changes": 10,
        "basis_refreshes": 0,
    }


def test_endpoint_validation_ambiguity_refreshes_same_solver_once(monkeypatch):
    prepared = prepare_highs_flux_region(_model(), 1.0)
    original_validate = highs._validate
    injected = False

    def fail_first_source_endpoint(lp, fluxes, reported, costs):
        nonlocal injected
        if not injected and tuple(costs) == (1.0, 0.0, 0.0, 0.0, 0.0):
            injected = True
            raise AnalysisError("synthetic endpoint primal ambiguity")
        return original_validate(lp, fluxes, reported, costs)

    monkeypatch.setattr(highs, "_validate", fail_first_source_endpoint)
    metrics = {}
    actual = run_prepared_highs_vffva(
        prepared,
        workers=1,
        instrumentation=metrics,
    )

    assert injected
    assert metrics["solver_instances"] == 1
    assert metrics["matrix_builds"] == 1
    assert metrics["basis_refreshes"] == 1
    pd.testing.assert_frame_equal(
        actual.ranges,
        run_highs_fva_reference(_model()).ranges,
        atol=1e-8,
        rtol=1e-8,
    )


def test_prepared_fva_does_not_recompile_or_resolve_biological_objective(monkeypatch):
    prepared = prepare_highs_flux_region(_model(), 0.9)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared FVA repeated model preparation")

    monkeypatch.setattr(highs, "compile_flux_lp", forbidden)
    monkeypatch.setattr(highs, "_run_compiled_fba", forbidden)
    actual = run_prepared_highs_vffva(prepared, workers=1)
    assert actual.objective_value == prepared.fba.objective_value
    assert actual.fraction_of_optimum == 0.9


def test_prepared_fva_rejects_inconsistent_retained_metadata():
    prepared = prepare_highs_flux_region(_model(), 0.9)
    forged = replace(
        prepared,
        retention=replace(prepared.retention, bound=prepared.retention.bound - 1.0),
    )
    with pytest.raises(AnalysisError, match="retained-objective metadata"):
        run_prepared_highs_vffva(forged, workers=1)


def test_prepared_fva_rejects_mismatched_fba_records():
    prepared = prepare_highs_flux_region(_model(), 0.9)
    reordered = prepared.fba.fluxes.iloc[::-1]
    bad_primal = prepared.fba.fluxes.copy()
    bad_primal.iloc[0] += 1.0
    forged_records = (
        replace(prepared, fba=replace(prepared.fba, fluxes=reordered)),
        replace(prepared, fba=replace(prepared.fba, status="infeasible")),
        replace(prepared, fba=replace(prepared.fba, objective_direction="min")),
        replace(prepared, fba=replace(prepared.fba, fluxes=bad_primal)),
    )
    for forged in forged_records:
        with pytest.raises(AnalysisError, match="prepared FBA|primal validation"):
            run_prepared_highs_vffva(forged, workers=1)


def test_prepared_fva_rejects_non_series_fba_primal_contextually():
    prepared = prepare_highs_flux_region(_model(), 0.9)
    forged = replace(
        prepared,
        fba=replace(prepared.fba, fluxes=tuple(prepared.fba.fluxes)),
    )

    with pytest.raises(AnalysisError, match="FBA fluxes.*pandas Series"):
        run_prepared_highs_vffva(forged, workers=1)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_prepared_fva_rejects_nonfinite_fba_primal_contextually(nonfinite):
    prepared = prepare_highs_flux_region(_model(), 0.9)
    fluxes = prepared.fba.fluxes.copy()
    fluxes.iloc[0] = nonfinite
    forged = replace(prepared, fba=replace(prepared.fba, fluxes=fluxes))

    with pytest.raises(AnalysisError, match="FBA primal.*non-finite|non-finite.*FBA primal"):
        run_prepared_highs_vffva(forged, workers=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fraction_of_optimum", 0.0),
        ("fraction_of_optimum", "bad"),
        ("sense", "<="),
        ("objective_direction", "min"),
        ("biological_optimum", 9.0),
        ("bound", 8.0),
    ],
)
def test_prepared_fva_rejects_forged_retention_fields(field, value):
    prepared = prepare_highs_flux_region(_model(), 0.9)
    forged = replace(
        prepared, retention=replace(prepared.retention, **{field: value})
    )
    with pytest.raises(AnalysisError, match="fraction_of_optimum|retained-objective"):
        run_prepared_highs_vffva(forged, workers=1)


def test_parallel_path_uses_unordered_unit_chunk_dynamic_queue(monkeypatch):
    observed = {}

    class RecordingPool:
        def __init__(self, workers, initializer, initargs):
            observed["workers"] = workers
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, function, tasks, chunksize):
            observed["chunksize"] = chunksize
            observed["tasks"] = tuple(tasks)
            return iter(reversed([function(task) for task in tasks]))

    class RecordingContext:
        Pool = RecordingPool

    monkeypatch.setattr(highs.multiprocessing, "get_context", lambda method: RecordingContext())
    result = run_highs_vffva(_model(), 0.9, workers=2)
    assert observed["workers"] == 2
    assert observed["chunksize"] == 1
    assert observed["tasks"] == tuple(
        (index, direction)
        for index in range(5)
        for direction in ("min", "max")
    )
    assert tuple(result.ranges.index) == (
        "source", "reversible", "export", "fixed", "blocked"
    )


@pytest.mark.parametrize(
    "worker_arguments",
    [pytest.param({}, id="omitted"), pytest.param({"workers": None}, id="none")],
)
def test_default_worker_mode_is_portable_serial(monkeypatch, worker_arguments):
    def forbidden_process_context(*args, **kwargs):
        raise AssertionError("default FastFVA unexpectedly attempted to spawn")

    monkeypatch.setattr(
        highs.multiprocessing, "get_context", forbidden_process_context
    )
    metrics = {}
    actual = run_highs_vffva(
        _model(), 0.9, instrumentation=metrics, **worker_arguments
    )

    assert metrics["solver_instances"] == 1
    assert metrics["matrix_builds"] == 1
    assert tuple(actual.ranges.index) == (
        "source", "reversible", "export", "fixed", "blocked"
    )
    pd.testing.assert_frame_equal(
        actual.ranges,
        run_highs_fva_reference(_model(), 0.9).ranges,
        atol=1e-8,
        rtol=1e-8,
    )


def test_parallel_queue_propagates_contextual_endpoint_failure(monkeypatch):
    original_run = highs._ReusableFVAWorker.solve

    def fail(self, task):
        if task == (1, "max"):
            self.solver.setOptionValue("time_limit", 0.0)
        return original_run(self, task)

    class ImmediatePool:
        def __init__(self, workers, initializer, initargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, function, tasks, chunksize):
            return map(function, tasks)

    class ImmediateContext:
        Pool = ImmediatePool

    monkeypatch.setattr(highs._ReusableFVAWorker, "solve", fail)
    monkeypatch.setattr(highs.multiprocessing, "get_context", lambda method: ImmediateContext())
    with pytest.raises(AnalysisError, match="FVA maximum for reaction 'reversible'.*solver status"):
        run_highs_vffva(_model(), workers=2)


def test_endpoint_solver_failure_identifies_reaction_direction_and_status(monkeypatch):
    original_run = highs._ReusableFVAWorker.solve

    def fail(self, task):
        if task == (1, "max"):
            self.solver.setOptionValue("time_limit", 0.0)
        return original_run(self, task)

    monkeypatch.setattr(highs._ReusableFVAWorker, "solve", fail)
    with pytest.raises(AnalysisError, match="FVA maximum for reaction 'reversible'.*solver status"):
        run_highs_vffva(_model(), workers=1)


def test_cold_fva_rejects_nonfinite_declared_retained_objective(monkeypatch):
    def synthetic_fba(lp):
        return FBAResult(
            0.0,
            "optimal",
            "max",
            pd.Series((2.0, 2.0), index=lp.reaction_ids, dtype=float),
        )

    monkeypatch.setattr(highs, "_run_compiled_fba", synthetic_fba)
    monkeypatch.setattr(
        highs,
        "_solve",
        lambda lp, costs, direction, operation, retention=None: (
            [2.0, 2.0],
            2.0,
            "Optimal",
        ),
    )
    monkeypatch.setattr(highs, "_validate", lambda *args, **kwargs: None)

    with pytest.raises(
        AnalysisError,
        match="FVA endpoint.*declared biological objective.*not finite",
    ):
        run_highs_fva_reference(_overflow_declared_objective_model())


def test_reusable_fva_endpoint_rejects_nonfinite_declared_retained_objective():
    lp = compile_flux_lp(_overflow_declared_objective_model())

    class Solution:
        col_value = (2.0, 2.0)

    class Solver:
        def changeColCost(self, *args):
            return None

        def setMinimize(self):
            return None

        def setMaximize(self):
            return None

        def getOptionValue(self, name):
            return None, float("inf")

        def run(self):
            return None

        def clearSolver(self):
            return None

        def getModelStatus(self):
            return highs._highspy().HighsModelStatus.kOptimal

        def modelStatusToString(self, status):
            return "Optimal"

        def getSolution(self):
            return Solution()

        def getObjectiveValue(self):
            return 2.0

    worker = object.__new__(highs._ReusableFVAWorker)
    worker.lp = lp
    worker.retention = RetainedObjectiveConstraint(
        1.0,
        0.0,
        0.0,
        ">=",
        "max",
        0.0,
        0.0,
    )
    worker.reference_fluxes = (2.0, 2.0)
    worker.endpoint_count = 0
    worker.solver = Solver()

    with pytest.raises(
        AnalysisError,
        match="FVA minimum for reaction 'X'.*declared biological objective.*not finite",
    ):
        worker.solve((0, "min"))


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_worker_count_is_rejected(workers):
    with pytest.raises(AnalysisError, match="workers must be a positive integer"):
        run_highs_vffva(_model(), workers=workers)


@pytest.mark.parametrize("objective", ["biomass", "acetate"])
def test_bundled_ecoli_fast_serial_matches_cold_reference(objective):
    model = load_ecoli_core_flux_model(objective)
    reference = run_highs_fva_reference(model, 0.9)
    fast = run_highs_vffva(model, 0.9, workers=1)
    pd.testing.assert_frame_equal(fast.ranges, reference.ranges, atol=1e-7, rtol=1e-7)


def test_bundled_ecoli_parallel_matches_cold_reference():
    model = load_ecoli_core_flux_model("biomass")
    reference = run_highs_fva_reference(model, 1.0)
    fast = run_highs_vffva(model, 1.0, workers=2)
    pd.testing.assert_frame_equal(fast.ranges, reference.ranges, atol=1e-7, rtol=1e-7)
