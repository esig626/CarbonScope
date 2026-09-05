"""Correctness and structural controls for reusable native HiGHS FVA."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import threading

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


def _dynamic_queue_model(reaction_count=8):
    """Independent bounded reactions make endpoint order easy to perturb."""

    reactions = tuple(
        FluxReaction(f"R{index:02d}", (), -float(index + 1), float(index + 2))
        for index in range(reaction_count)
    )
    return FluxModel(
        (),
        reactions,
        LinearObjective("maximise", (ObjectiveTerm("R00", 1.0),)),
    )


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


def test_analytical_fixture_has_expected_retained_region():
    result = run_highs_vffva(_model(), 0.8, workers=2, chunk_size=1)

    assert tuple(result.ranges.index) == (
        "source", "reversible", "export", "fixed", "blocked"
    )
    pd.testing.assert_frame_equal(
        result.ranges,
        pd.DataFrame(
            {
                "minimum": [8.0, 8.0, 8.0, 2.0, 0.0],
                "maximum": [10.0, 10.0, 10.0, 2.0, 0.0],
            },
            index=("source", "reversible", "export", "fixed", "blocked"),
        ),
        atol=1e-10,
        rtol=1e-10,
    )


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
    fba_calls = 0
    worker_builds = 0
    original_compile = highs.compile_flux_lp
    original_fba = highs._run_compiled_fba
    original_worker = highs._ReusableFVAWorker

    def compile_once(model):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(model)

    def fba_once(lp):
        nonlocal fba_calls
        fba_calls += 1
        return original_fba(lp)

    class RecordingWorker(original_worker):
        def __init__(self, *args, **kwargs):
            nonlocal worker_builds
            worker_builds += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(highs, "compile_flux_lp", compile_once)
    monkeypatch.setattr(highs, "_run_compiled_fba", fba_once)
    monkeypatch.setattr(highs, "_ReusableFVAWorker", RecordingWorker)
    metrics = {}
    run_highs_vffva(_model(), workers=1, instrumentation=metrics)
    assert compile_calls == fba_calls == worker_builds == 1
    assert metrics["solver_instances"] == 1
    assert metrics["matrix_builds"] == 1
    assert metrics["retention_rows"] == 1
    assert metrics["endpoint_solves"] == 10
    assert metrics["objective_changes"] == 10
    assert metrics["objective_clears"] == 10
    assert metrics["basis_refreshes"] == 0


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


def test_default_chunk_and_persistent_thread_lifecycle_are_real():
    metrics = {}
    result = run_highs_vffva(_model(), 0.9, workers=2, instrumentation=metrics)

    assert metrics["configured_workers"] == 2
    assert metrics["chunk_size"] == 50
    assert metrics["worker_threads"] == 2
    assert metrics["solver_instances"] == 2
    assert metrics["matrix_builds"] == 2
    assert metrics["retention_rows"] == 2
    assert metrics["highs_single_thread_workers"] == 2
    assert metrics["max_pass_workers"] == 2
    assert metrics["min_pass_workers"] == 2
    assert metrics["workers_surviving_pass_transition"] == 2
    assert metrics["worker_passes"] == (("max", "min"), ("max", "min"))

    thread_ids = metrics["worker_thread_ids"]
    solver_ids = metrics["worker_solver_ids"]
    assert len(set(thread_ids)) == len(set(solver_ids)) == 2
    assert threading.get_ident() not in thread_ids
    assert -1 not in thread_ids
    assert -1 not in solver_ids

    assert metrics["max_queue_accounted"] == 1
    assert metrics["min_released_after_max"] == 1
    assert sum(metrics["worker_max_chunk_counts"]) == 1
    assert sum(metrics["worker_min_chunk_counts"]) == 1
    assert metrics["endpoint_attempts"] == metrics["endpoint_solves"] == 10
    assert metrics["max_endpoint_solves"] == 5
    assert metrics["min_endpoint_solves"] == 5
    assert metrics["objective_change_attempts"] == 10
    assert metrics["objective_changes"] == 10
    assert metrics["objective_clear_attempts"] == 10
    assert metrics["objective_clears"] == 10
    assert metrics["joined_workers"] == metrics["solver_releases"] == 2
    assert metrics["endpoint_solves"] > sum(
        len(pass_directions) for pass_directions in metrics["worker_passes"]
    )
    assert tuple(result.ranges.index) == (
        "source", "reversible", "export", "fixed", "blocked"
    )


def test_dynamic_chunks_are_reacquired_and_completion_order_is_not_result_order(
    monkeypatch,
):
    model = _dynamic_queue_model()
    original_solve = highs._ReusableFVAWorker.solve
    later_endpoint_finished = threading.Event()
    completion_lock = threading.Lock()
    completion_order = []

    def force_out_of_order_completion(self, task):
        if task == (0, "max") and not later_endpoint_finished.wait(timeout=10):
            raise AssertionError("later maximum endpoint never ran concurrently")
        result = original_solve(self, task)
        with completion_lock:
            completion_order.append(task)
        if task == (1, "max"):
            later_endpoint_finished.set()
        return result

    monkeypatch.setattr(
        highs._ReusableFVAWorker, "solve", force_out_of_order_completion
    )
    metrics = {}
    actual = run_highs_vffva(
        model, 0.75, workers=2, chunk_size=1, instrumentation=metrics
    )

    assert completion_order.index((1, "max")) < completion_order.index((0, "max"))
    assert metrics["chunk_size"] == 1
    assert metrics["chunks_claimed"] == 2 * len(model.reactions)
    assert metrics["workers_claiming_multiple_chunks"] >= 1
    assert sum(metrics["worker_max_chunk_counts"]) == len(model.reactions)
    assert sum(metrics["worker_min_chunk_counts"]) == len(model.reactions)
    assert metrics["max_queue_accounted"] == 1
    assert metrics["min_released_after_max"] == 1
    assert tuple(actual.ranges.index) == tuple(
        reaction.reaction_id for reaction in model.reactions
    )
    pd.testing.assert_frame_equal(
        actual.ranges,
        run_highs_fva_reference(model, 0.75).ranges,
        atol=1e-8,
        rtol=1e-8,
    )


def test_threaded_bounds_are_deterministic_across_repeated_runs():
    model = _dynamic_queue_model()
    expected = run_highs_vffva(model, 0.8, workers=2, chunk_size=1)

    for _ in range(2):
        actual = run_highs_vffva(model, 0.8, workers=2, chunk_size=1)
        pd.testing.assert_frame_equal(actual.ranges, expected.ranges)
        assert actual.ranges_sha256 == expected.ranges_sha256


@pytest.mark.parametrize(
    "worker_arguments",
    [pytest.param({}, id="omitted"), pytest.param({"workers": None}, id="none")],
)
def test_default_worker_mode_uses_one_owning_thread(worker_arguments):
    metrics = {}
    actual = run_highs_vffva(
        _model(), 0.9, instrumentation=metrics, **worker_arguments
    )

    assert metrics["configured_workers"] == 1
    assert metrics["worker_threads"] == 1
    assert metrics["solver_instances"] == 1
    assert metrics["matrix_builds"] == 1
    assert metrics["worker_passes"] == (("max", "min"),)
    assert metrics["worker_thread_ids"] != (threading.get_ident(),)
    assert metrics["joined_workers"] == metrics["solver_releases"] == 1
    assert tuple(actual.ranges.index) == (
        "source", "reversible", "export", "fixed", "blocked"
    )
    pd.testing.assert_frame_equal(
        actual.ranges,
        run_highs_fva_reference(_model(), 0.9).ranges,
        atol=1e-8,
        rtol=1e-8,
    )


def test_threaded_failure_clears_objective_joins_workers_and_returns_no_partial_result(
    monkeypatch,
):
    original_run = highs._ReusableFVAWorker.solve
    original_canonicalize = highs._canonicalize_fva_endpoints
    canonicalized = False
    preexisting_worker_threads = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("fluxemu-vffva-")
    }

    def fail(self, task):
        if task == (1, "max"):
            self.solver.setOptionValue("time_limit", 0.0)
        return original_run(self, task)

    def record_canonicalize(*args, **kwargs):
        nonlocal canonicalized
        canonicalized = True
        return original_canonicalize(*args, **kwargs)

    monkeypatch.setattr(highs._ReusableFVAWorker, "solve", fail)
    monkeypatch.setattr(highs, "_canonicalize_fva_endpoints", record_canonicalize)
    metrics = {}
    with pytest.raises(AnalysisError, match="FVA maximum for reaction 'reversible'.*solver status"):
        run_highs_vffva(
            _model(), workers=2, chunk_size=1, instrumentation=metrics
        )

    assert not canonicalized
    assert metrics["endpoint_attempts"] > metrics["endpoint_solves"]
    assert metrics["min_endpoint_solves"] == 0
    assert metrics["objective_change_attempts"] == metrics["objective_changes"]
    assert metrics["objective_change_attempts"] == metrics["objective_clear_attempts"]
    assert metrics["objective_changes"] == metrics["objective_clears"]
    assert metrics["joined_workers"] == metrics["configured_workers"] == 2
    assert metrics["solver_releases"] == 2
    remaining_worker_threads = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("fluxemu-vffva-")
    }
    assert remaining_worker_threads == preexisting_worker_threads


def test_partial_worker_initialization_failure_cancels_before_any_endpoint(
    monkeypatch,
):
    original_worker = highs._ReusableFVAWorker
    original_canonicalize = highs._canonicalize_fva_endpoints
    construction_lock = threading.Lock()
    construction_calls = 0
    canonicalized = False
    preexisting_worker_threads = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("fluxemu-vffva-")
    }

    class OneInitializationFailure(original_worker):
        def __init__(self, *args, **kwargs):
            nonlocal construction_calls
            with construction_lock:
                construction_calls += 1
                fail_this_worker = construction_calls == 1
            if fail_this_worker:
                raise AnalysisError("synthetic worker constructor failure")
            super().__init__(*args, **kwargs)

    def record_canonicalize(*args, **kwargs):
        nonlocal canonicalized
        canonicalized = True
        return original_canonicalize(*args, **kwargs)

    monkeypatch.setattr(highs, "_ReusableFVAWorker", OneInitializationFailure)
    monkeypatch.setattr(highs, "_canonicalize_fva_endpoints", record_canonicalize)
    metrics = {}
    with pytest.raises(
        AnalysisError,
        match=r"FVA worker \d+ initialization failed: synthetic worker constructor failure",
    ):
        run_highs_vffva(
            _model(), workers=2, chunk_size=1, instrumentation=metrics
        )

    assert construction_calls == 2
    assert not canonicalized
    assert metrics["solver_instances"] == 1
    assert metrics["matrix_builds"] == metrics["retention_rows"] == 1
    assert metrics["endpoint_attempts"] == metrics["endpoint_solves"] == 0
    assert metrics["max_endpoint_solves"] == metrics["min_endpoint_solves"] == 0
    assert metrics["objective_change_attempts"] == 0
    assert metrics["objective_clear_attempts"] == 0
    assert metrics["worker_passes"] == ((), ())
    assert metrics["joined_workers"] == metrics["configured_workers"] == 2
    assert metrics["solver_releases"] == metrics["solver_instances"] == 1
    remaining_worker_threads = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("fluxemu-vffva-")
    }
    assert remaining_worker_threads == preexisting_worker_threads


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


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_invalid_dynamic_chunk_size_is_rejected(chunk_size):
    with pytest.raises(AnalysisError, match="chunk_size must be a positive integer"):
        run_highs_vffva(_model(), chunk_size=chunk_size)


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
