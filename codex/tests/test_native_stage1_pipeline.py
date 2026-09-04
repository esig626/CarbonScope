"""Focused integration gates for the complete native Stage 1 pipeline."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import importlib.util
import subprocess
import sys

import pandas as pd
import pytest

import fluxemu
import fluxemu.analysis
import fluxemu.analysis.stationary as pipeline_module
import fluxemu.flux_analysis.highs as highs_module
from fluxemu import NativeStationaryEnsembleResult, run_native_stationary_ensemble
from fluxemu.exceptions import AnalysisError
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


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("highspy") is None,
    reason="highspy is unavailable",
)


def _mapped_reaction(
    reaction_id: str, source: str, product: str
) -> IsotopeReaction:
    transition = AtomTransition(
        AtomPosition(source, 1), AtomPosition(product, 1)
    )
    return IsotopeReaction(
        reaction_id,
        "forward",
        True,
        (IsotopeParticipant(source, (1,)),),
        (IsotopeParticipant(product, (1,)),),
        (MappingBranch("declared", 1.0, (transition,)),),
    )


def _alternate_path_science(
) -> tuple[CanonicalModel, StationaryExperimentSemantics]:
    """One optimal face whose two labeled inputs produce different MIDs."""

    metabolites = (
        FluxMetabolite("S0", False),
        FluxMetabolite("S1", False),
        FluxMetabolite("I", True),
        FluxMetabolite("O", False),
    )
    reactions = (
        FluxReaction(
            "Z_IN",
            (StoichiometricTerm("S0", -1.0), StoichiometricTerm("I", 1.0)),
            0.0,
            10.0,
        ),
        FluxReaction(
            "A_IN",
            (StoichiometricTerm("S1", -1.0), StoichiometricTerm("I", 1.0)),
            0.0,
            10.0,
        ),
        FluxReaction(
            "M_OUT",
            (StoichiometricTerm("I", -1.0), StoichiometricTerm("O", 1.0)),
            0.0,
            10.0,
        ),
    )
    model = CanonicalModel(
        FluxModel(
            metabolites,
            reactions,
            LinearObjective("maximise", (ObjectiveTerm("M_OUT", 1.0),)),
        ),
        IsotopeModel(
            tuple(
                IsotopeMetabolite(item.metabolite_id, 1, True, False)
                for item in metabolites
            ),
            (
                _mapped_reaction("Z_IN", "S0", "I"),
                _mapped_reaction("A_IN", "S1", "I"),
                _mapped_reaction("M_OUT", "I", "O"),
            ),
        ),
    )
    experiment = StationaryExperimentSemantics(
        (
            Tracer("S0", (("#0", 1.0),), "no"),
            Tracer("S1", (("#1", 1.0),), "no"),
        ),
        (Target("O-mid", "O", (1,), "intermediate", "C1", "no"),),
    )
    return model, experiment


def test_composed_pipeline_reuses_flux_region_and_exact_sample_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, experiment = _alternate_path_science()
    calls = {
        "compile_lp": 0,
        "biological_fba": 0,
        "fast_fva": 0,
        "sampling": 0,
        "compile_emu": 0,
        "evaluate_emu": 0,
    }
    observed: dict[str, object] = {}

    original_compile_lp = highs_module.compile_flux_lp
    original_compiled_fba = highs_module._run_compiled_fba
    original_fast_fva = pipeline_module.run_prepared_highs_vffva
    original_sampling = pipeline_module.sample_prepared_highs_flux_states
    original_compile_emu = pipeline_module.compile_emu_plan
    original_evaluate_emu = pipeline_module.evaluate_stationary

    def record_compile_lp(flux_model):
        calls["compile_lp"] += 1
        return original_compile_lp(flux_model)

    def record_compiled_fba(lp):
        calls["biological_fba"] += 1
        return original_compiled_fba(lp)

    def record_fast_fva(prepared, **kwargs):
        calls["fast_fva"] += 1
        observed["fva_prepared"] = prepared
        observed["fva_workers"] = kwargs.get("workers")
        result = original_fast_fva(prepared, **kwargs)
        observed["fva_result"] = result
        return result

    def record_sampling(prepared, fva, count, **kwargs):
        calls["sampling"] += 1
        observed["sampling_prepared"] = prepared
        observed["sampling_fva"] = fva
        observed["sampling_count"] = count
        observed["sampling_kwargs"] = kwargs
        result = original_sampling(prepared, fva, count, **kwargs)
        observed["sampling_result"] = result
        return result

    def record_compile_emu(canonical_model, stationary_experiment):
        calls["compile_emu"] += 1
        return original_compile_emu(canonical_model, stationary_experiment)

    def record_evaluate_emu(plan, states):
        calls["evaluate_emu"] += 1
        observed["evaluated_states"] = states
        return original_evaluate_emu(plan, states)

    monkeypatch.setattr(highs_module, "compile_flux_lp", record_compile_lp)
    monkeypatch.setattr(highs_module, "_run_compiled_fba", record_compiled_fba)
    monkeypatch.setattr(
        pipeline_module, "run_prepared_highs_vffva", record_fast_fva
    )
    monkeypatch.setattr(
        pipeline_module,
        "sample_prepared_highs_flux_states",
        record_sampling,
    )
    monkeypatch.setattr(pipeline_module, "compile_emu_plan", record_compile_emu)
    monkeypatch.setattr(
        pipeline_module, "evaluate_stationary", record_evaluate_emu
    )

    result = run_native_stationary_ensemble(
        model,
        experiment,
        sample_count=4,
        seed=73,
        fraction_of_optimum=1.0,
        burn_in=5,
        thinning=2,
        max_direction_attempts=17,
    )

    assert calls == {
        "compile_lp": 1,
        "biological_fba": 1,
        "fast_fva": 1,
        "sampling": 1,
        "compile_emu": 1,
        "evaluate_emu": 1,
    }
    assert observed["fva_workers"] == 1
    assert observed["fva_prepared"] is observed["sampling_prepared"]
    assert observed["sampling_fva"] is observed["fva_result"] is result.fva
    assert observed["sampling_result"] is result.flux_sampling
    assert result.fba is observed["fva_prepared"].fba
    assert observed["evaluated_states"] is result.flux_sampling.states
    assert observed["sampling_count"] == 4
    assert observed["sampling_kwargs"] == {
        "seed": 73,
        "burn_in": 5,
        "thinning": 2,
        "max_direction_attempts": 17,
    }
    assert result.flux_sampling.provenance.fraction_of_optimum == 1.0
    endpoint_minima = tuple(float(value) for value in result.fva.ranges["minimum"])
    assert endpoint_minima == pytest.approx((0.0, 0.0, 10.0))
    assert all(
        tuple(value for _, value in state.values) != endpoint_minima
        for state in result.flux_sampling.states
    )


def test_fraction_one_ensemble_replays_ordered_distinct_mids() -> None:
    model, experiment = _alternate_path_science()
    arguments = dict(
        sample_count=7,
        seed=626,
        fraction_of_optimum=1.0,
        burn_in=8,
        thinning=2,
        fva_workers=1,
    )

    first = run_native_stationary_ensemble(model, experiment, **arguments)
    second = run_native_stationary_ensemble(model, experiment, **arguments)

    assert tuple(item.name for item in fields(first)) == (
        "fba",
        "fva",
        "flux_sampling",
        "mid_ensemble",
    )
    with pytest.raises(FrozenInstanceError):
        first.fba = second.fba  # type: ignore[misc]
    assert first.flux_sampling.states == second.flux_sampling.states
    assert first.flux_sampling.provenance == second.flux_sampling.provenance
    assert (
        first.mid_ensemble.forward.predictions
        == second.mid_ensemble.forward.predictions
    )
    pd.testing.assert_frame_equal(first.fva.ranges, second.fva.ranges)

    sampling = first.flux_sampling
    assert sampling.validation.valid
    assert sampling.validation.optimal_face_valid
    assert sampling.provenance.objective_face
    assert sampling.sample_count == 7
    reaction_order = ("Z_IN", "A_IN", "M_OUT")
    sample_ids = tuple(state.sample_id for state in sampling.states)
    assert sample_ids == tuple(
        f"native-sample-{index:06d}" for index in range(7)
    )
    assert all(
        tuple(reaction_id for reaction_id, _ in state.values) == reaction_order
        for state in sampling.states
    )
    assert all(
        dict(state.values)["M_OUT"] == pytest.approx(first.fba.objective_value)
        for state in sampling.states
    )
    assert len({state.values for state in sampling.states}) > 1

    predictions = first.mid_ensemble.forward.predictions
    assert len(predictions) == 7
    assert tuple(item.sample_id for item in predictions) == sample_ids
    assert len({item.fractions for item in predictions}) > 1
    for state, prediction in zip(sampling.states, predictions, strict=True):
        flux = dict(state.values)
        assert prediction.target_id == "O-mid"
        assert prediction.fractions == pytest.approx(
            (flux["Z_IN"] / 10.0, flux["A_IN"] / 10.0), abs=1e-9
        )
        assert min(prediction.fractions) >= -1e-10
        assert sum(prediction.fractions) == pytest.approx(1.0, abs=1e-10)
    assert tuple(item.sample_id for item in first.mid_ensemble.forward.values) == tuple(
        sample_id for sample_id in sample_ids for _ in range(2)
    )
    assert tuple(
        item.sample_id for item in first.mid_ensemble.layer_diagnostics
    ) == sample_ids


def test_existing_deterministic_path_also_prepares_flux_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, experiment = _alternate_path_science()
    calls = {"compile_lp": 0, "biological_fba": 0, "fast_fva": 0}
    original_compile_lp = highs_module.compile_flux_lp
    original_compiled_fba = highs_module._run_compiled_fba
    original_fast_fva = pipeline_module.run_prepared_highs_vffva

    def record_compile_lp(flux_model):
        calls["compile_lp"] += 1
        return original_compile_lp(flux_model)

    def record_compiled_fba(lp):
        calls["biological_fba"] += 1
        return original_compiled_fba(lp)

    def record_fast_fva(prepared, **kwargs):
        calls["fast_fva"] += 1
        return original_fast_fva(prepared, **kwargs)

    monkeypatch.setattr(highs_module, "compile_flux_lp", record_compile_lp)
    monkeypatch.setattr(highs_module, "_run_compiled_fba", record_compiled_fba)
    monkeypatch.setattr(
        pipeline_module, "run_prepared_highs_vffva", record_fast_fva
    )

    result = pipeline_module.run_native_stationary_analysis(model, experiment)

    assert calls == {"compile_lp": 1, "biological_fba": 1, "fast_fva": 1}
    assert result.flux_state.sample_id == "fba-optimum"
    assert len(result.mids.forward.predictions) == 1


def test_sampling_failure_stops_before_emu_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, experiment = _alternate_path_science()
    calls = {"compile_emu": 0, "evaluate_emu": 0}

    def fail_sampling(*args, **kwargs):
        raise AnalysisError("deliberate native sampling failure")

    def unexpected_compile(*args, **kwargs):
        calls["compile_emu"] += 1
        raise AssertionError("EMU compilation must follow successful sampling")

    def unexpected_evaluate(*args, **kwargs):
        calls["evaluate_emu"] += 1
        raise AssertionError("EMU evaluation must follow successful sampling")

    monkeypatch.setattr(
        pipeline_module, "sample_prepared_highs_flux_states", fail_sampling
    )
    monkeypatch.setattr(pipeline_module, "compile_emu_plan", unexpected_compile)
    monkeypatch.setattr(
        pipeline_module, "evaluate_stationary", unexpected_evaluate
    )

    with pytest.raises(AnalysisError, match="deliberate native sampling failure"):
        run_native_stationary_ensemble(
            model,
            experiment,
            sample_count=2,
            seed=1,
            burn_in=0,
            thinning=1,
        )

    assert calls == {"compile_emu": 0, "evaluate_emu": 0}


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("sample_count", 0, "sample_count must be a positive integer"),
        ("seed", -1, "seed must be a nonnegative integer"),
        ("burn_in", -1, "burn_in must be a nonnegative integer"),
        ("thinning", 0, "thinning must be a positive integer"),
        ("fva_workers", 0, "fva_workers must be a positive integer"),
        (
            "max_direction_attempts",
            0,
            "max_direction_attempts must be a positive integer",
        ),
    ],
)
def test_invalid_controls_fail_before_flux_compilation(
    keyword: str,
    value: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, experiment = _alternate_path_science()

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("invalid controls must fail before LP compilation")

    monkeypatch.setattr(highs_module, "compile_flux_lp", unexpected_compile)
    arguments = {"sample_count": 1, keyword: value}
    with pytest.raises(AnalysisError, match=message):
        run_native_stationary_ensemble(model, experiment, **arguments)


def test_ensemble_api_is_public_at_analysis_and_package_root() -> None:
    assert fluxemu.run_native_stationary_ensemble is run_native_stationary_ensemble
    assert (
        fluxemu.analysis.run_native_stationary_ensemble
        is run_native_stationary_ensemble
    )
    assert fluxemu.NativeStationaryEnsembleResult is NativeStationaryEnsembleResult
    assert (
        fluxemu.analysis.NativeStationaryEnsembleResult
        is NativeStationaryEnsembleResult
    )


def test_stage_b2_actual_sampled_batch_produces_twelve_mids_per_state() -> None:
    from fluxemu.real_model import (
        build_r1_acceptance_experiment,
        load_ecoli_core_stage_b2_model,
    )

    model = load_ecoli_core_stage_b2_model()
    experiment = build_r1_acceptance_experiment()
    result = run_native_stationary_ensemble(
        model,
        experiment,
        sample_count=2,
        seed=626,
        fraction_of_optimum=1.0,
        burn_in=8,
        thinning=2,
        fva_workers=1,
    )

    assert result.flux_sampling.validation.valid
    assert len(result.flux_sampling.states) == 2
    predictions = result.mid_ensemble.forward.predictions
    assert len(predictions) == 24
    sample_ids = tuple(state.sample_id for state in result.flux_sampling.states)
    assert tuple(item.sample_id for item in predictions) == tuple(
        sample_id for sample_id in sample_ids for _ in range(12)
    )
    assert all(min(item.fractions) >= -1e-10 for item in predictions)
    assert all(sum(item.fractions) == pytest.approx(1.0, abs=1e-10) for item in predictions)


def test_public_ensemble_executes_with_optional_solver_stacks_blocked() -> None:
    script = r'''
import sys
for name in ("cobra", "optlang", "mfapy"):
    sys.modules[name] = None

from fluxemu import NativeStationaryEnsembleResult, run_native_stationary_ensemble
from fluxemu.model import *

transition = AtomTransition(AtomPosition("S", 1), AtomPosition("T", 1))
model = CanonicalModel(
    FluxModel(
        (FluxMetabolite("S", False), FluxMetabolite("T", False)),
        (FluxReaction("R", (StoichiometricTerm("S", -1), StoichiometricTerm("T", 1)), 1, 1),),
        LinearObjective("maximise", (ObjectiveTerm("R", 1),)),
    ),
    IsotopeModel(
        (IsotopeMetabolite("S", 1, True, False), IsotopeMetabolite("T", 1, True, False)),
        (IsotopeReaction("R", "forward", True,
            (IsotopeParticipant("S", (1,)),),
            (IsotopeParticipant("T", (1,)),),
            (MappingBranch("declared", 1.0, (transition,)),)),),
    ),
)
experiment = StationaryExperimentSemantics(
    (Tracer("S", (("#1", 1.0),), "no"),),
    (Target("T-mid", "T", (1,), "intermediate", "C1", "no"),),
)
result = run_native_stationary_ensemble(
    model, experiment, sample_count=2, seed=4, burn_in=0, thinning=1
)
assert isinstance(result, NativeStationaryEnsembleResult)
assert result.flux_sampling.sample_count == 2
assert len(result.mid_ensemble.forward.predictions) == 2
assert all(sys.modules[name] is None for name in ("cobra", "optlang", "mfapy"))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
