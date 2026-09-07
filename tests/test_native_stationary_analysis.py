"""End-to-end contract for one-model native FBA/FVA to stationary MIDs."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
import subprocess
import sys

import pandas as pd
import pytest

import fluxemu.analysis.stationary as analysis_module
import fluxemu.flux_analysis.highs as highs_module
from fluxemu.analysis import run_native_fba, run_native_fva, run_native_stationary_analysis
from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.exceptions import AnalysisError
from fluxemu.execution import CanonicalFluxState
from fluxemu.flux_analysis import run_highs_fba, run_highs_fva_reference, run_highs_vffva
from fluxemu.model import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    CanonicalModelError,
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
    deterministic_serialise,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("highspy") is None,
    reason="highspy is unavailable",
)


def _mapped_reaction(reaction_id: str, source: str, product: str) -> IsotopeReaction:
    transition = AtomTransition(AtomPosition(source, 1), AtomPosition(product, 1))
    return IsotopeReaction(
        reaction_id,
        "forward",
        True,
        (IsotopeParticipant(source, (1,)),),
        (IsotopeParticipant(product, (1,)),),
        (MappingBranch("declared", 1.0, (transition,)),),
    )


def _alternate_optimum_science() -> tuple[CanonicalModel, StationaryExperimentSemantics]:
    # Deliberately non-lexical reaction order.  The two inputs are alternate
    # ways to sustain the same optimal drain, and carry different labels.
    metabolites = (
        FluxMetabolite("S0", False),
        FluxMetabolite("S1", False),
        FluxMetabolite("I", True),
        FluxMetabolite("O", False),
    )
    reactions = (
        FluxReaction(
            "Z_IN", (StoichiometricTerm("S0", -1), StoichiometricTerm("I", 1)), 0, 10
        ),
        FluxReaction(
            "A_IN", (StoichiometricTerm("S1", -1), StoichiometricTerm("I", 1)), 0, 10
        ),
        FluxReaction(
            "M_OUT", (StoichiometricTerm("I", -1), StoichiometricTerm("O", 1)), 0, 10
        ),
    )
    isotope_metabolites = tuple(
        IsotopeMetabolite(metabolite.metabolite_id, 1, True, False)
        for metabolite in metabolites
    )
    model = CanonicalModel(
        FluxModel(
            metabolites,
            reactions,
            LinearObjective("maximise", (ObjectiveTerm("M_OUT", 1.0),)),
        ),
        IsotopeModel(
            isotope_metabolites,
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


def test_manual_composition_parity_same_model_and_declared_order(monkeypatch):
    model, experiment = _alternate_optimum_science()
    direct_fba = run_highs_fba(model.flux_model)
    direct_fva = run_highs_fva_reference(model.flux_model, 1.0)
    direct_state = CanonicalFluxState(
        "fba-optimum", tuple((key, value) for key, value in direct_fba.fluxes.items())
    )
    direct_mids = evaluate_stationary(
        compile_emu_plan(model, experiment), (direct_state,)
    )

    seen: dict[str, object] = {}
    original_prepare = analysis_module.prepare_highs_flux_region
    original_fva = analysis_module.run_prepared_highs_vffva
    original_compile = analysis_module.compile_emu_plan

    def recording_prepare(flux_model, fraction):
        seen["fba_model"] = flux_model
        return original_prepare(flux_model, fraction)

    def recording_fva(prepared, *, workers=None):
        seen["fva_lp"] = prepared.lp
        return original_fva(prepared, workers=1)

    def recording_compile(canonical_model, stationary_experiment):
        seen["emu_model"] = canonical_model
        seen["experiment"] = stationary_experiment
        return original_compile(canonical_model, stationary_experiment)

    monkeypatch.setattr(analysis_module, "prepare_highs_flux_region", recording_prepare)
    monkeypatch.setattr(analysis_module, "run_prepared_highs_vffva", recording_fva)
    monkeypatch.setattr(analysis_module, "compile_emu_plan", recording_compile)
    result = run_native_stationary_analysis(model, experiment)

    assert seen["fba_model"] is model.flux_model
    assert seen["fva_lp"].reaction_ids == tuple(
        reaction.reaction_id for reaction in model.flux_model.reactions
    )
    assert seen["emu_model"] is model
    assert seen["experiment"] is experiment
    assert result.fba.objective_value == direct_fba.objective_value
    assert tuple(result.fba.fluxes.index) == ("Z_IN", "A_IN", "M_OUT")
    pd.testing.assert_series_equal(result.fba.fluxes, direct_fba.fluxes, check_exact=True)
    pd.testing.assert_frame_equal(result.fva.ranges, direct_fva.ranges, check_exact=True)
    assert result.flux_state == direct_state
    assert result.mids.forward.predictions == direct_mids.forward.predictions
    assert result.mids.forward.values == direct_mids.forward.values
    assert result.mids.layer_diagnostics == direct_mids.layer_diagnostics


def test_fva_endpoints_are_not_a_flux_vector_and_fba_primal_is_emu_input(monkeypatch):
    model, experiment = _alternate_optimum_science()
    direct_fba = run_highs_fba(model.flux_model)
    captured: list[CanonicalFluxState] = []
    original_evaluate = analysis_module.evaluate_stationary

    def recording_evaluate(plan, states):
        captured.extend(states)
        return original_evaluate(plan, states)

    monkeypatch.setattr(analysis_module, "evaluate_stationary", recording_evaluate)
    result = run_native_stationary_analysis(model, experiment)

    assert result.fva.ranges.loc["Z_IN", "maximum"] - result.fva.ranges.loc[
        "Z_IN", "minimum"
    ] > 1.0
    expected = tuple((key, value) for key, value in direct_fba.fluxes.items())
    assert result.flux_state.values == expected
    assert captured == [result.flux_state]
    assert tuple(value for _, value in captured[0].values) == tuple(direct_fba.fluxes)
    endpoint_mixture = tuple(result.fva.ranges["minimum"])
    assert endpoint_mixture != tuple(value for _, value in result.flux_state.values)
    # Independent minima do not even sustain the retained objective here.
    assert endpoint_mixture[2] == pytest.approx(10.0)
    assert endpoint_mixture[0] + endpoint_mixture[1] == pytest.approx(0.0)


def test_convenience_wrappers_delegate_without_reconstructing_model(monkeypatch):
    model, _ = _alternate_optimum_science()
    calls = []

    monkeypatch.setattr(
        analysis_module,
        "run_highs_fba",
        lambda value: calls.append(("fba", value)) or run_highs_fba(value),
    )
    monkeypatch.setattr(
        analysis_module,
        "run_highs_vffva",
        lambda value, fraction, *, workers=None: calls.append(("fva", value, fraction))
        or run_highs_vffva(value, fraction, workers=1),
    )
    run_native_fba(model)
    run_native_fva(model, 0.75)
    assert calls == [("fba", model.flux_model), ("fva", model.flux_model, 0.75)]


def test_composed_deterministic_path_compiles_and_solves_biological_objective_once(
    monkeypatch,
):
    model, experiment = _alternate_optimum_science()
    calls = {"compile": 0, "fba": 0}
    original_compile = highs_module.compile_flux_lp
    original_fba = highs_module._run_compiled_fba

    def recording_compile(flux_model):
        calls["compile"] += 1
        return original_compile(flux_model)

    def recording_fba(lp):
        calls["fba"] += 1
        return original_fba(lp)

    monkeypatch.setattr(highs_module, "compile_flux_lp", recording_compile)
    monkeypatch.setattr(highs_module, "_run_compiled_fba", recording_fba)
    run_native_stationary_analysis(model, experiment)
    assert calls == {"compile": 1, "fba": 1}


@pytest.mark.parametrize("fraction", [0, -1, 1.01, math.nan, True])
def test_invalid_fva_fraction_remains_explicit(fraction):
    model, experiment = _alternate_optimum_science()
    with pytest.raises(AnalysisError, match="fraction_of_optimum"):
        run_native_stationary_analysis(
            model, experiment, fva_fraction_of_optimum=fraction
        )


def test_invalid_model_tracer_and_target_remain_explicit():
    model, experiment = _alternate_optimum_science()
    malformed_reaction = replace(model.flux_model.reactions[0], lower_bound=11)
    malformed = replace(
        model,
        flux_model=replace(
            model.flux_model,
            reactions=(malformed_reaction,) + model.flux_model.reactions[1:],
        ),
    )
    with pytest.raises(CanonicalModelError, match="lower bound exceeds upper"):
        run_native_stationary_analysis(malformed, experiment)

    bad_tracer = replace(
        experiment,
        tracers=(Tracer("missing", (("#0", 1.0),), "no"),),
    )
    with pytest.raises(CanonicalModelError, match="unknown isotope metabolite"):
        run_native_stationary_analysis(model, bad_tracer)

    bad_target = replace(
        experiment,
        targets=(Target("bad", "missing", (1,), "intermediate", "C1", "no"),),
    )
    with pytest.raises(CanonicalModelError, match="unknown isotope metabolite"):
        run_native_stationary_analysis(model, bad_target)


def test_repeatable_and_nonmutating():
    model, experiment = _alternate_optimum_science()
    model_before = deterministic_serialise(model)
    experiment_before = repr(experiment)
    first = run_native_stationary_analysis(model, experiment)
    second = run_native_stationary_analysis(model, experiment)

    assert deterministic_serialise(model) == model_before
    assert repr(experiment) == experiment_before
    assert first.fba.objective_value == second.fba.objective_value
    pd.testing.assert_series_equal(first.fba.fluxes, second.fba.fluxes, check_exact=True)
    pd.testing.assert_frame_equal(first.fva.ranges, second.fva.ranges, check_exact=True)
    assert first.flux_state == second.flux_state
    assert first.mids == second.mids


def test_native_analysis_import_and_execution_firewall():
    script = r'''
import sys
from fluxemu.analysis import run_native_stationary_analysis
from fluxemu.model import *

def mapped(rid, source, product):
    transition = AtomTransition(AtomPosition(source, 1), AtomPosition(product, 1))
    return IsotopeReaction(rid, "forward", True, (IsotopeParticipant(source, (1,)),),
        (IsotopeParticipant(product, (1,)),), (MappingBranch("map", 1.0, (transition,)),))

model = CanonicalModel(
    FluxModel((FluxMetabolite("S", False), FluxMetabolite("T", False)),
        (FluxReaction("R", (StoichiometricTerm("S", -1), StoichiometricTerm("T", 1)), 1, 1),),
        LinearObjective("maximise", (ObjectiveTerm("R", 1),))),
    IsotopeModel((IsotopeMetabolite("S", 1, True, False), IsotopeMetabolite("T", 1, True, False)),
        (mapped("R", "S", "T"),)))
experiment = StationaryExperimentSemantics((Tracer("S", (("#1", 1.0),), "no"),),
    (Target("T", "T", (1,), "intermediate", "C1", "no"),))
run_native_stationary_analysis(model, experiment)
for forbidden in ("cobra", "mfapy", "matplotlib"):
    assert forbidden not in sys.modules, (forbidden, sys.modules.get(forbidden))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
