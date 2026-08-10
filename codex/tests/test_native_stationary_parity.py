"""Independent-oracle and temporary mfapy shadow gates for native stationary EMU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from cobra.io import read_sbml_model

from fluxemu.compat import (
    AuthoritativeTransitionAssignment,
    project_cobra_model,
    project_stationary_experiment,
)
from fluxemu.configuration import load_experiment
from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.execution import CanonicalFluxState, run_stationary_forward


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-12
TCA_REACTIONS = tuple(f"v{index}" for index in range(1, 9))
TCA_TRANSITIONS = (
    "antoniewicz.table5.v1.citrate_synthase", "antoniewicz.table5.v2.citrate_to_akg",
    "antoniewicz.table5.v3.akg_to_glutamate", "antoniewicz.table5.v4.akg_to_succinate",
    "antoniewicz.table5.v5.succinate_to_fumarate", "antoniewicz.table5.v6.fumarate_to_oaa",
    "antoniewicz.table5.v7.oaa_to_fumarate", "antoniewicz.table5.v8.aspartate_to_oaa",
)
UPSTREAM_REACTIONS = (
    "GLC_IN", "HEX", "PGI", "PFK", "FBA", "TPI", "GAPD", "PGK", "PGM", "ENO", "PYK", "LDH", "PDH",
)
UPSTREAM_TRANSITIONS = (
    "transport.glucose.identity", "glycolysis.hexokinase", "glycolysis.phosphoglucose_isomerase",
    "glycolysis.phosphofructokinase", "glycolysis.fructose_bisphosphate_aldolase",
    "glycolysis.triose_phosphate_isomerase", "glycolysis.gap_to_bpg", "glycolysis.bpg_to_3pg",
    "glycolysis.3pg_to_2pg", "glycolysis.2pg_to_pep", "glycolysis.pyruvate_kinase",
    "pyruvate.lactate_dehydrogenase", "pyruvate.pyruvate_dehydrogenase",
)
TCA_FLUXES = {"v1": 100.0, "v2": 100.0, "v3": 50.0, "v4": 50.0,
              "v5": 50.0, "v6": 125.0, "v7": 75.0, "v8": 50.0}
UPSTREAM_FLUXES = {"GLC_IN": 50.0, "HEX": 50.0, "PGI": 50.0, "PFK": 50.0,
                   "FBA": 50.0, "TPI": 50.0, "GAPD": 100.0, "PGK": 100.0,
                   "PGM": 100.0, "ENO": 100.0, "PYK": 100.0, "LDH": 0.0, "PDH": 100.0}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(directory, yaml_name, reaction_ids, transition_ids):
    base = ROOT / "examples" / directory
    model = read_sbml_model(str(base / f"{directory}.xml"))
    assignments = tuple(
        AuthoritativeTransitionAssignment(reaction, transition, "forward")
        for reaction, transition in zip(reaction_ids, transition_ids)
    )
    return (
        project_cobra_model(model, assignments, require_complete=True),
        project_stationary_experiment(load_experiment(base / yaml_name)),
    )


def _report(name, native, oracle, threshold):
    differences = [
        abs(value - reference)
        for prediction, reference_mid in zip(native.forward.predictions, oracle)
        for value, reference in zip(prediction.fractions, reference_mid)
    ]
    payload = {
        "benchmark": name,
        "samples": len({item.sample_id for item in native.forward.predictions}),
        "targets": len(native.forward.predictions),
        "mid_components": sum(len(item.fractions) for item in native.forward.predictions),
        "max_native_vs_oracle": float(max(differences, default=0.0)),
        "max_native_vs_mfapy": None,
        "max_normalization_error": native.forward.max_normalization_error,
        "max_solve_residual": max((item.max_absolute_residual for item in native.layer_diagnostics), default=0.0),
        "worst_condition_number": max((item.condition_number for item in native.layer_diagnostics), default=0.0),
        "threshold": threshold,
        "passed": bool(max(differences, default=0.0) <= threshold),
    }
    print("FLUXEMU_PARITY " + json.dumps(payload, sort_keys=True))
    return payload


def test_antoniewicz_native_matches_independent_full_isotopomer_solver():
    model, experiment = _canonical(
        "antoniewicz_tca", "experiment_stationary.yaml", TCA_REACTIONS, TCA_TRANSITIONS
    )
    direct = _load(
        ROOT / "examples" / "antoniewicz_tca" / "direct_isotopomer_solver.py",
        "_native_antoniewicz_direct_solver",
    )
    native = evaluate_stationary(compile_emu_plan(model, experiment), TCA_FLUXES, mid_tolerance=TOLERANCE)
    oracle = (direct.glutamate_mid(),)
    report = _report("antoniewicz-table-5-6", native, oracle, TOLERANCE)
    assert report["max_native_vs_oracle"] <= TOLERANCE
    official = json.loads((ROOT / "tests" / "fixtures" / "official_mfapy_example_0.json").read_text())
    example_zero = np.asarray(official["expected_mdvs"]["Glue"])
    assert np.max(np.abs(np.asarray(native.forward.predictions[0].fractions) - example_zero)) <= TOLERANCE
    published = np.array((0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039))
    assert np.max(np.abs(np.asarray(native.forward.predictions[0].fractions) - published)) <= 4.625e-5
    assert np.max(np.abs(direct.glutamate_mid(symmetric=False) - oracle[0])) > 1e-3


def test_glucose_tca_native_matches_independent_reference_for_all_tca_targets():
    model, experiment = _canonical(
        "antoniewicz_tca_glucose", "experiment_u13c6_glucose.yaml",
        UPSTREAM_REACTIONS + TCA_REACTIONS, UPSTREAM_TRANSITIONS + TCA_TRANSITIONS,
    )
    direct = _load(
        ROOT / "examples" / "antoniewicz_tca_glucose" / "independent_tca_solver.py",
        "_native_glucose_tca_direct_solver",
    )
    fluxes = {**UPSTREAM_FLUXES, **TCA_FLUXES}
    native = evaluate_stationary(compile_emu_plan(model, experiment), fluxes, mid_tolerance=TOLERANCE)
    predictions = {item.target_id: np.asarray(item.fractions) for item in native.forward.predictions}
    reference = direct.mass_isotopomer_distributions()
    checked = tuple(reference)
    subset = tuple(item for item in native.forward.predictions if item.target_id in checked)
    oracle = tuple(reference[item.target_id] for item in subset)
    subset_result = replace_native_predictions(native, subset)
    report = _report("glucose-to-tca", subset_result, oracle, TOLERANCE)
    assert report["max_native_vs_oracle"] <= TOLERANCE
    for target in ("glucose_c", "G6P", "F6P", "FBP"):
        assert predictions[target][-1] == 1.0
    for target in ("DHAP", "GAP", "BPG", "3PG", "2PG", "PEP", "pyruvate"):
        assert predictions[target][-1] == 1.0
    corrupted = direct.mass_isotopomer_distributions(symmetric=False)
    assert max(np.max(np.abs(reference[key] - corrupted[key])) for key in checked) > 1e-3


def replace_native_predictions(native, predictions):
    from dataclasses import replace
    return replace(native, forward=replace(native.forward, predictions=predictions))


@pytest.mark.parametrize(
    ("directory", "yaml_name", "reactions", "transitions", "fluxes", "sample_id"),
    (
        ("antoniewicz_tca", "experiment_stationary.yaml", TCA_REACTIONS, TCA_TRANSITIONS, TCA_FLUXES, "control-0"),
        ("antoniewicz_tca_glucose", "experiment_u13c6_glucose.yaml", UPSTREAM_REACTIONS + TCA_REACTIONS,
         UPSTREAM_TRANSITIONS + TCA_TRANSITIONS, {**UPSTREAM_FLUXES, **TCA_FLUXES}, "control-1"),
    ),
)
def test_native_matches_current_canonical_mfapy_shadow(
    directory, yaml_name, reactions, transitions, fluxes, sample_id
):
    pytest.importorskip("scipy", reason="mfapy shadow requires the optional SciPy backend")
    model, experiment = _canonical(directory, yaml_name, reactions, transitions)
    state = CanonicalFluxState.from_mapping(sample_id, fluxes)
    plan = compile_emu_plan(model, experiment)
    native = evaluate_stationary(plan, (state,), mid_tolerance=TOLERANCE)
    shadow = run_stationary_forward(model, experiment, (state,), mid_tolerance=TOLERANCE)
    differences = [
        abs(a - b)
        for actual, expected in zip(native.forward.predictions, shadow.predictions)
        for a, b in zip(actual.fractions, expected.fractions)
    ]
    maximum = max(differences, default=0.0)
    print("FLUXEMU_PARITY " + json.dumps({
        "benchmark": sample_id, "samples": 1, "targets": len(shadow.predictions),
        "mid_components": len(shadow.values), "max_native_vs_oracle": None,
        "max_native_vs_mfapy": maximum,
        "max_normalization_error": native.forward.max_normalization_error,
        "max_solve_residual": max((x.max_absolute_residual for x in native.layer_diagnostics), default=0.0),
        "worst_condition_number": max((x.condition_number for x in native.layer_diagnostics), default=0.0),
        "threshold": TOLERANCE, "passed": maximum <= TOLERANCE,
    }, sort_keys=True))
    assert maximum <= TOLERANCE
