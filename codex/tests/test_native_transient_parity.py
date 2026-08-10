"""Frozen historical shadow gates for the native transient EMU engine."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from cobra.io import read_sbml_model

from fluxemu.compat import (
    AuthoritativeTransitionAssignment,
    project_cobra_model,
    project_stationary_experiment,
)
from fluxemu.configuration import parse_experiment_config
from fluxemu.emu import (
    compile_emu_plan,
    compile_transient_emu_plan,
    evaluate_stationary,
    evaluate_transient,
)
from fluxemu.model import (
    MappingBranch,
    PoolQuantity,
    StationaryExperimentSemantics,
    TransientExperimentSemantics,
)


ROOT = Path(__file__).resolve().parents[1]
ANTON_SHADOW_TOLERANCE = 2.0e-6
GLUCOSE_SHADOW_TOLERANCE = 1.0e-5
RTOL = 1.0e-9
ATOL = 1.0e-12
TCA_REACTIONS = tuple(f"v{index}" for index in range(1, 9))
TCA_TRANSITIONS = (
    "antoniewicz.table5.v1.citrate_synthase",
    "antoniewicz.table5.v2.citrate_to_akg",
    "antoniewicz.table5.v3.akg_to_glutamate",
    "antoniewicz.table5.v4.akg_to_succinate",
    "antoniewicz.table5.v5.succinate_to_fumarate",
    "antoniewicz.table5.v6.fumarate_to_oaa",
    "antoniewicz.table5.v7.oaa_to_fumarate",
    "antoniewicz.table5.v8.aspartate_to_oaa",
)
UPSTREAM_REACTIONS = (
    "GLC_IN", "HEX", "PGI", "PFK", "FBA", "TPI", "GAPD", "PGK",
    "PGM", "ENO", "PYK", "LDH", "PDH",
)
UPSTREAM_TRANSITIONS = (
    "transport.glucose.identity", "glycolysis.hexokinase",
    "glycolysis.phosphoglucose_isomerase", "glycolysis.phosphofructokinase",
    "glycolysis.fructose_bisphosphate_aldolase",
    "glycolysis.triose_phosphate_isomerase", "glycolysis.gap_to_bpg",
    "glycolysis.bpg_to_3pg", "glycolysis.3pg_to_2pg",
    "glycolysis.2pg_to_pep", "glycolysis.pyruvate_kinase",
    "pyruvate.lactate_dehydrogenase", "pyruvate.pyruvate_dehydrogenase",
)
TCA_FLUXES = {
    "v1": 100.0, "v2": 100.0, "v3": 50.0, "v4": 50.0,
    "v5": 50.0, "v6": 125.0, "v7": 75.0, "v8": 50.0,
}
UPSTREAM_FLUXES = {
    "GLC_IN": 50.0, "HEX": 50.0, "PGI": 50.0, "PFK": 50.0,
    "FBA": 50.0, "TPI": 50.0, "GAPD": 100.0, "PGK": 100.0,
    "PGM": 100.0, "ENO": 100.0, "PYK": 100.0, "LDH": 0.0,
    "PDH": 100.0,
}


def _canonical(directory, yaml_name, reactions, transitions, times, pool_size):
    base = ROOT / "examples" / directory
    cobra_model = read_sbml_model(str(base / f"{directory}.xml"))
    assignments = tuple(
        AuthoritativeTransitionAssignment(reaction, transition, "forward")
        for reaction, transition in zip(reactions, transitions)
    )
    model = project_cobra_model(cobra_model, assignments, require_complete=True)
    raw = yaml.safe_load((base / yaml_name).read_text(encoding="utf-8"))
    raw.pop("timecourse")
    stationary = project_stationary_experiment(parse_experiment_config(raw))
    topology = compile_emu_plan(model, stationary)
    pool_ids = tuple(dict.fromkeys(
        emu.metabolite_id for layer in topology.layers for emu in layer.unknowns
    ))
    experiment = TransientExperimentSemantics(
        stationary.tracers,
        stationary.targets,
        tuple(float(item) for item in times),
        tuple(PoolQuantity(item, pool_size) for item in pool_ids),
        "unlabelled",
    )
    return model, experiment


def _predictions(result):
    return {
        (float(item.time), item.target_id): np.asarray(item.fractions, dtype=float)
        for item in result.forward.predictions
    }


def _frozen(path):
    frame = pd.read_csv(path)
    return {
        (float(time), metabolite): group.sort_values(
            "mass_isotopologue", key=lambda values: values.str[2:].astype(int)
        )["fraction"].to_numpy(dtype=float)
        for (time, metabolite), group in frame.groupby(
            ["time", "metabolite"], sort=False
        )
    }


def _stationary_predictions(model, experiment, fluxes):
    stationary = StationaryExperimentSemantics(experiment.tracers, experiment.targets)
    result = evaluate_stationary(compile_emu_plan(model, stationary), fluxes)
    return {item.target_id: np.asarray(item.fractions) for item in result.forward.predictions}


def _report(benchmark, native, frozen, stationary, threshold):
    predicted = _predictions(native)
    shadow_error = max(
        float(np.max(np.abs(predicted[key] - reference)))
        for key, reference in frozen.items()
    )
    final_time = max(time for time, _ in predicted)
    stationary_error = max(
        float(np.max(np.abs(predicted[(final_time, target)] - reference)))
        for target, reference in stationary.items()
    )
    diagnostic = native.diagnostics[0]
    payload = {
        "benchmark": benchmark,
        "sample_count": 1,
        "time_point_count": len({key[0] for key in predicted}),
        "target_count": len(stationary),
        "total_mid_components": len(native.forward.values),
        "max_native_vs_analytic_difference": None,
        "max_native_vs_frozen_mfapy_difference": shadow_error,
        "max_late_native_vs_native_stationary_difference": stationary_error,
        "maximum_normalization_error": native.forward.max_normalization_error,
        "minimum_component": diagnostic.minimum_mid_component,
        "solver_method": diagnostic.solver_method,
        "rtol": RTOL,
        "atol": ATOL,
        "threshold": threshold,
        "passed": shadow_error <= threshold and stationary_error <= threshold,
    }
    print("FLUXEMU_TRANSIENT_PARITY " + json.dumps(payload, sort_keys=True))
    return payload


def _mid(predictions, time, target):
    return predictions[(float(time), target)]


def test_antoniewicz_native_transient_matches_frozen_shadow_and_stationary_limit():
    pytest.importorskip("scipy", reason="native transient execution requires the transient extra")
    base = ROOT / "examples" / "antoniewicz_tca"
    frozen = _frozen(base / "timecourse_mids.csv")
    times = tuple(dict.fromkeys(time for time, _ in frozen))
    model, experiment = _canonical(
        "antoniewicz_tca", "experiment_timecourse.yaml", TCA_REACTIONS,
        TCA_TRANSITIONS, times, 1.0,
    )
    native = evaluate_transient(
        compile_transient_emu_plan(model, experiment), TCA_FLUXES,
        rtol=RTOL, atol=ATOL,
    )
    predicted = _predictions(native)
    stationary = _stationary_predictions(model, experiment, TCA_FLUXES)
    report = _report(
        "antoniewicz-table-5-transient", native, frozen, stationary,
        ANTON_SHADOW_TOLERANCE,
    )
    assert report["max_native_vs_frozen_mfapy_difference"] <= ANTON_SHADOW_TOLERANCE
    assert report["max_late_native_vs_native_stationary_difference"] <= ANTON_SHADOW_TOLERANCE
    assert all(mid[0] == 1.0 and np.count_nonzero(mid[1:]) == 0 for (time, _), mid in predicted.items() if time == 0.0)
    early = _mid(predicted, 0.001, "glutamate")
    assert early[1:3].sum() > 0.0
    assert early[3:].sum() < 1.0e-12
    assert np.all(_mid(predicted, 0.05, "glutamate")[3:] > 0.0)


def test_glucose_tca_native_transient_matches_actual_frozen_grid_and_stationary_limit():
    pytest.importorskip("scipy", reason="native transient execution requires the transient extra")
    base = ROOT / "examples" / "antoniewicz_tca_glucose"
    frozen = _frozen(base / "timecourse_mids.csv")
    times = tuple(dict.fromkeys(time for time, _ in frozen))
    model, experiment = _canonical(
        "antoniewicz_tca_glucose", "experiment_u13c6_glucose_timecourse.yaml",
        UPSTREAM_REACTIONS + TCA_REACTIONS, UPSTREAM_TRANSITIONS + TCA_TRANSITIONS,
        times, 100.0,
    )
    fluxes = {**UPSTREAM_FLUXES, **TCA_FLUXES}
    native = evaluate_transient(
        compile_transient_emu_plan(model, experiment), fluxes, rtol=RTOL, atol=ATOL,
    )
    predicted = _predictions(native)
    stationary = _stationary_predictions(model, experiment, fluxes)
    report = _report(
        "glucose-to-tca-transient", native, frozen, stationary,
        GLUCOSE_SHADOW_TOLERANCE,
    )
    assert report["max_native_vs_frozen_mfapy_difference"] <= GLUCOSE_SHADOW_TOLERANCE
    assert report["max_late_native_vs_native_stationary_difference"] <= GLUCOSE_SHADOW_TOLERANCE
    assert max(_mid(predicted, time, "pyruvate")[3] for time in times[1:]) > 0.0
    assert max(_mid(predicted, time, "AcCoA")[2] for time in times[1:]) > 0.0
    first = next(time for time in times[1:] if _mid(predicted, time, "citrate")[2] > 1.0e-10)
    assert _mid(predicted, first, "citrate")[2] > _mid(predicted, first, "citrate")[4]
    assert max(_mid(predicted, time, "citrate")[4] for time in times if time > first) > 0.0
    assert _mid(predicted, 10.0, "glutamate")[3:].sum() > 0.0


def test_transient_shadow_does_not_heal_a_removed_symmetry_orientation():
    pytest.importorskip("scipy", reason="native transient execution requires the transient extra")
    base = ROOT / "examples" / "antoniewicz_tca"
    frozen = _frozen(base / "timecourse_mids.csv")
    times = tuple(dict.fromkeys(time for time, _ in frozen))
    model, experiment = _canonical(
        "antoniewicz_tca", "experiment_timecourse.yaml", TCA_REACTIONS,
        TCA_TRANSITIONS, times, 1.0,
    )
    reactions = list(model.isotope_model.reactions)
    index = next(i for i, item in enumerate(reactions) if item.reaction_id == "v5")
    reaction = reactions[index]
    branch = reaction.mapping_branches[0]
    reactions[index] = replace(
        reaction,
        mapping_branches=(MappingBranch(branch.branch_id, 1.0, branch.transitions),),
    )
    corrupted = replace(
        model, isotope_model=replace(model.isotope_model, reactions=tuple(reactions))
    )
    result = evaluate_transient(
        compile_transient_emu_plan(corrupted, experiment), TCA_FLUXES,
        rtol=RTOL, atol=ATOL,
    )
    disagreement = max(
        float(np.max(np.abs(_predictions(result)[key] - reference)))
        for key, reference in frozen.items()
    )
    assert disagreement > ANTON_SHADOW_TOLERANCE
