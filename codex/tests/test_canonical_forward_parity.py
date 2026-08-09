"""Real mfapy parity gates for the checked Control 0 and Control 1 models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from cobra.io import read_sbml_model

from fluxemu.compat import AuthoritativeTransitionAssignment, project_cobra_model, project_stationary_experiment
from fluxemu.configuration import load_experiment
from fluxemu.execution import CanonicalFluxState, run_stationary_forward


PARITY_TOLERANCE = 1e-12
ROOT = Path(__file__).resolve().parents[1]
CONTROL_0_REACTIONS = tuple(f"v{index}" for index in range(1, 9))
CONTROL_0_TRANSITIONS = (
    "antoniewicz.table5.v1.citrate_synthase",
    "antoniewicz.table5.v2.citrate_to_akg",
    "antoniewicz.table5.v3.akg_to_glutamate",
    "antoniewicz.table5.v4.akg_to_succinate",
    "antoniewicz.table5.v5.succinate_to_fumarate",
    "antoniewicz.table5.v6.fumarate_to_oaa",
    "antoniewicz.table5.v7.oaa_to_fumarate",
    "antoniewicz.table5.v8.aspartate_to_oaa",
)
GLYCOLYSIS_REACTIONS = (
    "GLC_IN", "HEX", "PGI", "PFK", "FBA", "TPI", "GAPD", "PGK", "PGM", "ENO", "PYK", "LDH", "PDH",
)
GLYCOLYSIS_TRANSITIONS = (
    "transport.glucose.identity",
    "glycolysis.hexokinase",
    "glycolysis.phosphoglucose_isomerase",
    "glycolysis.phosphofructokinase",
    "glycolysis.fructose_bisphosphate_aldolase",
    "glycolysis.triose_phosphate_isomerase",
    "glycolysis.gap_to_bpg",
    "glycolysis.bpg_to_3pg",
    "glycolysis.3pg_to_2pg",
    "glycolysis.2pg_to_pep",
    "glycolysis.pyruvate_kinase",
    "pyruvate.lactate_dehydrogenase",
    "pyruvate.pyruvate_dehydrogenase",
)


def _builder(directory: str):
    name = f"_canonical_parity_{directory}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / directory / "build_model.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assignments(reactions, transitions):
    return tuple(
        AuthoritativeTransitionAssignment(reaction, transition, "forward")
        for reaction, transition in zip(reactions, transitions)
    )


@pytest.mark.parametrize(
    ("directory", "yaml_name", "reaction_ids", "transition_ids", "sample_id"),
    [
        ("antoniewicz_tca", "experiment_stationary.yaml", CONTROL_0_REACTIONS, CONTROL_0_TRANSITIONS, "figure12"),
        (
            "antoniewicz_tca_glucose",
            "experiment_u13c6_glucose.yaml",
            GLYCOLYSIS_REACTIONS + CONTROL_0_REACTIONS,
            GLYCOLYSIS_TRANSITIONS + CONTROL_0_TRANSITIONS,
            "glucose_to_tca_fixed",
        ),
    ],
)
def test_real_control_canonical_forward_matches_legacy(
    directory, yaml_name, reaction_ids, transition_ids, sample_id
):
    builder = _builder(directory)
    model_path = ROOT / "examples" / directory / f"{directory}.xml"
    config_path = ROOT / "examples" / directory / yaml_name
    cobra_model = read_sbml_model(str(model_path))
    config = load_experiment(config_path)

    # This is the reference path and retains its legacy COBRA/configuration science.
    if directory == "antoniewicz_tca":
        legacy_bundle, legacy_mid, _ = builder.run_stationary()
        legacy = {"glutamate": np.asarray(legacy_mid)}
        frame = builder.ground_truth_flux_frame(cobra_model)
        assert legacy_bundle is not None
    else:
        legacy_bundle, legacy, _ = builder.run_stationary()
        frame = builder.fixed_flux_frame(cobra_model)
        assert legacy_bundle is not None

    canonical = project_cobra_model(
        cobra_model, _assignments(reaction_ids, transition_ids), require_complete=True
    )
    experiment = project_stationary_experiment(config)
    state = CanonicalFluxState.from_mapping(sample_id, frame.iloc[0].to_dict())
    result = run_stationary_forward(
        canonical, experiment, (state,), mid_tolerance=PARITY_TOLERANCE
    )
    canonical_predictions = {item.target_id: np.asarray(item.fractions) for item in result.predictions}
    assert tuple(canonical_predictions) == tuple(item.target_id for item in experiment.targets)
    differences = []
    for target in experiment.targets:
        actual = canonical_predictions[target.target_id]
        reference = np.asarray(legacy[target.target_id])
        assert actual.shape == reference.shape
        differences.extend(np.abs(actual - reference))
    maximum = max(differences, default=0.0)
    assert maximum <= PARITY_TOLERANCE

    repeated = run_stationary_forward(
        canonical, experiment, (state,), mid_tolerance=PARITY_TOLERANCE
    )
    assert repeated == result


def test_weighted_branch_records_compile_in_exact_canonical_order():
    from fluxemu.backends.mfapy import compile_stationary_model

    directory = "antoniewicz_tca"
    cobra_model = read_sbml_model(str(ROOT / "examples" / directory / f"{directory}.xml"))
    canonical = project_cobra_model(
        cobra_model,
        _assignments(CONTROL_0_REACTIONS, CONTROL_0_TRANSITIONS),
        require_complete=True,
    )
    experiment = project_stationary_experiment(
        load_experiment(ROOT / "examples" / directory / "experiment_stationary.yaml")
    )
    compiled = compile_stationary_model(canonical, experiment)
    by_reaction = {}
    for branch in compiled.branches:
        by_reaction.setdefault(branch.reaction_id, []).append(branch)
    for reaction_id in ("v5", "v6", "v7"):
        source = next(item for item in canonical.isotope_model.reactions if item.reaction_id == reaction_id)
        actual = by_reaction[reaction_id]
        assert tuple((item.branch_id, item.weight) for item in actual) == tuple(
            (item.branch_id, float(item.weight)) for item in source.mapping_branches
        )
        assert tuple(item.transitions for item in actual) == tuple(
            tuple(
                (t.source.metabolite_id, t.source.position, t.destination.metabolite_id, t.destination.position)
                for t in branch.transitions
            )
            for branch in source.mapping_branches
        )
