"""Regression coverage for the reusable normalized carbon-transition library."""

from __future__ import annotations

from dataclasses import replace
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction
from cobra.io import read_sbml_model

from fluxemu.carbon_transitions import (
    AtomRef,
    AtomTransition,
    MappingBranch,
    export_mfapy_atom_map,
    load_default_library,
    resolve_model_metadata,
    validate_transition,
)
from fluxemu.carbon_transitions.gold import generated_table5_forward_maps
from fluxemu.exceptions import CarbonTransitionValidationError, MappingError
from fluxemu.isotope_metadata import (
    IsotopeMetadataCollection,
    collect_isotope_metadata,
    encode_metadata_note,
)


LIBRARY = load_default_library()
CODEX_ROOT = Path(__file__).resolve().parents[1]


def _load_builder(directory: str, name: str):
    path = CODEX_ROOT / "examples" / directory / "build_model.py"
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_every_packaged_entry_loads_with_strict_schema_and_registry_counts() -> None:
    assert len(LIBRARY.metabolites) >= 35
    assert len(LIBRARY.transitions) >= 44
    assert {entry.validation_status for entry in LIBRARY.transitions} == {"gold", "curated"}
    for entry in LIBRARY.transitions:
        validate_transition(entry, LIBRARY.metabolites)


def test_machine_readable_inventory_covers_every_library_entry() -> None:
    inventory = CODEX_ROOT / "results" / "carbon_transition_library" / "reaction_inventory.csv"
    with inventory.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["canonical_id"] for row in rows] == sorted(LIBRARY.by_id)
    assert set(rows[0]) == {
        "canonical_id", "name", "pathway", "carbon_substrates", "carbon_products",
        "reversible", "symmetry", "validation_status", "primary_source",
        "source_identifier", "aliases", "tests",
    }


def test_exporter_generates_mfapy_abc_only_from_normalized_positions() -> None:
    pdh = LIBRARY.by_id["pyruvate.pyruvate_dehydrogenase"]
    exported = export_mfapy_atom_map(pdh, LIBRARY.metabolites)
    assert exported.atommap == "abc-->bc+a"
    assert all(len(label) == 1 for _, labels in exported.substrate_labels for label in labels)

    succinate = LIBRARY.by_id["antoniewicz.table5.v5.succinate_to_fumarate"]
    symmetric = export_mfapy_atom_map(succinate, LIBRARY.metabolites)
    assert symmetric.atommap == "abcd-->abcd"
    assert [branch.weight for branch in symmetric.branches] == [0.5, 0.5]
    assert symmetric.symmetry_metabolites == ("succinate", "fumarate")


def test_antoniewicz_gold_maps_are_generated_directly_from_frozen_table5() -> None:
    generated = generated_table5_forward_maps()
    for number in range(1, 9):
        entry = LIBRARY.by_id[f"antoniewicz.table5.v{number}." + {
            1: "citrate_synthase", 2: "citrate_to_akg", 3: "akg_to_glutamate",
            4: "akg_to_succinate", 5: "succinate_to_fumarate", 6: "fumarate_to_oaa",
            7: "oaa_to_fumarate", 8: "aspartate_to_oaa",
        }[number]]
        assert {
            (str(item.source), str(item.destination)) for item in entry.forward_atom_map
        } == {
            (str(item.source), str(item.destination)) for item in generated[f"v{number}"]
        }


def test_reversible_maps_double_invert_to_the_original_normalized_map() -> None:
    for entry in LIBRARY.transitions:
        if not entry.reversible:
            continue
        forward = {
            (branch.branch_id, branch.weight, tuple(sorted((str(item.source), str(item.destination)) for item in branch.atom_map)))
            for branch in entry.forward_branches
        }
        recovered = {
            (branch.branch_id, branch.weight, tuple(sorted((str(item.source), str(item.destination)) for item in branch.atom_map)))
            for branch in (item.inverted() for item in entry.reverse_branches)
        }
        assert recovered == forward


def test_corruption_tests_reject_aldolase_pdh_citrate_and_symmetry_errors() -> None:
    aldolase = LIBRARY.by_id["glycolysis.fructose_bisphosphate_aldolase"]
    broken_aldolase = replace(
        aldolase,
        forward_atom_map=(
            AtomTransition(AtomRef("fructose_1_6_bisphosphate", 1), AtomRef("dihydroxyacetone_phosphate", 1)),
            *aldolase.forward_atom_map[1:],
        ),
    )
    with pytest.raises(CarbonTransitionValidationError, match="multiple origins"):
        validate_transition(broken_aldolase, LIBRARY.metabolites)

    pdh = LIBRARY.by_id["pyruvate.pyruvate_dehydrogenase"]
    with pytest.raises(CarbonTransitionValidationError, match="does not account"):
        validate_transition(replace(pdh, forward_atom_map=pdh.forward_atom_map[:-1]), LIBRARY.metabolites)

    citrate = LIBRARY.by_id["antoniewicz.table5.v1.citrate_synthase"]
    broken_citrate = replace(
        citrate,
        forward_atom_map=(
            AtomTransition(AtomRef("oxaloacetate", 1), AtomRef("citrate", 1)),
            *citrate.forward_atom_map[1:],
        ),
    )
    with pytest.raises(CarbonTransitionValidationError, match="multiple origins"):
        validate_transition(broken_citrate, LIBRARY.metabolites)

    symmetric = LIBRARY.by_id["antoniewicz.table5.v5.succinate_to_fumarate"]
    broken_branches = (
        MappingBranch("canonical_orientation", 0.6, symmetric.mapping_branches[0].atom_map),
        symmetric.mapping_branches[1],
    )
    with pytest.raises(CarbonTransitionValidationError, match="sum to 1"):
        validate_transition(replace(symmetric, mapping_branches=broken_branches), LIBRARY.metabolites)


def _pdh_model() -> Model:
    model = Model("library_resolution")
    pyr = Metabolite("pyr_c", compartment="c")
    ac = Metabolite("accoa_c", compartment="c")
    co2 = Metabolite("co2_c", compartment="c")
    reaction = Reaction("unfamiliar_model_id")
    reaction.add_metabolites({pyr: -1.0, ac: 1.0, co2: 1.0})
    model.add_reactions([reaction])
    return model


def test_matcher_uses_explicit_id_then_database_alias_then_exact_carbon_chemistry() -> None:
    model = _pdh_model()
    reaction = model.reactions.get_by_id("unfamiliar_model_id")
    reaction.annotation["fluxemu_transition_id"] = "pyruvate.pyruvate_dehydrogenase"
    assert LIBRARY.resolve_reaction(reaction).matched_by == "explicit_id"  # type: ignore[union-attr]

    del reaction.annotation["fluxemu_transition_id"]
    reaction.annotation["bigg.reaction"] = "PDH"
    assert LIBRARY.resolve_reaction(reaction).matched_by == "database_identifier"  # type: ignore[union-attr]

    del reaction.annotation["bigg.reaction"]
    assert LIBRARY.resolve_reaction(reaction).matched_by == "carbon_chemistry"  # type: ignore[union-attr]

    reaction.name = "pyruvate dehydrogenase"
    lactate = Metabolite("lac__D_c", compartment="c")
    reaction.add_metabolites({model.metabolites.get_by_id("accoa_c"): -1.0, model.metabolites.get_by_id("co2_c"): -1.0, lactate: 1.0})
    matched = LIBRARY.resolve_reaction(reaction)
    assert matched is not None
    assert matched.transition.canonical_id == "pyruvate.lactate_dehydrogenase"

    separated = _pdh_model()
    separated.metabolites.get_by_id("accoa_c").compartment = "m"
    separated.reactions.get_by_id("unfamiliar_model_id").annotation["fluxemu_transition_id"] = "pyruvate.pyruvate_dehydrogenase"
    with pytest.raises(MappingError, match="carbon chemistry or direction"):
        LIBRARY.resolve_reaction(separated.reactions.get_by_id("unfamiliar_model_id"))


def test_library_resolution_generates_metadata_but_explicit_sbml_metadata_wins() -> None:
    model = _pdh_model()
    result = resolve_model_metadata(model, IsotopeMetadataCollection({}, {}), LIBRARY)
    generated = result.metadata.reactions["unfamiliar_model_id"]
    assert "".join(generated.substrates[0].atom_labels) == "abc"
    assert ["".join(item.atom_labels) for item in generated.products] == ["bc", "a"]
    assert result.provenance[0].mapping_source == "library"

    overridden = IsotopeMetadataCollection(
        reactions={"unfamiliar_model_id": replace(generated, directional_id="sbml:override")},
        metabolites=result.metadata.metabolites,
    )
    second = resolve_model_metadata(model, overridden, LIBRARY)
    assert second.metadata.reactions["unfamiliar_model_id"].directional_id == "sbml:override"
    assert second.provenance[0].mapping_source == "SBML"


def _install_library_generated_reactions(model: Model) -> None:
    original = IsotopeMetadataCollection({}, collect_isotope_metadata(model).metabolites)
    result = resolve_model_metadata(model, original, LIBRARY)
    for reaction_id, metadata in result.metadata.reactions.items():
        model.reactions.get_by_id(reaction_id).notes["FLUXEMU_REACTION_METADATA_V1"] = encode_metadata_note(metadata)


def test_library_generated_tca_maps_reproduce_the_frozen_glutamate_mid() -> None:
    benchmark = _load_builder("antoniewicz_tca", "_library_tca_builder")
    model = read_sbml_model(str(benchmark.MODEL_PATH))
    _install_library_generated_reactions(model)
    bundle = benchmark.build_table5_mfapy_bundle(model, benchmark.load_experiment(benchmark.STATIONARY_EXPERIMENT))
    result = benchmark.run_batch_forward(bundle, benchmark._fixed_flux_frame(model), benchmark.load_experiment(benchmark.STATIONARY_EXPERIMENT))
    observed = np.asarray(result.predictions["figure12"]["glutamate"], dtype=float)
    assert np.allclose(observed, benchmark.PUBLISHED_TABLE6_MID, rtol=0.0, atol=5.1e-5)


def test_library_generated_glucose_maps_reproduce_frozen_stationary_results() -> None:
    benchmark = _load_builder("antoniewicz_tca_glucose", "_library_glucose_builder")
    model = read_sbml_model(str(benchmark.MODEL_PATH))
    _install_library_generated_reactions(model)
    bundle = benchmark.build_glucose_tca_mfapy_bundle(
        model, benchmark.load_experiment(benchmark.STATIONARY_EXPERIMENT), symmetric=True
    )
    result = benchmark.run_batch_forward(bundle, benchmark.fixed_flux_frame(model), benchmark.load_experiment(benchmark.STATIONARY_EXPERIMENT))
    assert result.predictions["glucose_to_tca_fixed"]["pyruvate"][3] == pytest.approx(1.0, abs=1.0e-12)
    assert result.predictions["glucose_to_tca_fixed"]["AcCoA"][2] == pytest.approx(1.0, abs=1.0e-12)
    assert result.predictions["glucose_to_tca_fixed"]["citrate"][4] > 0.0
    reference = benchmark.independent_tca_mids()
    for metabolite in benchmark.INDEPENDENT_TCA_TARGETS:
        assert np.allclose(result.predictions["glucose_to_tca_fixed"][metabolite], reference[metabolite], rtol=0.0, atol=1.0e-12)

    timepoints = list(benchmark.BASE_TIMEPOINTS)
    while True:
        dynamic = benchmark._timecourse_frame(bundle, timepoints)
        final = timepoints[-1]
        maximum_error = max(
            float(np.max(np.abs(benchmark._mid_at(dynamic, final, target) - result.predictions["glucose_to_tca_fixed"][target])))
            for target in benchmark.STATIONARY_TARGETS
        )
        if maximum_error <= benchmark.DYNAMIC_CONVERGENCE_TOLERANCE:
            break
        assert final < 640.0
        timepoints.append(final * 2.0)
    assert maximum_error <= benchmark.DYNAMIC_CONVERGENCE_TOLERANCE
