from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tarfile

import pytest

from fluxemu.flux_analysis import run_highs_fba, run_highs_fva_reference
from fluxemu.real_model import TARGET_COVERAGE, load_ecoli_core_flux_model, load_ecoli_core_stage_b2_model

ROOT = Path(__file__).parents[2]
ARCHIVE = ROOT / "codex/handoff/fluxemu_r1_ecoli_stage_b2_handoff.tar.gz"
EXPECTED_ARCHIVE_SHA256 = "9acc0ed52cf6099e4f74760c20463d8a8d1973e6b213ab803916d76e070cef09"
REACTION_ORDER_SHA256 = "7c375cbe07c3f7b496de65500cb0c7852caff7b709c6bcae6622e3a1e214c364"


def test_handoff_archive_integrity_and_native_model_contract():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == EXPECTED_ARCHIVE_SHA256
    required = (
        "vendor/cobrapy/tests/data/e_coli_core.xml",
        "codex/results/hypothesis_testing/r1_ecoli_cobra_flux_classes/model_inventory.json",
        "codex/results/hypothesis_testing/r1_ecoli_atom_map_crosswalk/mapping_completeness.json",
        "vendor/mfapy/sample/Example_2_Ecoli_model.txt",
        "vendor/mfapy/sample/Tutorial 1_13C-MFAEcoli_model.txt",
        "vendor/mfapy/sample/Example_3_MCF7_model.txt",
        "vendor/mfapy/sample/Example_7_CancerCell_model.txt",
        "codex/src/fluxemu/carbon_transitions/data/metabolites.yaml",
    )
    with tarfile.open(ARCHIVE) as handoff:
        names = tuple(item.name for item in handoff.getmembers())
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert all(any(name.endswith(component) for name in names) for component in required)
        inventory_name = next(name for name in names if name.endswith("r1_ecoli_cobra_flux_classes/model_inventory.json"))
        completeness_name = next(name for name in names if name.endswith("r1_ecoli_atom_map_crosswalk/mapping_completeness.json"))
        import json
        inventory = json.load(handoff.extractfile(inventory_name))
        completeness = json.load(handoff.extractfile(completeness_name))
        assert (inventory["reaction_count"], inventory["metabolite_count"]) == (95, 72)
        assert completeness["mapped_isotope_directional_components"] == 48
        assert completeness["missing_isotope_directional_components"] == 0
        assert completeness["projection_balance_failures"] == 0
    biomass = load_ecoli_core_flux_model("biomass")
    acetate = load_ecoli_core_flux_model("acetate")
    assert len(biomass.reactions) == 95
    assert len(biomass.metabolites) == 72
    order = tuple(item.reaction_id for item in biomass.reactions)
    assert hashlib.sha256(("\n".join(order) + "\n").encode()).hexdigest() == REACTION_ORDER_SHA256
    assert biomass.reactions == acetate.reactions
    glucose = next(item for item in biomass.reactions if item.reaction_id == "EX_glc__D_e")
    viability = next(item for item in biomass.reactions if item.reaction_id == "BIOMASS_Ecoli_core_w_GAM")
    assert (glucose.lower_bound, glucose.upper_bound) == (-10.0, -10.0)
    assert viability.lower_bound == 0.5243529041810849


def test_target_coverage_is_ordered_deterministic_and_fails_closed():
    expected = ("Pyruvate", "Alanine", "Lactate", "Citrate", "AKG", "Succinate", "Fumarate", "Malate", "Glutamine", "Aspartate", "Glycine", "Serine")
    assert tuple(item.display_name for item in TARGET_COVERAGE) == expected
    assert TARGET_COVERAGE == tuple(TARGET_COVERAGE)
    assert all(item.source_complete for item in TARGET_COVERAGE)
    overlays = tuple(item.display_name for item in TARGET_COVERAGE if item.target_type == "observation_overlay")
    assert overlays == ("Alanine", "Lactate", "Citrate", "Aspartate", "Glycine", "Serine")


def test_new_observation_extension_maps_are_present_verbatim_in_checked_sources():
    with tarfile.open(ARCHIVE) as handoff:
        names = tuple(item.name for item in handoff.getmembers())
        example2 = next(name for name in names if name.endswith("vendor/mfapy/sample/Example_2_Ecoli_model.txt"))
        example7 = next(name for name in names if name.endswith("vendor/mfapy/sample/Example_7_CancerCell_model.txt"))
        text2 = handoff.extractfile(example2).read().decode()
        text7 = handoff.extractfile(example7).read().decode()
    assert "r41_ldh" in text7 and "Pyr --> Lac" in text7 and "ABC --> ABC" in text7
    assert "r17_cs" in text7 and "AcCOAmit + Oxa --> Cit" in text7 and "AB + CDEF --> FEDBAC" in text7
    assert "r67" in text2 and "PGA --> Ser" in text2 and "ABC --> ABC" in text2
    assert "r68" in text2 and "Ser --> Gly + MEETHF" in text2 and "ABC --> AB + C" in text2


def test_complete_stage_b2_projection_uses_admissible_domain_certificate():
    from fluxemu.emu import compile_emu_plan
    from fluxemu.exceptions import MappingError
    from fluxemu.model import StationaryExperimentSemantics, Target, Tracer

    model = load_ecoli_core_stage_b2_model()
    assert len(model.flux_model.reactions) == 95
    assert len(model.isotope_model.reactions) == 49  # GLUSy has two skeleton maps.
    assert len({item.flux_projection.projection_id for item in model.isotope_model.reactions}) == 48
    experiment = StationaryExperimentSemantics(
        (Tracer("glc__D_e", (("#000000", 0.5), ("#111111", 0.5)), "no"),),
        (Target("Pyruvate", "pyr_c", (1, 2, 3), "whole", "C3", "no"),),
    )
    plan = compile_emu_plan(model, experiment)
    assert plan.model is model
    fru = next(r for r in model.flux_model.reactions if r.reaction_id == "FRUpts2")
    activity = next(r for r in model.isotope_model.direction_activity_certificate.activities if r.reaction_id == "FRUpts2")
    assert fru.upper_bound == 1000.0
    assert not activity.forward_active and not activity.reverse_active
    from dataclasses import replace
    strict = replace(model, isotope_model=replace(model.isotope_model, direction_activity_certificate=None))
    with pytest.raises(MappingError, match="FRUpts2.*forward.*pyr_c"):
        compile_emu_plan(strict, experiment)
    from fluxemu.model import model_fingerprint
    changed_activity = replace(activity, forward_active=True)
    activities = tuple(changed_activity if x.reaction_id == "FRUpts2" else x for x in model.isotope_model.direction_activity_certificate.activities)
    changed_certificate = replace(model.isotope_model.direction_activity_certificate, activities=activities)
    changed = replace(model, isotope_model=replace(model.isotope_model, direction_activity_certificate=changed_certificate))
    assert model_fingerprint(model) != model_fingerprint(changed)


def test_biomass_all_observations_evaluate_but_acetate_zero_turnover_is_singular():
    import tarfile, csv, io
    from fluxemu.emu import compile_emu_plan, evaluate_stationary
    from fluxemu.exceptions import ForwardEMUError
    from fluxemu.execution import CanonicalFluxState
    from fluxemu.real_model import build_r1_acceptance_experiment
    model = load_ecoli_core_stage_b2_model()
    plan = compile_emu_plan(model, build_r1_acceptance_experiment())
    with tarfile.open(ARCHIVE) as handoff:
        def state(suffix, name):
            member = next(x.name for x in handoff.getmembers() if x.name.endswith(suffix))
            rows = csv.DictReader(io.TextIOWrapper(handoff.extractfile(member), encoding="utf8"))
            return CanonicalFluxState(name, tuple((r["reaction_id"], float(r["lp_optimum_flux"])) for r in rows))
        biomass = state("biomass_optimum_fluxes.csv", "biomass")
        acetate = state("acetate_optimum_fluxes.csv", "acetate")
    first = evaluate_stationary(plan, (biomass,))
    second = evaluate_stationary(plan, (biomass,))
    assert first == second
    assert tuple(x.target_id for x in first.forward.predictions) == tuple(x.display_name for x in TARGET_COVERAGE)
    assert len(first.forward.predictions[10].fractions) == 3  # positional PGA C1,C2 Glycine EMU
    assert len(first.forward.predictions[3].fractions) == 7  # AcCoA + OAA condensation
    with pytest.raises(ForwardEMUError, match="layer 4.*oaa_c.*mal__L_c.*fum_c.*succ_c"):
        evaluate_stationary(plan, (acetate,))


def test_deterministic_positive_fixtures_both_produce_twelve_mids():
    import importlib.resources, json
    from fluxemu.emu import compile_emu_plan, evaluate_stationary
    from fluxemu.execution import CanonicalFluxState
    from fluxemu.real_model import build_r1_acceptance_experiment
    resource = importlib.resources.files("fluxemu.real_model") / "data/e_coli_core_positive_fixtures.json"
    fixtures = json.loads(resource.read_text())["selection"]
    assert fixtures["acetate"]["sample_id"] == "theta1_sample_000"
    model = load_ecoli_core_stage_b2_model()
    plan = compile_emu_plan(model, build_r1_acceptance_experiment())
    for condition in ("biomass", "acetate"):
        fixture = fixtures[condition]
        state = CanonicalFluxState(fixture["sample_id"], tuple(map(tuple, fixture["fluxes"])))
        first = evaluate_stationary(plan, (state,))
        second = evaluate_stationary(plan, (state,))
        assert first == second
        assert len(first.forward.predictions) == 12
        assert all(abs(sum(item.fractions) - 1) <= 1e-10 for item in first.forward.predictions)


def test_frozen_optimum_projection_parity_is_exact():
    import csv
    parity = ROOT / "codex/results/real_model_50_50_u13c_glucose/frozen_projection_parity.csv"
    with parity.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 96
    assert {row["condition"] for row in rows} == {"biomass", "acetate"}
    assert all(row["passed"] == "true" for row in rows)
    assert max(float(row["absolute_residual"]) for row in rows) <= 1e-9


@pytest.mark.skipif(importlib.util.find_spec("highspy") is None, reason="highspy is unavailable")
def test_native_fba_parity_and_fva_completion_without_cobra_or_mfapy_imports():
    before = set(sys.modules)
    biomass_model = load_ecoli_core_flux_model("biomass")
    acetate_model = load_ecoli_core_flux_model("acetate")
    biomass = run_highs_fba(biomass_model)
    acetate = run_highs_fba(acetate_model)
    assert biomass.status == acetate.status == "optimal"
    assert biomass.objective_value == pytest.approx(0.8739215069684307, abs=1e-7)
    assert acetate.objective_value == pytest.approx(12.030622385803783, abs=1e-7)
    assert acetate.fluxes["BIOMASS_Ecoli_core_w_GAM"] == pytest.approx(0.524352904181085, abs=1e-7)
    assert biomass.diagnostics.max_mass_balance_residual <= 1e-7
    assert acetate.diagnostics.max_mass_balance_residual <= 1e-7
    for model in (biomass_model, acetate_model):
        fva = run_highs_fva_reference(model, 1.0)
        assert fva.ranges.shape == (95, 2)
    imported = set(sys.modules) - before
    assert not any(name == "cobra" or name.startswith("cobra.") for name in imported)
    assert not any(name == "mfapy" or name.startswith("mfapy.") for name in imported)
