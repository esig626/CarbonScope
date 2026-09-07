from __future__ import annotations

import csv
import hashlib
import importlib.resources
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.exceptions import ForwardEMUError, MappingError
from fluxemu.execution import CanonicalFluxState
from fluxemu.flux_analysis import run_highs_fba, run_highs_fva_reference
from fluxemu.model import StationaryExperimentSemantics, Target, Tracer, model_fingerprint
from fluxemu.real_model import (
    TARGET_COVERAGE,
    build_r1_acceptance_experiment,
    load_ecoli_core_flux_model,
    load_ecoli_core_stage_b2_model,
)


ROOT = Path(__file__).resolve().parents[1]
REACTION_ORDER_SHA256 = "7c375cbe07c3f7b496de65500cb0c7852caff7b709c6bcae6622e3a1e214c364"


def _json_resource(name: str):
    path = importlib.resources.files("fluxemu.real_model") / f"data/{name}"
    return json.loads(path.read_text())


def test_packaged_native_model_contract_is_complete_and_ordered():
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


def test_target_coverage_is_ordered_complete_and_fails_closed():
    expected = (
        "Pyruvate", "Alanine", "Lactate", "Citrate", "AKG", "Succinate",
        "Fumarate", "Malate", "Glutamine", "Aspartate", "Glycine", "Serine",
    )
    assert tuple(item.display_name for item in TARGET_COVERAGE) == expected
    assert all(item.source_complete for item in TARGET_COVERAGE)
    overlays = tuple(
        item.display_name for item in TARGET_COVERAGE if item.target_type == "observation_overlay"
    )
    assert overlays == ("Alanine", "Lactate", "Citrate", "Aspartate", "Glycine", "Serine")


def test_complete_stage_b2_projection_uses_admissible_domain_direction_activity():
    from dataclasses import replace

    model = load_ecoli_core_stage_b2_model()
    assert len(model.flux_model.reactions) == 95
    assert len(model.isotope_model.reactions) == 49
    assert len({item.flux_projection.projection_id for item in model.isotope_model.reactions}) == 48
    experiment = StationaryExperimentSemantics(
        (Tracer("glc__D_e", (("#000000", 0.5), ("#111111", 0.5)), "no"),),
        (Target("Pyruvate", "pyr_c", (1, 2, 3), "whole", "C3", "no"),),
    )
    plan = compile_emu_plan(model, experiment)
    assert plan.model is model
    fru = next(r for r in model.flux_model.reactions if r.reaction_id == "FRUpts2")
    activity = next(
        r for r in model.isotope_model.direction_activity_certificate.activities
        if r.reaction_id == "FRUpts2"
    )
    assert fru.upper_bound == 1000.0
    assert not activity.forward_active and not activity.reverse_active
    strict = replace(model, isotope_model=replace(model.isotope_model, direction_activity_certificate=None))
    with pytest.raises(MappingError, match="FRUpts2.*forward.*pyr_c"):
        compile_emu_plan(strict, experiment)
    changed_activity = replace(activity, forward_active=True)
    activities = tuple(
        changed_activity if x.reaction_id == "FRUpts2" else x
        for x in model.isotope_model.direction_activity_certificate.activities
    )
    changed_direction_activity = replace(
        model.isotope_model.direction_activity_certificate, activities=activities
    )
    changed = replace(
        model,
        isotope_model=replace(
            model.isotope_model,
            direction_activity_certificate=changed_direction_activity,
        ),
    )
    assert model_fingerprint(model) != model_fingerprint(changed)


def test_packaged_biomass_fixture_and_frozen_acetate_negative_oracle():
    fixtures = _json_resource("e_coli_core_positive_fixtures.json")["selection"]
    frozen = _json_resource("e_coli_core_frozen_optima.json")
    model = load_ecoli_core_stage_b2_model()
    plan = compile_emu_plan(model, build_r1_acceptance_experiment())

    biomass_data = fixtures["biomass"]
    biomass = CanonicalFluxState(
        biomass_data["sample_id"], tuple(map(tuple, biomass_data["fluxes"]))
    )
    first = evaluate_stationary(plan, (biomass,))
    second = evaluate_stationary(plan, (biomass,))
    assert first == second
    assert tuple(item.target_id for item in first.forward.predictions) == tuple(
        item.display_name for item in TARGET_COVERAGE
    )
    assert len(first.forward.predictions[10].fractions) == 3
    assert len(first.forward.predictions[3].fractions) == 7

    acetate_data = frozen["acetate"]
    acetate = CanonicalFluxState(
        acetate_data["sample_id"], tuple(map(tuple, acetate_data["fluxes"]))
    )
    with pytest.raises(ForwardEMUError, match="layer 4.*oaa_c.*mal__L_c.*fum_c.*succ_c"):
        evaluate_stationary(plan, (acetate,))


def test_deterministic_positive_fixtures_both_produce_twelve_mids():
    fixtures = _json_resource("e_coli_core_positive_fixtures.json")["selection"]
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


def test_frozen_projection_parity_table_is_exact_and_complete():
    parity = ROOT / "results" / "real_model_50_50_u13c_glucose" / "frozen_projection_parity.csv"
    with parity.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 96
    assert {row["condition"] for row in rows} == {"biomass", "acetate"}
    assert all(row["passed"] == "true" for row in rows)
    assert max(float(row["absolute_residual"]) for row in rows) <= 1e-9


@pytest.mark.skipif(importlib.util.find_spec("highspy") is None, reason="highspy is unavailable")
def test_native_fba_parity_and_fva_completion_without_legacy_imports():
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
    blocked = ("cobra", "mfapy", "optlang", "nlopt")
    assert not any(name.split(".", 1)[0] in blocked for name in imported)
