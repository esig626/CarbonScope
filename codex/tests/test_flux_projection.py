from __future__ import annotations

from dataclasses import replace

import pytest

from fluxemu.emu.projection import evaluate_flux_projection
from fluxemu.emu.stationary import _turnover
from fluxemu.exceptions import MappingError
from fluxemu.model import (
    AtomPosition, AtomTransition, CanonicalModel, FluxMetabolite, FluxModel,
    FluxProjectionExpression, FluxProjectionRule, FluxProjectionTerm, FluxReaction,
    IsotopeMetabolite, IsotopeModel, IsotopeParticipant, IsotopeReaction,
    LinearObjective, MappingBranch, ObjectiveTerm, PhysicalDirectionRef,
    StationaryExperimentSemantics, StoichiometricTerm, Target, Tracer,
    deterministic_serialise, model_fingerprint,
)
from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.execution import CanonicalFluxState


def rule(name, coefficient=1.0, equivalents=(), covered=()):
    return FluxProjectionRule(
        name,
        FluxProjectionExpression((FluxProjectionTerm("R", coefficient),)),
        "positive_part", 1e-9, equivalents, covered, (("source", "test"),),
    )


def test_positive_reverse_zero_scaled_and_lump_checks():
    assert evaluate_flux_projection(rule("f"), {"R": 3.0}) == 3.0
    assert evaluate_flux_projection(rule("r", -1.0), {"R": -3.0}) == 3.0
    assert evaluate_flux_projection(rule("z"), {"R": 5e-10}) == 0.0
    assert evaluate_flux_projection(rule("scaled", 4.1182), {"R": 2.0}) == 8.2364
    equivalent = FluxProjectionExpression((FluxProjectionTerm("Q", 1.0),))
    lump = rule("lump", equivalents=(equivalent,))
    assert evaluate_flux_projection(lump, {"R": 2.0, "Q": 2.0}) == 2.0
    with pytest.raises(MappingError, match="equivalent-expression residual"):
        evaluate_flux_projection(lump, {"R": 2.0, "Q": 2.1})


def projected_model():
    metabolites = (FluxMetabolite("S", False), FluxMetabolite("A", False), FluxMetabolite("B", False))
    physical = FluxReaction("R", (StoichiometricTerm("A", -1), StoichiometricTerm("B", 1)), -10, 10)
    flux = FluxModel(metabolites, (physical,), LinearObjective("maximise", (ObjectiveTerm("R", 1),)))
    iso_met = tuple(IsotopeMetabolite(m.metabolite_id, 1, True, False) for m in metabolites)
    def component(name, source, product, coefficient, covered):
        return IsotopeReaction(
            name, "forward", True, (IsotopeParticipant(source, (1,)),),
            (IsotopeParticipant(product, (1,)),),
            (MappingBranch("map", 1, (AtomTransition(AtomPosition(source, 1), AtomPosition(product, 1)),)),),
            flux_projection=rule(name, coefficient, covered=(PhysicalDirectionRef("R", covered),)),
        )
    isotope = IsotopeModel(iso_met, (
        component("R_forward", "A", "B", 1, "forward"),
        component("R_reverse", "B", "A", -1, "reverse"),
    ))
    return CanonicalModel(flux, isotope)


def test_two_components_share_reversible_physical_flux_without_bound_mutation():
    model = projected_model()
    bounds = (model.flux_model.reactions[0].lower_bound, model.flux_model.reactions[0].upper_bound)
    forward = StationaryExperimentSemantics((Tracer("A", (("#1", 1.0),), "no"),), (Target("B", "B", (1,), "x", "C1", "no"),))
    result = evaluate_stationary(compile_emu_plan(model, forward), (CanonicalFluxState("p", (("R", 2.0),)),))
    assert result.forward.predictions[0].fractions == pytest.approx((0, 1))
    reverse = StationaryExperimentSemantics((Tracer("B", (("#1", 1.0),), "no"),), (Target("A", "A", (1,), "x", "C1", "no"),))
    result = evaluate_stationary(compile_emu_plan(model, reverse), (CanonicalFluxState("n", (("R", -2.0),)),))
    assert result.forward.predictions[0].fractions == pytest.approx((0, 1))
    assert bounds == (-10, 10) == (model.flux_model.reactions[0].lower_bound, model.flux_model.reactions[0].upper_bound)


def test_turnover_uses_actual_positive_and_negative_net_flux():
    model = projected_model()
    experiment = StationaryExperimentSemantics((Tracer("A", (("#0", 1.0),), "no"),), (Target("B", "B", (1,), "x", "C1", "no"),))
    plan = compile_emu_plan(model, experiment)
    from fluxemu.emu.graph import EMU
    assert _turnover(plan, EMU("A", (1,)), {"R": 3.0}) == 3.0
    assert _turnover(plan, EMU("B", (1,)), {"R": -4.0}) == 4.0


def test_projection_serialisation_and_fingerprint_are_deterministic_and_sensitive():
    model = projected_model()
    assert deterministic_serialise(model) == deterministic_serialise(model)
    changed_rule = replace(model.isotope_model.reactions[0].flux_projection, zero_tolerance=1e-8)
    changed_rxn = replace(model.isotope_model.reactions[0], flux_projection=changed_rule)
    changed = replace(model, isotope_model=replace(model.isotope_model, reactions=(changed_rxn,) + model.isotope_model.reactions[1:]))
    assert model_fingerprint(model) != model_fingerprint(changed)
