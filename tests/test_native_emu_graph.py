from __future__ import annotations

from fluxemu.emu import EMU, compile_emu_plan
from fluxemu.model import (
    AtomPosition, AtomTransition, CanonicalModel, FluxMetabolite, FluxModel, FluxReaction,
    IsotopeMetabolite, IsotopeModel, IsotopeParticipant, IsotopeReaction, LinearObjective,
    MappingBranch, StationaryExperimentSemantics, StoichiometricTerm, Target, Tracer,
)


def simple_science():
    flux = FluxModel(
        (FluxMetabolite("source", False), FluxMetabolite("product", False)),
        (FluxReaction("R", (StoichiometricTerm("source", -1), StoichiometricTerm("product", 1)), 0, 10),),
        LinearObjective("maximise", ()),
    )
    isotope = IsotopeModel(
        (IsotopeMetabolite("source", 2, True, False), IsotopeMetabolite("product", 2, True, False)),
        (IsotopeReaction(
            "R", "forward", True, (IsotopeParticipant("source", (1, 2)),),
            (IsotopeParticipant("product", (1, 2)),),
            (MappingBranch("primary", 1, (
                AtomTransition(AtomPosition("source", 1), AtomPosition("product", 2)),
                AtomTransition(AtomPosition("source", 2), AtomPosition("product", 1)),
            )),),
        ),),
    )
    experiment = StationaryExperimentSemantics(
        (Tracer("source", (("#00", 0.5), ("#10", 0.5)), "no"),),
        (Target("fragment", "product", (1, 2), "intermediate", "C2", "no"),),
    )
    return CanonicalModel(flux, isotope), experiment


def test_compiler_traces_explicit_atoms_and_preserves_scientific_order():
    model, experiment = simple_science()
    plan = compile_emu_plan(model, experiment)
    assert plan.targets == (("fragment", EMU("product", (1, 2))),)
    assert plan.emus == (EMU("product", (1, 2)), EMU("source", (2, 1)))
    assert plan.source_emus == (EMU("source", (2, 1)),)
    assert plan.contributions[0].precursors == (EMU("source", (2, 1)),)
    assert len(plan.model_fingerprint) == len(plan.experiment_fingerprint) == 64


def test_emu_identity_retains_requested_atom_order():
    assert EMU("x", (2, 1)) != EMU("x", (1, 2))
