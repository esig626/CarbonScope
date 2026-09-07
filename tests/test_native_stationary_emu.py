from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fluxemu.emu import EMU, compile_emu_plan, evaluate_stationary
from fluxemu.exceptions import ForwardEMUError, MappingError
from fluxemu.execution import CanonicalFluxState
from fluxemu.model import (
    AtomPosition, AtomTransition, CanonicalModel, FluxMetabolite, FluxModel, FluxReaction,
    IsotopeMetabolite, IsotopeModel, IsotopeParticipant, IsotopeReaction, LinearObjective,
    MappingBranch, StationaryExperimentSemantics, StoichiometricTerm, Target, Tracer,
)


def _reaction(reaction_id, substrates, product, *, branches=None):
    source_atoms = tuple((source, position) for source, positions in substrates for position in positions)
    transitions = tuple(
        AtomTransition(AtomPosition(source, position), AtomPosition(product, destination))
        for destination, (source, position) in enumerate(source_atoms, 1)
    )
    if branches is None:
        branches = (MappingBranch("primary", 1.0, transitions),)
    return IsotopeReaction(
        reaction_id, "forward", True,
        tuple(IsotopeParticipant(source, tuple(positions)) for source, positions in substrates),
        (IsotopeParticipant(product, tuple(range(1, len(transitions) + 1))),), tuple(branches),
    )


def _science(metabolites, reactions, isotope_reactions, tracers, targets):
    flux_metabolites = tuple(FluxMetabolite(mid, balanced) for mid, _, balanced in metabolites)
    isotope_metabolites = tuple(IsotopeMetabolite(mid, count, True, False) for mid, count, _ in metabolites)
    model = CanonicalModel(
        FluxModel(flux_metabolites, tuple(reactions), LinearObjective("maximise", ())),
        IsotopeModel(isotope_metabolites, tuple(isotope_reactions)),
    )
    experiment = StationaryExperimentSemantics(tuple(tracers), tuple(targets))
    return model, experiment


def _flux_reaction(reaction_id, substrates, products, bound=10):
    return FluxReaction(
        reaction_id,
        tuple(StoichiometricTerm(item, -1) for item in substrates)
        + tuple(StoichiometricTerm(item, 1) for item in products),
        0, bound,
    )


def _target(mid, count=1):
    return Target(mid, mid, tuple(range(1, count + 1)), "intermediate", f"C{count}", "no")


def test_n0_source_to_terminal_target_and_batch_order():
    model, experiment = _science(
        (("S", 1, False), ("T", 1, False)),
        (_flux_reaction("R", ("S",), ("T",)),),
        (_reaction("R", (("S", (1,)),), "T"),),
        (Tracer("S", (("#1", 1.0),), "no"),), (_target("T"),),
    )
    plan = compile_emu_plan(model, experiment)
    states = (
        CanonicalFluxState.from_mapping("first", {"R": 2.0}),
        CanonicalFluxState.from_mapping("second", {"R": 3.0}),
    )
    result = evaluate_stationary(plan, states).forward
    assert tuple(item.sample_id for item in result.predictions) == ("first", "second")
    assert tuple(item.fractions for item in result.predictions) == ((0.0, 1.0), (0.0, 1.0))


def test_n1_internal_chain_and_internal_balanced_target():
    model, experiment = _science(
        (("S", 1, False), ("I", 1, True), ("T", 1, False)),
        (_flux_reaction("R1", ("S",), ("I",)), _flux_reaction("R2", ("I",), ("T",))),
        (_reaction("R1", (("S", (1,)),), "I"), _reaction("R2", (("I", (1,)),), "T")),
        (Tracer("S", (("#0", 0.25), ("#1", 0.75)), "no"),), (_target("I"), _target("T")),
    )
    result = evaluate_stationary(compile_emu_plan(model, experiment), {"R1": 2, "R2": 2})
    assert tuple(item.fractions for item in result.forward.predictions) == ((0.25, 0.75),) * 2
    assert result.layer_diagnostics[0].max_absolute_residual == 0.0


def test_n2_two_precursor_condensation():
    model, experiment = _science(
        (("A", 1, False), ("B", 1, False), ("T", 2, False)),
        (_flux_reaction("R", ("A", "B"), ("T",)),),
        (_reaction("R", (("A", (1,)), ("B", (1,))), "T"),),
        (Tracer("A", (("#0", 0.5), ("#1", 0.5)), "no"), Tracer("B", (("#1", 1.0),), "no")),
        (_target("T", 2),),
    )
    result = evaluate_stationary(compile_emu_plan(model, experiment), {"R": 1}).forward
    assert result.predictions[0].fractions == (0.0, 0.5, 0.5)


def test_n3_mixed_producers_are_flux_weighted():
    model, experiment = _science(
        (("A", 1, False), ("B", 1, False), ("T", 1, False)),
        (_flux_reaction("RA", ("A",), ("T",)), _flux_reaction("RB", ("B",), ("T",))),
        (_reaction("RA", (("A", (1,)),), "T"), _reaction("RB", (("B", (1,)),), "T")),
        (Tracer("A", (("#0", 1.0),), "no"), Tracer("B", (("#1", 1.0),), "no")),
        (_target("T"),),
    )
    result = evaluate_stationary(compile_emu_plan(model, experiment), {"RA": 1, "RB": 3}).forward
    assert result.predictions[0].fractions == (0.25, 0.75)


def test_n4_same_size_coupled_layer_matches_independent_solution():
    metabolites = (("S0", 1, False), ("S1", 1, False), ("A", 1, True), ("B", 1, True),
                   ("OA", 1, False), ("OB", 1, False))
    specs = (("IN_A", ("S0",), ("A",)), ("IN_B", ("S1",), ("B",)),
             ("A_B", ("A",), ("B",)), ("B_A", ("B",), ("A",)),
             ("OUT_A", ("A",), ("OA",)), ("OUT_B", ("B",), ("OB",)))
    flux_reactions = tuple(_flux_reaction(*item) for item in specs)
    isotope_reactions = tuple(_reaction(rid, tuple((s, (1,)) for s in subs), products[0]) for rid, subs, products in specs)
    model, experiment = _science(
        metabolites, flux_reactions, isotope_reactions,
        (Tracer("S0", (("#0", 1.0),), "no"), Tracer("S1", (("#1", 1.0),), "no")),
        (_target("A"), _target("B")),
    )
    fluxes = {item[0]: 1.0 for item in specs}
    result = evaluate_stationary(compile_emu_plan(model, experiment), fluxes)
    np.testing.assert_allclose(result.forward.predictions[0].fractions, (2 / 3, 1 / 3), atol=1e-12)
    np.testing.assert_allclose(result.forward.predictions[1].fractions, (1 / 3, 2 / 3), atol=1e-12)
    assert result.layer_diagnostics[0].matrix_dimension == 2


def test_n5_explicit_weighted_branches_apply_once_and_retain_order():
    direct = (
        AtomTransition(AtomPosition("S", 1), AtomPosition("T", 1)),
        AtomTransition(AtomPosition("S", 2), AtomPosition("T", 2)),
    )
    reverse = (
        AtomTransition(AtomPosition("S", 2), AtomPosition("T", 1)),
        AtomTransition(AtomPosition("S", 1), AtomPosition("T", 2)),
    )
    branches = (MappingBranch("first", 0.25, direct), MappingBranch("second", 0.75, reverse))
    model, experiment = _science(
        (("S", 2, False), ("T", 2, False)), (_flux_reaction("R", ("S",), ("T",)),),
        (IsotopeReaction("R", "forward", True, (IsotopeParticipant("S", (1, 2)),),
                         (IsotopeParticipant("T", (1, 2)),), branches),),
        (Tracer("S", (("#10", 1.0),), "no"),),
        (Target("T-1", "T", (1,), "intermediate", "C", "no"),),
    )
    plan = compile_emu_plan(model, experiment)
    assert tuple(item.branch_id for item in plan.contributions) == ("first", "second")
    result = evaluate_stationary(plan, {"R": 1}).forward
    assert result.predictions[0].fractions == (0.75, 0.25)


def test_n6_invalid_unbalanced_nontracer_substrate_fails():
    model, experiment = _science(
        (("U", 1, False), ("T", 1, False)), (_flux_reaction("R", ("U",), ("T",)),),
        (_reaction("R", (("U", (1,)),), "T"),), (), (_target("T"),),
    )
    with pytest.raises(MappingError, match="unbalanced isotope substrate"):
        compile_emu_plan(model, experiment)


def test_n7_missing_production_mapping_fails():
    model, experiment = _science(
        (("S", 1, False), ("T", 1, False)),
        (_flux_reaction("mapped", ("S",), ("T",)), _flux_reaction("missing", ("S",), ("T",))),
        (_reaction("mapped", (("S", (1,)),), "T"),),
        (Tracer("S", (("#0", 1.0),), "no"),), (_target("T"),),
    )
    with pytest.raises(MappingError, match="without an explicit isotope mapping"):
        compile_emu_plan(model, experiment)


def test_n8_zero_flux_terminal_target_fails():
    model, experiment = _science(
        (("S", 1, False), ("T", 1, False)), (_flux_reaction("R", ("S",), ("T",)),),
        (_reaction("R", (("S", (1,)),), "T"),),
        (Tracer("S", (("#0", 1.0),), "no"),), (_target("T"),),
    )
    with pytest.raises(ForwardEMUError, match="zero productive flux"):
        evaluate_stationary(compile_emu_plan(model, experiment), {"R": 0})


def test_n9_singular_layer_fails_without_pseudoinverse():
    model, experiment = _science(
        (("S", 1, False), ("A", 1, True), ("O", 1, False)),
        (_flux_reaction("IN", ("S",), ("A",)), _flux_reaction("OUT", ("A",), ("O",))),
        (_reaction("IN", (("S", (1,)),), "A"), _reaction("OUT", (("A", (1,)),), "O")),
        (Tracer("S", (("#0", 1.0),), "no"),), (_target("A"),),
    )
    with pytest.raises(ForwardEMUError, match="singular"):
        evaluate_stationary(compile_emu_plan(model, experiment), {"IN": 0, "OUT": 0})
