from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fluxemu.emu import (
    compile_transient_emu_plan,
    convolve_mids,
    evaluate_stationary,
    evaluate_transient,
)
from fluxemu.exceptions import MappingError
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
    PoolQuantity,
    StationaryExperimentSemantics,
    StoichiometricTerm,
    Target,
    Tracer,
    TransientExperimentSemantics,
)


def _flux_reaction(reaction_id, substrates, products):
    return FluxReaction(
        reaction_id,
        tuple(StoichiometricTerm(item, -1) for item in substrates)
        + tuple(StoichiometricTerm(item, 1) for item in products),
        0,
        100,
    )


def _mapped_reaction(reaction_id, substrates, product, *, branches=None):
    atoms = tuple((source, position) for source, positions in substrates for position in positions)
    transitions = tuple(
        AtomTransition(AtomPosition(source, position), AtomPosition(product, destination))
        for destination, (source, position) in enumerate(atoms, 1)
    )
    return IsotopeReaction(
        reaction_id,
        "forward",
        True,
        tuple(IsotopeParticipant(source, tuple(positions)) for source, positions in substrates),
        (IsotopeParticipant(product, tuple(range(1, len(atoms) + 1))),),
        tuple(branches or (MappingBranch("primary", 1.0, transitions),)),
    )


def _target(metabolite_id, carbon_count=1, *, target_id=None, positions=None):
    return Target(
        target_id or metabolite_id,
        metabolite_id,
        tuple(positions or range(1, carbon_count + 1)),
        "intermediate",
        f"C{carbon_count}",
        "no",
    )


def _model(metabolites, reaction_specs, isotope_reactions=None):
    flux_reactions = tuple(_flux_reaction(*spec) for spec in reaction_specs)
    if isotope_reactions is None:
        counts = {item[0]: item[1] for item in metabolites}
        isotope_reactions = tuple(
            _mapped_reaction(
                rid,
                tuple((source, tuple(range(1, counts[source] + 1))) for source in substrates),
                products[0],
            )
            for rid, substrates, products in reaction_specs
        )
    return CanonicalModel(
        FluxModel(
            tuple(FluxMetabolite(mid, balanced) for mid, _, balanced in metabolites),
            flux_reactions,
            LinearObjective("maximise", ()),
        ),
        IsotopeModel(
            tuple(IsotopeMetabolite(mid, count, True, False) for mid, count, _ in metabolites),
            tuple(isotope_reactions),
        ),
    )


def _experiment(tracers, targets, times, pools, initial="unlabelled"):
    return TransientExperimentSemantics(
        tuple(tracers), tuple(targets), tuple(times),
        tuple(PoolQuantity(mid, quantity) for mid, quantity in pools), initial,
    )


def _trajectory(result, target_id):
    return np.asarray(
        [item.fractions for item in result.forward.predictions if item.target_id == target_id],
        dtype=float,
    )


def _single_pool(
    source_fractions, *, times=(0.0, 0.2, 1.0, 3.0), q=2.0, v=3.0, run=True
):
    model = _model(
        (("S", 1, False), ("A", 1, True), ("T", 1, False)),
        (("IN", ("S",), ("A",)), ("OUT", ("A",), ("T",))),
    )
    tracer = Tracer("S", tuple(source_fractions), "no")
    experiment = _experiment((tracer,), (_target("A"),), times, (("A", q),))
    result = (
        evaluate_transient(
            compile_transient_emu_plan(model, experiment), {"IN": v, "OUT": v}
        )
        if run
        else None
    )
    return model, experiment, result


def test_stationary_emu_import_isolated_from_transient_optional_backends():
    source = "import sys; import fluxemu.emu; assert 'scipy' not in sys.modules; assert 'mfapy' not in sys.modules; assert 'matplotlib' not in sys.modules"
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_t0_fully_labelled_step_matches_single_pool_exponential():
    _, experiment, result = _single_pool((("#1", 1.0),))
    time = np.asarray(experiment.time_points, dtype=float)
    labelled = 1.0 - np.exp(-(3.0 / 2.0) * time)
    np.testing.assert_allclose(_trajectory(result, "A"), np.column_stack((1 - labelled, labelled)), atol=2e-10)


def test_t1_partially_labelled_step_matches_exact_solution():
    p = 0.37
    _, experiment, result = _single_pool((("#0", 1 - p), ("#1", p)))
    time = np.asarray(experiment.time_points, dtype=float)
    labelled = p * (1.0 - np.exp(-(3.0 / 2.0) * time))
    np.testing.assert_allclose(_trajectory(result, "A"), np.column_stack((1 - labelled, labelled)), atol=2e-10)


def test_t2_two_serial_pools_match_independent_compartment_formula():
    times = np.asarray((0.0, 0.1, 0.7, 2.0, 5.0))
    model = _model(
        (("S", 1, False), ("A", 1, True), ("B", 1, True), ("T", 1, False)),
        (("IN", ("S",), ("A",)), ("AB", ("A",), ("B",)), ("OUT", ("B",), ("T",))),
    )
    experiment = _experiment(
        (Tracer("S", (("#1", 1.0),), "no"),), (_target("A"), _target("B")),
        times, (("A", 1.0), ("B", 2.0)),
    )
    result = evaluate_transient(
        compile_transient_emu_plan(model, experiment), {"IN": 2.0, "AB": 2.0, "OUT": 2.0}
    )
    k1, k2 = 2.0, 1.0
    a = 1.0 - np.exp(-k1 * times)
    b = 1.0 - (k2 * np.exp(-k1 * times) - k1 * np.exp(-k2 * times)) / (k2 - k1)
    np.testing.assert_allclose(_trajectory(result, "A")[:, 1], a, atol=3e-10)
    np.testing.assert_allclose(_trajectory(result, "B")[:, 1], b, atol=3e-10)


def test_t3_dynamic_condensation_is_same_time_discrete_convolution():
    times = np.asarray((0.0, 0.25, 1.0, 3.0))
    metabolites = (("S1", 1, False), ("S2", 1, False), ("A", 1, True), ("B", 1, True), ("P", 2, False), ("OA", 1, False), ("OB", 1, False))
    specs = (("IA", ("S1",), ("A",)), ("IB", ("S2",), ("B",)), ("OA", ("A",), ("OA",)), ("OB", ("B",), ("OB",)), ("COND", ("A", "B"), ("P",)))
    model = _model(metabolites, specs)
    experiment = _experiment(
        (Tracer("S1", (("#1", 1.0),), "no"), Tracer("S2", (("#0", 0.6), ("#1", 0.4)), "no")),
        (_target("P", 2),), times, (("A", 1.0), ("B", 2.0)),
    )
    fluxes = {"IA": 2.0, "IB": 2.0, "OA": 1.0, "OB": 1.0, "COND": 1.0}
    result = evaluate_transient(compile_transient_emu_plan(model, experiment), fluxes)
    a1 = 1 - np.exp(-2 * times)
    b1 = 0.4 * (1 - np.exp(-times))
    expected = np.asarray([convolve_mids(((1 - x, x), (1 - y, y))) for x, y in zip(a1, b1)])
    np.testing.assert_allclose(_trajectory(result, "P"), expected, atol=3e-10)


def test_t4_same_size_bidirectional_coupling_matches_matrix_exponential():
    from scipy.linalg import expm

    times = (0.0, 0.2, 1.0, 4.0)
    metabolites = (("S0", 1, False), ("S1", 1, False), ("A", 1, True), ("B", 1, True), ("OA", 1, False), ("OB", 1, False))
    specs = (("IA", ("S0",), ("A",)), ("IB", ("S1",), ("B",)), ("AB", ("A",), ("B",)), ("BA", ("B",), ("A",)), ("OA", ("A",), ("OA",)), ("OB", ("B",), ("OB",)))
    model = _model(metabolites, specs)
    experiment = _experiment(
        (Tracer("S0", (("#0", 1.0),), "no"), Tracer("S1", (("#1", 1.0),), "no")),
        (_target("A"), _target("B")), times, (("A", 1.0), ("B", 1.0)),
    )
    result = evaluate_transient(compile_transient_emu_plan(model, experiment), {item[0]: 1.0 for item in specs})
    matrix = np.asarray(((-2.0, 1.0), (1.0, -2.0)))
    steady = np.asarray((1 / 3, 2 / 3))
    expected = np.asarray([steady - expm(matrix * time) @ steady for time in times])
    actual = np.column_stack((_trajectory(result, "A")[:, 1], _trajectory(result, "B")[:, 1]))
    np.testing.assert_allclose(actual, expected, atol=4e-10)


def test_t5_explicit_weighted_mapping_branch_is_applied_once_over_time():
    direct = (AtomTransition(AtomPosition("S", 1), AtomPosition("A", 1)), AtomTransition(AtomPosition("S", 2), AtomPosition("A", 2)))
    reverse = (AtomTransition(AtomPosition("S", 2), AtomPosition("A", 1)), AtomTransition(AtomPosition("S", 1), AtomPosition("A", 2)))
    inlet = IsotopeReaction(
        "IN", "forward", True, (IsotopeParticipant("S", (1, 2)),),
        (IsotopeParticipant("A", (1, 2)),),
        (MappingBranch("direct", 0.25, direct), MappingBranch("reverse", 0.75, reverse)),
    )
    outlet = _mapped_reaction("OUT", (("A", (1, 2)),), "T")
    model = _model(
        (("S", 2, False), ("A", 2, True), ("T", 2, False)),
        (("IN", ("S",), ("A",)), ("OUT", ("A",), ("T",))), (inlet, outlet),
    )
    times = np.asarray((0.0, 0.3, 2.0))
    experiment = _experiment(
        (Tracer("S", (("#10", 1.0),), "no"),),
        (_target("A", target_id="A-C1", positions=(1,)),), times, (("A", 1.0),),
    )
    result = evaluate_transient(compile_transient_emu_plan(model, experiment), {"IN": 1.0, "OUT": 1.0})
    expected = 0.25 * (1 - np.exp(-times))
    np.testing.assert_allclose(_trajectory(result, "A-C1")[:, 1], expected, atol=2e-10)


def test_t6_time_zero_is_exactly_unlabelled_for_every_dynamic_emu():
    _, _, result = _single_pool((("#1", 1.0),), times=(0.0,))
    assert result.forward.predictions[0].fractions == (1.0, 0.0)
    assert result.diagnostics[0].rhs_evaluation_count == 0


def test_t7_missing_required_dynamic_pool_fails_explicitly():
    model, experiment, _ = _single_pool((("#1", 1.0),), run=False)
    with pytest.raises(MappingError, match="missing pool quantity.*A"):
        compile_transient_emu_plan(model, replace(experiment, pool_quantities=()))


@pytest.mark.parametrize("quantity", [0.0, -1.0, math.nan, math.inf])
def test_t8_invalid_pool_quantity_is_rejected(quantity):
    model, experiment, _ = _single_pool((("#1", 1.0),), run=False)
    with pytest.raises(CanonicalModelError, match="pool quantity"):
        compile_transient_emu_plan(
            model, replace(experiment, pool_quantities=(PoolQuantity("A", quantity),))
        )


@pytest.mark.parametrize("times", [(0.0, 0.0), (-1.0, 1.0), (0.0, math.nan), (0.0, math.inf)])
def test_t8_invalid_time_grid_is_rejected(times):
    model, experiment, _ = _single_pool((("#1", 1.0),), run=False)
    with pytest.raises(CanonicalModelError, match="time point"):
        compile_transient_emu_plan(model, replace(experiment, time_points=times))


def test_t9_long_time_limit_matches_native_stationary_solution():
    model, experiment, transient = _single_pool((("#0", 0.2), ("#1", 0.8)), times=(0.0, 30.0))
    stationary_experiment = StationaryExperimentSemantics(experiment.tracers, experiment.targets)
    stationary = evaluate_stationary(
        compile_transient_emu_plan(model, experiment).stationary_plan,
        {"IN": 3.0, "OUT": 3.0},
    )
    np.testing.assert_allclose(
        transient.forward.predictions[-1].fractions,
        stationary.forward.predictions[0].fractions,
        atol=2e-10,
    )
    assert stationary_experiment.targets == experiment.targets


def test_t10_terminal_target_is_instantaneous_production_not_accumulated_pool():
    times = np.asarray((0.0, 0.5, 1.0, 2.0))
    model = _model(
        (("S", 1, False), ("A", 1, True), ("T", 1, False)),
        (("IN", ("S",), ("A",)), ("OUT", ("A",), ("T",))),
    )
    experiment = _experiment(
        (Tracer("S", (("#1", 1.0),), "no"),),
        (_target("A"), _target("T")), times, (("A", 1.0),),
    )
    result = evaluate_transient(compile_transient_emu_plan(model, experiment), {"IN": 1.0, "OUT": 1.0})
    internal = _trajectory(result, "A")[:, 1]
    terminal = _trajectory(result, "T")[:, 1]
    np.testing.assert_allclose(terminal, internal, atol=2e-10)
    np.testing.assert_allclose(terminal, 1 - np.exp(-times), atol=2e-10)
    accumulated_external_pool = 1 - (1 + times) * np.exp(-times)
    assert abs(terminal[2] - accumulated_external_pool[2]) > 0.1
