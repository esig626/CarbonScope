"""Independent published Antoniewicz TCA validation for the native EMU engine."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.model import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    IsotopeMetabolite,
    IsotopeModel,
    IsotopeParticipant,
    IsotopeReaction,
    LinearObjective,
    MappingBranch,
    StationaryExperimentSemantics,
    StoichiometricTerm,
    Target,
    Tracer,
)


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "antoniewicz_tca"
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))
from direct_isotopomer_solver import FLUXES, glutamate_mid  # noqa: E402

PUBLISHED_TABLE6_MID = np.asarray((0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039))
PAPER_TOLERANCE = 5.1e-5

CARBONS = {
    "OAC": 4,
    "AcCoA": 2,
    "citrate": 6,
    "AKG": 5,
    "glutamate": 5,
    "succinate": 4,
    "fumarate": 4,
    "aspartate": 4,
    "CO2": 1,
}


def _transitions(source: str, product: str, destinations: tuple[int, ...]):
    return tuple(
        AtomTransition(AtomPosition(source, source_position), AtomPosition(product, product_position))
        for source_position, product_position in enumerate(destinations, 1)
    )


def _reaction(
    reaction_id: str,
    substrates: tuple[str, ...],
    products: tuple[str, ...],
    branches: tuple[MappingBranch, ...],
) -> IsotopeReaction:
    return IsotopeReaction(
        reaction_id,
        "forward",
        True,
        tuple(IsotopeParticipant(item, tuple(range(1, CARBONS[item] + 1))) for item in substrates),
        tuple(IsotopeParticipant(item, tuple(range(1, CARBONS[item] + 1))) for item in products),
        branches,
    )


def _identity_branch(branch_id: str, source: str, product: str, weight=1.0):
    return MappingBranch(
        branch_id,
        weight,
        tuple(
            AtomTransition(AtomPosition(source, index), AtomPosition(product, index))
            for index in range(1, CARBONS[source] + 1)
        ),
    )


def _model() -> CanonicalModel:
    balanced = {"OAC", "citrate", "AKG", "succinate", "fumarate"}
    flux_metabolites = tuple(
        FluxMetabolite(item, item in balanced) for item in CARBONS
    )

    reaction_specs = (
        ("v1", ("OAC", "AcCoA"), ("citrate",)),
        ("v2", ("citrate",), ("AKG", "CO2")),
        ("v3", ("AKG",), ("glutamate",)),
        ("v4", ("AKG",), ("succinate", "CO2")),
        ("v5", ("succinate",), ("fumarate",)),
        ("v6", ("fumarate",), ("OAC",)),
        ("v7", ("OAC",), ("fumarate",)),
        ("v8", ("aspartate",), ("OAC",)),
    )
    flux_reactions = tuple(
        FluxReaction(
            reaction_id,
            tuple(StoichiometricTerm(item, -1) for item in substrates)
            + tuple(StoichiometricTerm(item, 1) for item in products),
            0,
            1_000_000,
        )
        for reaction_id, substrates, products in reaction_specs
    )

    v1 = MappingBranch(
        "published",
        1.0,
        (
            AtomTransition(AtomPosition("OAC", 1), AtomPosition("citrate", 6)),
            AtomTransition(AtomPosition("OAC", 2), AtomPosition("citrate", 3)),
            AtomTransition(AtomPosition("OAC", 3), AtomPosition("citrate", 2)),
            AtomTransition(AtomPosition("OAC", 4), AtomPosition("citrate", 1)),
            AtomTransition(AtomPosition("AcCoA", 1), AtomPosition("citrate", 5)),
            AtomTransition(AtomPosition("AcCoA", 2), AtomPosition("citrate", 4)),
        ),
    )
    v2 = MappingBranch(
        "published",
        1.0,
        tuple(
            AtomTransition(AtomPosition("citrate", index), AtomPosition("AKG", index))
            for index in range(1, 6)
        )
        + (AtomTransition(AtomPosition("citrate", 6), AtomPosition("CO2", 1)),),
    )
    v4 = MappingBranch(
        "published",
        1.0,
        (AtomTransition(AtomPosition("AKG", 1), AtomPosition("CO2", 1)),)
        + tuple(
            AtomTransition(AtomPosition("AKG", index + 1), AtomPosition("succinate", index))
            for index in range(1, 5)
        ),
    )

    def symmetric(source: str, product: str):
        return (
            _identity_branch("canonical_orientation", source, product, 0.5),
            MappingBranch(
                "reversed_orientation",
                0.5,
                tuple(
                    AtomTransition(
                        AtomPosition(source, index),
                        AtomPosition(product, CARBONS[product] + 1 - index),
                    )
                    for index in range(1, CARBONS[source] + 1)
                ),
            ),
        )

    isotope_reactions = (
        _reaction("v1", ("OAC", "AcCoA"), ("citrate",), (v1,)),
        _reaction("v2", ("citrate",), ("AKG", "CO2"), (v2,)),
        _reaction("v3", ("AKG",), ("glutamate",), (_identity_branch("published", "AKG", "glutamate"),)),
        _reaction("v4", ("AKG",), ("succinate", "CO2"), (v4,)),
        _reaction("v5", ("succinate",), ("fumarate",), symmetric("succinate", "fumarate")),
        _reaction("v6", ("fumarate",), ("OAC",), symmetric("fumarate", "OAC")),
        _reaction("v7", ("OAC",), ("fumarate",), symmetric("OAC", "fumarate")),
        _reaction("v8", ("aspartate",), ("OAC",), (_identity_branch("published", "aspartate", "OAC"),)),
    )

    isotope_metabolites = tuple(
        IsotopeMetabolite(
            item,
            count,
            True,
            item in {"succinate", "fumarate"},
        )
        for item, count in CARBONS.items()
    )
    return CanonicalModel(
        FluxModel(flux_metabolites, flux_reactions, LinearObjective("maximise", ())),
        IsotopeModel(isotope_metabolites, isotope_reactions),
    )


def _experiment() -> StationaryExperimentSemantics:
    return StationaryExperimentSemantics(
        tracers=(
            Tracer("AcCoA", (("#00", 0.50), ("#01", 0.25), ("#11", 0.25)), "no"),
            Tracer("aspartate", (("#0000", 1.0),), "no"),
        ),
        targets=(Target("glutamate", "glutamate", (1, 2, 3, 4, 5), "native", "C5H9NO4", "no"),),
    )


def test_native_emu_matches_independent_full_isotopomer_solver() -> None:
    native = evaluate_stationary(compile_emu_plan(_model(), _experiment()), FLUXES)
    actual = np.asarray(native.forward.predictions[0].fractions)
    independent = np.asarray(glutamate_mid())
    np.testing.assert_allclose(actual, independent, rtol=0.0, atol=1e-12)
    assert actual.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(actual >= 0.0)


def test_native_emu_matches_published_table6_mid() -> None:
    actual = np.asarray(
        evaluate_stationary(compile_emu_plan(_model(), _experiment()), FLUXES)
        .forward.predictions[0].fractions
    )
    np.testing.assert_allclose(actual, PUBLISHED_TABLE6_MID, rtol=0.0, atol=PAPER_TOLERANCE)
    assert np.all(actual[3:] > 0.0)


def test_symmetry_branches_are_scientifically_material() -> None:
    correct = np.asarray(glutamate_mid())
    corrupted = np.asarray(glutamate_mid(symmetric=False))
    assert np.max(np.abs(correct - corrupted)) > 1e-3
