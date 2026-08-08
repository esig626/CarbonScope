"""Regression tests for the exact Antoniewicz et al. TCA EMU benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from cobra.io import read_sbml_model


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "antoniewicz_tca"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from build_model import (  # noqa: E402
    CANONICAL_FLUX_BOUNDS,
    FIGURE12_PAGE,
    GROUND_TRUTH_FLUXES,
    MODEL_PATH,
    PUBLISHED_TABLE6_MID,
    STATIONARY_EXPERIMENT,
    TABLE5_BY_ID,
    TABLE5_REACTIONS,
    TIMEPOINTS,
    _fixed_flux_frame,
    build_table5_mfapy_bundle,
    run_stationary,
    run_timecourse,
    validate_table5_model,
)
from direct_isotopomer_solver import glutamate_mid, solve_stationary  # noqa: E402
from fluxemu.configuration import load_experiment  # noqa: E402
from fluxemu.forward import run_batch_forward  # noqa: E402
from fluxemu.isotope_metadata import collect_isotope_metadata  # noqa: E402


PAPER_TOLERANCE = 5.1e-5


@pytest.fixture(scope="module")
def table5_model():
    model = read_sbml_model(str(MODEL_PATH))
    validate_table5_model(model)
    return model


def test_every_atom_transition_is_the_exact_table5_transcription(table5_model) -> None:
    assert tuple(reaction.id for reaction in table5_model.reactions) == tuple(
        item.reaction_id for item in TABLE5_REACTIONS
    )
    metadata = collect_isotope_metadata(table5_model)
    for spec in TABLE5_REACTIONS:
        item = metadata.reactions[spec.reaction_id]
        observed = (
            tuple((part.metabolite_id, "".join(part.atom_labels)) for part in item.substrates),
            tuple((part.metabolite_id, "".join(part.atom_labels)) for part in item.products),
        )
        assert observed == (spec.substrates, spec.products)
        assert item.direction == "forward"
        reaction = table5_model.reactions.get_by_id(spec.reaction_id)
        assert reaction.bounds == CANONICAL_FLUX_BOUNDS
        assert GROUND_TRUTH_FLUXES[spec.reaction_id] == pytest.approx(
            _fixed_flux_frame(table5_model).iloc[0][spec.reaction_id]
        )
    assert TABLE5_BY_ID["v5"].printed_transition == (
        "0.5 abcd + 0.5 dcba -> 0.5 abcd + 0.5 dcba"
    )
    assert TABLE5_BY_ID["v6"].printed_transition == "0.5 abcd + 0.5 dcba -> abcd"
    assert TABLE5_BY_ID["v7"].printed_transition == "abcd -> 0.5 abcd + 0.5 dcba"
    assert FIGURE12_PAGE == "paper page 29, Figure 12"


def test_carbon_conservation_and_symmetric_half_branches(table5_model) -> None:
    metadata = collect_isotope_metadata(table5_model)
    for reaction_id, item in metadata.reactions.items():
        assert set(item.substrate_atom_labels) == set(item.product_atom_labels)
        if reaction_id in {"v2", "v4"}:
            assert any(part.metabolite_id == "CO2" for part in item.products)
        else:
            assert all(part.metabolite_id != "CO2" for part in item.products)
    assert metadata.metabolites["succinate"].symmetry is True
    assert metadata.metabolites["fumarate"].symmetry is True


def test_stationary_fluxemu_mfapy_direct_and_table6_agree() -> None:
    _, mfapy_mid, _ = run_stationary()
    direct_mid = glutamate_mid()
    for values in (mfapy_mid, direct_mid):
        assert np.isfinite(values).all()
        assert np.all(values >= 0.0)
        assert values.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(mfapy_mid, direct_mid, rtol=0.0, atol=1e-12)
    assert np.allclose(mfapy_mid, PUBLISHED_TABLE6_MID, rtol=0.0, atol=PAPER_TOLERANCE)
    assert np.all(mfapy_mid[3:] > 0.0)


def test_direct_solver_enumerates_the_full_published_state_space() -> None:
    solution = solve_stationary()
    assert {name: len(values) for name, values in solution.items()} == {
        "OAC": 16,
        "citrate": 64,
        "AKG": 32,
        "glutamate": 32,
        "succinate": 16,
        "fumarate": 16,
    }
    assert all(np.isfinite(values).all() and np.all(values >= 0.0) for values in solution.values())
    assert all(values.sum() == pytest.approx(1.0, abs=1e-12) for values in solution.values())


def test_removing_symmetric_reverse_orientation_breaks_the_published_mid(table5_model) -> None:
    experiment = load_experiment(STATIONARY_EXPERIMENT)
    without_symmetry = build_table5_mfapy_bundle(
        table5_model, experiment, symmetric=False
    )
    result = run_batch_forward(
        without_symmetry, _fixed_flux_frame(table5_model), experiment
    )
    unsymmetrical_mid = np.asarray(
        result.predictions["figure12"]["glutamate"], dtype=float
    )
    assert np.allclose(
        unsymmetrical_mid, glutamate_mid(symmetric=False), rtol=0.0, atol=1e-12
    )
    assert not np.allclose(
        unsymmetrical_mid, PUBLISHED_TABLE6_MID, rtol=0.0, atol=PAPER_TOLERANCE
    )


def test_diffmdv_timecourse_is_valid_and_converges_to_stationary() -> None:
    timecourse = run_timecourse()
    assert set(timecourse["time"]) == set(TIMEPOINTS)
    assert np.isfinite(timecourse["fraction"].to_numpy()).all()
    assert (timecourse["fraction"] >= -1e-12).all()
    grouped = timecourse.groupby(["time", "metabolite"])["fraction"].sum()
    assert np.allclose(grouped.to_numpy(), 1.0, rtol=0.0, atol=1e-9)

    zero = timecourse[timecourse["time"] == 0.0]
    assert (zero[zero["mass_isotopologue"] == "M+0"]["fraction"] == 1.0).all()
    assert (zero[zero["mass_isotopologue"] != "M+0"]["fraction"] == 0.0).all()

    early = timecourse[(timecourse["time"] == 0.001) & (timecourse["metabolite"] == "glutamate")]
    assert early.loc[early["mass_isotopologue"].isin(["M+1", "M+2"]), "fraction"].sum() > 0.0
    assert early.loc[early["mass_isotopologue"].isin(["M+3", "M+4", "M+5"]), "fraction"].sum() < 1e-12

    recirculated = timecourse[(timecourse["time"] == 0.05) & (timecourse["metabolite"] == "glutamate")]
    assert (recirculated.loc[recirculated["mass_isotopologue"].isin(["M+3", "M+4", "M+5"]), "fraction"] > 0.0).all()

    late = timecourse[(timecourse["time"] == 2.0) & (timecourse["metabolite"] == "glutamate")]
    late_mid = late.sort_values("mass_isotopologue")["fraction"].to_numpy()
    assert np.allclose(late_mid, glutamate_mid(), rtol=0.0, atol=2e-6)


def test_checked_csv_outputs_match_the_calculated_stationary_mid() -> None:
    stationary = pd.read_csv(EXAMPLE_DIR / "stationary_mids.csv")
    assert np.allclose(stationary["fluxemu_mfapy"], glutamate_mid(), rtol=0.0, atol=1e-12)
    assert np.allclose(stationary["direct_isotopomer"], glutamate_mid(), rtol=0.0, atol=1e-12)
