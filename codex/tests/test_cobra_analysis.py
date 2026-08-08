"""Tests for constrained stable-COBRApy flux analysis."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cobra import Metabolite, Model, Reaction

import fluxemu.cobra_analysis as analysis
from fluxemu.cobra_analysis import (
    OBJECTIVE_FLOOR_CONSTRAINT,
    run_fba,
    run_fva,
    sample_fluxes,
    validate_flux_samples,
)
from fluxemu.exceptions import AnalysisError


@pytest.fixture
def tolerances() -> SimpleNamespace:
    return SimpleNamespace(bounds=1e-7, mass_balance=1e-7, objective_floor=1e-7)


def _build_toy_model() -> Model:
    model = Model("fluxemu_sampling_toy")
    a = Metabolite("a_c", compartment="c")
    b = Metabolite("b_c", compartment="c")

    source = Reaction("SOURCE")
    source.bounds = (0.0, 10.0)
    source.add_metabolites({a: 1.0})

    path_1 = Reaction("PATH_1")
    path_1.bounds = (0.0, 10.0)
    path_1.add_metabolites({a: -1.0, b: 1.0})

    path_2 = Reaction("PATH_2")
    path_2.bounds = (0.0, 10.0)
    path_2.add_metabolites({a: -1.0, b: 1.0})

    biomass = Reaction("BIOMASS")
    biomass.bounds = (0.0, 10.0)
    biomass.add_metabolites({b: -1.0})

    model.add_reactions([source, path_1, path_2, biomass])
    model.objective = biomass
    model.objective_direction = "max"
    return model


@pytest.fixture
def toy_model() -> Model:
    """Return a bounded model with a two-dimensional feasible optimum region."""

    return _build_toy_model()


def test_run_fba_requires_and_returns_optimal_finite_solution(toy_model: Model) -> None:
    result = run_fba(toy_model)

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(10.0)
    assert result.objective_direction == "max"
    assert list(result.fluxes.index) == [reaction.id for reaction in toy_model.reactions]
    assert np.isfinite(result.fluxes.to_numpy()).all()
    assert list(result.to_frame().columns) == [
        "flux",
        "objective_value",
        "solver_status",
    ]


def test_run_fba_rejects_nonoptimal_model() -> None:
    model = Model("infeasible")
    metabolite = Metabolite("blocked_c", compartment="c")
    demand = Reaction("DEMAND")
    demand.bounds = (1.0, 2.0)
    demand.add_metabolites({metabolite: -1.0})
    model.add_reactions([demand])
    model.objective = demand

    with pytest.warns(UserWarning, match="Solver status is 'infeasible'"):
        with pytest.raises(AnalysisError, match="did not return an optimal solution"):
            run_fba(model)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, np.nan, np.inf, True])
def test_run_fva_rejects_invalid_fraction(toy_model: Model, fraction: float) -> None:
    with pytest.raises(AnalysisError, match="fraction of optimum"):
        run_fva(toy_model, fraction)


def test_run_fva_uses_fraction_of_optimum(toy_model: Model) -> None:
    result = run_fva(toy_model, 0.9)

    assert list(result.ranges.index) == [reaction.id for reaction in toy_model.reactions]
    assert list(result.ranges.columns) == ["minimum", "maximum"]
    assert result.fraction_of_optimum == 0.9
    assert result.objective_value == pytest.approx(10.0)
    assert result.ranges.loc["BIOMASS", "minimum"] == pytest.approx(9.0)
    assert result.ranges.loc["BIOMASS", "maximum"] == pytest.approx(10.0)
    assert np.isfinite(result.ranges.to_numpy()).all()


def test_sampling_adds_floor_before_achr_construction_and_keeps_it(
    toy_model: Model,
    tolerances: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sampler = analysis.ACHRSampler
    observed: dict[str, float | None] = {}

    def checked_sampler(model: Model, **kwargs: object) -> object:
        constraint = model.constraints[OBJECTIVE_FLOOR_CONSTRAINT]
        observed["lb"] = constraint.lb
        observed["ub"] = constraint.ub
        return original_sampler(model, **kwargs)

    monkeypatch.setattr(analysis, "ACHRSampler", checked_sampler)
    result = sample_fluxes(toy_model, 4, 0.9, 17, "achr", tolerances)

    assert observed == {"lb": pytest.approx(9.0), "ub": None}
    assert OBJECTIVE_FLOOR_CONSTRAINT in toy_model.constraints
    floor = toy_model.constraints[OBJECTIVE_FLOOR_CONSTRAINT]
    assert floor.lb == pytest.approx(9.0)
    assert floor.ub is None
    assert result.objective_floor == pytest.approx(9.0)
    assert result.validation.valid
    assert list(result.samples.index) == [
        "sample_0000",
        "sample_0001",
        "sample_0002",
        "sample_0003",
    ]
    assert list(result.samples.columns) == [
        reaction.id for reaction in toy_model.reactions
    ]


def test_optgp_is_forced_to_one_process(
    toy_model: Model,
    tolerances: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sampler = analysis.OptGPSampler
    observed: dict[str, int] = {}

    def checked_sampler(model: Model, **kwargs: object) -> object:
        observed["processes"] = int(kwargs["processes"])
        return original_sampler(model, **kwargs)

    monkeypatch.setattr(analysis, "OptGPSampler", checked_sampler)
    result = sample_fluxes(toy_model, 2, 0.9, 19, "optgp", tolerances)

    assert observed["processes"] == 1
    assert len(result.samples) == 2
    assert result.validation.valid


def test_sampling_seed_is_reproducible(tolerances: SimpleNamespace) -> None:
    first_model = _build_toy_model()
    second_model = _build_toy_model()

    first = sample_fluxes(first_model, 5, 0.9, 42, "achr", tolerances)
    second = sample_fluxes(second_model, 5, 0.9, 42, "achr", tolerances)

    pd.testing.assert_frame_equal(first.samples, second.samples)


def _controlled_samples() -> pd.DataFrame:
    return pd.DataFrame(
        [[9.5, 4.0, 5.5, 9.5]],
        index=pd.Index(["sample_0000"], name="sample_id"),
        columns=["SOURCE", "PATH_1", "PATH_2", "BIOMASS"],
    )


def test_validation_accepts_finite_balanced_bounded_floor_sample(
    toy_model: Model, tolerances: SimpleNamespace
) -> None:
    report = validate_flux_samples(toy_model, _controlled_samples(), 9.0, tolerances)

    assert report.valid
    assert report.finite_values_valid
    assert report.bounds_valid
    assert report.mass_balance_valid
    assert report.objective_floor_valid
    assert report.max_mass_balance_residual == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("row", "invalid_field"),
    [
        ([np.inf, 4.0, 5.5, 9.5], "finite_values_valid"),
        ([10.5, 4.0, 5.5, 9.5], "bounds_valid"),
        ([9.5, 4.25, 5.5, 9.5], "mass_balance_valid"),
        ([8.0, 3.0, 5.0, 8.0], "objective_floor_valid"),
    ],
)
def test_validation_reports_numerical_failures(
    toy_model: Model,
    tolerances: SimpleNamespace,
    row: list[float],
    invalid_field: str,
) -> None:
    samples = _controlled_samples()
    samples.iloc[0] = row

    report = validate_flux_samples(toy_model, samples, 9.0, tolerances)

    assert not report.valid
    assert not getattr(report, invalid_field)


def test_validation_rejects_duplicate_missing_and_out_of_order_columns(
    toy_model: Model, tolerances: SimpleNamespace
) -> None:
    duplicate = _controlled_samples()
    duplicate.columns = ["SOURCE", "PATH_1", "PATH_1", "BIOMASS"]
    duplicate_report = validate_flux_samples(toy_model, duplicate, 9.0, tolerances)
    assert not duplicate_report.valid
    assert duplicate_report.duplicate_columns == ("PATH_1",)
    assert duplicate_report.missing_reactions == ("PATH_2",)

    reordered = _controlled_samples()[["BIOMASS", "PATH_2", "PATH_1", "SOURCE"]]
    reordered_report = validate_flux_samples(toy_model, reordered, 9.0, tolerances)
    assert not reordered_report.valid
    assert not reordered_report.column_order_valid

    unexpected = _controlled_samples().rename(columns={"PATH_2": "UNKNOWN"})
    unexpected_report = validate_flux_samples(toy_model, unexpected, 9.0, tolerances)
    assert not unexpected_report.valid
    assert unexpected_report.missing_reactions == ("PATH_2",)
    assert unexpected_report.unexpected_reactions == ("UNKNOWN",)
