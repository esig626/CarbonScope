"""Hard compatibility gates for shared prepared native HiGHS regions."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math

import pandas as pd
import pytest

import fluxemu.flux_analysis.highs as highs
from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis import (
    prepare_highs_flux_region,
    run_highs_fva_reference,
    run_highs_vffva,
    run_prepared_highs_vffva,
)
from fluxemu.model import (
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    LinearObjective,
    ObjectiveTerm,
    StoichiometricTerm,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("highspy") is None, reason="highspy is unavailable"
)


def _coupled_model() -> FluxModel:
    return FluxModel(
        (FluxMetabolite("A", True),),
        (
            FluxReaction("source", (StoichiometricTerm("A", 1.0),), 0.0, 10.0),
            FluxReaction("sink", (StoichiometricTerm("A", -1.0),), 0.0, 10.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("sink", 1.0),)),
    )


def _with_source_model(prepared, model: FluxModel):
    """Forge a self-consistent source record while retaining the old compiled LP."""

    return replace(
        prepared,
        flux_model=model,
        flux_model_fingerprint=highs._flux_model_fingerprint(model),
    )


def _wrong_sign_model(direction: str) -> tuple[FluxModel, float]:
    if direction == "maximise":
        bounds = (-10.0, -2.0)
        optimum = -2.0
    else:
        bounds = (2.0, 10.0)
        optimum = 2.0
    return (
        FluxModel(
            (),
            (FluxReaction("objective", (), *bounds),),
            LinearObjective(direction, (ObjectiveTerm("objective", 1.0),)),
        ),
        optimum,
    )


def test_prepared_region_rejects_forged_source_fingerprint() -> None:
    prepared = prepare_highs_flux_region(_coupled_model())
    forged = replace(prepared, flux_model_fingerprint="0" * 64)

    with pytest.raises(AnalysisError, match="canonical model fingerprint"):
        run_prepared_highs_vffva(forged, workers=1)


def test_prepared_region_rejects_forged_compiled_lp_fingerprint() -> None:
    prepared = prepare_highs_flux_region(_coupled_model())
    forged = replace(
        prepared,
        lp=replace(prepared.lp, fingerprint="0" * 64),
    )

    with pytest.raises(AnalysisError, match="compiled-LP fingerprint"):
        run_prepared_highs_vffva(forged, workers=1)


def test_prepared_region_rejects_subtle_source_stoichiometry_mismatch() -> None:
    prepared = prepare_highs_flux_region(_coupled_model())
    changed_sink = replace(
        prepared.flux_model.reactions[1],
        stoichiometric_terms=(StoichiometricTerm("A", -2.0),),
    )
    changed_model = replace(
        prepared.flux_model,
        reactions=(prepared.flux_model.reactions[0], changed_sink),
    )

    with pytest.raises(AnalysisError, match="stoichiometry.*compiled LP"):
        run_prepared_highs_vffva(
            _with_source_model(prepared, changed_model), workers=1
        )


@pytest.mark.parametrize("mismatch", ["objective", "bounds", "order"])
def test_prepared_region_rejects_source_model_metadata_mismatch(mismatch: str) -> None:
    prepared = prepare_highs_flux_region(_coupled_model())
    model = prepared.flux_model
    if mismatch == "objective":
        changed_model = replace(
            model,
            objective=LinearObjective(
                "maximise", (ObjectiveTerm("sink", 2.0),)
            ),
        )
        message = "objective.*compiled LP"
    elif mismatch == "bounds":
        changed_source = replace(model.reactions[0], upper_bound=9.0)
        changed_model = replace(
            model, reactions=(changed_source, model.reactions[1])
        )
        message = "canonical model.*compiled LP"
    else:
        changed_model = replace(model, reactions=tuple(reversed(model.reactions)))
        message = "canonical model.*compiled LP"

    with pytest.raises(AnalysisError, match=message):
        run_prepared_highs_vffva(
            _with_source_model(prepared, changed_model), workers=1
        )


def test_prepared_region_rejects_non_series_fba_primal_contextually() -> None:
    prepared = prepare_highs_flux_region(_coupled_model())
    forged_fba = replace(prepared.fba, fluxes=tuple(prepared.fba.fluxes))

    with pytest.raises(AnalysisError, match="FBA fluxes must be a pandas Series"):
        run_prepared_highs_vffva(replace(prepared, fba=forged_fba), workers=1)


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_prepared_region_rejects_nonfinite_fba_primal(nonfinite: float) -> None:
    prepared = prepare_highs_flux_region(_coupled_model())
    fluxes = prepared.fba.fluxes.copy()
    fluxes.iloc[0] = nonfinite
    assert isinstance(fluxes, pd.Series)
    forged_fba = replace(prepared.fba, fluxes=fluxes)

    with pytest.raises(AnalysisError, match="FBA primal contains non-finite"):
        run_prepared_highs_vffva(replace(prepared, fba=forged_fba), workers=1)


@pytest.mark.parametrize(
    ("direction", "short_direction"),
    (("maximise", "max"), ("minimise", "min")),
)
def test_wrong_sign_fraction_fails_before_any_fva_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    short_direction: str,
) -> None:
    model, _ = _wrong_sign_model(direction)
    operations: list[str] = []
    original_solve = highs._solve

    def recording_solve(lp, costs, solve_direction, operation, retention=None):
        operations.append(operation)
        return original_solve(
            lp, costs, solve_direction, operation, retention=retention
        )

    monkeypatch.setattr(highs, "_solve", recording_solve)
    message = rf"retained objective region is empty: direction={short_direction}"

    with pytest.raises(AnalysisError, match=message):
        run_highs_fva_reference(model, 0.9)
    assert operations == ["FBA"]

    operations.clear()
    with pytest.raises(AnalysisError, match=message):
        run_highs_vffva(model, 0.9, workers=1)
    assert operations == ["FBA"]


@pytest.mark.parametrize("direction", ["maximise", "minimise"])
def test_wrong_sign_objective_fraction_one_remains_a_valid_optimal_face(
    direction: str,
) -> None:
    model, optimum = _wrong_sign_model(direction)

    reference = run_highs_fva_reference(model, 1.0)
    fast = run_highs_vffva(model, 1.0, workers=1)

    assert reference.objective_value == pytest.approx(optimum)
    assert fast.objective_value == pytest.approx(optimum)
    assert reference.ranges.loc["objective"].tolist() == pytest.approx(
        [optimum, optimum]
    )
    pd.testing.assert_frame_equal(fast.ranges, reference.ranges, check_exact=True)


def test_default_fast_fva_never_requests_multiprocessing_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _coupled_model()
    prepared = prepare_highs_flux_region(model)

    def forbidden_context(*args, **kwargs):
        pytest.fail("default FastFVA unexpectedly requested multiprocessing")

    monkeypatch.setattr(highs.multiprocessing, "get_context", forbidden_context)

    # Omitting workers and explicitly passing the documented sentinel must both
    # select the benchmark-proven serial reusable path.
    run_prepared_highs_vffva(prepared)
    run_prepared_highs_vffva(prepared, workers=None)
    run_highs_vffva(model)
    run_highs_vffva(model, workers=None)
