from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fluxemu.forward as forward_module
from fluxemu._mfapy import load_mfapy
from fluxemu.cobra_analysis import sample_fluxes
from fluxemu.cobra_to_mfapy import build_mfapy_model
from fluxemu.configuration import load_experiment
from fluxemu.exceptions import ForwardEMUError, ValidationError
from fluxemu.forward import MID_COLUMNS, run_batch_forward
from fluxemu.isotope_metadata import collect_isotope_metadata
from fluxemu.toy import build_toy_model


EXPERIMENT_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "toy_experiment.yaml"
)


@pytest.fixture(scope="module")
def toy_case() -> SimpleNamespace:
    model = build_toy_model()
    experiment = load_experiment(EXPERIMENT_PATH)
    sampling = sample_fluxes(
        model,
        count=3,
        fraction=experiment.fraction_of_optimum,
        seed=experiment.seed,
        sampler=experiment.sampler,
        tolerances=experiment.tolerances,
    )
    assert sampling.validation.valid
    bundle = build_mfapy_model(
        model, collect_isotope_metadata(model), experiment
    )
    return SimpleNamespace(
        model=model,
        experiment=experiment,
        samples=sampling.samples,
        bundle=bundle,
    )


@pytest.fixture(scope="module")
def forward_result(toy_case: SimpleNamespace):
    return run_batch_forward(
        toy_case.bundle, toy_case.samples, toy_case.experiment
    )


def test_batch_forward_mids_are_finite_nonnegative_and_normalized(
    forward_result,
) -> None:
    for targets in forward_result.predictions.values():
        assert "X_list" not in targets
        for mid in targets.values():
            values = np.asarray(mid, dtype=float)
            assert np.isfinite(values).all()
            assert np.all(values >= -1e-8)
            assert np.all(values <= 1.0 + 1e-8)
            assert values.sum() == pytest.approx(1.0, abs=1e-8)

    assert forward_result.validation["valid"] is True
    assert forward_result.validation["max_normalization_error"] <= 1e-8


def test_batch_forward_preserves_batch_and_sample_ids(
    toy_case: SimpleNamespace, forward_result
) -> None:
    expected_sample_ids = list(toy_case.samples.index)
    assert list(forward_result.predictions) == expected_sample_ids
    assert forward_result.validation["sample_count"] == len(toy_case.samples)
    assert forward_result.validation["sample_ids"] == expected_sample_ids
    assert list(forward_result.mids.columns) == MID_COLUMNS
    assert forward_result.mids["sample_id"].drop_duplicates().tolist() == (
        expected_sample_ids
    )
    assert set(forward_result.mids["target_fragment_id"]) == {"Glue"}
    assert len(forward_result.mids) == len(toy_case.samples) * 6


def test_one_mfapy_model_is_constructed_for_the_whole_batch(
    toy_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_toy_model()
    experiment = toy_case.experiment
    mfapy = load_mfapy()
    original_constructor = mfapy.metabolicmodel.MetabolicModel
    construction_count = 0

    def counted_constructor(*args, **kwargs):
        nonlocal construction_count
        construction_count += 1
        return original_constructor(*args, **kwargs)

    monkeypatch.setattr(
        mfapy.metabolicmodel, "MetabolicModel", counted_constructor
    )
    bundle = build_mfapy_model(
        model, collect_isotope_metadata(model), experiment
    )
    generated_function = bundle.model.func["calmdv"]

    result = run_batch_forward(bundle, toy_case.samples, experiment)

    assert construction_count == 1
    assert bundle.model.func["calmdv"] is generated_function
    assert result.validation["sample_count"] == len(toy_case.samples)


def test_missing_predicted_fragment_fails_validation(
    toy_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    mfapy = load_mfapy()

    def missing_fragment(*args, **kwargs):
        return np.asarray([], dtype=float), {"X_list": []}

    monkeypatch.setattr(mfapy.optimize, "calc_MDV_from_flux", missing_fragment)

    with pytest.raises(ValidationError, match="missing requested target"):
        run_batch_forward(
            toy_case.bundle,
            toy_case.samples.iloc[:1].copy(),
            toy_case.experiment,
        )


def test_mfapy_calculation_error_is_wrapped(
    toy_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    mfapy = load_mfapy()

    def failed_calculation(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(
        mfapy.optimize, "calc_MDV_from_flux", failed_calculation
    )

    with pytest.raises(ForwardEMUError, match="synthetic solver failure"):
        run_batch_forward(
            toy_case.bundle,
            toy_case.samples.iloc[:1].copy(),
            toy_case.experiment,
        )
