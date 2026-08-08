from __future__ import annotations

import json

import numpy as np
import pytest

from fluxemu.exceptions import ValidationError
from fluxemu.validation import validate_mid_batch


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_mid_finite_values_are_required(invalid_value: float) -> None:
    predictions = {"sample-1": {"fragment": [0.5, invalid_value]}}

    with pytest.raises(ValidationError, match="non-finite"):
        validate_mid_batch(predictions, {"fragment": 2}, tolerance=1e-9)


def test_mid_nonnegativity_is_required() -> None:
    predictions = {"sample-1": {"fragment": [1.01, -0.01]}}

    with pytest.raises(ValidationError, match="negative"):
        validate_mid_batch(predictions, {"fragment": 2}, tolerance=1e-6)


def test_mid_values_cannot_exceed_one() -> None:
    predictions = {"sample-1": {"fragment": [1.01, 0.0]}}

    with pytest.raises(ValidationError, match="above one"):
        validate_mid_batch(predictions, {"fragment": 2}, tolerance=1e-6)


def test_mid_normalization_is_required() -> None:
    predictions = {"sample-1": {"fragment": [0.6, 0.5]}}

    with pytest.raises(ValidationError, match="not normalized"):
        validate_mid_batch(predictions, {"fragment": 2}, tolerance=1e-9)


def test_requested_fragment_must_be_present() -> None:
    predictions = {"sample-1": {"other-fragment": [1.0]}}

    with pytest.raises(ValidationError, match="missing requested target"):
        validate_mid_batch(predictions, {"fragment": 2}, tolerance=1e-9)


def test_internal_x_list_must_not_be_exposed() -> None:
    predictions = {
        "sample-1": {"fragment": [0.75, 0.25], "X_list": [0.5, 0.5]}
    }

    with pytest.raises(ValidationError, match="internal mfapy entry 'X_list'"):
        validate_mid_batch(predictions, {"fragment": 2}, tolerance=1e-9)


def test_mid_length_must_match_requested_length() -> None:
    predictions = {"sample-1": {"fragment": [0.5, 0.25, 0.25]}}

    with pytest.raises(ValidationError, match="has length 3; expected 2"):
        validate_mid_batch(predictions, {"fragment": 2}, tolerance=1e-9)


def test_nonempty_batch_is_required() -> None:
    with pytest.raises(ValidationError, match="nonempty"):
        validate_mid_batch({}, {"fragment": 2}, tolerance=1e-9)


def test_success_preserves_batch_size_and_sample_ids() -> None:
    predictions = {
        "sample-z": {
            "fragment-a": [0.25, 0.75],
            "fragment-b": [1.0, 0.0, 0.0],
        },
        "sample-a": {
            "fragment-a": [0.5, 0.5],
            "fragment-b": [0.1, 0.2, 0.7],
        },
    }

    summary = validate_mid_batch(
        predictions,
        {"fragment-a": 2, "fragment-b": 3},
        tolerance=1e-12,
    )

    assert summary == {
        "valid": True,
        "sample_count": 2,
        "sample_ids": ["sample-z", "sample-a"],
        "target_count": 2,
        "target_ids": ["fragment-a", "fragment-b"],
        "validated_mid_count": 4,
        "max_normalization_error": 0.0,
        "tolerance": 1e-12,
    }
    assert json.loads(json.dumps(summary)) == summary
