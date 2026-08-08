"""Numerical validation for forward-EMU mass isotopomer distributions."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import operator
from typing import Any

import numpy as np

from .exceptions import ValidationError


def _validated_lengths(requested_lengths: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(requested_lengths, Mapping) or not requested_lengths:
        raise ValidationError("at least one requested target fragment is required")
    if "X_list" in requested_lengths:
        raise ValidationError("internal mfapy entry 'X_list' cannot be requested")

    lengths: dict[str, int] = {}
    for target_id, expected_length in requested_lengths.items():
        if not isinstance(target_id, str) or not target_id:
            raise ValidationError("target fragment IDs must be nonempty strings")
        try:
            length = operator.index(expected_length)
        except TypeError as exc:
            raise ValidationError(
                f"expected MID length for target {target_id!r} must be an integer"
            ) from exc
        if isinstance(expected_length, bool) or length <= 0:
            raise ValidationError(
                f"expected MID length for target {target_id!r} must be positive"
            )
        lengths[target_id] = length
    return lengths


def _mid_array(values: Any, sample_id: Any, target_id: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValidationError(
            f"MID for sample {sample_id!r}, target {target_id!r} is not a numeric sequence"
        )
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"MID for sample {sample_id!r}, target {target_id!r} is not numeric"
        ) from exc
    if array.ndim != 1:
        raise ValidationError(
            f"MID for sample {sample_id!r}, target {target_id!r} must be one-dimensional"
        )
    return array


def validate_mid_batch(
    predictions: Mapping[Any, Mapping[str, Any]],
    requested_lengths: Mapping[str, int],
    tolerance: float,
) -> dict[str, Any]:
    """Validate a batch of predicted MIDs and return a JSON-safe summary.

    ``predictions`` preserves its mapping iteration order in the returned
    ``sample_ids``. Values within ``tolerance`` of zero, one, or a normalized
    total are accepted but are not clipped or otherwise modified.
    """

    try:
        numeric_tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValidationError("MID validation tolerance must be numeric") from exc
    if not math.isfinite(numeric_tolerance) or numeric_tolerance < 0.0:
        raise ValidationError("MID validation tolerance must be finite and nonnegative")

    if not isinstance(predictions, Mapping) or not predictions:
        raise ValidationError("predicted MID batch must be a nonempty mapping")
    lengths = _validated_lengths(requested_lengths)

    sample_ids = list(predictions.keys())
    try:
        json.dumps(sample_ids)
    except (TypeError, ValueError) as exc:
        raise ValidationError("sample IDs must be JSON-serializable") from exc

    max_normalization_error = 0.0
    validated_mid_count = 0
    for sample_id, sample_predictions in predictions.items():
        if not isinstance(sample_predictions, Mapping):
            raise ValidationError(
                f"predictions for sample {sample_id!r} must be a target mapping"
            )
        if "X_list" in sample_predictions:
            raise ValidationError(
                f"sample {sample_id!r} exposes internal mfapy entry 'X_list'"
            )

        missing = [target for target in lengths if target not in sample_predictions]
        if missing:
            raise ValidationError(
                f"sample {sample_id!r} is missing requested target fragment(s): "
                + ", ".join(missing)
            )

        for target_id, expected_length in lengths.items():
            array = _mid_array(sample_predictions[target_id], sample_id, target_id)
            if len(array) != expected_length:
                raise ValidationError(
                    f"MID for sample {sample_id!r}, target {target_id!r} has length "
                    f"{len(array)}; expected {expected_length}"
                )
            if not np.all(np.isfinite(array)):
                raise ValidationError(
                    f"MID for sample {sample_id!r}, target {target_id!r} "
                    "contains non-finite values"
                )
            if np.any(array < -numeric_tolerance):
                raise ValidationError(
                    f"MID for sample {sample_id!r}, target {target_id!r} "
                    "contains a negative value beyond tolerance"
                )
            if np.any(array > 1.0 + numeric_tolerance):
                raise ValidationError(
                    f"MID for sample {sample_id!r}, target {target_id!r} "
                    "contains a value above one beyond tolerance"
                )

            normalization_error = abs(float(np.sum(array)) - 1.0)
            if normalization_error > numeric_tolerance:
                raise ValidationError(
                    f"MID for sample {sample_id!r}, target {target_id!r} is not "
                    f"normalized (error {normalization_error})"
                )
            max_normalization_error = max(
                max_normalization_error, normalization_error
            )
            validated_mid_count += 1

    summary: dict[str, Any] = {
        "valid": True,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "target_count": len(lengths),
        "target_ids": list(lengths),
        "validated_mid_count": validated_mid_count,
        "max_normalization_error": float(max_normalization_error),
        "tolerance": numeric_tolerance,
    }
    # This is part of the public contract; fail explicitly rather than return a
    # summary that an output writer cannot serialize.
    try:
        json.dumps(summary)
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
        raise ValidationError("MID validation summary is not JSON-serializable") from exc
    return summary
