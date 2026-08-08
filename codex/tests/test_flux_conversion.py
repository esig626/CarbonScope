from __future__ import annotations

import pandas as pd
import pytest

from fluxemu.exceptions import MappingError
from fluxemu.flux_conversion import FluxConverter, ReactionMapping


def _entries() -> list[ReactionMapping]:
    return [
        ReactionMapping("cobra-b", "fr_b", 1),
        ReactionMapping("cobra-a", "fr_a", 0),
    ]


def test_exact_reaction_order_conversion() -> None:
    converter = FluxConverter(_entries(), ["fr_a", "fr_b"])
    row = pd.Series({"unmapped": 99.0, "cobra-b": 2.0, "cobra-a": 1.0})
    assert converter.convert_row(row).tolist() == [1.0, 2.0]


def test_reverse_reaction_direction_flips_flux_sign() -> None:
    mappings = [
        ReactionMapping("cobra-b", "fr_b", 0, direction=-1),
        ReactionMapping("cobra-a", "fr_a", 1, direction=1),
    ]
    converter = FluxConverter(mappings, ["fr_a", "fr_b"])
    row = pd.Series({"cobra-a": 1.0, "cobra-b": -3.0})
    assert converter.convert_row(row).tolist() == [1.0, 3.0]


def test_reverse_reaction_direction_flips_frame_columns() -> None:
    mappings = [
        ReactionMapping("cobra-b", "fr_b", 0, direction=-1),
        ReactionMapping("cobra-a", "fr_a", 1, direction=1),
    ]
    converter = FluxConverter(mappings, ["fr_a", "fr_b"])
    frame = pd.DataFrame({"cobra-b": [2.0, -1.0], "cobra-a": [1.0, 4.0]})
    converted = converter.convert_frame(frame)
    assert converted.tolist() == [[1.0, -2.0], [4.0, 1.0]]


def test_missing_reaction_fails_without_zero_fill() -> None:
    converter = FluxConverter(_entries(), ["fr_a", "fr_b"])
    with pytest.raises(MappingError, match="missing required directional"):
        converter.convert_row(pd.Series({"cobra-a": 1.0}))


def test_duplicate_reaction_column_fails() -> None:
    converter = FluxConverter(_entries(), ["fr_a", "fr_b"])
    duplicate = pd.DataFrame([[1.0, 2.0]], columns=["cobra-a", "cobra-a"])
    with pytest.raises(MappingError, match="duplicate reaction columns"):
        converter.convert_frame(duplicate)


def test_ambiguous_internal_mapping_fails() -> None:
    entries = [
        ReactionMapping("cobra-a", "fr_same", 0),
        ReactionMapping("cobra-b", "fr_same", 1),
    ]
    with pytest.raises(MappingError, match="ambiguous mapping"):
        FluxConverter(entries, ["fr_same"])


def test_duplicate_original_mapping_fails() -> None:
    entries = [
        ReactionMapping("cobra-a", "fr_a", 0),
        ReactionMapping("cobra-a", "fr_b", 1),
    ]
    with pytest.raises(MappingError, match="duplicate original"):
        FluxConverter(entries, ["fr_a", "fr_b"])
