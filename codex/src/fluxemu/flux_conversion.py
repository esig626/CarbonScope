"""Strict in-memory conversion from COBRA flux columns to mfapy order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .exceptions import MappingError


@dataclass(frozen=True)
class ReactionMapping:
    """A one-to-one mapping for one explicit directional reaction."""

    original_cobra_reaction_id: str
    internal_mfapy_reaction_id: str
    mfapy_reaction_order: int
    direction: int = 1

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


class FluxConverter:
    """Convert complete COBRA samples without defaults or inferred fluxes."""

    def __init__(
        self,
        mappings: Iterable[ReactionMapping],
        mfapy_reaction_order: Sequence[str],
    ) -> None:
        entries = tuple(mappings)
        if not entries:
            raise MappingError("the isotope reaction mapping is empty")

        original_ids = [entry.original_cobra_reaction_id for entry in entries]
        internal_ids = [entry.internal_mfapy_reaction_id for entry in entries]
        order = tuple(mfapy_reaction_order)
        if len(set(original_ids)) != len(original_ids):
            raise MappingError("duplicate original COBRA reaction mapping")
        if len(set(internal_ids)) != len(internal_ids):
            raise MappingError("ambiguous mapping: multiple reactions use one mfapy ID")
        if len(set(order)) != len(order):
            raise MappingError("duplicate reaction in mfapy reaction order")
        if set(order) != set(internal_ids) or len(order) != len(internal_ids):
            raise MappingError("reaction mappings are incompatible with mfapy order")

        by_internal = {entry.internal_mfapy_reaction_id: entry for entry in entries}
        ordered_entries = tuple(by_internal[internal_id] for internal_id in order)
        mapping_positions = {entry.mfapy_reaction_order for entry in ordered_entries}
        if mapping_positions != set(range(len(ordered_entries))):
            raise MappingError(
                "recorded mfapy position is incompatible with reaction count"
            )

        self.mappings = ordered_entries
        self.direction_factors = tuple(entry.direction for entry in ordered_entries)
        self.mfapy_reaction_order = order
        self.original_reaction_order = tuple(
            entry.original_cobra_reaction_id for entry in ordered_entries
        )

    def convert_row(self, fluxes: pd.Series) -> np.ndarray:
        """Return a finite-independent vector in exact mfapy reaction order."""

        if fluxes.index.has_duplicates:
            raise MappingError("duplicate reaction IDs in sampled flux vector")
        missing = [rid for rid in self.original_reaction_order if rid not in fluxes.index]
        if missing:
            raise MappingError(
                "sample is missing required directional reaction(s): "
                + ", ".join(missing)
            )
        # No fill/reindex default is used: every value is selected explicitly.
        return np.asarray(
            [float(fluxes.at[rid]) * factor for rid, factor in zip(
                self.original_reaction_order, self.direction_factors
            )],
            dtype=float,
        )

    def convert_frame(self, samples: pd.DataFrame) -> np.ndarray:
        """Convert a batch while preserving its row/sample order."""

        if samples.columns.has_duplicates:
            raise MappingError("duplicate reaction columns in sampled flux batch")
        if samples.index.has_duplicates:
            raise MappingError("duplicate sample IDs in sampled flux batch")
        missing = [rid for rid in self.original_reaction_order if rid not in samples]
        if missing:
            raise MappingError(
                "sample batch is missing required directional reaction(s): "
                + ", ".join(missing)
            )
        return (
            samples.loc[:, list(self.original_reaction_order)]
            .to_numpy(dtype=float, copy=True)
            * np.array(self.direction_factors)[None, :]
        )
