"""Native tracer marginalisation and EMU condensation algebra."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from fluxemu.exceptions import MappingError
from fluxemu.model import Tracer

from .graph import EMU


def source_emu_mid(tracer: Tracer, emu: EMU, carbon_count: int) -> np.ndarray:
    if tracer.metabolite_id != emu.metabolite_id:
        raise MappingError("tracer and source EMU metabolite IDs differ")
    if tracer.correction != "no":
        raise MappingError("native tracer evaluation supports only no correction")
    if any(position < 1 or position > carbon_count for position in emu.atom_positions):
        raise MappingError(f"source EMU atom position is out of bounds for {emu.metabolite_id!r}")
    result = np.zeros(emu.size + 1, dtype=float)
    total = 0.0
    for pattern, fraction in tracer.isotopomers:
        if not isinstance(pattern, str) or not pattern.startswith("#") or len(pattern) != carbon_count + 1:
            raise MappingError(f"invalid tracer isotopomer code {pattern!r}")
        if any(bit not in "01" for bit in pattern[1:]):
            raise MappingError(f"invalid tracer isotopomer code {pattern!r}")
        value = float(fraction)
        if not math.isfinite(value) or value < 0.0:
            raise MappingError("tracer fractions must be finite and nonnegative")
        mass = sum(int(pattern[position]) for position in emu.atom_positions)
        result[mass] += value
        total += value
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise MappingError("tracer fractions must sum to one")
    return result


def convolve_mids(mids: Sequence[np.ndarray]) -> np.ndarray:
    if not mids:
        raise MappingError("condensation requires at least one precursor MID")
    result = np.array((1.0,), dtype=float)
    for mid in mids:
        vector = np.asarray(mid, dtype=float)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise MappingError("precursor MID must be a finite vector")
        result = np.convolve(result, vector)
    return result


__all__ = ["convolve_mids", "source_emu_mid"]
