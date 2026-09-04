"""Result records shared by flux-analysis engines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pandas as pd


def _fva_ranges_sha256(ranges: pd.DataFrame, model_fingerprint: str) -> str:
    """Digest ordered scalar FVA output so later DataFrame mutation is detectable."""

    values = ranges.to_numpy(dtype=float, copy=False)
    payload = (
        model_fingerprint,
        tuple(str(value) for value in ranges.index),
        tuple(str(value) for value in ranges.columns),
        tuple(tuple(float(value).hex() for value in row) for row in values),
    )
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PrimalDiagnostics:
    max_lower_bound_violation: float
    max_upper_bound_violation: float
    max_mass_balance_residual: float
    objective_recalculation_error: float
    max_raw_mass_balance_residual: float | None = None
    conditioned_row_space_residual: float | None = None


@dataclass(frozen=True)
class FBAResult:
    objective_value: float
    status: str
    objective_direction: str
    fluxes: pd.Series
    diagnostics: PrimalDiagnostics | None = None

    def to_frame(self) -> pd.DataFrame:
        frame = self.fluxes.rename("flux").to_frame()
        frame.index.name = "reaction_id"
        frame["objective_value"] = self.objective_value
        frame["solver_status"] = self.status
        return frame


@dataclass(frozen=True)
class FVAResult:
    ranges: pd.DataFrame
    fraction_of_optimum: float
    objective_value: float
    objective_direction: str
    model_fingerprint: str | None = None
    ranges_sha256: str | None = None

    def to_frame(self) -> pd.DataFrame:
        frame = self.ranges.copy()
        frame.index.name = "reaction_id"
        frame["fraction_of_optimum"] = self.fraction_of_optimum
        return frame
