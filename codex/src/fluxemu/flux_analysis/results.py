"""Result records shared by flux-analysis engines."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True, slots=True)
class PrimalDiagnostics:
    max_lower_bound_violation: float
    max_upper_bound_violation: float
    max_mass_balance_residual: float
    objective_recalculation_error: float


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

    def to_frame(self) -> pd.DataFrame:
        frame = self.ranges.copy()
        frame.index.name = "reaction_id"
        frame["fraction_of_optimum"] = self.fraction_of_optimum
        return frame
