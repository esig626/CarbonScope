"""FluxEMU standalone native metabolic isotope simulation."""

__version__ = "0.2.0"

from .analysis import run_native_fba, run_native_fva, run_native_stationary_analysis
from .model import load_sbml_flux_model

__all__ = ["load_sbml_flux_model", "run_native_fba", "run_native_fva", "run_native_stationary_analysis"]
