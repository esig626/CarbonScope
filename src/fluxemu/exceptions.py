"""FluxEMU exception hierarchy for native public boundaries."""

from __future__ import annotations


class FluxEMUError(Exception):
    """Base class for all expected FluxEMU failures."""


class InputValidationError(FluxEMUError, ValueError):
    """Base class for invalid user-controlled input."""


class ConfigurationError(InputValidationError):
    """Raised when an input configuration is missing or invalid."""


class MetadataError(InputValidationError):
    """Raised when scientific metadata are missing, malformed, or incompatible."""


class AnalysisError(FluxEMUError):
    """Raised when native flux analysis cannot produce a valid result."""


class MappingError(InputValidationError):
    """Raised for missing, duplicate, incompatible, or ambiguous mappings."""


class CarbonTransitionValidationError(MappingError):
    """Raised when a curated carbon-transition record is internally invalid."""


class ForwardEMUError(FluxEMUError):
    """Raised when native stationary or transient EMU evaluation fails."""


class ValidationError(FluxEMUError):
    """Raised when calculated fluxes or MIDs fail numerical validation."""
