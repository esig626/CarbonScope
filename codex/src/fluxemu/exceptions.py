"""FluxEMU exception hierarchy.

The package raises these exceptions at its public boundaries so callers do not
need to depend on implementation-specific exceptions from COBRApy, PyYAML, or
mfapy.
"""

from __future__ import annotations


class FluxEMUError(Exception):
    """Base class for all expected FluxEMU failures."""


class InputValidationError(FluxEMUError, ValueError):
    """Base class for invalid user-controlled input."""


class ConfigurationError(InputValidationError):
    """Raised when an experiment configuration is missing or invalid."""


class MetadataError(InputValidationError):
    """Raised when isotope metadata is missing, malformed, or incompatible."""


class AnalysisError(FluxEMUError):
    """Raised when a COBRA analysis cannot produce a valid result."""


class MappingError(InputValidationError):
    """Raised for missing, duplicate, incompatible, or ambiguous mappings."""


class CarbonTransitionValidationError(MappingError):
    """Raised when a curated carbon-transition record is internally invalid."""


class ForwardEMUError(FluxEMUError):
    """Raised when mfapy forward-EMU calculation fails."""


class ValidationError(FluxEMUError):
    """Raised when calculated fluxes or MIDs fail numerical validation."""
