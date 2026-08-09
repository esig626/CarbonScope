"""One-way adapters from external engines into the canonical FluxEMU model."""

from .cobra import (
    AuthoritativeTransitionAssignment,
    ProjectionError,
    project_cobra_model,
    resolve_authoritative_transitions,
)

__all__ = [
    "AuthoritativeTransitionAssignment",
    "ProjectionError",
    "project_cobra_model",
    "resolve_authoritative_transitions",
]
