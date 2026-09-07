"""Direct SBML Level 3 FBC ingestion into FluxEMU's native flux model."""
from __future__ import annotations
from pathlib import Path
import math

from fluxemu.exceptions import MappingError
from .schema import FluxMetabolite, FluxModel, FluxReaction, LinearObjective, ObjectiveTerm, StoichiometricTerm
from .validation import validate_flux_model


def load_sbml_flux_model(path: str | Path) -> FluxModel:
    """Load an SBML FBC model without COBRApy, preserving document ordering."""
    try:
        import libsbml
    except ImportError as exc:  # pragma: no cover
        raise MappingError("SBML loading requires the default python-libsbml dependency") from exc
    document = libsbml.readSBMLFromFile(str(path))
    fatal = [document.getError(i).getMessage() for i in range(document.getNumErrors())
             if document.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR]
    if fatal:
        raise MappingError("invalid SBML: " + "; ".join(fatal))
    model = document.getModel()
    if model is None:
        raise MappingError("SBML document contains no model")
    fbc = model.getPlugin("fbc")
    if fbc is None:
        raise MappingError("SBML model must use the FBC package")
    parameters = {p.getId(): float(p.getValue()) for p in model.getListOfParameters()}
    metabolites = tuple(FluxMetabolite(s.getId(), not s.getBoundaryCondition()) for s in model.getListOfSpecies())
    reactions = []
    for reaction in model.getListOfReactions():
        plugin = reaction.getPlugin("fbc")
        if plugin is None or not plugin.getLowerFluxBound() or not plugin.getUpperFluxBound():
            raise MappingError(f"reaction {reaction.getId()!r} has no unambiguous FBC bounds")
        try:
            lower, upper = parameters[plugin.getLowerFluxBound()], parameters[plugin.getUpperFluxBound()]
        except KeyError as exc:
            raise MappingError(f"reaction {reaction.getId()!r} references an unknown bound parameter") from exc
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise MappingError(f"reaction {reaction.getId()!r} has non-finite bounds")
        coefficients: dict[str, float] = {}
        for ref in reaction.getListOfReactants():
            if not ref.isSetStoichiometry() or ref.isSetStoichiometryMath():
                raise MappingError(f"reaction {reaction.getId()!r} has unsupported stoichiometry")
            coefficients[ref.getSpecies()] = coefficients.get(ref.getSpecies(), 0.0) - float(ref.getStoichiometry())
        for ref in reaction.getListOfProducts():
            if not ref.isSetStoichiometry() or ref.isSetStoichiometryMath():
                raise MappingError(f"reaction {reaction.getId()!r} has unsupported stoichiometry")
            coefficients[ref.getSpecies()] = coefficients.get(ref.getSpecies(), 0.0) + float(ref.getStoichiometry())
        reactions.append(FluxReaction(reaction.getId(), tuple(StoichiometricTerm(k, v) for k, v in coefficients.items() if v), lower, upper))
    active = fbc.getActiveObjectiveId()
    objective = fbc.getObjective(active) if active else None
    if objective is None:
        raise MappingError("SBML FBC model must select exactly one active objective")
    direction = {"maximize": "maximise", "minimize": "minimise"}.get(objective.getType())
    if direction is None:
        raise MappingError(f"unsupported FBC objective type {objective.getType()!r}")
    terms = tuple(ObjectiveTerm(item.getReaction(), float(item.getCoefficient())) for item in objective.getListOfFluxObjectives())
    result = FluxModel(metabolites, tuple(reactions), LinearObjective(direction, terms))
    validate_flux_model(result)
    return result

__all__ = ["load_sbml_flux_model"]
