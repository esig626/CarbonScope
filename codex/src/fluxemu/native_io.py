"""Strict native experiment ingestion and authoritative transition binding."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import yaml
from fluxemu.carbon_transitions import load_default_library
from fluxemu.exceptions import MappingError
from fluxemu.model import (AtomPosition, AtomTransition, CanonicalModel, FluxProjectionExpression, FluxProjectionRule,
 FluxProjectionTerm, IsotopeMetabolite, IsotopeModel, IsotopeParticipant, IsotopeReaction, MappingBranch,
 ObservationPrecursor, ObservationTarget, StationaryExperimentSemantics, Target, Tracer, validate_stationary_experiment)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping): raise MappingError(f"{name} must be a mapping")
    return value

def load_native_stationary_spec(path: str | Path, flux_model):
    """Load schema_version 1 YAML, returning canonical model, experiment and FVA fraction."""
    try: raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc: raise MappingError(f"cannot load experiment YAML: {exc}") from exc
    root = _mapping(raw, "experiment")
    if root.get("schema_version") != 1: raise MappingError("experiment must declare schema_version: 1")
    isotope = _mapping(root.get("isotope_model"), "isotope_model")
    declarations = _mapping(isotope.get("metabolites"), "isotope_model.metabolites")
    mets = tuple(IsotopeMetabolite(str(mid), int(_mapping(v, f"metabolite {mid}")["carbon_count"]), bool(v.get("isotope_visible", True)), bool(v.get("symmetry", False))) for mid,v in declarations.items())
    model_met_ids = {m.metabolite_id for m in flux_model.metabolites}
    if any(m.metabolite_id not in model_met_ids for m in mets): raise MappingError("isotope model references an unknown model metabolite")
    library = load_default_library(); reactions=[]
    assignments = isotope.get("assignments", ())
    if not isinstance(assignments, list): raise MappingError("isotope_model.assignments must be a list")
    physical = {r.reaction_id:r for r in flux_model.reactions}
    for index, item in enumerate(assignments):
        item=_mapping(item, f"assignment {index}"); rid=str(item.get("reaction_id", "")); tid=str(item.get("transition_id", "")); direction=str(item.get("direction", ""))
        if rid not in physical: raise MappingError(f"mapping assignment references unknown reaction {rid!r}")
        transition=library.by_id.get(tid)
        if transition is None: raise MappingError(f"reaction {rid!r} has unknown authoritative transition {tid!r}")
        if direction not in {"forward","reverse"}: raise MappingError(f"reaction {rid!r} mapping direction must be forward or reverse")
        names=_mapping(item.get("metabolite_map"), f"assignment {rid} metabolite_map")
        required=set(transition.substrates+transition.products)
        if set(names) != required: raise MappingError(f"reaction {rid!r} metabolite_map must map exactly {sorted(required)}")
        if len(set(map(str,names.values()))) != len(names): raise MappingError(f"reaction {rid!r} metabolite_map is not one-to-one")
        coeff={t.metabolite_id:float(t.coefficient) for t in physical[rid].stoichiometric_terms}
        canonical_subs, canonical_prods=(transition.substrates,transition.products) if direction=="forward" else (transition.products,transition.substrates)
        mapped_subs=tuple(str(names[x]) for x in canonical_subs); mapped_prods=tuple(str(names[x]) for x in canonical_prods)
        if any(coeff.get(x,0)>=0 for x in mapped_subs) or any(coeff.get(x,0)<=0 for x in mapped_prods): raise MappingError(f"reaction {rid!r} participants do not match authoritative transition direction")
        counts={str(names[k]):v for k,v in transition.carbon_counts.items()}
        declared={m.metabolite_id:m.carbon_count for m in mets}
        if any(declared.get(k)!=v for k,v in counts.items()): raise MappingError(f"reaction {rid!r} authoritative carbon counts disagree with isotope declarations")
        participants=lambda ids: tuple(IsotopeParticipant(x,tuple(range(1,declared[x]+1))) for x in ids)
        branches=[]
        for branch in transition.branch_for_direction(direction):
            branches.append(MappingBranch(branch.branch_id,branch.weight,tuple(AtomTransition(AtomPosition(str(names[a.source.metabolite]),a.source.position),AtomPosition(str(names[a.destination.metabolite]),a.destination.position)) for a in branch.atom_map)))
        projection=FluxProjectionRule(f"{rid}:{direction}",FluxProjectionExpression((FluxProjectionTerm(rid,1.0),)),"positive_part",1e-12)
        reactions.append(IsotopeReaction(rid,direction,True,participants(mapped_subs),participants(mapped_prods),tuple(branches),flux_projection=projection,provenance=(("transition_id",tid),)))
    required = isotope.get("required_reactions", ())
    if not isinstance(required, list) or not all(isinstance(x, str) for x in required): raise MappingError("isotope_model.required_reactions must be a list of reaction IDs")
    missing = [x for x in required if x not in {r.reaction_id for r in reactions}]
    if missing: raise MappingError("missing authoritative mapping for isotope-active reaction(s): " + ", ".join(missing))
    iso_model=IsotopeModel(mets,tuple(reactions))
    exp=_mapping(root.get("experiment"),"experiment")
    tracers=tuple(Tracer(str(x["metabolite_id"]),tuple((str(k),float(v)) for k,v in _mapping(x["isotopomers"],"isotopomers").items()),str(x.get("correction","no"))) for x in exp.get("tracers",()))
    targets=tuple(Target(str(x["target_id"]),str(x["metabolite_id"]),tuple(map(int,x["atom_positions"])),str(x.get("analytical_method","native")),str(x.get("formula","unspecified")),str(x.get("correction","no"))) for x in exp.get("targets",()))
    observations=tuple(ObservationTarget(str(x["target_id"]),int(x["carbon_count"]),tuple(ObservationPrecursor(str(p["metabolite_id"]),tuple(map(int,p["atom_positions"]))) for p in x["precursors"])) for x in exp.get("observation_targets",()))
    canonical=CanonicalModel(flux_model,iso_model); semantics=StationaryExperimentSemantics(tracers,targets,observations); validate_stationary_experiment(canonical,semantics)
    fraction=float(root.get("fva_fraction_of_optimum",1.0))
    if not 0 < fraction <= 1: raise MappingError("fva_fraction_of_optimum must be in (0, 1]")
    return canonical,semantics,fraction
