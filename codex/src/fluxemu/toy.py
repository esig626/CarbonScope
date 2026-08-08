"""Construction helpers for the complete annotated FluxEMU toy network."""

from __future__ import annotations

from pathlib import Path

from cobra import Metabolite, Model, Reaction
from cobra.io import write_sbml_model

from .isotope_metadata import (
    AtomMappedParticipant,
    MetaboliteIsotopeMetadata,
    ReactionIsotopeMetadata,
    set_metabolite_metadata,
    set_reaction_metadata,
)


def _participant(metabolite_id: str, labels: str) -> AtomMappedParticipant:
    return AtomMappedParticipant(metabolite_id, tuple(labels))


def build_toy_model() -> Model:
    """Return the bounded explicit-direction network used by tests and smoke runs."""

    model = Model("fluxemu_toy")
    carbon_counts = {
        "CO2ex": 1,
        "AcCoA": 2,
        "OAC": 4,
        "Cit": 6,
        "AKG": 5,
        "Suc": 4,
        "Fum": 4,
        "Glu": 5,
        "Gluex": 5,
        "Asp": 4,
    }
    metabolites = {
        metabolite_id: Metabolite(metabolite_id, compartment="c")
        for metabolite_id in carbon_counts
    }
    model.add_metabolites(list(metabolites.values()))
    for metabolite_id, carbon_count in carbon_counts.items():
        set_metabolite_metadata(
            metabolites[metabolite_id],
            MetaboliteIsotopeMetadata(
                original_cobra_metabolite_id=metabolite_id,
                carbon_count=carbon_count,
                is_carbon_source=metabolite_id in {"AcCoA", "Asp"},
                is_excreted=metabolite_id in {"CO2ex", "Gluex"},
                symmetry=metabolite_id in {"Suc", "Fum"},
                include_in_isotope_model=True,
            ),
        )

    reaction_specs = [
        ("SRC_AcCoA", 0.0, 100.0, {"AcCoA": 1.0}),
        ("SRC_Asp", 0.0, 100.0, {"Asp": 1.0}),
        ("v1", 0.0, 100.0, {"AcCoA": -1.0, "OAC": -1.0, "Cit": 1.0}),
        ("v2", 0.0, 100.0, {"Cit": -1.0, "AKG": 1.0, "CO2ex": 1.0}),
        ("v3", 0.0, 100.0, {"AKG": -1.0, "Glu": 1.0}),
        ("v4", 0.0, 100.0, {"AKG": -1.0, "Suc": 1.0, "CO2ex": 1.0}),
        ("v5", 0.0, 100.0, {"Suc": -1.0, "Fum": 1.0}),
        ("v6", 0.0, 300.0, {"Fum": -1.0, "OAC": 1.0}),
        ("v7", 0.0, 300.0, {"OAC": -1.0, "Fum": 1.0}),
        ("v8", 0.0, 100.0, {"Asp": -1.0, "OAC": 1.0}),
        ("v9", 0.0, 100.0, {"Glu": -1.0, "Gluex": 1.0}),
        ("DM_CO2", 0.0, 200.0, {"CO2ex": -1.0}),
        ("DM_Gluex", 0.0, 100.0, {"Gluex": -1.0}),
    ]
    for reaction_id, lower, upper, stoichiometry in reaction_specs:
        reaction = Reaction(reaction_id, lower_bound=lower, upper_bound=upper)
        reaction.add_metabolites(
            {metabolites[mid]: coefficient for mid, coefficient in stoichiometry.items()}
        )
        model.add_reactions([reaction])

    mappings = {
        "v1": (("AcCoA", "AB"), ("OAC", "CDEF"), ("Cit", "FEDBAC")),
        "v2": (("Cit", "ABCDEF"), ("AKG", "ABCDE"), ("CO2ex", "F")),
        "v3": (("AKG", "ABCDE"), ("Glu", "ABCDE")),
        "v4": (("AKG", "ABCDE"), ("Suc", "BCDE"), ("CO2ex", "A")),
        "v5": (("Suc", "ABCD"), ("Fum", "ABCD")),
        "v6": (("Fum", "ABCD"), ("OAC", "ABCD")),
        "v7": (("OAC", "ABCD"), ("Fum", "ABCD")),
        "v8": (("Asp", "ABCD"), ("OAC", "ABCD")),
        "v9": (("Glu", "ABCDE"), ("Gluex", "ABCDE")),
    }
    side_counts = {
        "v1": 2,
        "v2": 1,
        "v3": 1,
        "v4": 1,
        "v5": 1,
        "v6": 1,
        "v7": 1,
        "v8": 1,
        "v9": 1,
    }
    for reaction in model.reactions:
        included = reaction.id in mappings
        participants = mappings.get(reaction.id, ())
        split = side_counts.get(reaction.id, 0)
        set_reaction_metadata(
            reaction,
            ReactionIsotopeMetadata(
                original_cobra_reaction_id=reaction.id,
                directional_id=f"{reaction.id}:forward",
                direction="forward",
                include_in_isotope_model=included,
                substrates=tuple(
                    _participant(metabolite_id, labels)
                    for metabolite_id, labels in participants[:split]
                ),
                products=tuple(
                    _participant(metabolite_id, labels)
                    for metabolite_id, labels in participants[split:]
                ),
            ),
        )

    model.objective = model.reactions.DM_Gluex
    model.objective_direction = "max"
    model.notes["FluxEMU"] = "annotated explicit-direction toy model"
    model.annotation["doi"] = "10.1016/j.ymben.2006.09.001"
    return model


def write_toy_sbml(path: str | Path) -> Path:
    """Write the deterministic annotated toy model to SBML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_sbml_model(build_toy_model(), destination)
    return destination


__all__ = ["build_toy_model", "write_toy_sbml"]
