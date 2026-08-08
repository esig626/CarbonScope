"""Tests for FluxEMU JSON metadata in stable COBRApy SBML notes."""

from __future__ import annotations

from pathlib import Path

import pytest
from cobra import Metabolite, Model, Reaction
from cobra.io import read_sbml_model, write_sbml_model

from fluxemu.exceptions import MetadataError
from fluxemu.isotope_metadata import (
    METABOLITE_METADATA_NOTE_KEY,
    REACTION_METADATA_NOTE_KEY,
    AtomMappedParticipant,
    MetaboliteIsotopeMetadata,
    ReactionIsotopeMetadata,
    collect_isotope_metadata,
    decode_metadata_note,
    get_metabolite_metadata,
    get_reaction_metadata,
    set_metabolite_metadata,
    set_reaction_metadata,
)


def _annotated_model() -> tuple[
    Model, ReactionIsotopeMetadata, MetaboliteIsotopeMetadata
]:
    model = Model("fluxemu_metadata_roundtrip")
    source = Metabolite("source_c", compartment="c")
    product = Metabolite("product_c", compartment="c")
    reaction = Reaction("R1", lower_bound=0.0, upper_bound=10.0)
    reaction.add_metabolites({source: -1.0, product: 1.0})
    model.add_reactions([reaction])
    model.objective = reaction

    reaction_metadata = ReactionIsotopeMetadata(
        original_cobra_reaction_id="R1",
        directional_id='R1 forward <mapped> & "quoted"',
        direction="forward",
        include_in_isotope_model=True,
        substrates=(
            AtomMappedParticipant("source_c", ("carbon&1", "carbon<2>")),
        ),
        products=(
            AtomMappedParticipant("product_c", ("carbon&1", "carbon<2>")),
        ),
    )
    metabolite_metadata = MetaboliteIsotopeMetadata(
        original_cobra_metabolite_id="source_c",
        carbon_count=2,
        is_carbon_source=True,
        is_excreted=False,
        symmetry=False,
        include_in_isotope_model=True,
    )
    set_reaction_metadata(reaction, reaction_metadata)
    set_metabolite_metadata(source, metabolite_metadata)
    return model, reaction_metadata, metabolite_metadata


def test_reaction_and_metabolite_metadata_sbml_roundtrip(tmp_path: Path) -> None:
    model, expected_reaction, expected_metabolite = _annotated_model()
    reaction_note = model.reactions.R1.notes[REACTION_METADATA_NOTE_KEY]
    metabolite_note = model.metabolites.source_c.notes[
        METABOLITE_METADATA_NOTE_KEY
    ]

    # This proves the JSON crossed the XHTML boundary in escaped form.
    assert "&lt;" in reaction_note
    assert "&amp;" in reaction_note
    assert "&quot;" in reaction_note
    assert decode_metadata_note(reaction_note) == expected_reaction.to_dict()

    path = tmp_path / "annotated.xml"
    write_sbml_model(model, path)
    loaded = read_sbml_model(path)

    # COBRApy preserves the escaped canonical note string exactly.
    assert loaded.reactions.R1.notes[REACTION_METADATA_NOTE_KEY] == reaction_note
    assert (
        loaded.metabolites.source_c.notes[METABOLITE_METADATA_NOTE_KEY]
        == metabolite_note
    )
    assert get_reaction_metadata(loaded.reactions.R1, required=True) == expected_reaction
    assert (
        get_metabolite_metadata(loaded.metabolites.source_c, required=True)
        == expected_metabolite
    )

    collected = collect_isotope_metadata(loaded)
    assert collected.reactions["R1"] == expected_reaction
    assert collected.metabolites["source_c"] == expected_metabolite
    assert collected.included_reactions == (expected_reaction,)
    assert collected.included_metabolites == (expected_metabolite,)


def test_metadata_rejects_wrong_cobra_object() -> None:
    model, reaction_metadata, metabolite_metadata = _annotated_model()

    other_reaction = Reaction("R2")
    model.add_reactions([other_reaction])
    with pytest.raises(MetadataError, match="attached to 'R2'"):
        set_reaction_metadata(other_reaction, reaction_metadata)

    other_metabolite = Metabolite("other_c", compartment="c")
    model.add_metabolites([other_metabolite])
    with pytest.raises(MetadataError, match="attached to 'other_c'"):
        set_metabolite_metadata(other_metabolite, metabolite_metadata)


def test_required_metadata_reports_missing_object() -> None:
    reaction = Reaction("unannotated")
    metabolite = Metabolite("unannotated_c")

    with pytest.raises(MetadataError, match="missing isotope metadata"):
        get_reaction_metadata(reaction, required=True)
    with pytest.raises(MetadataError, match="missing isotope metadata"):
        get_metabolite_metadata(metabolite, required=True)
