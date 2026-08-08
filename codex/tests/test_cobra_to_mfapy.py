from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fluxemu._mfapy import load_mfapy
from fluxemu.cobra_to_mfapy import build_mfapy_model, deterministic_internal_id
from fluxemu.configuration import load_experiment
from fluxemu.exceptions import MappingError, MetadataError
from fluxemu.isotope_metadata import (
    REACTION_METADATA_NOTE_KEY,
    collect_isotope_metadata,
    get_reaction_metadata,
    IsotopeMetadataCollection,
    set_reaction_metadata,
)
from fluxemu.toy import build_toy_model


EXPERIMENT = Path(__file__).resolve().parents[1] / "examples" / "toy_experiment.yaml"


def _inputs():
    model = build_toy_model()
    return model, collect_isotope_metadata(model), load_experiment(EXPERIMENT)


def test_deterministic_safe_id_generation() -> None:
    value = deterministic_internal_id("reaction", "unsafe-id.with spaces")
    assert value == deterministic_internal_id("reaction", "unsafe-id.with spaces")
    assert value.startswith("fr")
    assert value.isidentifier()
    assert value != deterministic_internal_id("reaction", "another")


def test_builds_all_four_mfapy_objects_directly_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, metadata, experiment = _inputs()
    mfapy = load_mfapy()

    def forbidden_parser(*args, **kwargs):
        raise AssertionError("runtime called mfapy text parser")

    monkeypatch.setattr(mfapy.mfapyio, "load_metabolic_model", forbidden_parser)
    bundle = build_mfapy_model(model, metadata, experiment)

    assert [entry.original_cobra_reaction_id for entry in bundle.reaction_mappings] == [
        f"v{number}" for number in range(1, 10)
    ]
    assert tuple(bundle.model.reaction_ids) == bundle.converter.mfapy_reaction_order
    assert len(bundle.reactions) == 9
    assert bundle.reversible_reactions == {}
    assert len(bundle.metabolites) == 10
    assert len(bundle.target_fragments) == 1
    assert all(entry["atommap"] for entry in bundle.reactions.values())


def test_missing_object_metadata_fails() -> None:
    model = build_toy_model()
    del model.reactions.v1.notes[REACTION_METADATA_NOTE_KEY]
    with pytest.raises(MetadataError, match="explicitly declare"):
        build_mfapy_model(
            model, collect_isotope_metadata(model), load_experiment(EXPERIMENT)
        )


def test_ambiguous_directional_identity_fails() -> None:
    model = build_toy_model()
    first = get_reaction_metadata(model.reactions.v1, required=True)
    second = get_reaction_metadata(model.reactions.v2, required=True)
    assert first is not None and second is not None
    set_reaction_metadata(
        model.reactions.v2, replace(second, directional_id=first.directional_id)
    )
    with pytest.raises(MappingError, match="ambiguous directional"):
        build_mfapy_model(
            model, collect_isotope_metadata(model), load_experiment(EXPERIMENT)
        )


def test_signed_reaction_requires_directional_annotation() -> None:
    model, _, experiment = _inputs()
    model.reactions.v7.lower_bound = -10.0
    metadata = collect_isotope_metadata(model)
    reaction_metadata = metadata.reactions["v7"]
    assert reaction_metadata.direction == "forward"
    # The same model is now permitted as long as direction is explicit and
    # metadata participants are declared as the intended reverse direction.
    updated = replace(
        reaction_metadata,
        direction="reverse",
        substrates=reaction_metadata.products,
        products=reaction_metadata.substrates,
    )
    updated_metadata = IsotopeMetadataCollection(
        reactions={**metadata.reactions, "v7": updated},
        metabolites=metadata.metabolites,
    )
    bundle = build_mfapy_model(model, updated_metadata, experiment)
    assert bundle.reaction_mappings[6].direction == -1
