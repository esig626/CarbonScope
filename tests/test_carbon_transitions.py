"""Native regression coverage for the curated carbon-transition library."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fluxemu.carbon_transitions import (
    AtomRef,
    AtomTransition,
    CarbonTransition,
    MappingBranch,
    load_default_library,
    validate_transition,
)
from fluxemu.exceptions import CarbonTransitionValidationError


LIBRARY = load_default_library()
ROOT = Path(__file__).resolve().parents[1]


def test_every_packaged_entry_loads_with_strict_schema_and_unique_ids() -> None:
    assert len(LIBRARY.metabolites) >= 35
    assert len(LIBRARY.transitions) >= 44
    assert len(LIBRARY.by_id) == len(LIBRARY.transitions)
    assert {entry.validation_status for entry in LIBRARY.transitions} == {"gold", "curated"}
    for entry in LIBRARY.transitions:
        validate_transition(entry, LIBRARY.metabolites)


def test_machine_readable_inventory_covers_every_library_entry() -> None:
    inventory = ROOT / "results" / "carbon_transition_library" / "reaction_inventory.csv"
    with inventory.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["canonical_id"] for row in rows] == sorted(LIBRARY.by_id)
    assert set(rows[0]) == {
        "canonical_id", "name", "pathway", "carbon_substrates", "carbon_products",
        "reversible", "symmetry", "validation_status", "primary_source",
        "source_identifier", "aliases", "tests",
    }


def test_published_antoniewicz_weighted_symmetry_branches_are_retained_exactly() -> None:
    for transition_id in (
        "antoniewicz.table5.v5.succinate_to_fumarate",
        "antoniewicz.table5.v6.fumarate_to_oaa",
        "antoniewicz.table5.v7.oaa_to_fumarate",
    ):
        entry = LIBRARY.by_id[transition_id]
        branches = entry.branch_for_direction("forward")
        assert len(branches) == 2
        assert tuple(float(branch.weight) for branch in branches) == (0.5, 0.5)
        assert branches[0].atom_map != branches[1].atom_map


def test_core_glycolysis_and_tca_entries_preserve_known_atom_fates() -> None:
    pdh = LIBRARY.by_id["pyruvate.pyruvate_dehydrogenase"]
    pdh_map = pdh.branch_for_direction("forward")[0].atom_map
    assert AtomTransition(AtomRef("pyruvate", 1), AtomRef("carbon_dioxide", 1)) in pdh_map
    assert AtomTransition(AtomRef("pyruvate", 2), AtomRef("acetyl_coa", 1)) in pdh_map
    assert AtomTransition(AtomRef("pyruvate", 3), AtomRef("acetyl_coa", 2)) in pdh_map

    citrate = LIBRARY.by_id["antoniewicz.table5.v2.citrate_to_akg"]
    citrate_map = citrate.branch_for_direction("forward")[0].atom_map
    assert AtomTransition(AtomRef("citrate", 6), AtomRef("carbon_dioxide", 1)) in citrate_map
    assert len(citrate_map) == 6


def test_validator_rejects_incomplete_atom_conservation() -> None:
    entry = LIBRARY.by_id["pyruvate.pyruvate_dehydrogenase"]
    branch = entry.branch_for_direction("forward")[0]
    broken_branch = MappingBranch(branch.branch_id, branch.weight, branch.atom_map[:-1])
    broken = CarbonTransition(
        canonical_id=entry.canonical_id,
        name=entry.name,
        pathway=entry.pathway,
        substrates=entry.substrates,
        products=entry.products,
        carbon_counts=entry.carbon_counts,
        forward_atom_map=entry.forward_atom_map[:-1],
        reverse_atom_map=entry.reverse_atom_map,
        mapping_branches=(broken_branch,),
        reversible=entry.reversible,
        symmetry=entry.symmetry,
        stereochemistry_notes=entry.stereochemistry_notes,
        numbering_convention=entry.numbering_convention,
        aliases=entry.aliases,
        ec_numbers=entry.ec_numbers,
        identifiers=entry.identifiers,
        provenance=entry.provenance,
        validation_status=entry.validation_status,
        comments=entry.comments,
    )
    with pytest.raises(CarbonTransitionValidationError):
        validate_transition(broken, LIBRARY.metabolites)
