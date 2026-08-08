"""Print the deterministic Version-1 transition inventory as CSV to stdout."""

from __future__ import annotations

import csv
import sys

from fluxemu.carbon_transitions import load_default_library


def main() -> None:
    library = load_default_library()
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=(
            "canonical_id", "name", "pathway", "carbon_substrates",
            "carbon_products", "reversible", "symmetry", "validation_status",
            "primary_source", "source_identifier", "aliases", "tests",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for entry in sorted(library.transitions, key=lambda item: item.canonical_id):
        writer.writerow({
            "canonical_id": entry.canonical_id,
            "name": entry.name,
            "pathway": entry.pathway,
            "carbon_substrates": "+".join(entry.substrates),
            "carbon_products": "+".join(entry.products),
            "reversible": str(entry.reversible).lower(),
            "symmetry": entry.symmetry,
            "validation_status": entry.validation_status,
            "primary_source": entry.provenance.citation,
            "source_identifier": entry.provenance.source_identifier,
            "aliases": ";".join(entry.aliases),
            "tests": "tests/test_carbon_transitions.py",
        })


if __name__ == "__main__":
    main()
