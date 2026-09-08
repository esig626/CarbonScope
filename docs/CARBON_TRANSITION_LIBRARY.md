# Carbon transition library

CarbonScope ships an authoritative, source audited library of central carbon atom transitions under `src/fluxemu/carbon_transitions/data/`.

## Contract

Each transition declares:

* a canonical reaction ID and biochemical participants;
* carbon counts;
* explicit forward atom fate;
* reverse atom fate where scientifically supported;
* explicit weighted mapping branches where required by symmetry;
* numbering and stereochemistry notes;
* aliases and external identifiers for provenance only;
* source citation and location and validation status.

The runtime loader strictly parses and validates these records. It does not infer a transition from stoichiometry, reaction names, molecular formulae or external databases.

## Model binding

Native experiment YAML explicitly names the transition ID assigned to each isotope active physical reaction and explicitly maps canonical transition participants to the model's metabolite IDs. The physical direction must agree with the canonical flux bounds and projection semantics.

This explicit binding is deliberate: identifiers and aliases in the library are provenance and search metadata, not permission for automatic scientific mapping.

## Symmetry

Symmetry is represented through explicit mapping branches and weights where scientifically required. Boolean symmetry metadata does not generate branches or probabilities.

The Antoniewicz Table 5 v5 to v7 entries retain their explicit 0.5/0.5 orientation branches and are independently checked against the direct full isotopomer benchmark in `tests/test_antoniewicz_native.py`.

## Validation

`load_default_library()` validates the full packaged dataset at load time. `tests/test_carbon_transitions.py` additionally checks registry completeness, source inventory coverage, selected carbon fates and weighted branch preservation.

The machine readable source inventory is retained under `results/carbon_transition_library/reaction_inventory.csv`.
