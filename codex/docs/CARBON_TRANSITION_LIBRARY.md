# Carbon transition library

FluxEMU separates a stoichiometric COBRA model from a carbon-transition model.
Stoichiometry says which metabolites react and carries flux constraints; a
transition says exactly where every tracked carbon goes. The reusable library
in `src/fluxemu/carbon_transitions/` supplies the latter without tying it to a
single COBRA model.

Canonical library maps use explicit references such as
`pyruvate.C1 -> carbon_dioxide.C1` and
`pyruvate.C2 -> acetyl_coa.C1`. Letter strings such as `ABC-->BC+A` are
generated only at the mfapy boundary. They are never canonical stored data.

## Numbering and provenance

`data/metabolites.yaml` is the single metabolite-carbon registry. It fixes a
global C1..Cn order, aliases, database identifiers, source, stereochemical
notes, and symmetry. See the library's
[`NUMBERING_CONVENTIONS.md`](../src/fluxemu/carbon_transitions/docs/NUMBERING_CONVENTIONS.md).

Every reaction has exact provenance. `gold` means direct frozen Antoniewicz
Table 5 coverage with benchmark regression; `curated` means an explicit map
from an established isotope-MFA source and structural validation; `provisional`
is reserved for incomplete validation. Unresolved maps are recorded rather
than guessed.

## Symmetry and reverse maps

An entry carries a primary map plus explicit weighted branches where needed.
The Antoniewicz succinate/fumarate reactions retain their 0.5/0.5 orientation
branches. The mfapy exporter converts the primary normalized map to its ABC
form and retains the symmetric metabolite treatment that reproduces the frozen
benchmark. Reversible records use a stored reverse map or `derived`; the
validator double-inverts it.

## Matching and SBML precedence

The matcher uses, in order: `fluxemu_transition_id`, database identifiers,
known reaction-ID aliases, and exact normalized carbon chemistry. It never
accepts a reaction just because a name resembles a library name. Compartment
IDs are retained in the resulting map; carbon-free water/proton/cofactor
differences may be ignored, but recognized carbon substrates and products must
match exactly.

When resolving a model, precedence is fixed:

1. An explicit `FLUXEMU_REACTION_METADATA_V1` SBML map.
2. A validated library map.
3. A loud mapping error.

Library resolution can generate SBML-compatible metadata, so users do not
hand-write ABC labels for a supported reaction. Carbon-source/excretion model
roles remain model-specific metadata.

Each pipeline output now contains `mapping_provenance.json`; its per-reaction
records identify the model reaction, canonical transition, source identifier,
status, SBML/library origin, symmetry treatment, and warnings.

## Adding a reaction safely

Add its globally numbered metabolites first. Add a complete normalized map
where every input carbon has one fate and every product carbon has one origin,
declare/reproduce the reverse map, record exact provenance, and add aliases
only when the carbon chemistry is actually the same. Run schema, accounting,
reverse, symmetry, corruption, and appropriate benchmark tests before
committing. If no source supports the map, add it to `UNRESOLVED_MAPPINGS.md`
instead of adding a speculative transition.
