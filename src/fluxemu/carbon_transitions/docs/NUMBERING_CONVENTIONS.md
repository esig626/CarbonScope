# Carbon numbering conventions

Every library metabolite has one immutable `C1` through `Cn` sequence in
`data/metabolites.yaml`. A map always names positions as
`canonical_metabolite.C<n>`; it never relies on the local labels from another
reaction.

For the Antoniewicz entries, the sequence is exactly the Table 5 source-map
order. For example, Table 5 v1 `abcd + ef -> dcbfea` is represented as
`oxaloacetate.C4 -> citrate.C1`, `acetyl_coa.C2 -> citrate.C4`, and so on.
This preserves the frozen numerical benchmark byte-for-byte at the atom-map
level.

The synthetic glycolysis benchmark likewise fixes its own source-map order.
It is explicitly curated, not re-labelled as a claim about an external
database's structural numbering. The registry records this distinction in
each metabolite's `numbering_source` and reaction's `numbering_convention`.

Compartment does not create a new carbon-numbering convention. An identity
transport entry maps, for example, mitochondrial citrate C1..C6 to cytosolic
citrate C1..C6. The matcher retains the actual COBRA metabolite IDs for the
two participants while the library keeps one canonical carbon sequence.

Succinate and fumarate use the same C1..C4 reference orientation but are
flagged symmetric. Their entries retain the two equally weighted orientations
instead of choosing a single arbitrary direction.
