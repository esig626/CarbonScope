# Unresolved mappings

These reactions are intentionally not supplied with a guessed mapping in
Version 1:

- Individual glucose-6-phosphate dehydrogenase and
  6-phosphogluconolactonase maps. The official mfapy MCF7 source exposes a
  mapped G6P-to-`m6PG` net reaction, so V1 provides that net transition rather
  than inventing an unobserved lactone intermediate map.
- A standalone aconitase map compatible with the frozen Antoniewicz citrate
  convention. Table 5 validates a net citrate-to-alpha-ketoglutarate
  transition, not a separate aconitase entry.
- Carbon maps for carrier exchange steps of the malate-aspartate shuttle that
  involve multiple simultaneously transported metabolites. V1 includes the
  carbon-skeleton identity transport, not an unsupported carrier mechanism.
- Additional serine/glycine/folate reactions beyond the directly sourced
  3PG-to-serine and serine-to-glycine plus transferred one-carbon unit maps.

An unresolved reaction must either retain explicit SBML mapping metadata or
fail loudly during bridge construction. It must not fall through to a
name-based or automatic mapping.
