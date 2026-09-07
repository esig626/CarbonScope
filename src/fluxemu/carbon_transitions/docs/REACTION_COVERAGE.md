# Version-1 reaction coverage

The library covers these source-backed modules:

- Glycolysis, lactate, pyruvate dehydrogenase, and acetate activation.
- Oxidative PPP net entry, 6PGD, RPI, RPE, both transketolase transfers, and
  transaldolase.
- The complete frozen Antoniewicz Table 5 TCA transition system, plus sourced
  fumarate hydratase and malate dehydrogenase.
- Pyruvate/PEP carboxylation, PEP carboxykinase, and malic enzyme.
- Glutaminase, glutamate dehydrogenase, alanine/aspartate skeleton transfer,
  and malate/aspartate carbon transport.
- Citrate identity transport, ATP-citrate lyase, and the direct 3PG/serine
  and serine/glycine entry maps.

Coverage deliberately means an atom-transition record, not just a reaction
name. Carbon-free cofactors may be absent from the normalized reaction; the
matcher verifies the complete recognized carbon chemistry and ignores only
non-carbon cofactor, proton, and water differences.
