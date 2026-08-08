# Antoniewicz TCA benchmark paper transcription

## Sole source

All biochemical statements and numerical reference values in this example were
transcribed only from [`nihms16238.pdf`](/workspace/references/nihms16238.pdf):
Antoniewicz, Kelleher and Stephanopoulos, *Elementary Metabolite Units (EMU): a
novel framework for modeling isotopic distributions*.

The manuscript labels the relevant material as follows:

- Section 3.2, “Tricarboxylic acid cycle”: paper pages 12–14.
- Figure 12, simplified TCA model and flux state: paper page 29.
- Table 5, stoichiometry and carbon transitions: paper page 37.
- Table 6, glutamate reference MID: paper page 38.

## Figure 12 flux state

All fluxes have the paper’s arbitrary units: `v1=100`, `v2=100`, `v3=50`,
`v4=50`, `v5=50`, `v6=125`, `v7=75`, and `v8=50` (Figure 12, paper page 29).

## Table 5 carbon transitions

| Reaction | Metabolite transition | Carbon transition | Paper source |
| --- | --- | --- | --- |
| v1 | OAC + AcCoA -> citrate | `abcd + ef -> dcbfea` | Table 5, p. 37 |
| v2 | citrate -> AKG + CO2 | `abcdef -> abcde + f` | Table 5, p. 37 |
| v3 | AKG -> glutamate | `abcde -> abcde` | Table 5, p. 37 |
| v4 | AKG -> succinate + CO2 | `abcde -> bcde + a` | Table 5, p. 37 |
| v5 | succinate -> fumarate | `0.5 abcd + 0.5 dcba -> 0.5 abcd + 0.5 dcba` | Table 5, p. 37 |
| v6 | fumarate -> OAC | `0.5 abcd + 0.5 dcba -> abcd` | Table 5, p. 37 |
| v7 | OAC -> fumarate | `abcd -> 0.5 abcd + 0.5 dcba` | Table 5, p. 37 |
| v8 | aspartate -> OAC | `abcd -> abcd` | Table 5, p. 37 |

The SBML encodes the exact non-symmetric atom order in its FluxEMU notes.  The
published 0.5 branches for v5–v7 are represented by marking **both** succinate
and fumarate as rotationally symmetric; mfapy then applies the equal forward
and reverse orientations.  The direct isotopomer solver writes those 0.5
branches explicitly.

## Substrate labelling and expected MID

Section 3.2 (paper page 12) specifies 25% `[2-13C]AcCoA` and 25%
`[1,2-13C]AcCoA`; the remaining 50% is unlabelled AcCoA.  Aspartate is 100%
unlabelled.  Natural isotope abundance is ignored, exactly as stated there.

Table 6 (paper page 38) reports the following complete glutamate MID, stored
in `published_reference_mid.csv`: M+0 0.3464, M+1 0.2695, M+2 0.2708, M+3
0.0807, M+4 0.0286, and M+5 0.0039.
