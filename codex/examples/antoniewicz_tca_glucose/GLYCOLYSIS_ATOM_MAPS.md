# Synthetic glycolytic atom maps

This benchmark adds a deliberately small carbon-propagation layer ahead of the
frozen Antoniewicz TCA cycle. It is not a reconstruction of E. coli energetics:
there is no PTS (`GLCpts`), ATP, NAD(H), phosphate, water, or proton bookkeeping.

The Antoniewicz paper is the source for the TCA transition system. The
glycolytic transitions below are conventional carbon mappings added for this
synthetic numerical benchmark and are not claimed to be transcribed from
Antoniewicz et al.

| Reaction | Carbon skeleton | Atom transition |
| --- | --- | --- |
| GLC_IN | glucose_ext -> glucose_c | `abcdef -> abcdef` |
| HEX | glucose_c -> G6P | `abcdef -> abcdef` |
| PGI | G6P -> F6P | `abcdef -> abcdef` |
| PFK | F6P -> FBP | `abcdef -> abcdef` |
| FBA | FBP -> DHAP + GAP | `abcdef -> cba + def` |
| TPI | DHAP -> GAP | `abc -> abc` |
| GAPD | GAP -> BPG | `abc -> abc` |
| PGK | BPG -> 3PG | `abc -> abc` |
| PGM | 3PG -> 2PG | `abc -> abc` |
| ENO | 2PG -> PEP | `abc -> abc` |
| PYK | PEP -> pyruvate | `abc -> abc` |
| LDH | pyruvate -> lactate | `abc -> abc` |
| PDH | pyruvate -> AcCoA + CO2 | `abc -> bc + a` |

`build_model.py` imports the exact frozen `TABLE5_REACTIONS` objects for v1–v8
from the separately runnable `antoniewicz_tca` benchmark. It does not carry a
second editable transcription of those mappings.

The map-level positional result of FBA, TPI, and the downstream identities is
two pyruvate origin patterns, in pyruvate C1–C3 order:

`(glucose C3, C2, C1)` and `(glucose C4, C5, C6)`.

Thus glucose C1/C6 reach pyruvate C3, glucose C2/C5 reach pyruvate C2, and
glucose C3/C4 reach pyruvate C1. PDH then sends pyruvate C1 to CO2 and
pyruvate C2/C3 to AcCoA C1/C2. The tests derive these relationships from the
stored atom maps, rather than inferring them from mass counts.
