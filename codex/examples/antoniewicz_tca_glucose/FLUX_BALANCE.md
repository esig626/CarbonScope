# Fixed flux vector and carbon-pool balance

This benchmark does not optimise or sample fluxes. Its one ground-truth,
complete vector is separate from canonical SBML bounds and is passed explicitly
to mfapy for forward evaluation:

| Reaction group | Fixed flux |
| --- | ---: |
| GLC_IN, HEX, PGI, PFK, FBA, TPI | 50 each |
| GAPD, PGK, PGM, ENO, PYK, PDH | 100 each |
| LDH | 0 |
| v1, v2 | 100 each |
| v3, v4, v5, v8 | 50 each |
| v6 | 125 |
| v7 | 75 |

The required glycolytic relation is therefore explicit: 50 FBP produces 50
DHAP and 50 direct GAP; 50 DHAP becomes the other 50 GAP; and the downstream
triose flux is 100. PYK produces 100 pyruvate, PDH produces 100 AcCoA, and
v1 consumes exactly those 100 AcCoA.

`build_model.carbon_mass_balance()` evaluates the complete vector against the
SBML stoichiometry. Each explicitly balanced carbon metabolite has residual
zero (absolute tolerance `1e-12`):

| Pool set | Net molecular flux |
| --- | ---: |
| glucose_c, G6P, F6P, FBP, DHAP, GAP, BPG, 3PG, 2PG, PEP, pyruvate, AcCoA | 0 each |
| OAC, citrate, AKG, succinate, fumarate | 0 each |

`glucose_ext` and `aspartate` are specified carbon sources; `CO2` and lactate
are external products. Glutamate is the intentional terminal measured product
already present in the frozen eight-reaction Antoniewicz construction, with
its implicit measurement outflow. No extra glucose, glutamate, or carbon sink
reaction is added merely for bookkeeping.
