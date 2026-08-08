# Stationary Table 5 / Table 6 benchmark validation

`build_model.py` writes the exact eight-reaction Table 5 SBML and runs the
Figure 12 flux vector through FluxEMU’s in-memory mfapy forward path.  It also
runs the independent full-isotopomer solver.  The latter imports neither
FluxEMU nor mfapy and solves the 176 simultaneous isotopomer balance variables
directly.

The full-precision stationary result is:

| MID | FluxEMU/mfapy | Direct isotopomer solver | Table 6 (p. 38) |
| --- | ---: | ---: | ---: |
| M+0 | 0.34635416666666663 | 0.34635416666666663 | 0.3464 |
| M+1 | 0.26953124999999994 | 0.26953124999999994 | 0.2695 |
| M+2 | 0.27083333333333326 | 0.27083333333333330 | 0.2708 |
| M+3 | 0.08072916666666664 | 0.08072916666666664 | 0.0807 |
| M+4 | 0.028645833333333325 | 0.028645833333333325 | 0.0286 |
| M+5 | 0.003906249999999999 | 0.003906249999999998 | 0.0039 |

The direct and mfapy paths agree to floating-point precision.  Comparison with
the four-decimal Table 6 entries uses an absolute tolerance of `5.1e-5`, which
is sufficient for rounding at four decimal places.  All MID components are
finite, nonnegative, normalised, and M+3/M+4/M+5 are strictly positive.

The terminal glutamate product is kept as the Table 5 endpoint.  A narrow
benchmark-local mfapy subclass bypasses only mfapy’s otherwise useful
requirement that every non-excreted compound have a downstream reaction; no
ninth reaction or artificial carbon transition is added.
