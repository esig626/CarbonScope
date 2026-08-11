# Time-resolved recirculation validation

The time course uses mfapy’s existing generated `diffmdv` route for the same
eight Table 5 reactions. It introduces the Section 3.2 acetyl-CoA mixture at
time zero, starts every internal EMU as unlabelled, and uses positive pool
quantities of 1.0 for OAC, citrate, AKG, glutamate, succinate, and fumarate.
These are artificial numerical pool quantities only; the time coordinates have
no biological-unit claim.

Time points are `0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0`. The generated
`timecourse_mids.csv` includes OAC, citrate, AKG, succinate, fumarate, and
glutamate.

- At time zero, every internal MID is exactly M+0.
- At 0.001, glutamate M+1 and M+2 are positive while M+3–M+5 are zero.
- By 0.05, glutamate M+3, M+4, and M+5 are positive, demonstrating delayed
  recirculation into higher mass isotopologues.
- At 2.0, the historical glutamate MID is within `2e-6` of the stationary Table 6
  benchmark calculated by both stationary methods.
- Every reported time-course MID is finite, nonnegative, and normalised.

The biochemical transitions and tracer mixture are from Section 3.2, Figure
12, and Table 5 (paper pages 12–14, 29, and 37); the stationary limiting MID
is Table 6 (paper page 38).

## Historical fixture, terminal semantics, and numerical accuracy

This file and `timecourse_mids.csv` preserve the historical mfapy `diffmdv`
calculation. FluxEMU's compatibility bridge temporarily compiled terminal
glutamate as a dynamic intermediate with its declared pool quantity, producing a
finite mixing lag. Native FluxEMU V1 instead defines an unbalanced terminal target
as its instantaneous flux-weighted production MID and does not invent an external
pool. Native glutamate is therefore checked separately against its instantaneous-production contract.

The historical shared-target trajectory also reflects mfapy's generated `odeint`
settings (`rtol=1e-3`, `atol=1e-3`). A GitHub convergence diagnostic reintegrated
the exact same generated mfapy equations without changing their state ordering,
fluxes, pools, tracer, or requested times. The maximum shared-semantics native/reference
difference decreased from about `4.80e-4` with the historical settings to about
`4.62e-8` at `rtol=1e-6`, `atol=1e-9`, and to order `1e-10` at high accuracy.
High-accuracy `odeint`, LSODA, RK45, and DOP853 integrations agreed at roughly
`1e-10` to `1e-9` scale.

Accordingly, `timecourse_mids.csv` remains an immutable **historical provenance
fixture**, and live historical mfapy must continue to reproduce it. The strict
shared-semantics scientific parity gate instead compares native FluxEMU with the
same mfapy-generated ODE system integrated with LSODA at `rtol=1e-9`,
`atol=1e-12`. The original `2e-6` parity threshold is unchanged. The historical
native/frozen discrepancy remains reported diagnostically rather than being hidden
or used as a high-precision reference.
