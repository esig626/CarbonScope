# Time-resolved recirculation validation

The time course uses mfapy’s existing generated `diffmdv` route for the same
eight Table 5 reactions.  It introduces the Section 3.2 acetyl-CoA mixture at
time zero, starts every internal EMU as unlabelled, and uses positive pool
quantities of 1.0 for OAC, citrate, AKG, glutamate, succinate, and fumarate.
These are artificial numerical pool quantities only; the time coordinates have
no biological-unit claim.

Time points are `0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0`.  The generated
`timecourse_mids.csv` includes OAC, citrate, AKG, succinate, fumarate, and
glutamate.

- At time zero, every internal MID is exactly M+0.
- At 0.001, glutamate M+1 and M+2 are positive while M+3–M+5 are zero.
- By 0.05, glutamate M+3, M+4, and M+5 are positive, demonstrating delayed
  recirculation into higher mass isotopologues.
- At 2.0, the glutamate MID is within `2e-6` of the stationary Table 6
  benchmark calculated by both stationary methods.
- Every reported time-course MID is finite, nonnegative, and normalised.

The biochemical transitions and tracer mixture are from Section 3.2, Figure
12, and Table 5 (paper pages 12–14, 29, and 37); the stationary limiting MID
is Table 6 (paper page 38).
