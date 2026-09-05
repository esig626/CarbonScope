# Stationary MFA architecture decision

This decision implements only prompt 03A and its expressly incorporated prompt
03. Before production implementation, MFAPY-REFERENCE inspected the vendored
source and MFA-GUARDIAN approved the twelve-row transfer table in
[mfapy engineering comparison](MFAPY_ENGINEERING_COMPARISON.md). The reference
snapshot and its license remain unchanged.

## Engineering lineage and native replacements

Retain mfapy's reduced-coordinate reconstruction, feasible initialization,
multiple tracer experiments sharing one complete state, independent starts,
best-fit selection, and forward-generated synthetic data. Replace its mutable
`matrixinv`/`Rm` representation with the existing native affine geometry,
initialization with native complete-state sampling, and generated forward code
with one native compiled stationary EMU plan per experiment. No second
coordinate system, atom-map workflow, sampler, or EMU engine is introduced.

Replace weighted RSS, covariance weighting, and full-state bound penalties with
the plain sum of explicit MID divergences `D_alpha(observed || predicted)`.
Exclude mfapy's statistical testing and uncertainty procedures entirely.
Sequential deterministic multistart SLSQP is the first backend. Additional
global optimizers and parallel scheduling are deferred: the native sampler
already supplies diverse feasible starts, and neither extra backend is needed
to validate this core. Local multistart provides no global-optimum certificate.

Exact reaction/exchange measurements use explicitly chosen canonical equal
bounds, and hard admissible intervals use canonical bounds. They are constraints,
not weighted residuals. Soft measured-flux/pool observations are deferred until
their semantics are explicitly defined; the canonical problem/constraint
boundary is the extension point. No standard deviation is silently converted
to an interval. Noisy synthetic observations are also deferred.

The basic workflow has no accidental single-experiment or single-start
restriction. It includes explicit whole MID observations and replicates,
complete feasible states, constrained fitting, all-start diagnostics, and
independent original-model validation.

## Scientific data and objective boundary

Frozen records describe `StationaryMIDObservation`, `StationaryMFAExperiment`,
`StationaryMFAProblem`, `DivergenceObjectiveConfig`, `MFAOptimizationConfig`,
`MFAObservationDivergence`, `MFAObjectiveEvaluation`, `MFAStartDiagnostic`, and
`StationaryMFAResult`. Experiments and their targets/replicates retain declared
tuple ordering. Observations bind to declared regular or composite targets;
unknown or duplicate identities and incorrect dimensions fail. Native model
and stationary experiment validation remains authoritative.

One shared probability validator rejects nonfinite/negative entries and
normalization errors beyond the explicitly recorded tolerance, without
renormalization or mutation. Every selected observation is a complete MID;
there are no per-mass use flags, hidden weights, or invisible replicate means.
Fingerprints bind the ordered canonical model, experiments, observations,
divergence configuration, optimizer configuration, and supplied starts.

KL uses its exact zero/support conventions, and `alpha == 1` dispatches exactly
to KL. Every finite positive real order is accepted, excluding booleans.
For order above one, observed positive mass outside predicted support gives
infinity. For order below one, partial support overlap can give a finite value;
disjoint support gives infinity. Stable log-domain arithmetic, including
near-one evaluation without snapping the order, must retain these semantics.
For numerical divergence evaluation, both sums must also lie within 16 ulps
of one (approximately `3.55e-15`). This explicit machine-simplex requirement
is stricter than the configurable schema tolerance: a schema-valid record is
not a promise that materially nonunit mass can be evaluated as a distribution.
Wider discrepancies raise a contextual error. Stable `expm1`/`log1p` identities
treat only these machine-roundoff sums as simplex arithmetic; neither input
vector is changed. This avoids an artificial `log(sum(p))/(alpha-1)` pole.

The total objective is the plain sum of per-observation components, each with
experiment, target, replicate, observed MID, predicted MID, and divergence.
This is a fitting criterion on MID distributions, not a likelihood or an
experimental observation law `P(Y | v)`.

## Shared feasible geometry and optimization

Generalize the existing `PreparedFluxRegion` to allow absent retention while
preserving Stage 1 public defaults. `fraction_of_optimum=None` means only the
canonical mass balances and bounds. The original biological objective remains
in the model; its FBA primal may anchor geometry but imposes no constraint on
unrestricted MFA. Explicit fractions retain the native validated max/min
objective row, never a contribution to the MID loss.

Generalize existing FVA, reduced geometry, hit-and-run sampling, and independent
state validation to that optional-retention case. Share one prepared geometry
and center between generated starts and fitting. FVA endpoints remain solely
affine-collapse/geometry evidence; they are never sampled independently or
assembled into a trial flux state.

SciPy SLSQP operates on `v = particular + basis @ theta` with the shared explicit
linear inequalities. SciPy is imported only when fitting needs it and belongs
to an optional `mfa` extra. Each trial reconstructs a complete canonical state
and is checked before any native EMU evaluation. Each experiment plan is
compiled once. Invalid states never enter EMU. True support mismatch retains
infinite scientific loss; undefined forward calculations and infeasible trials
are distinguished in diagnostics. No finite penalty or repaired final vector
is allowed. An initially undefined or nonfinite objective ends that start.
During optimization, expected infeasible, undefined-forward, or support-failing
trials are recorded and return exact `+inf` to SLSQP so its line search may
backtrack. A backend that cannot recover fails that start transparently.

Generated starts use a fixed recorded PCG64 seed and native sampling controls.
When complete starts are supplied, every supplied start is attempted in order;
`n_starts` controls generation only. Malformed supplied states fail validation.
Each attempt retains initial/final state and loss, backend status/success,
iterations, evaluations, validation outcome, and contextual trial failures.
An unevaluated loss is `None`, distinct from an exactly infinite divergence.
If no start succeeds, `MFAFitError` exposes all start diagnostics.

The selected result is the lowest exact loss among successful independently
validated fits. The final state is validated again against the original
compiled model and any explicit retention, and its native MIDs and exact loss
are reevaluated. Backend termination alone never establishes acceptance.

## Recovery and release gates

The identifiable programme uses the existing explicitly mapped complementary
tracer inflow topology, canonical reaction order `(Z_IN, A_IN, M_OUT)`, fixed
throughput `M_OUT=10`, and truth `(3,7,10)`. Its MID is
`(Z_IN/M_OUT, A_IN/M_OUT)`; mass balance and fixed throughput identify the full
state. Distinct nontruth feasible starts must recover the identified values
for KL and orders 0.5 and 2.

The non-identifiable programme allows throughput in `[2,10]`. Complete states
`(1.5,3.5,5)` and `(3,7,10)` have the same MID. Acceptance demonstrates matching
MIDs and compatible distinct states; it does not require unique scale recovery.
Both programmes generate exact observations through native EMU without noise.

Milestones are separately tested, guardian-reviewed, and committed: (1) data
and divergence; (2) shared geometry and multistart fitting; (3) recovery
acceptance; (4) documentation, dependency isolation, public API, and CI.
Existing FBA/FVA/sampling/EMU regression suites remain required. Final approval
requires the actual pushed SHA, green relevant CI, a clean branch, a mergeable
PR, and explicit guardian confirmation of the auditable mfapy lineage.
