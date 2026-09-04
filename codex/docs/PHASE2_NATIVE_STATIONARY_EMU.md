# Phase 2 native stationary EMU engine

> **Historical delivery notes.** Native stationary EMU began as a parity engine
> in this milestone. It is now the public forward engine used by deterministic
> and sampled Stage 1 orchestration; see
> [Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md).

## Execution boundary and identity

The native engine consumes only a validated `CanonicalModel`, a validated
`StationaryExperimentSemantics`, and complete `CanonicalFluxState` records.
`EMU(metabolite_id, atom_positions)` is immutable. Atom positions retain the
order obtained by following the target's declared positions through explicit
`AtomTransition` records; they are not globally sorted. Targets, reactions,
mapping branches, substrate participants, flux states, and output
isotopologues retain declared order.

`compile_emu_plan` starts at each requested target and follows every explicit
mapping branch backwards. Selected source atoms are grouped in declared
substrate-participant order. Tracers terminate traversal. Non-source precursor
EMUs are recursively compiled. A missing atom origin is a `MappingError`; no
stoichiometric atom-map inference or Boolean-symmetry expansion exists.

The immutable plan records model and experiment fingerprints, target order,
EMU discovery order, contribution order, source EMUs, and size-ordered layers.
It is reusable for every complete flux state in a batch.

## Native numerical rules

Tracer subset MIDs are native marginals of declared full-metabolite binary
isotopomers. FluxEMU counts labels at the selected positions and applies no
natural-abundance correction. Multiple precursor MIDs are combined by ordered
discrete convolution.

For each EMU size, one matrix contains balanced unknown EMUs in discovery
order. Physical consumption flux supplies the diagonal turnover. Explicit
same-size one-precursor branches supply off-diagonal coefficients. Tracer and
lower-size condensation contributions supply the right-hand side. All mass
columns are solved together with `numpy.linalg.solve`. Rank-deficient or
singular layers fail; the implementation never uses a pseudoinverse,
least-squares fallback, clipping, or silent normalization.

An unbalanced target remains a boundary product. Its MID is the mixture of
explicit mapped production contributions weighted by directed reaction flux
and exact branch weight. Zero productive flux fails explicitly. No balance
role, excreted flag, or reaction is synthesized.

Each layer reports dimension, rank, condition number, maximum absolute
residual, minimum component, and maximum normalization error. Public native
predictions reuse the existing FluxEMU stationary result records; matrices
remain internal.

## Supported V1 semantics

For a direct isotope mapping without a `FluxProjectionRule`, the physical flux
reaction must be directionally unambiguous from its canonical bounds:
nonnegative bounds mean forward and nonpositive bounds mean reverse, and the
corresponding `IsotopeReaction.direction` must agree. The final native model can
also express directional isotope components of signed or sign-spanning physical
fluxes through explicit `positive_part` projection rules, covered-direction
metadata, and direction-activity certificates. No direction or gross flux is
inferred from a signed net value.

Declared tracers are fixed isotope sources. Balanced metabolites are solved
unknown pools. Unbalanced non-tracers are allowed only as requested terminal
products; encountering one as a precursor is invalid. Any physical flux that
produces a planned pool without an explicit isotope mapping is rejected.
Only `correction: no` is supported in this milestone.

## Current public status and benchmark inventory

`compile_emu_plan` and `evaluate_stationary` are public native APIs, and
`run_native_stationary_analysis` plus `run_native_stationary_ensemble` compose
them with native HiGHS flux analysis. Native execution imports neither mfapy
nor COBRApy; SciPy is not required for stationary EMU. The older
`run_stationary_forward` function remains a separate compatibility path rather
than the primary Stage 1 route.

The repository retains independent Antoniewicz full-isotopomer and
glucose-to-TCA oracles plus the official mfapy Example 0 frozen Glue MID.
Example 0 maps to the same checked authoritative Antoniewicz chemistry, with
the native boundary-target rule replacing mfapy's artificial export workaround.
It also now includes a persisted 95-reaction E. coli Stage B2 canonical model
and ordered 12-target native experiment used for sampled end-to-end acceptance.
