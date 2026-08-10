# Phase 2 native stationary EMU shadow engine

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

Flux reactions must be directionally unambiguous from their canonical bounds:
nonnegative bounds mean forward and nonpositive bounds mean reverse. The
corresponding `IsotopeReaction.direction` must agree. Bounds spanning negative
and positive flux are rejected rather than interpreted as net/exchange flux.

Declared tracers are fixed isotope sources. Balanced metabolites are solved
unknown pools. Unbalanced non-tracers are allowed only as requested terminal
products; encountering one as a precursor is invalid. Any physical flux that
produces a planned pool without an explicit isotope mapping is rejected.
Only `correction: no` is supported in this milestone.

## Shadow status and benchmark inventory

The public `run_stationary_forward` route remains mfapy-backed. Native
execution is a parallel internal shadow and imports neither mfapy, COBRApy,
nor Matplotlib. The repository contains the independent Antoniewicz full-
isotopomer solver, the independent glucose-to-TCA solver, and the official
mfapy Example 0 frozen expected Glue MID. Example 0 maps to the same checked
authoritative Antoniewicz chemistry, with the native boundary-target rule
replacing mfapy's artificial export workaround.

No frozen seven-target E. coli stationary forward fixture or frozen complete
E. coli flux vectors are present. Curated E. coli-related transition-library
entries and vendor samples are not a shadow fixture, so Gate E is unavailable
and was not reconstructed.
