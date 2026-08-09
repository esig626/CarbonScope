# Phase 2: canonical stationary forward execution

## Phase 1 foundation

Phase 1 introduced immutable `CanonicalModel` and
`StationaryExperimentSemantics` records, validation, deterministic identity,
and an authoritative one-way projection from checked COBRA models and YAML
configuration. It did not make those canonical records executable: the old
forward entry point still accepted an `MfapyModelBundle` and
`ExperimentConfig`.

## Canonical execution boundary

`fluxemu.execution.run_stationary_forward` accepts only:

- a validated `CanonicalModel`;
- validated `StationaryExperimentSemantics`; and
- one complete reaction-ID-to-flux mapping, or ordered
  `CanonicalFluxState` records for a batch.

Every flux reaction is required, unknown reactions are rejected, values are
checked against canonical bounds, and backend vectors are assembled in
declared reaction and mapping-branch order. The canonical inputs are immutable
and are not changed during compilation or execution.

The result is a `StationaryForwardResult` containing ordered FluxEMU-owned
`StationaryMID` and long-form `StationaryMIDValue` records. Raw mfapy objects
are not returned. Every MID is checked for its expected length, finite and
non-negative fractions, and normalization within the caller's declared
tolerance. Invalid results are not renormalized.

## Temporary mfapy backend

`fluxemu.backends.mfapy` compiles canonical records directly into mfapy's
in-memory dictionaries. It does not read COBRA objects, legacy isotope
metadata, reaction-map CSV files, or mfapy text models.

Each canonical mapping branch becomes one ordered backend reaction carrying
that branch's exact weight. Atom labels are a backend encoding derived only
from explicit `AtomTransition` records. Canonical symmetry booleans never
create branches or weights; mfapy symmetry generation is disabled for this
route. Tracers define carbon sources, canonical flux balance semantics define
boundary metabolites, and targets retain their experiment order.

mfapy remains responsible only for compiling and numerically evaluating the
stationary EMU equations. It no longer owns the model chemistry, tracer
semantics, target semantics, complete flux-state contract, or result contract.
Backend imports are lazy, so importing `fluxemu.model` or the public execution
records does not import mfapy or SciPy.

## Dependency contract

The audited mfapy source imports NumPy and SciPy for stationary execution.
FluxEMU already requires NumPy 2.x. SciPy 1.13 is the first SciPy release line
supporting NumPy 2.0, so the `mfapy` optional dependency group declares
`scipy>=1.13`. The dependency is backend-scoped because canonical model use
does not require an execution engine.

## Scientific parity gate

`test_canonical_forward_parity.py` projects the checked Control 0 and Control 1
models once, then runs canonical science and the same complete frozen flux
state through the new backend. It independently runs the existing legacy
COBRA/configuration route and compares every requested target and isotopologue
at an absolute tolerance of `1e-12`, matching the existing forward regression
tolerance. It also checks deterministic repeated execution and exact compiled
branch order, IDs, weights, and transitions for v5, v6, and v7.

Codex Cloud lacks SciPy and therefore cannot execute mfapy. The GitHub Actions
workflow **Canonical stationary forward parity** installs the `mfapy` extra on
Python 3.11 and is the authoritative Control 0 and Control 1 execution gate.
The milestone is complete only after that workflow passes.

## Remaining work before a native backend

FluxEMU still depends on mfapy and SciPy for numerical stationary EMU
execution. A future native backend would replace that numerical compiler and
solver behind the same canonical boundary. Native EMU execution, inverse MFA,
optimization, time-course simulation, and inference are outside this
milestone.
