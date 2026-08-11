# Phase 3B: one-model native flux to stationary MID

## Supported public path

FluxEMU now supports a deterministic, native, forward analysis from one
scientific model definition:

```text
CanonicalModel
    |-- FluxModel ----> native HiGHS FBA
    |              `--> native cold-reference HiGHS FVA
    `-- IsotopeModel --\
                         exact complete FBA optimum flux state
                                      |
                                      v
                         native stationary EMU
                                      |
                                      v
                               predicted MIDs
```

```python
from fluxemu.analysis import run_native_stationary_analysis

result = run_native_stationary_analysis(
    model,
    experiment,
    fva_fraction_of_optimum=1.0,
)

result.fba
result.fva
result.flux_state
result.mids
```

The model is defined once. FBA and FVA consume the `FluxModel`, while EMU
topology and explicit atom mappings come from the `IsotopeModel` held by the
same `CanonicalModel` object. The separate `StationaryExperimentSemantics`
contains only experiment-specific tracers and requested targets. The analysis
does not construct COBRApy or mfapy models.

For flux-only use, `run_native_fba(model)` and
`run_native_fva(model, fraction_of_optimum=...)` validate the complete
canonical model and delegate to the established native HiGHS routines.

## FVA is not a flux state

Every FVA minimum or maximum can come from a different LP solution. A column
of FVA minima, a column of maxima, or any mixture of endpoints is therefore not
guaranteed to be one jointly feasible state. The workflow never converts FVA
ranges into a `CanonicalFluxState` and never passes them to EMU. FVA is returned
only as diagnostic characterization.

The exposed `result.flux_state` is copied, in declared canonical reaction
order and without clipping, rounding, normalization, or sorting, exclusively
from the complete `result.fba.fluxes` primal selected by HiGHS. That exact state
is the input to the native stationary EMU evaluator. Consequently, the
predicted MIDs are conditional on that particular FBA optimum.

If the optimum is non-unique, another complete optimal flux state can produce
different MIDs. FVA can reveal reaction-level underdetermination, but it does
not describe a joint flux distribution and does not quantify the resulting
MID distribution.

## Scope and next milestone

This milestone adds orchestration, not a solver or a new scientific engine. It
uses the existing cold-reference HiGHS FBA/FVA implementation and the existing
native stationary EMU compiler/evaluator unchanged. It does not perform inverse
MFA, fit isotope measurements, or propagate uncertainty.

Native feasible-flux sampling is the next milestone. It will sample complete
jointly feasible states so that flux-space uncertainty can later be propagated
into MID ensembles; independently sampling or combining FVA intervals will not
be used.
