# Phase 3B: deterministic one-model native flux to stationary MID

This document records the preserved deterministic API delivered before the
sampled Stage 1 path. It remains supported alongside the complete
[Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md).

## Supported public path

FluxEMU supports a deterministic, native, forward analysis from one
scientific model definition:

```text
CanonicalModel
    |-- FluxModel ----> native HiGHS FBA
    |              `--> reusable native FastFVA
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
`run_highs_fva_reference` remains the cold correctness oracle, but production
orchestration uses the reusable FastFVA path.

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

## Relationship to the sampled Stage 1 path

This deterministic result deliberately contains `fba`, `fva`, `flux_state`, and
`mids`. The FBA primal is one complete solver-selected optimum, and its MIDs are
conditional on that state.

The completed sampled API, `run_native_stationary_ensemble`, instead returns
`fba`, reaction-wise `fva`, `flux_sampling`, and `mid_ensemble` as structurally
separate products. It samples complete jointly feasible states and evaluates
them with one compiled native EMU plan; it never independently samples or
combines FVA intervals. See [Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md)
for its algorithm, provenance, validation, limitations, and example.

Neither path performs inverse MFA, fits isotope measurements, or reports
confidence intervals or biological probabilities.
