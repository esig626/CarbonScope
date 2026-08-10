# Phase 2 native transient EMU shadow engine

## V1 scientific contract

Transient experiments use their own immutable contract; they do not add hidden fields to stationary experiments. An explicit `experiment_mode: transient` selects the transient parser. V1 accepts fixed complete physical flux states, fixed positive metabolite pool quantities, a tracer step at time zero, constant tracer distributions thereafter, stationary metabolite amount balances, and internal pools initially labelled exactly `unlabelled`.

Times are finite, nonnegative, strictly increasing, and returned in declared order. Integration always starts at zero even when the first requested time is later. Pool and flux units must be mutually consistent: `pool quantity / flux` sets the numerical time scale, and FluxEMU performs no unit conversion. One pool quantity applies to every EMU of a balanced dynamic metabolite. Source pools and purely terminal unbalanced targets do not receive fake pools.

```yaml
schema_version: 1
experiment_mode: transient
tracers: [...]
targets: [...]
timecourse:
  time_points: [0.0, 0.1, 1.0]
  initial_internal_mids: unlabelled
  pool_quantities:
    - {metabolite_id: citrate, quantity: 1.0}
  tolerances: {rtol: 1.0e-9, atol: 1.0e-12, mid: 1.0e-8}
```

Pool quantities are a list, not a mapping, so duplicate declarations remain detectable. Unknown, duplicate, missing-required, nonfinite, zero, and negative pool quantities fail validation. Tracer, target, time, pool, and initial-state ordering and values participate in the separate transient fingerprint.

## Equations and extraction

The compiled global state concatenates every dynamic EMU MID in deterministic layer, EMU, then mass-isotopologue order. The isotope material balance is `D dX/dt = B(X_lower(t), tracer, v) - A(v) X`.

`D` repeats the metabolite pool quantity for every component of each of its EMUs. `A` is the same turnover and same-size mapped-dependency operator used by the native stationary engine. `B` evaluates native source marginalisation and lower-size contributions at the current ODE state. Condensations use the same exact discrete MID convolution as stationary evaluation. Same-size cycles are coupled directly in the ODE, not replaced by stationary substitutions. At `dX/dt = 0`, the equation is the established stationary system `A X = B`; stable long-time trajectories are checked against native stationary evaluation independently of mfapy.

An unbalanced terminal target is the **instantaneous production MID** formed from flux-weighted mapped contributions at that time. It is not accumulated extracellular material or a medium concentration. Such behaviour requires an explicit future dynamic external-pool model.

## Numerics and boundaries

The native engine uses SciPy `solve_ivp` (declared by the `transient` optional extra), with defaults `rtol=1e-9` and `atol=1e-12`. Solver failure, nonfinite output, unacceptable negative components, or normalization error fails explicitly. The engine does not clip, renormalise, or project MIDs onto a simplex. SciPy is imported locally by the transient path so importing the stationary EMU package does not depend on SciPy.

V1 does not support time-varying fluxes or pool sizes, pulse-chase schedules, tracer switches after zero, arbitrary prelabelled internal pools, abundance dynamics, growth dilution not represented by turnover, or non-steady metabolite concentrations. It does not perform inverse MFA.

mfapy remains only a temporary historical shadow for differential validation. It performs no native ODE construction, graph construction, source evaluation, atom propagation, condensation, integration, or target extraction.
