# Native transient forward EMU

CarbonScope includes a native fixed flux transient EMU engine for explicit tracer step experiments.

Install SciPy support with:

```bash
python -m pip install '.[transient]'
```

## Experiment semantics

`TransientExperimentSemantics` declares:

* stationary tracer definitions;
* ordered targets;
* ordered requested time points beginning at zero;
* one explicit positive pool quantity for every required dynamic balanced metabolite;
* the initial internal MID policy (`unlabelled` in the current V1 contract).

`compile_transient_emu_plan(...)` reuses the native stationary EMU topology and creates a deterministic global state vector layout. `evaluate_transient(...)` integrates each complete fixed flux state with SciPy `solve_ivp`.

## Numerical policy

Default integration uses RK45 with explicit `rtol` and `atol`. The result reports solver success, internal time point count, RHS evaluations, minimum MID component, maximum normalisation error and maximum MID sum derivative error.

No clipping, simplex projection or hidden renormalisation is performed. Invalid or singular configurations fail explicitly.

## Terminal targets

For an unbalanced terminal target, CarbonScope reports the instantaneous flux weighted production MID rather than inventing an unmodelled extracellular mixing pool.

## Validation

`tests/test_native_transient_emu.py` includes an analytic first order labelling control, complete state batch ordering, terminal target semantics, required pool validation and long time convergence to the native stationary solution.

The transient engine is forward simulation only. Transient inverse MFA is not implemented.
