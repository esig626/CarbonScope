# Native stationary workflow

## Public sequence

CarbonScope's native stationary path is

```text
SBML Level 3 FBC
  -> FluxModel
  -> explicit authoritative isotope mapping
  -> CanonicalModel
  -> HiGHS FBA / FVA
  -> complete feasible flux state(s)
  -> stationary EMU
  -> ordered MID predictions
```

The command line entry point remains:

```bash
fluxemu run --model model.xml --experiment experiment.yaml --output results
```

The equivalent Python orchestration lives in `fluxemu.analysis`.

## FBA and FVA

The physical flux model is solved natively with HiGHS. CarbonScope provides a cold reference FVA implementation and a reusable FastFVA implementation based on the shared memory computational architecture of VFFVA, adapted to HiGHS.

FVA gives reaction wise feasible extrema subject to the declared retained objective constraint. Those extrema are diagnostics. They are not generally jointly feasible and are never assembled into a flux vector.

## Complete feasible states

When an ensemble is requested, CarbonScope samples complete vectors from the constrained feasible region. The sampler operates in a numerically reduced affine hull and validates every returned state against reaction order, bounds, steady state mass balance and the retained objective.

A sampled ensemble is a finite Markov chain output, not proof of independence or convergence.

## Stationary EMU

The EMU compiler uses only explicit atom mappings, mapping branches, direction semantics and flux projection rules in the canonical model. It does not infer carbon fate from reaction names or stoichiometry.

The evaluator solves each required stationary EMU layer directly and reports rank, conditioning, residual and normalisation diagnostics. Singular or undefined systems fail explicitly; no pseudoinverse or hidden repair is used.

## Ordering and provenance

Declared reaction, experiment, target, replicate and mass class order are retained throughout. Model and experiment fingerprints bind calculations to their scientific inputs.

## Dependencies

The standard native stationary path requires the base installation only:

```bash
python -m pip install .
```

It does not require COBRApy or mfapy.
