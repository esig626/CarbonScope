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

## Declarative finite hypothesis testing

To compare explicit H0/H1 constraints on a common model, use the separate workflow command:

```bash
fluxemu test-hypotheses \
  --specification tests/fixtures/hypothesis_workflow/workflow.yaml \
  --output results/hypothesis_workflow
```

`--model MODEL.xml` can override the specification's model path. Experiment and model paths declared in the YAML resolve relative to that YAML. The workflow applies reaction-bound restrictions, samples complete feasible states using declared counts and seeds, evaluates stationary EMU and forms ordered genuine-count law families. It writes `report.json` and `summary.txt`. The existing `fluxemu run` interface and its forward-analysis outputs are unchanged.

The Python entry point is `fluxemu.run_hypothesis_testing_workflow(...)`. Exact finite minimax uses the optional `testing` installation extra. Invalid input or failed workflow construction returns CLI exit code 2; a valid report containing explicit refusals of optional statistical procedures returns 0. See [HYPOTHESIS_WORKFLOW.md](HYPOTHESIS_WORKFLOW.md) for a complete example and the distinction between bounds, minimax values and achieved errors.

## FBA and FVA

The physical flux model is solved natively with HiGHS. CarbonScope provides a cold reference FVA implementation and a reusable FastFVA implementation based on the shared memory computational architecture of VFFVA, adapted to HiGHS.

FVA gives reaction wise feasible extrema subject to the declared retained objective constraint. Those extrema are diagnostics. They are not generally jointly feasible and are never assembled into a flux vector.

## Complete feasible states

When an ensemble is requested, CarbonScope samples complete vectors from the constrained feasible region. The sampler operates in a numerically reduced affine hull and validates every returned state against reaction order, bounds, steady state mass balance and the retained objective.

The hypothesis-testing workflow uses each declared H0/H1 steady-state region without retaining an objective fraction. Its bounds are intersections with the common physical model, and every state is validated against that hypothesis's constrained bounds. The native experiment's `fva_fraction_of_optimum` applies to the forward-analysis path only.

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
