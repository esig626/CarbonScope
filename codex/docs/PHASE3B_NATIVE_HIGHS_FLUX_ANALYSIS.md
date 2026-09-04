# Phase 3B: native HiGHS flux analysis

This milestone provides native FBA, a correctness-first cold FVA oracle, and a
reusable dynamically scheduled FastFVA engine. `compile_flux_lp` translates
`FluxModel` directly to immutable CSR-style row arrays. Reaction and
balanced-metabolite orders are their declared tuple orders; only metabolites
whose `steady_state_balanced` value is literally true produce `S v = 0` rows.
Reaction bounds are copied unchanged. Duplicate terms are summed without
creating a dense matrix. A SHA-256 fingerprint covers ordering, sparse values,
bounds, objective, and direction.

COBRA flux balance and isotope source/terminal roles are separate concepts.
`project_cobra_flux_model` faithfully preserves every COBRA metabolite balance row;
isotope metadata can never silently alter native FBA feasibility. In contrast,
`project_cobra_model` retains the established isotope-forward policy in which
reviewed tracer-source and excreted pools may be unbalanced. The implementations
select these policies explicitly rather than conflating them.

The public `run_highs_fba`, `run_highs_fva_reference`, and `run_highs_vffva` APIs
import highspy only when execution begins. The standard install supplies
`highspy>=1.11,<1.13`. CI prints
the exact installed version. Solver output is disabled, one thread and simplex are
selected, and every non-optimal status becomes an explicit `AnalysisError`.

FBA independently checks vector length, finiteness, bounds, balanced mass residuals,
and the recomputed objective at fixed `1e-7` tolerances. It never clips fluxes.

Reference FVA first solves the biological objective. For fraction `f`, it retains
`c^T v >= f z*` for maximisation and `c^T v <= f z*` for minimisation. It then constructs
a cold independent LP for each reaction minimum and maximum, in canonical order,
with exactly one endpoint cost. Endpoint feasibility and objective retention are
independently checked. Thus it performs two endpoint solves per reaction plus FBA.

Production `run_highs_vffva` compiles once, solves the biological objective once,
and constructs the retained-objective row once in every worker-local HiGHS model.
Each of the `2N` endpoint jobs changes only one column cost and objective sense.
Repeated `Highs.run()` calls on the same simplex model retain HiGHS' incumbent
basis and reoptimize it; explicit `getBasis`/`setBasis` round trips are deliberately
avoided because the basis never leaves its owning solver. HiGHS internal threads
are fixed at one. Omitting `workers` or passing `workers=None` uses one reusable
worker in the calling process; this is the portable default. An explicit
`workers>1` opts into spawned processes consuming
`imap_unordered(..., chunksize=1)`, which dynamically assigns the next endpoint
to the next available process, and results are restored to canonical reaction
order. The composed deterministic path is serial. The ensemble path defaults
to serial and exposes explicit opt-in through `fva_workers>1`, while callers
of the lower-level FVA APIs use `workers>1`.

Python scripts that request multiple workers must guard the call so spawned
children can import the module safely:

```python
if __name__ == "__main__":
    fva = run_highs_vffva(model, workers=2)
```

The architecture is inspired by M. Ben Guebila, *VFFVA: dynamic load balancing
enables large-scale flux variability analysis*, BMC Bioinformatics 21, 519 (2020),
DOI: 10.1186/s12859-020-03711-2. FluxEMU neither contains nor executes original
VFFVA code. This is VFFVA-style dynamically scheduled native HiGHS FVA.

COBRApy remains a compatibility and parity oracle, not a native dependency. No
CPLEX, GLPK, external VFFVA executable, or MPI runtime is used.

## FastFVA acceptance evidence

The non-gating benchmark at
`codex/benchmarks/benchmark_highs_fva_reference.py` measures cold native FVA, reusable
native FVA with one worker, reusable native FVA with multiple workers when
available, and an explicitly requested optional COBRApy comparator. It records
raw repetitions, medians, model dimensions, worker counts, software versions,
correctness differences, and speedup ratios as JSON. It fails before reporting
timings if fast and cold results differ in order or beyond solver tolerance.

From the repository root, reproduce the tracked run with:

```bash
cd codex
PYTHONPATH=src python benchmarks/benchmark_highs_fva_reference.py \
  ecoli-core --fraction 0.9 --repeats 5 --parallel-workers 2 \
  > /tmp/stage1_native_fva.json
```

On the bundled 95-reaction E. coli core problem at source commit
`d1056facd9d41a1bebeb2bc21eeede727766c843`, Python 3.12.13 and HiGHS
1.12.0, two independent invocations with one warm-up and five measured
repetitions per path produced cold medians of `0.253045` and `0.249138` seconds
and reusable one-worker medians of `0.099676` and `0.099711` seconds. The
per-run serial speedups were `2.538659x` and `2.498593x`, with a median of
`2.518626x`; maximum absolute serial endpoint difference was
`3.652189662e-12`. Two process workers remained correct (maximum difference
`2.903455254e-12`) but were slower on this small model because spawn and IPC
overhead dominated. The machine-readable record is
[`benchmarks/results/stage1_native_fva_linux_x86_64.json`](../benchmarks/results/stage1_native_fva_linux_x86_64.json);
timing ratios are diagnostic and are not CI gates.

Native feasible-state sampling and sampled stationary-MID integration are
documented in the [Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md). FVA
endpoints remain independent diagnostics and are never treated as complete
flux states.
