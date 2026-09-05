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

Production `run_highs_vffva` compiles once and solves the declared biological
objective once. `run_prepared_highs_vffva` then starts explicit shared-memory
Python worker threads, capped by the reaction count. Each thread constructs,
owns, uses, and releases exactly one HiGHS instance. That instance receives the
prepared LP and retained-objective row once, uses simplex with HiGHS internal
`threads=1`, and remains alive across the complete maximum and minimum passes.

The coordinator first releases a global maximum pass over a shared queue of
reaction chunks. It waits until every maximum chunk is accounted before
releasing the minimum pass over a second queue. A worker that finishes one
chunk claims another dynamically; the faithful baseline and public default are
`chunk_size=50`. During a normal endpoint solve the worker changes the active
reaction's column cost to one, calls `Highs.run()`, reads the objective value,
and clears that cost in a `finally` block. The pass sets objective sense once,
so endpoint LPs are reoptimised on the same solver instead of rebuilt.
Completion order does not affect the canonical output order.

Worker initialization and endpoint failures are contextual `AnalysisError`s.
Cancellation stops further endpoint solves, drains both work queues, releases
each solver in its owning thread, joins every started thread, and returns no
partial FVA table. Omitting `workers` or passing `workers=None` selects one
solver-owning thread; an explicit integer greater than one selects that many
shared-memory workers. Multiprocessing, process IPC, and a Python
`if __name__ == "__main__"` spawn guard are not part of the production path.
The composed deterministic and ensemble APIs remain serial by default; the
ensemble exposes an explicit `fva_workers` choice.

The low-level APIs require `audit_endpoints` to be literally Boolean. Its
default, `False`, validates the prepared region once and keeps solver status,
finite endpoint values, the independently validated FBA witness, endpoint
completeness, and canonical minimum/maximum consistency in the production
path. `audit_endpoints=True` additionally retrieves each complete primal and
rechecks its length, finiteness, bounds, balanced mass residuals, declared
objective equivalence, and retained-objective constraint. The cold reference
continues to rebuild and independently validate every endpoint. This separates
expensive diagnostic work from the normal hot loop without changing the
correctness oracle or its `1e-7` acceptance tolerance.

## Relationship to VFFVA

This execution design is a HiGHS-native port/adaptation of the computational
architecture developed by Marouen Ben Guebila in
[VFFVA](https://github.com/marouenbg/VFFVA/tree/7cf7b82505bf99aed38a2073e3ed308f79e95802),
audited at pinned upstream commit
[`7cf7b82505bf99aed38a2073e3ed308f79e95802`](https://github.com/marouenbg/VFFVA/commit/7cf7b82505bf99aed38a2073e3ed308f79e95802).
See Ben Guebila, *VFFVA: dynamic load balancing enables large-scale flux
variability analysis*, BMC Bioinformatics 21, 519 (2020),
[DOI 10.1186/s12859-020-03711-2](https://doi.org/10.1186/s12859-020-03711-2).
FluxEMU translates the strategy rather than copying the C source or claiming a
new FastFVA algorithm.

| VFFVA computational operation | FluxEMU HiGHS adaptation |
| --- | --- |
| one persistent solver clone per OpenMP worker | one persistent `Highs` instance per owning Python thread |
| solver clone configured for one internal thread | each worker verifies HiGHS `threads=1` |
| outer maximum pass, then outer minimum pass | coordinator-enforced maximum queue, barrier, then minimum queue |
| OpenMP dynamic scheduling, Python wrapper default `dynamic,50` | shared-memory queues of dynamically claimed reaction chunks, default 50 |
| change one objective coefficient and reoptimise | `changeColCost(1)`, `Highs.run()`, read objective, `changeColCost(0)` |
| collect results independently of completion order | assemble the final DataFrame in declared reaction order |

FluxEMU deliberately preserves stronger scientific semantics. The biological
objective may contain any number of declared linear terms and may be maximised
or minimised. Its retained row remains `c^T v >= f z*` for maximisation or
`c^T v <= f z*` for minimisation, including fixed and signed/reversible
coordinates. It does not replace that row with upstream VFFVA's single
objective-reaction bound shortcut and does not adopt upstream objective
rounding. Porting the worker architecture therefore does not weaken model
semantics or canonical ordering.

COBRApy remains an optional compatibility and parity oracle, not a native
dependency. Production uses highspy directly and neither calls nor requires
the original VFFVA executable, CPLEX, GLPK, MPI, or multiprocessing.

## FastFVA acceptance evidence

The diagnostic driver
[`benchmark_vffva_highs_scaling.py`](../benchmarks/benchmark_vffva_highs_scaling.py)
and its tracked
[`vffva_highs_scaling_linux_x86_64.json`](../benchmarks/results/vffva_highs_scaling_linux_x86_64.json)
bind the measurement to the exact engine source, verify cold-reference parity
before timing, exercise `audit_endpoints=True`, record worker lifecycle
instrumentation, and keep three different comparisons separate. The recorded
environment was CPython 3.12.13, highspy/HiGHS 1.12.0, NumPy 2.5.2, nine allowed
CPUs on an AMD EPYC 9V74, with HiGHS and numerical-library thread counts fixed
at one. Each reusable path had one untimed warm-up and three measured calls at
fraction 0.9 with dynamic chunks of 50. Timing values are diagnostic, not CI
thresholds.

The portable command template and byte-identical model acquisition checks are
stored in the JSON. With those model files available, its shape is:

```bash
cd codex
PYTHONPATH=src python benchmarks/benchmark_vffva_highs_scaling.py \
  /path/to/iLJ478.xml.gz \
  --attempted-large-model /path/to/iJO1366.xml.gz \
  --remove-redundant-balance-rows --fraction 0.9 \
  --workers 1 2 4 8 --chunk-size 50 --warmups 1 --repeats 3 \
  --scaling-cold-repeats 1 \
  --source-remote-commit b672e3551e707e5dd6e7f003f60b04cfc040fbdd
```

### Internal cold-reference comparison

This ratio compares FluxEMU's deliberately cold, endpoint-rebuilding oracle
with its reusable one-worker engine. It is internal engineering evidence, not
a comparison with original VFFVA.

| Model | Cold median (s) | Reusable T1 median (s) | Cold / reusable T1 |
| --- | ---: | ---: | ---: |
| bundled E. coli core, 95 reactions / 190 endpoints | `0.264785` | `0.044715` | `5.922x` |
| iLJ478 benchmark-only algebraically equivalent independent equality-row basis, 652 reactions / 1,304 endpoints | `15.599710` | `1.894390` | `8.235x` |

The E. coli cold median contains three measured calls. The iLJ478 cold value is
only one measured sample after one warm-up, so its ratio is descriptive rather
than a stable estimate. Fresh iLJ478 preparation was about 74.92 seconds and is
reported separately; the reusable timings are warm-cache, full public-API
calls, not fresh-process end-to-end latency.

### Shared-memory threaded scaling

`T1/Tp` below compares only reusable public-API medians. The official BiGG
iLJ478 input failed the unchanged production rank gate because it contains
redundant steady-state rows. The benchmark used a certified, benchmark-only,
algebraically equivalent independent equality-row basis with 528 balanced rows;
this was not a production compiler change or a general model transformation.

| Workers | E. coli median (s) | E. coli `T1/Tp` | iLJ478 median (s) | iLJ478 `T1/Tp` | iLJ478 efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.044715` | `1.000x` | `1.894390` | `1.000x` | `1.000` |
| 2 | `0.029710` | `1.505x` | `1.110843` | `1.705x` | `0.853` |
| 4 | `0.030751` | `1.454x` | `0.637523` | `2.971x` | `0.743` |
| 8 | `0.034583` | `1.293x` | `0.484070` | `3.913x` | `0.489` |

The 95-reaction result exposes small-workload overhead and is not evidence for
general large-model scaling. The tracked record also documents rejected iIT341
and iSB619 candidates and the iJO1366 production-preparation timeout; it makes
no timing claim for a workload that failed the required numerical gate.

For historical context only, the older same-family E. coli Stage 1 artifact
recorded about `0.598091` seconds for two spawned process workers, whereas this
run recorded `0.029710` seconds for two shared-memory worker threads. This is
not a controlled attribution of the full difference to IPC removal: the engine
and measurement source changed, and full per-endpoint primal validation also
moved behind explicit audit mode. It supports only the narrower observation
that the production architecture no longer operates in the old spawn/IPC
overhead regime.

### Direct original-VFFVA attempt

The upstream repository was pinned to
`7cf7b82505bf99aed38a2073e3ed308f79e95802`. In the required order, the
benchmark attempted `make` for CPLEX and then `make SOLVER=glpk` from upstream's
`lib` directory. Both returned code 2 because `mpicc` was unavailable. The
environment also lacked `mpirun`, the CPLEX executable/Python API/installation
and licence, and the GLPK executable/headers/libraries. Therefore no upstream
binary ran and there is no direct original-VFFVA parity, timing, or speedup
claim. Exact commands, dependency probes, build output, and log hashes are in
the tracked JSON; the cold/reusable and `T1/Tp` results above are not substitutes
for that blocked comparison.

Native feasible-state sampling and sampled stationary-MID integration are
documented in the [Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md). FVA
endpoints remain independent diagnostics and are never treated as complete
flux states.
