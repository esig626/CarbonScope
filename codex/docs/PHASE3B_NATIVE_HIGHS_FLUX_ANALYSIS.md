# Phase 3B: native HiGHS flux analysis

This milestone provides a correctness-first LP reference engine. `compile_flux_lp`
translates `FluxModel` directly to immutable CSR-style row arrays. Reaction and
balanced-metabolite orders are their declared tuple orders; only metabolites whose
`steady_state_balanced` value is literally true produce `S v = 0` rows. Reaction
bounds are copied unchanged. Duplicate terms are summed without creating a dense
matrix. A SHA-256 fingerprint covers ordering, sparse values, bounds, objective,
and direction.

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
`c v >= f z*` for maximisation and `c v <= f z*` for minimisation. It then constructs
a cold independent LP for each reaction minimum and maximum, in canonical order,
with exactly one endpoint cost. Endpoint feasibility and objective retention are
independently checked. Thus it performs two endpoint solves per reaction plus FBA.

Production `run_highs_vffva` compiles once, solves the biological objective once,
and constructs the retained-objective row once in every worker-local HiGHS model.
Each of the `2N` endpoint jobs changes only one column cost and objective sense.
Repeated `Highs.run()` calls on the same simplex model retain HiGHS' incumbent
basis and reoptimize it; explicit `getBasis`/`setBasis` round trips are deliberately
avoided because the basis never leaves its owning solver. HiGHS internal threads
are fixed at one. A single worker uses the same reusable worker object. Multiple
spawned processes consume `imap_unordered(..., chunksize=1)`, which dynamically
assigns the next endpoint to the next available process, and results are restored
to canonical reaction order.

The architecture is inspired by M. Ben Guebila, *VFFVA: dynamic load balancing
enables large-scale flux variability analysis*, BMC Bioinformatics 21, 519 (2020),
DOI: 10.1186/s12859-020-03711-2. FluxEMU neither contains nor executes original
VFFVA code. This is VFFVA-style dynamically scheduled native HiGHS FVA.

COBRApy remains a compatibility and parity oracle, not a native dependency. No
CPLEX, GLPK, external VFFVA executable, or MPI runtime is used.

## FastFVA acceptance evidence

The non-gating benchmark at
`benchmarks/benchmark_highs_fva_reference.py` measures cold native FVA, reusable
native FVA with one worker, reusable native FVA with multiple workers when
available, and an explicitly requested optional COBRApy comparator. It records
raw repetitions, medians, model dimensions, worker counts, software versions,
correctness differences, and speedup ratios as JSON. It fails before reporting
timings if fast and cold results differ in order or beyond solver tolerance.

On the bundled 95-reaction E. coli core problem, Python 3.12.13 and HiGHS 1.12.0,
five repetitions after one warm-up produced a cold median of 0.253030 seconds and
a reusable one-worker median of 0.066404 seconds: a 3.810x speedup, with maximum
absolute endpoint difference 8.38e-12. Two process workers remained correct but
were slower on this small model because spawn and IPC overhead dominated. The
machine-readable record is
`benchmarks/results/stage1_native_fva_linux_x86_64.json`; timing ratios are not CI
gates.

Native feasible-state sampling and sampled stationary-MID integration are
documented in the final Stage 1 workflow. FVA endpoints remain independent
diagnostics and are never treated as complete flux states.
