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

The public `run_highs_fba` and `run_highs_fva_reference` APIs import highspy only
when execution begins. Install `fluxemu[highs]` (`highspy>=1.11,<1.13`). CI prints
the exact installed version. Solver output is disabled, one thread and simplex are
selected, and every non-optimal status becomes an explicit `AnalysisError`.

FBA independently checks vector length, finiteness, bounds, balanced mass residuals,
and the recomputed objective at fixed `1e-7` tolerances. It never clips fluxes.

Reference FVA first solves the biological objective. For fraction `f`, it retains
`c v >= f z*` for maximisation and `c v <= f z*` for minimisation. It then constructs
a cold independent LP for each reaction minimum and maximum, in canonical order,
with exactly one endpoint cost. Endpoint feasibility and objective retention are
independently checked. Thus it performs two endpoint solves per reaction plus FBA.

COBRApy remains a compatibility and parity oracle, not a native dependency. The
benchmark emits diagnostic JSON medians after warm-up; runner timings are not a
scientific performance result and no performance superiority is claimed.

## Deliberate limitations and next boundary

This phase does not implement native sampling, inverse MFA, or the VFFVA-inspired
performance layer. A later FastFVA milestone is reserved for one persistent HiGHS
LP, cheap objective switching, basis reuse/warm starts, strategic solve ordering,
dynamic scheduling, and multiprocessing without rebuilding unnecessary Python
state. Stationary/transient EMU and atom-mapping behavior are unchanged.
