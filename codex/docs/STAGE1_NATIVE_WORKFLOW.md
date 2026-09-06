# Stage 1 native workflow

FluxEMU Stage 1 provides a public, native forward path from a canonical
constraint-based model to an ensemble of stationary mass-isotopomer
distributions (MIDs):

```text
CanonicalModel
  -> native HiGHS FBA
  -> reusable native FastFVA
  -> complete feasible flux-state sampling
  -> one compiled native stationary EMU plan
  -> sample-indexed MID ensemble
```

The standard package includes FluxEMU, NumPy, pandas, PyYAML, python-libSBML,
and HiGHS. The in-memory Stage 1 computation itself uses the native FluxEMU,
NumPy, pandas, and HiGHS path; YAML and libSBML support the file/CLI boundary.
It does not import or require COBRApy, optlang, mfapy, or SciPy. Those packages
remain confined to optional compatibility, parity, or transient paths.

## Keep the four products separate

| Product | Meaning | Valid downstream use |
| --- | --- | --- |
| FBA primal | One complete optimum selected by HiGHS | The deterministic MID path |
| FVA ranges | Independent minimum and maximum LP optima for each reaction | Diagnosing reaction-wise flexibility |
| Flux samples | Ordered, complete, jointly feasible `CanonicalFluxState` records | Batch stationary EMU evaluation |
| MID ensemble | Deterministic MID predictions conditional on each sampled state | Comparing forward predictions across the sampled chain |

An FVA minimum for one reaction and an FVA maximum for another can come from
different LP solutions. A column of minima, a column of maxima, or any mixture
of FVA endpoints is therefore not a flux state. FluxEMU never turns FVA ranges
into `CanonicalFluxState` records and never supplies them to EMU.

The deterministic result is conditional on one solver-selected FBA optimum.
The ensemble result is a collection of conditional forward predictions, not a
confidence interval, an inverse-MFA result, or a calibrated biological
probability distribution.

## Retained biological objective

Let `z*` be the native FBA optimum, `c` the declared linear biological
objective, and `f` the requested `fraction_of_optimum` in `(0, 1]`. The retained
region is

```text
S v = 0
l <= v <= u
c^T v >= f z*    for maximisation
c^T v <= f z*    for minimisation
```

Within the composed ensemble path, FVA and sampling use the same
`PreparedFluxRegion`, so they share the same compiled LP, validated FBA primal,
objective direction, optimum, retained bound, and flux-model fingerprint.
Internally, equality-implied and fixed-coordinate objective terms are removed
and the remaining objective is scaled for numerical conditioning without
changing the declared constraint.

At `f == 1`, the sampler deliberately adds the nonconstant part of `c^T v = z*`
to the affine equalities and samples relative to the optimal face. Any other
case in which the represented retained bound is exactly the biological optimum,
including a zero optimum, is treated the same way. Otherwise, `f < 1` keeps the
retained max/min inequality in the sampled polytope. A fractional max problem
with a negative optimum, or a fractional min problem with a positive optimum,
would make the literal multiplicative bound stricter than the optimum; FluxEMU
rejects that infeasible arithmetic rather than silently changing its meaning.

## FastFVA architecture and oracle

`run_highs_fva_reference` is the permanent cold correctness oracle. It builds a
fresh HiGHS problem for every reaction minimum and maximum and independently
checks every returned primal.

Production `run_highs_vffva` and `run_prepared_highs_vffva` create at most one
reusable HiGHS model per explicit shared-memory worker thread. Each owning
thread constructs its LP and retained-objective row once, verifies HiGHS
internal `threads=1`, and keeps the same solver alive for a dynamically
scheduled maximum pass followed by a minimum pass. Each successful endpoint
changes one column cost, calls `Highs.run()`, reads the objective, and clears
the cost; it does not rebuild the LP. The default scheduling baseline is
dynamic reaction chunks of 50. Results are restored to exact canonical
reaction order regardless of completion order.

Omitting `workers` or passing `workers=None` selects one solver-owning thread.
An explicit integer greater than one selects that many worker threads, capped
by the reaction count; no multiprocessing spawn guard is needed. A failure
names the reaction and min/max direction, cancels further solves, drains the
queues, releases solvers in their owning threads, and joins all started
workers. The composed deterministic path is serial by default. The ensemble
path also defaults to serial and exposes explicit opt-in through
`fva_workers>1`.

The lower-level FVA APIs also expose `chunk_size=50` and the strict-Boolean
`audit_endpoints=False` default. Production keeps solver status, finite result,
validated-FBA-witness, completeness, and canonical range checks in the hot
path. `audit_endpoints=True` additionally retrieves and independently validates
every endpoint primal, including bounds, balanced mass residuals, and the
retained objective. The cold oracle always performs full endpoint validation.

This is a HiGHS-native port/adaptation of Marouen Ben Guebila's
[VFFVA computational architecture](https://github.com/marouenbg/VFFVA/tree/7cf7b82505bf99aed38a2073e3ed308f79e95802),
audited at pinned upstream commit
`7cf7b82505bf99aed38a2073e3ed308f79e95802`, not a claim that FluxEMU invented
dynamic FastFVA. FluxEMU preserves its stronger arbitrary multi-term,
maximisation/minimisation retained-objective row and does not use VFFVA's
single-objective-reaction bound or rounding shortcuts. Production calls only
HiGHS; it does not call or require multiprocessing, the original VFFVA binary,
CPLEX, GLPK, or MPI. The detailed operation mapping and attribution are in
[Phase 3B: native HiGHS flux analysis](PHASE3B_NATIVE_HIGHS_FLUX_ANALYSIS.md).

The composed deterministic and ensemble APIs prepare the LP and solve the
biological objective once. The ensemble then uses one FastFVA result for both
reporting and sampling. The sampler constructs its reduced retained geometry
once from those prepared objects; it does not recompile the flux LP or solve the
biological objective a second time.

## Native sampling algorithm

The sampler is `random-direction-hit-and-run`, algorithm version `1`, with
NumPy's `PCG64` bit generator. It operates as follows:

1. Validate the prepared LP, FBA primal, retained-objective metadata, and the
   ordered FastFVA result.
2. Use FVA for collapsed-coordinate facial reduction, numerical-degeneracy and
   unboundedness checks, and binding the evidence by its ordered range digest.
   FVA endpoints are not warm-up states, coordinate distributions, or
   independent sampling intervals.
3. Build the affine hull from balanced-metabolite equalities, fixed/collapsed
   coordinates, and the objective equality whenever the retained bound exactly
   equals the optimum (including `f == 1`).
4. Factor an orthonormal null-space basis and solve a native HiGHS Chebyshev-
   centre problem in that reduced hull.
5. Draw an isotropic Gaussian direction, normalize it, compute the complete
   feasible chord, and draw one uniform point on that chord. Finite numerically
   zero-width chords are retained as self-loop Markov transitions rather than
   redrawn.
6. Apply the requested burn-in and thinning, then convert each ambient vector
   directly to a complete state in canonical reaction order.

A zero-dimensional retained region returns the requested number of records
containing the same unique feasible vector under distinct ordered sample IDs.
Non-finite bounds, an unbounded reduced direction, an ambiguous numerical rank,
a degenerate positive-dimensional interior, or inability to make the next
transition raises `AnalysisError`. FluxEMU does not invent a truncation, clip a
state, repair it, or replace it with a different draw.

Preparation also fails closed when material objective activity lies below
HiGHS coefficient resolution, when an exact retained optimal face is
solver-ambiguous, or when the retained bound cannot be represented without
material numerical distortion. This is especially important for exact-face
sampling; such cases are not approximated as a nearby polytope.

### Sampling law and limits of the claim

The random-direction hit-and-run transition kernel has relative-volume uniform
stationary target on the numerically reduced retained affine polytope. A finite
run from the computed centre is a correlated Markov chain. Burn-in and thinning
do not prove convergence, adequate mixing, independence, representativeness, or
biological probability. Exact replay for a fixed seed is tested within the same
algorithm and numerical-library environment; it is not promised across NumPy,
HiGHS, BLAS, platform, or algorithm-version changes.

## Independent validation and provenance

Every returned state is independently validated against the original compiled
model rather than the sampler's reduced geometry. Validation requires exact
reaction membership and order, unique sample IDs, finite values, both reaction
bounds, balanced-metabolite steady state, and the retained max/min objective.
Invalid states produce contextual per-state diagnostics; the public sampler
fails rather than returning an invalid batch.

The principal numerical thresholds recorded in provenance are:

| Check | Tolerance |
| --- | ---: |
| Reaction bounds | `1e-7` |
| Normalized/conditioned mass residual and retained-objective checks | `1e-7` |
| FVA collapsed-coordinate width | `1e-10` absolute flux units |
| FVA numerically ambiguous width | greater than `1e-10` and at most `1e-8` absolute flux units |
| Affine rank | `sqrt(float epsilon)` after row normalization |
| Reduced direction | `1e-14` |
| Chord interval and centre radius | `1e-12` |

Diagnostics retain raw and conditioned mass-balance residuals and declared,
stable, and effective objective values/violations. Provenance also records the
algorithm/version, `PCG64` seed, sample count, burn-in, thinning, transition
attempt limit, objective direction/optimum/bounds/scales, compiled flux-LP
fingerprint, FVA digest, exact reaction order, equality rank, affine dimension,
fixed-reaction count, collapsed reaction IDs, optimal-face flag, centre
method/radius, accepted and self-loop steps, rejected directions, the
sampling/validation tolerances listed above, and NumPy and HiGHS versions. The
high-level ensemble result separately exposes the full canonical-model and
stationary-experiment fingerprints used by the compiled EMU plan.

## Stationary EMU composition

The ensemble API compiles one flux-independent EMU topology and calls
`evaluate_stationary` once with the exact ordered tuple of sampled states.
Predictions are sample-major and then follow declared target order. Sample IDs
such as `sample_0000` propagate unchanged into MID predictions, long-form MID
values, and layer diagnostics.

Every MID must pass existing finiteness, nonnegativity, and normalization gates.
If one otherwise feasible flux state makes a stationary EMU layer singular or a
target undefined, evaluation stops with that sample ID in the error. FluxEMU
does not drop, reject, or resample states based on their EMU output, because such
conditioning would silently change the declared flux-chain law.

## Public APIs

Given a validated `CanonicalModel` and `StationaryExperimentSemantics`, the
high-level deterministic API remains available:

```python
from fluxemu import run_native_stationary_analysis

result = run_native_stationary_analysis(
    model,
    experiment,
    fva_fraction_of_optimum=1.0,
)

result.fba          # biological optimum and one complete primal
result.fva          # independent reaction-wise extrema
result.flux_state   # the exact FBA primal supplied to EMU
result.mids         # MIDs conditional on that one state
```

The bundled nontrivial acceptance fixture provides an executable example of the
complete sampled Stage 1 path:

```python
from fluxemu import run_native_stationary_ensemble
from fluxemu.real_model import (
    build_r1_acceptance_experiment,
    load_ecoli_core_stage_b2_model,
)

model = load_ecoli_core_stage_b2_model()
experiment = build_r1_acceptance_experiment()

result = run_native_stationary_ensemble(
    model,
    experiment,
    sample_count=2,
    seed=626,
    fraction_of_optimum=1.0,
    burn_in=8,
    thinning=2,
    fva_workers=1,
)

result.fba
result.fva
result.flux_sampling.states
result.flux_sampling.validation
result.flux_sampling.provenance
result.mid_ensemble.forward.predictions
result.model_fingerprint
result.experiment_fingerprint

assert result.flux_sampling.validation.valid
assert len(result.mid_ensemble.forward.predictions) == 2 * 12
```

`run_native_stationary_ensemble_analysis` is an identity alias for
`run_native_stationary_ensemble`; `NativeStationaryEnsembleAnalysisResult` is an
identity alias for `NativeStationaryEnsembleResult`.

Lower-level native APIs live under `fluxemu.flux_analysis`:

- `prepare_highs_flux_region`
- `run_highs_fba`, `run_highs_fva_reference`, `run_highs_vffva`, and
  `run_prepared_highs_vffva`
- `sample_highs_flux_states` and `sample_prepared_flux_states`
- `validate_flux_states`
- `PreparedFluxRegion`, `NativeFluxSamplingResult`,
  `RetainedObjectiveConstraint`, `FBAResult`, `FVAResult`,
  `FluxSamplingProvenance`, `FluxSampleValidationReport`, and
  `FluxStateValidationDiagnostics`

The current YAML/CLI route is deliberately deterministic and does not expose
the sampler controls. Use the Python ensemble API above for sampled Stage 1
execution; YAML fields from the historical COBRA/mfapy compatibility schema do
not configure this native sampler.

## Installation and dependency boundary

From the repository root:

```bash
python -m pip install ./codex
```

The standard installation supplies NumPy, pandas, PyYAML, highspy, and
python-libSBML. `highspy` is a default dependency, not a `highs` extra. The
optional `compat` extra installs COBRApy for compatibility/parity work, and the
optional `transient` extra installs SciPy for transient integration. The
optional `mfa` extra installs SciPy for the separate native stationary fitting
layer; importing FluxEMU or evaluating a stationary MFA objective does not
load SciPy. The
`mfapy` extra supplies the compatible SciPy dependency for an externally
available or vendored mfapy backend; it does not distribute mfapy itself. None
of those optional stacks participates in native Stage 1.

## Reproducible FastFVA evidence

The current diagnostic driver and tracked result are
[`benchmark_vffva_highs_scaling.py`](../benchmarks/benchmark_vffva_highs_scaling.py)
and
[`vffva_highs_scaling_linux_x86_64.json`](../benchmarks/results/vffva_highs_scaling_linux_x86_64.json).
They record cold-reference parity, worker lifecycle, warm reusable scaling, the
separate first-preparation cost, exact environment/model provenance, and the
attempt to execute original VFFVA. From the repository root, inspect the
available controls with:

```bash
PYTHONPATH=codex/src python codex/benchmarks/benchmark_vffva_highs_scaling.py \
  --help
```

Keep the evidence classes distinct:

- **Internal cold-reference ratio:** cold endpoint rebuilds divided by reusable
  one-worker time. The tracked ratios were `5.922x` for the bundled 95-reaction
  E. coli core and `8.235x` for the iLJ478 benchmark-only algebraically
  equivalent independent equality-row basis. The latter used only one measured
  cold sample and is descriptive.
- **Shared-memory scaling (`T1/Tp`):** the 652-reaction, 1,304-endpoint iLJ478
  benchmark-only basis recorded medians of `1.894390`, `1.110843`, `0.637523`,
  and `0.484070` seconds at 1, 2, 4, and 8 workers, respectively: `1.000x`,
  `1.705x`, `2.971x`, and `3.913x`. The small E. coli model is retained to show
  overhead, not to generalize scaling.
- **Direct original-VFFVA comparison:** upstream CPLEX `make` followed by GLPK
  `make SOLVER=glpk` was attempted at the pinned commit. Both stopped because
  `mpicc` was unavailable; the required MPI/runtime and CPLEX/GLPK components
  were also absent. No upstream binary ran, so no direct parity, timing, or
  speedup result is claimed.

The earlier
[`stage1_native_fva_linux_x86_64.json`](../benchmarks/results/stage1_native_fva_linux_x86_64.json)
recorded about `0.598091` seconds for two process workers; the current artifact
records `0.029710` seconds for two threads on the same-family E. coli workload.
That historical contrast is not a controlled estimate of IPC cost: hot-loop
validation and other source details changed too. The full difference must not
be attributed solely to replacing processes. Exact scaling tables, cold-sample
limitations, benchmark-only row-basis certification, environment metadata,
source binding, and original-VFFVA build logs are documented in
[Phase 3B: native HiGHS flux analysis](PHASE3B_NATIVE_HIGHS_FLUX_ANALYSIS.md).
No timing ratio is a CI gate.

Stage 1 ends at forward flux-to-MID ensembles. It does not implement inverse
MFA, fitting, confidence intervals, Monte-Carlo MFA uncertainty, information
theory, or hypothesis testing.
The separate [stationary MFA layer](STATIONARY_MFA_RENYI_CORE.md) now reuses
this forward stack and shared feasible geometry for KL/Rényi MID fitting;
it does not change the scope or default retention of the Stage 1 APIs.
