# Stationary MFA with KL/Rényi MID fitting

FluxEMU fits complete stationary flux states to one or more stationary tracer
experiments using its native canonical model, feasible-state geometry, and EMU
engine. The criterion is a plain sum of divergences between observed and
predicted mass isotopologue distributions (MIDs):

```text
L_alpha(v) = sum_j D_alpha(p_observed,j || p_predicted,j(v))
```

This is a fitting criterion on MID distributions. It is not a likelihood or
the experimental observation law `P(Y | v)`. Statistical testing between those
observation laws is a separate later layer. The biological CBM objective in
`FluxModel` remains separate from this MID loss.

## Installation and public API

From the repository root:

```bash
python -m pip install './codex[mfa]'
```

The `mfa` extra adds `scipy>=1.13` for SLSQP optimization. Ordinary `import
fluxemu`, `import fluxemu.mfa`, schema validation, divergence evaluation, and
native stationary objective evaluation work with the base installation.
Only a fitting action loads SciPy; a missing backend raises an `AnalysisError`
with the `fluxemu[mfa]` installation hint. Native MFA never imports COBRApy or
mfapy, and requires neither an mfapy text model nor a reaction-map CSV.

`fluxemu.fit_stationary_mfa` and `fluxemu.evaluate_stationary_mfa` directly
reexport the same functions available from `fluxemu.mfa`. Scientific records,
divergence functions, validators, and fingerprints are exported from
`fluxemu.mfa`:

| Public object | Contract |
| --- | --- |
| `StationaryMIDObservation(target_id, fractions, replicate_id="0")` | One complete MID tuple in declared mass-class order. |
| `StationaryMFAExperiment(experiment_id, experiment, observations)` | A uniquely identified native stationary tracer experiment and ordered observation tuple. |
| `StationaryMFAProblem(model, experiments)` | One canonical model shared by all experiment blocks. |
| `DivergenceObjectiveConfig(alpha=1.0, mid_tolerance=1e-9)` | Divergence order and declared input-validation tolerance. |
| `MFAOptimizationConfig(...)` | Native initialization, local optimizer controls, and optional biological retention. |
| `MFAObjectiveEvaluation` | Complete state, total loss, and ordered `MFAObservationDivergence` components. |
| `StationaryMFAResult` | Best accepted state, exact loss/components, all starts, validation, and provenance. |

Given your validated `CanonicalModel` named `model` and
`StationaryExperimentSemantics` named `experiment`, with a declared one-carbon
target named `product-mid`, a fit is constructed explicitly:

```python
from fluxemu.mfa import (
    DivergenceObjectiveConfig,
    MFAOptimizationConfig,
    StationaryMFAExperiment,
    StationaryMFAProblem,
    StationaryMIDObservation,
    evaluate_stationary_mfa,
    fit_stationary_mfa,
)

problem = StationaryMFAProblem(
    model,
    (StationaryMFAExperiment(
        "tracer-1",
        experiment,
        (StationaryMIDObservation("product-mid", (0.3, 0.7), "replicate-1"),),
    ),),
)
result = fit_stationary_mfa(
    problem,
    objective=DivergenceObjectiveConfig(alpha=0.73),
    optimization=MFAOptimizationConfig(n_starts=8, seed=626),
)
assert result.validation.valid
accepted = tuple(d for d in result.start_diagnostics if d.accepted)
recomputed = evaluate_stationary_mfa(problem, result.state, objective=result.objective_config)
assert recomputed.total_loss == result.total_loss
```

The complete runnable [synthetic recovery example](../examples/stationary_mfa_recovery.py)
builds its canonical model and explicit atom maps, forward-generates exact
observations, fits three divergence orders, and reports every start:

```bash
python codex/examples/stationary_mfa_recovery.py
```

## Observations and divergence semantics

Each experiment must contain at least one complete observed MID. Observation
targets may be declared ordinary or composite native targets. Unknown target
IDs, duplicate `(experiment_id, target_id, replicate_id)` identities, wrong
dimensions, negative/nonfinite entries, and invalid normalization fail.
Experiment, observation, replicate, target-atom, and mass-class order remain
declared order. Replicates are explicit observations; they are never silently
averaged. Selecting individual mass classes is not part of this contract.

Every component records its experiment/target/replicate identity, observed
fractions, predicted fractions, and divergence. `total_loss` is their plain
sum, with no target weights or division by mass-class count.

At `alpha == 1` the implementation dispatches exactly to
`sum_i p_i log(p_i / q_i)`. Other finite real orders `alpha > 0` use
`log(sum_i p_i**alpha * q_i**(1-alpha)) / (alpha-1)`. The direction is always
observed `p` versus predicted `q`; logarithms are natural. The order is a
continuous API, not a predefined grid. Booleans, nonpositive orders, NaN,
and infinite orders are rejected.

| Support case | KL / order greater than one | Order between zero and one |
| --- | --- | --- |
| `p_i = 0` | Zero contribution, including when `q_i = 0`. | Zero contribution to the power sum. |
| Some `p_i > 0, q_i = 0` | Infinite divergence. | Can remain finite when positive support overlaps elsewhere. |
| Disjoint supports | Infinite divergence. | Infinite divergence. |
| Identical valid distributions | Zero. | Zero. |

Stable log-domain arithmetic and near-one formulas preserve these conventions.
No pseudocounts, epsilon floors, clipping, or silent normalization are applied.
Schema validation uses `mid_tolerance`; numerical divergence evaluation also
requires each vector's total mass within **16 ulps of one**, approximately
`3.55e-15`. This stricter machine-simplex requirement prevents near-one formulas
from treating a normalization discrepancy as divergence. A vector can pass a
broader schema tolerance and still fail divergence evaluation contextually;
increasing `mid_tolerance` does not authorize mass repair. Inputs remain unchanged.

## Feasibility, initialization, and optimization

The default feasible region is exactly canonical steady-state constraints and
reaction bounds: `S v = 0`, `lower <= v <= upper`. Default MFA has
`fraction_of_optimum=None`. A native FBA solve provides a geometry anchor but
does not impose biological optimality on the fit.

An explicitly requested fraction in `(0, 1]` adds the existing native retained
objective constraint: a maximization floor or minimization ceiling using the
same sign, scaling, and feasibility checks as Stage 1. It never adds a loss
term. Existing Stage 1 public defaults remain fraction one.

Exact measured reaction/exchange values can be specified as equal canonical
bounds; explicitly chosen admissible intervals can be bounds. These are hard
constraints. Soft measured-flux observations have no residual term in this
MID-only core, and standard deviations are not interpreted as bounds.

SLSQP operates on `v = particular + basis @ theta` using the same reduced
affine geometry, complete-state inequalities, and native validation as the
sampler. FVA supplies collapse/geometry evidence; its coordinate extrema are
never assembled into candidate states. Each experiment's native EMU plan is
compiled once per fit. Every evaluated trial is a complete canonical state
checked against the original model before EMU execution.

Generated starts reuse native complete-state hit-and-run with a recorded
PCG64 seed. Defaults are eight starts, seed zero, 100 burn-in transitions,
thinning ten, and 100 maximum direction attempts. These starts are correlated
chain states; neither independence nor adequate mixing is claimed. Fixed-seed
replay is scoped to the same algorithm and numerical-library environment.
To supply starts, pass `initial_states=(state_a, state_b, ...)`; all supplied
states are attempted in order, and `n_starts` then does not control their count.
Nonserializable requests fail before fitting. Finite but infeasible/misordered
starts remain explicit failed-start diagnostics.

The local backend defaults to `maxiter=1000` and `ftol=1e-10`. Exact infinite
initial losses and undefined initial predictions reject that start. Later
infeasible, undefined-forward, or support-failing trials are recorded and
return `+inf` to SLSQP for backtracking. There is no finite penalty substitute.
Finite differences at feasible boundaries can fail; the backend may report
failure or a nonfinite gradient even at a state with matching MIDs. Those
attempts remain visible, and other starts continue.

## Results and acceptance

Use `MFAStartDiagnostic.accepted` to identify eligible fit candidates.
`optimizer_success` preserves the raw backend flag and `validated` records
original-model feasibility; neither flag alone implies acceptance. A malformed
or nonfinite final gradient is rejected even if the backend reports success.
Acceptance requires backend success, finite numerical diagnostics, independent
full-state validation, and finite exact re-evaluation. It does not certify
stationarity, unique identifiability, or a global optimum.

Every start retains its initial/final state and loss, status/message,
iterations/evaluations, acceptance, and contextual trial failures. An
unevaluated loss is `None`; an exact divergent loss is `inf`. If every attempt
fails, `MFAFitError.start_diagnostics` exposes them all.

The selected state has the smallest exact loss among accepted candidates. It
is independently checked again for canonical ordering, bounds, mass balance,
and any requested retained objective, then its MIDs and loss are recomputed.
The result binds model, experiment, problem, and fit fingerprints; divergence
and optimizer controls; supplied start content; backend/version; and affine
dimension. The problem fingerprint includes ordered observations; the fit
fingerprint additionally binds optimizer controls and supplied starts.

## Synthetic acceptance and scope

The example uses explicitly atom-mapped unlabelled and labelled inputs feeding
one balanced carbon pool. Canonical order is `(Z_IN, A_IN, M_OUT)` and the MID
is `(Z_IN/M_OUT, A_IN/M_OUT)`, with `Z_IN + A_IN = M_OUT`.

| Programme | Identified quantity | Acceptance |
| --- | --- | --- |
| Exact throughput constraint `M_OUT = 10` and truth `(3,7,10)` | Full state, jointly from MID and the exact throughput constraint. | Three distinct nontruth starts recover for KL and Rényi orders 0.5 and 2; test tolerances are `1e-10` loss and `5e-5` flux error. |
| Throughput allowed in `[2,10]` | The input split ratio; absolute throughput remains undetermined. | `(1.5,3.5,5)` and `(3,7,10)` have identical native MIDs; multiple accepted fitted scales match those MIDs. |

The numerical example retains rejected boundary attempts as well as successful
interior fits. Failure to recover one particular absolute scale in the second
programme is consistent with the model's non-identifiability. These exact
noise-free software fixtures make no biological model-validity claim.

Engineering lineage, including exact vendored source locations and reasons for
each native replacement, is recorded in the
[mfapy engineering comparison](MFAPY_ENGINEERING_COMPARISON.md) and
[architecture decision](STATIONARY_MFA_ARCHITECTURE.md). The implementation
adapts reduced-state reconstruction, feasible initialization, multistart
orchestration, multiple experiments, and forward-generated recovery from the
reference workflow. It uses native FluxEMU geometry, complete states, and EMU,
and replaces weighted RSS/covariance and bound penalties with exact MID
divergence and explicit feasibility. `vendor/mfapy` remains read-only and is
not a production dependency.

Global-search backends, parallel start scheduling, soft measured-flux
observations, and noisy-data generation are explicitly deferred. This core
contains no experimental observation-law testing or uncertainty inference,
transient MFA, or JAX. Native stationary EMU restrictions, including explicit
atom mappings and `correction: no`, still apply. CI checks the base-only
boundary, the installed MFA extra, synthetic recovery, and existing native
Stage 1 FBA/FVA/sampling/EMU regression suites.
