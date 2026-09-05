# Codex task: implement the first production stationary MFA core with KL/Rényi fitting

Repository: `esig626/fluxemu-standalone`

Work on branch: `codex/stationary-mfa-renyi-core`

Read `AGENTS.md` first and obey it exactly.

Then execute this prompt in full.

---

## 0. Goal and stopping boundary

Implement the first **production-quality stationary MFA vertical slice** on top of the already completed native FluxEMU Stage 1 pipeline.

This task is deliberately **narrow in scope, not toy-quality**.

The required scientific path is:

```text
canonical metabolic model
        + stationary tracer experiment(s)
        + observed MID distribution(s)
                |
                v
complete feasible flux state v
        -> native stationary EMU
        -> predicted MID distribution(s) p_v
        -> KL / Rényi divergence from observed MID(s)
        -> constrained multistart optimization
        -> fitted complete CanonicalFluxState
        -> fitted predicted MID(s) + transparent divergence diagnostics
```

The canonical fitting objective for one MID is

```text
D_alpha(p_obs || p_v)
```

with exact KL at `alpha = 1` and finite-order Rényi divergence for every finite real `alpha > 0`, `alpha != 1`.

For multiple observed MIDs, expose every per-observation divergence and use their plain sum as the default aggregate loss. Do not introduce hidden weights, arbitrary chi-square weights, or goodness-of-fit thresholds.

### This task must implement

- first-class stationary MFA problem/data/config/result objects;
- strict observed-MID validation;
- exact KL divergence and the full finite positive-real Rényi-order API;
- native EMU-backed objective evaluation;
- constrained multistart stationary MFA;
- complete fitted canonical flux states;
- independent feasibility validation of final fitted states;
- synthetic known-truth recovery and a non-identifiability control;
- tests, docs, CI, public exports, and a clean optional MFA dependency boundary.

### This task must NOT implement

- Bayesian priors or posterior inference;
- a reference-MID regularization term or arbitrary lambda penalty;
- chi-square goodness-of-fit tests;
- p-values;
- INCA-style goodness-of-fit acceptance criteria;
- observation/noise laws `P(Y|v)`;
- likelihood-ratio tests;
- finite-sample hypothesis-testing bounds;
- the results of arXiv:2608.28068 or arXiv:2601.09550;
- confidence regions obtained by test inversion;
- profile likelihood;
- Monte-Carlo uncertainty;
- Sibson mutual information;
- KL/Rényi experiment-design analysis;
- transient MFA;
- JAX reimplementation of the EMU engine;
- information-theoretic research claims beyond the fitting objective itself.

Those belong to later tasks.

Stop after the stationary KL/Rényi MFA core is implemented, validated, documented, committed, pushed, and opened as a PR.

---

# 1. Scientific distinction that must never be blurred

There are two different probability objects in the long-term programme.

This task concerns MID vectors themselves:

```text
p_obs = observed normalized MID on isotopologue mass classes
p_v   = EMU-predicted normalized MID at flux state v
```

and uses

```text
D_alpha(p_obs || p_v)
```

as an information-divergence fitting criterion on the simplex.

A later task will introduce the experimental observation law

```text
P_v = P(Y | v)
```

and finite-sample hypothesis testing between such laws/families.

**Do not claim in this task that `D_alpha(p_obs || p_v)` is already the finite-sample testing divergence between observation laws.**

Likewise, do not call the divergence objective a likelihood unless a later explicit measurement model makes that interpretation valid.

---

# 2. Existing infrastructure to reuse

Before editing, inspect `main` and verify the actual APIs rather than assuming them.

At minimum read:

```text
AGENTS.md
codex/docs/STAGE1_NATIVE_WORKFLOW.md
codex/docs/PHASE3B_NATIVE_HIGHS_FLUX_ANALYSIS.md
codex/docs/PHASE3B_NATIVE_FLUX_TO_MID.md
codex/docs/PHASE2_NATIVE_STATIONARY_EMU.md
codex/docs/API_MAP.md
codex/docs/KNOWN_LIMITATIONS.md
codex/src/fluxemu/analysis/stationary.py
codex/src/fluxemu/emu/stationary.py
codex/src/fluxemu/emu/graph.py
codex/src/fluxemu/flux_analysis/highs.py
codex/src/fluxemu/flux_analysis/sampling.py
codex/src/fluxemu/flux_analysis/results.py
codex/src/fluxemu/execution.py
codex/src/fluxemu/model/schema.py
codex/src/fluxemu/model/validation.py
codex/pyproject.toml
codex/tests/test_native_stage1_ensemble_integration.py
codex/tests/test_native_highs_sampling.py
codex/tests/test_native_stationary_analysis.py
```

Important existing facts to preserve:

- Stage 1 now has native FBA, faithful VFFVA-style HiGHS FVA, complete feasible-state sampling, stationary EMU, and MID ensembles.
- Complete sampled states are `CanonicalFluxState` records in canonical reaction order.
- Native sampling already constructs reduced affine feasible geometry. Reuse or carefully extract shared geometry rather than implementing a second inconsistent null-space system for MFA.
- Native stationary EMU already evaluates complete canonical flux states.
- The biological CBM objective belongs to `FluxModel` and is **not** the MFA loss.

---

# 3. Mandatory persistent supervisor: MFA-GUARDIAN

Before spawning any implementation specialist, create one persistent supervisor agent named conceptually **MFA-GUARDIAN**.

Its job is to prevent scope drift, easy shortcuts, scientifically invalid substitutions, and token-wasting duplicate work.

The MFA-GUARDIAN must remain active until final acceptance.

## MFA-GUARDIAN authority

The guardian must review and explicitly issue one of:

```text
APPROVE
REJECT
BLOCKED
```

for:

1. the initial architecture plan;
2. the divergence semantics;
3. the feasible-optimization design;
4. the synthetic recovery design;
5. every milestone before the lead proceeds;
6. the final PR state.

The lead agent may not override a `REJECT` without fixing the cited issue.

## MFA-GUARDIAN must reject any proposal that

- replaces KL/Rényi fitting with least squares, chi-square, RMSE, or another surrogate;
- silently adds pseudocounts, clipping, renormalization, or epsilon floors to avoid support problems;
- discretizes Rényi orders into a finite grid as the API;
- silently reverses the requested divergence direction;
- forces all MFA solutions to the FBA optimum merely because Stage 1 sampling already has retained-objective machinery;
- mixes the CBM biological objective into the MFA loss;
- optimizes reaction FVA endpoints as if they formed a feasible flux state;
- evaluates EMU on independently sampled FVA coordinates;
- adds arbitrary regularization weights or a prior/reference MID penalty;
- calls the divergence a likelihood without an explicit measurement model;
- implements hypothesis-testing/error bounds in this task;
- copies a second ad-hoc nullspace/constraint representation when the existing reduced geometry can be safely generalized/reused;
- returns a fitted flux vector without independent original-model feasibility validation;
- claims exact flux recovery on a non-identifiable synthetic system;
- treats optimizer success flags as scientific validation;
- hides optimizer failures or failed starts;
- leaves required functionality as TODO/future work while declaring completion;
- expands into JAX, Bayesian MFA, transient MFA, experiment design, or uncertainty quantification.

## Token-discipline duty

The guardian must also prevent redundant exploration:

- assign non-overlapping tasks;
- stop agents from independently redesigning the same API;
- require concise evidence-based reports;
- terminate a specialist once its deliverable has been integrated or rejected;
- do not spawn extra agents merely to restate existing findings.

---

# 4. Mandatory specialist agents

After the guardian is active, spawn specialists with non-overlapping responsibilities.

## Agent A — divergence mathematics referee

Owns only the mathematics/numerics of the MID loss.

Must specify and test:

### KL

```text
D_KL(p || q) = sum_i p_i log(p_i / q_i)
```

using standard exact support conventions:

- `p_i = 0` contributes zero;
- `p_i > 0, q_i = 0` gives `+inf`;
- no hidden pseudocounts.

### finite-order Rényi

For every finite real `alpha > 0`, `alpha != 1`:

```text
D_alpha(p || q)
  = 1/(alpha - 1) * log(sum_i p_i^alpha q_i^(1-alpha))
```

with mathematically correct support behavior for `0 < alpha < 1` and `alpha > 1`.

Requirements:

- exact `alpha == 1` dispatch to KL;
- numerically stable log-domain evaluation;
- continuity tests near `alpha = 1`;
- finite positive-real alpha accepted without a grid;
- reject booleans, NaN, infinity, nonpositive alpha;
- nonnegative result up to controlled floating-point tolerance;
- exact zero for equal valid distributions within numerical precision;
- preserve requested direction `observed || predicted` everywhere.

The agent must report formulas, support cases, numerical strategy, and tests before implementation is accepted.

## Agent B — MFA scientific data/API architect

Owns the scientific objects and validation boundary.

Design production-quality objects conceptually equivalent to:

```text
StationaryMIDObservation
StationaryMFAExperiment
StationaryMFAProblem
DivergenceObjectiveConfig
MFAOptimizationConfig
StationaryMFAResult
MFAStartDiagnostic
```

Exact names may differ if repository conventions justify it.

Required semantics:

- one canonical metabolic model;
- one or more stationary tracer experiment blocks;
- each block has explicit experiment ID and stationary experiment semantics;
- one or more observed MID records keyed unambiguously to declared experiment targets/observation targets;
- replicate observations may be represented explicitly rather than averaged invisibly;
- every observed MID is validated as a probability vector:
  - finite;
  - nonnegative;
  - correct isotopologue dimension;
  - sum to one within a declared tolerance;
  - no silent renormalization;
- duplicate/unknown target IDs must fail;
- preserve declared target/isotopologue ordering;
- fingerprints/provenance must bind model, experiment, observations, divergence order, and optimizer settings.

Do not introduce measurement standard deviations or covariance as fake inputs unless actually used by a scientifically defined measurement model. That is later work.

## Agent C — constrained optimization / geometry engineer

Owns the feasible optimization layer.

Key scientific requirement:

```text
MFA feasible region = canonical steady-state/bounds constraints
                    + optional explicitly requested retained biological objective
```

The biological objective restriction is **optional**, not automatic.

Default MFA must not force `fraction_of_optimum = 1` simply because FBA exists.

The agent must inspect the native sampler's reduced affine geometry and either:

1. extract/generalize a shared internal feasible-region representation usable by both sampling and MFA; or
2. justify with concrete evidence why safe reuse is impossible.

Do not create two inconsistent mathematical representations of the same feasible region.

The optimization should work in reduced feasible coordinates where practical:

```text
v = v0 + N theta
```

with original reaction bounds and any optional retained-objective inequality preserved as explicit constraints.

Use an established constrained optimizer from SciPy as the first backend unless repository evidence demonstrates a better existing option. Keep SciPy behind an optional `mfa` dependency extra; importing ordinary FluxEMU must not import SciPy.

Requirements:

- compile each EMU plan once per stationary experiment block, not per objective evaluation;
- reconstruct a complete `CanonicalFluxState` for every EMU evaluation;
- no FVA-endpoint vector construction;
- multistart fitting;
- feasible starting states, preferably reusing native feasible-state sampling where appropriate;
- explicit deterministic seed for generated starts;
- support user-supplied complete feasible starts if cleanly compatible with the API;
- record every start's initial/final loss, optimizer status, iterations/evaluations, and failure message;
- select the best **successfully validated** fitted state, not merely the numerically smallest malformed result;
- independently validate the final complete flux state against the original model constraints;
- do not claim a global optimum from multistart local optimization;
- no silent penalty repair of infeasible final states.

### Support mismatch during optimization

The exact divergence may legitimately be `+inf` when predicted support excludes observed positive mass.

Do not change the mathematical objective with hidden epsilons.

Choose an optimizer handling strategy that preserves exact reported objective semantics. If a backend cannot robustly handle exact support mismatch, fail clearly or use a documented optimizer-interface strategy whose finite internal guard cannot be confused with the reported scientific loss. The guardian must approve this design explicitly.

## Agent D — synthetic identifiability/recovery engineer

Owns the acceptance fixtures.

Build at least two deterministic synthetic programmes.

### Fixture 1: identifiable known-truth recovery

- choose a small stationary canonical model/experiment with a known complete feasible truth `v*`;
- forward simulate exact observed MID(s):

```text
v* -> native EMU -> p_obs
```

- fit from several distinct feasible starts;
- require near-zero divergence;
- require recovery of `v*` only for flux coordinates actually identifiable in that fixture;
- test KL (`alpha=1`) and representative Rényi orders on both sides of one, e.g. `0.5` and `2.0`, while keeping the production API continuous in alpha.

### Fixture 2: non-identifiability control

Construct or use a system where different feasible flux states produce the same relevant MID(s).

Acceptance must demonstrate:

- fitted predicted MIDs can match exactly/near-exactly;
- multiple flux solutions may remain compatible;
- tests and docs must **not** falsely call failure to recover the unique flux vector an optimizer failure.

This distinction is mandatory because MFA uncertainty/identifiability is scientific, not merely numerical.

No random measurement noise in this task. Noise laws belong to the next stage.

## Agent E — regression/dependency/docs reviewer

Owns:

- base-package import isolation;
- optional `mfa` extra;
- CI selection;
- public exports;
- docs;
- regression against Stage 1.

Must ensure:

- existing FBA/FVA/sampling/EMU behavior is unchanged;
- no COBRApy or mfapy runtime dependency is introduced;
- ordinary `import fluxemu` does not import SciPy merely because MFA support exists;
- MFA docs clearly say that KL/Rényi is the fitting criterion on MID distributions;
- docs explicitly distinguish this from later observation-law hypothesis testing;
- no goodness-of-fit p-value language appears;
- no Bayesian-prior language appears.

---

# 5. Required architecture

The public structure should be clean and small enough to remain stable.

A reasonable module layout is conceptually:

```text
fluxemu/mfa/
    __init__.py
    schema.py          # MFA data/config/result records and validation
    divergence.py      # KL/Rényi on MID probability vectors
    stationary.py      # objective compilation/evaluation + multistart fit
```

You may adapt names to repository conventions, but do not bury MFA inside `analysis/stationary.py` as one giant function.

The runtime flow should be:

```text
problem validation
      |
      +--> compile stationary EMU plan(s) once
      |
      +--> prepare reusable feasible optimization geometry once
      |
      +--> obtain/validate multistart feasible states
      |
      +--> for each optimizer trial:
              reduced coordinates theta
                    -> complete v
                    -> CanonicalFluxState
                    -> stationary EMU
                    -> predicted MID(s)
                    -> per-observation D_alpha(p_obs || p_v)
                    -> exact sum
      |
      +--> independently validate fitted complete state
      |
      +--> re-evaluate fitted MID(s) and exact divergence
      |
      +--> StationaryMFAResult
```

No trial may use a vector assembled from independent FVA intervals.

---

# 6. Divergence objective contract

For observations indexed by `j`, define

```text
L_alpha(v) = sum_j D_alpha(p_obs,j || p_v,j)
```

for this first task.

Requirements:

- return total loss;
- return every per-observation component;
- preserve experiment/target/replicate identity;
- no hidden target weighting;
- no division by isotopologue count;
- no arbitrary regularizer;
- no goodness-of-fit cutoff;
- no model ranking claim.

Because multiplying the whole loss by a positive constant would not change the optimizer, do not waste effort debating sum versus mean for a fixed problem. Use the plain sum and expose the components.

---

# 7. Optional retained biological objective

The CBM biological objective remains separate from MFA.

Support two cases:

### Default

```text
S v = 0
l <= v <= u
```

with no biological optimality restriction.

### Explicit optional restriction

If the user requests a retained fraction `f`, add the existing FluxEMU retained-objective constraint using the same mathematically validated semantics as Stage 1.

Never add the biological objective to `L_alpha(v)`.

If existing Stage 1 geometry is currently inseparable from mandatory retention, refactor the shared internal geometry cleanly rather than faking unrestricted MFA with an arbitrary tiny fraction.

---

# 8. Optimizer acceptance

This is a correctness-first first backend, not the final performance backend.

Hard requirements:

- multistart;
- deterministic replay for a fixed seed/config where starts are generated;
- complete diagnostics for failed starts;
- at least one successful validated solution required;
- best validated solution chosen by exact reported divergence;
- final state preserves exact canonical reaction ordering;
- final state satisfies original bounds/mass balance and any explicit retained objective within existing repository tolerances;
- exact predicted MID(s) and divergence are re-evaluated after optimizer termination;
- no optimizer success flag can bypass FluxEMU validation.

Do not implement JAX/autodiff in this task. Design the objective boundary so a differentiable evaluator can replace/augment the backend later without changing scientific data objects.

---

# 9. Hard test matrix

At minimum add focused tests for:

## MID/schema validation

- valid MID;
- negative entry rejected;
- NaN/inf rejected;
- wrong dimension rejected;
- sum not one rejected;
- unknown target rejected;
- duplicate observation identity rejected;
- ordering preserved;
- multiple stationary experiment blocks where supported by the chosen public contract.

## KL/Rényi mathematics

- identical distributions -> zero;
- KL known analytical examples;
- Rényi known analytical examples for `alpha < 1` and `alpha > 1`;
- exact alpha=1 KL dispatch;
- convergence toward KL near alpha=1;
- correct support mismatch -> infinity;
- zero-mass terms handled correctly;
- invalid alpha rejected;
- input vectors not mutated;
- directionality test proving `D(p||q)` was not silently reversed.

## Feasible optimization

- unrestricted MFA does not force FBA optimum;
- optional retained-objective MFA does enforce it;
- maximizing and minimizing biological objective semantics remain correct when optional retention is used;
- fixed/blocked/signed reactions;
- canonical ordering;
- generated starts are complete and feasible;
- malformed supplied starts rejected;
- final result independently validated;
- failed starts retained in diagnostics rather than hidden;
- no FVA interval sampling.

## End-to-end synthetic MFA

- identifiable noise-free truth recovery;
- non-identifiability control;
- KL fit;
- Rényi `alpha=0.5` fit;
- Rényi `alpha=2` fit;
- deterministic replay where promised;
- more than one restart;
- fit result predicted MIDs exactly correspond to fitted flux state;
- no COBRApy/mfapy import on native MFA path.

## Regression

Run all relevant existing Stage 1 suites plus the new MFA suite.

---

# 10. Dependency boundary

Add a dedicated optional dependency extra if SciPy is required, conceptually:

```toml
mfa = ["scipy>=1.13"]
```

Do not move SciPy into unconditional base dependencies merely for MFA unless the guardian finds a compelling repository-wide reason and documents it.

Base installation and ordinary forward FluxEMU use must remain functional without the MFA extra.

Importing `fluxemu.mfa` may give a clear dependency error only when an optimization action actually requires the missing backend; pure schema/divergence functionality should remain lightweight if practical.

---

# 11. Milestones and commit discipline

Follow `AGENTS.md`: focused tests before each implementation commit; commit each coherent milestone immediately.

## Milestone 1 — mathematical/data contract

Implement:

- MFA schema/validation;
- KL/Rényi divergence engine;
- focused mathematical tests.

Guardian must approve divergence direction/support/zero semantics.

Suggested commit theme:

```text
Add stationary MFA data and information-divergence core
```

## Milestone 2 — feasible multistart optimizer

Implement:

- reusable reduced feasible geometry/shared refactor if needed;
- stationary EMU objective compilation;
- multistart optimizer;
- result/start diagnostics;
- independent final-state validation.

Guardian must approve that unrestricted MFA is genuinely unrestricted by the biological objective unless explicitly requested.

Suggested commit theme:

```text
Add constrained multistart stationary MFA
```

## Milestone 3 — synthetic recovery acceptance

Implement:

- identifiable known-truth fixture;
- non-identifiability fixture;
- end-to-end KL/Rényi recovery tests.

Guardian must approve identifiability claims.

Suggested commit theme:

```text
Validate stationary MFA synthetic recovery
```

## Milestone 4 — docs/CI/public API

Finish:

- docs;
- optional dependency boundary;
- CI;
- public exports;
- regression suite;
- clean install checks.

Suggested commit theme:

```text
Finalize stationary Rényi MFA core
```

---

# 12. Final adversarial audit

Before opening/finalizing the PR, MFA-GUARDIAN must inspect the actual diff and test evidence and explicitly answer all of the following.

1. Is the fitted objective really `D_alpha(observed || predicted)` everywhere?
2. Is `alpha=1` exactly KL?
3. Does the API accept the positive-real continuum of finite Rényi orders rather than a finite grid?
4. Are zeros/support mismatches handled mathematically rather than with hidden pseudocounts?
5. Are observed MIDs validated as probability vectors without silent renormalization?
6. Does default MFA avoid imposing FBA optimality?
7. If objective retention is requested, is it a feasibility constraint rather than a loss term?
8. Does every EMU evaluation receive a complete canonical flux state?
9. Was reduced feasible geometry reused/generalized rather than duplicated inconsistently?
10. Is the final fitted state independently validated against the original model?
11. Are all failed starts visible in diagnostics?
12. Does the identifiable fixture recover truth for genuinely identifiable coordinates?
13. Does the non-identifiable fixture avoid false unique-flux claims?
14. Is SciPy optional and base FluxEMU still clean?
15. Are there zero goodness-of-fit p-values/chi-square acceptance criteria?
16. Are there zero Bayesian priors/posteriors/reference-MID regularizers?
17. Are there zero finite-sample hypothesis-testing claims in this task?
18. Are docs explicit that later `P(Y|v)` testing is a different layer?
19. Are all relevant tests green at the final remote SHA?
20. Is the branch clean, pushed, and PR mergeable?

The final verdict must be exactly one of:

```text
FINAL APPROVE
FINAL REJECT
FINAL BLOCKED
```

The lead may only declare completion on `FINAL APPROVE`.

---

# 13. Final report format

Keep the final report concise and evidence-based.

Include only:

- branch;
- final SHA;
- PR number/state;
- guardian verdict;
- public MFA API added;
- exact divergence semantics;
- optimizer backend and feasible-coordinate strategy;
- synthetic recovery results, separating identifiable from non-identifiable behavior;
- test counts and CI state;
- any genuine remaining limitations that belong to future tasks.

Do not spend the final report on future hypothesis-testing ideas.

Stop.