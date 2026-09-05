# Codex task 03A: stationary MFA KL/Rényi core, mfapy-informed implementation

Repository: `esig626/fluxemu-standalone`

Work on branch: `codex/stationary-mfa-renyi-core`

Read `AGENTS.md` first and obey it exactly.

Then read, treat as incorporated, and execute the complete specification in:

```text
codex/prompts/03_stationary_mfa_renyi_core_multiagent.md
```

This 03A prompt **supersedes only the execution strategy where stated below**. All scientific scope, stopping conditions, KL/Rényi semantics, feasibility requirements, tests, documentation, and MFA-GUARDIAN authority from prompt 03 remain mandatory.

Do not execute any other project prompt.

---

## Why this addendum exists

FluxEMU already vendors a complete mfapy source snapshot under:

```text
vendor/mfapy/
```

The first stationary MFA implementation must therefore **not reinvent established MFA engineering** merely because FluxEMU is replacing mfapy's statistical objective.

The intended principle is:

```text
mfapy: reference implementation for mature MFA mechanics
FluxEMU: native model/EMU/feasibility stack + KL/Rényi statistical core
```

Use mfapy to learn what is already solved well. Reimplement only what must differ because FluxEMU has a different canonical model, native EMU engine, native feasible-region machinery, or information-divergence objective.

`vendor/` is read-only third-party source. Do not edit it. Do not import it from production FluxEMU. Do not make mfapy a runtime dependency.

---

# 1. MFA-GUARDIAN must still be created first

Before any other specialist, create the persistent **MFA-GUARDIAN** required by prompt 03.

The guardian remains active through final acceptance and retains veto authority.

In addition to the rejection criteria in prompt 03, the guardian must reject any plan that:

- reimplements an MFA mechanism from scratch without first checking whether mfapy already contains a mature version of that mechanism;
- blindly copies mfapy code or data structures when FluxEMU already has a cleaner native equivalent;
- imports `vendor/mfapy` into production runtime;
- preserves mfapy's RSS / weighted least-squares objective merely because its optimizer plumbing expects it;
- preserves mfapy's bound-violation penalty as the scientific MFA objective when explicit FluxEMU feasibility constraints can be maintained instead;
- preserves chi-square goodness-of-fit, F-test confidence machinery, p-values, or other mfapy statistical conclusions that are outside this task;
- throws away useful mfapy multistart/global-local/initialization engineering without a documented reason;
- creates a second, incompatible independent-flux parameterization when FluxEMU's native affine feasible-region representation can carry the same role.

The guardian may not approve implementation until the mfapy forensic audit below is complete.

---

# 2. Mandatory first specialist: MFAPY-REFERENCE

Immediately after MFA-GUARDIAN, create a specialist agent named conceptually **MFAPY-REFERENCE**, **mfapy forensic auditor**, or equivalent.

This agent performs **read-only source archaeology** before any MFA production code is written.

At minimum inspect:

```text
vendor/mfapy/mfapy/metabolicmodel.py
vendor/mfapy/mfapy/optimize.py
vendor/mfapy/mfapy/mdv.py
vendor/mfapy/mfapy/carbonsource.py
vendor/mfapy/mfapy/mfapyio.py
vendor/mfapy/LICENSE
```

Search the source for the actual implementation of at least:

- independent/free flux selection and reconstruction of complete flux states;
- `matrixinv`, `Rm_initial`, `independent_flux`, independent bounds, and related bookkeeping;
- generation of randomized feasible starting states, including `initializing_Rm_fitting` and related callers;
- stationary fitting entry points in `MetabolicModel`;
- `fit_r_mdv_scipy`;
- `fit_r_mdv_nlopt` and any global-to-local fitting sequence;
- `calc_MDV_residue_scipy` / corresponding residual functions;
- handling of multiple experiments;
- inclusion of measured fluxes and/or other non-MID observations in the fit vector;
- use flags / observation selection;
- standard-deviation weighting / covariance construction;
- bound handling and penalties;
- optimizer methods, tolerances, iteration limits, restart logic, and failure reporting;
- parallel fitting architecture, if present;
- synthetic/noisy MDV generation utilities that are relevant to later recovery tests;
- any confidence/profile/goodness-of-fit machinery, only to classify it as later/rejected rather than to implement it now.

The agent must not rely on mfapy documentation alone. Inspect the vendored source code and identify exact functions/classes and behavior.

---

# 3. Mandatory mfapy transfer table

Before implementation begins, MFAPY-REFERENCE must produce a concise transfer table with exactly these categories:

| mfapy mechanism | exact vendored source location | what mfapy does | FluxEMU decision | reason |
|---|---|---|---|---|
| independent flux coordinates | ... | ... | REUSE CONCEPT / REPLACE | ... |
| full flux reconstruction | ... | ... | ... | ... |
| feasible start generation | ... | ... | ... | ... |
| multistart fitting | ... | ... | ... | ... |
| global -> local optimization | ... | ... | ... | ... |
| multiple experiments | ... | ... | ... | ... |
| measured flux observations | ... | ... | ... | ... |
| bounds/feasibility handling | ... | ... | ... | ... |
| RSS/covariance objective | ... | ... | REPLACE | KL/Rényi objective |
| goodness-of-fit/CI machinery | ... | ... | DEFER/REJECT | later testing task |
| parallel execution | ... | ... | ... | ... |
| synthetic-data utilities | ... | ... | ... | ... |

Allowed decisions are:

```text
REUSE CONCEPT
REUSE WITH ADAPTATION
REPLACE WITH EXISTING FLUXEMU NATIVE MECHANISM
REPLACE FOR SCIENTIFIC REASON
DEFER
REJECT
```

For every `REPLACE`, the reason must be explicit.

MFA-GUARDIAN must review this table and issue `APPROVE` or `REJECT` before Agents A-E from prompt 03 begin implementation.

---

# 4. What we expect to preserve from mfapy unless source inspection disproves it

These are hypotheses to verify, not assumptions to copy blindly.

## 4.1 Reduced independent coordinates

mfapy reconstructs a complete metabolic state from independent variables rather than optimizing every reaction independently.

FluxEMU should preserve this **principle**, but preferentially use its already-native affine feasible-region/null-space machinery rather than importing mfapy's `matrixinv` representation.

The required conceptual mapping is:

```text
mfapy independent flux vector
        -> mfapy matrix reconstruction
        -> complete metabolic state

FluxEMU reduced coordinate theta
        -> existing native affine geometry
        -> complete CanonicalFluxState
```

Do not duplicate the mathematics unnecessarily.

## 4.2 Feasible initial states

mfapy contains explicit machinery for generating randomized admissible initial states before fitting.

FluxEMU already has a native complete feasible-state sampler. Compare the two approaches and reuse the stronger existing native mechanism when it satisfies the same purpose.

Do not invent a third unrelated initialization system.

## 4.3 Multistart and global/local fitting

Determine exactly how mfapy performs repeated starts and whether it combines global and local optimizers.

Preserve useful orchestration ideas such as:

- many independent starts;
- robust failure isolation;
- retaining best successful solution;
- optional global-search stage followed by local refinement;
- deterministic/reproducible orchestration where FluxEMU can improve on mfapy.

Do not add an optimizer merely because mfapy exposes it. The first FluxEMU implementation should use the smallest justified optimizer set that faithfully reproduces the useful workflow.

## 4.4 Multiple experiments

If mfapy composes several tracer experiments into one fit, FluxEMU must not regress to a one-experiment-only architecture unless there is a hard implementation blocker.

The KL/Rényi objective should naturally aggregate explicit per-observation divergences across experiment blocks as required by prompt 03.

## 4.5 Measured flux observations

Determine how mfapy includes measured reaction/exchange flux values in fitting.

This task's primary statistical innovation concerns MID fitting. Do **not** silently discard a mature capability that practical MFA requires.

If measured flux observations can be supported cleanly without introducing the later observation-law machinery, design them as explicit optional constraints/observations with transparent semantics. If this would materially expand prompt 03 beyond a coherent first vertical slice, document the exact mfapy mechanism and leave a clean extension point rather than inventing an ad-hoc weighted residual term.

MFA-GUARDIAN decides whether inclusion belongs in this task after the source audit.

## 4.6 What must be replaced

mfapy's fitting objective includes a weighted residual sum of squares of measured versus predicted values, using standard deviations/covariance, and its low-level optimizer path can add a large penalty for bound violations.

FluxEMU must **not** inherit that objective by inertia.

For MID observations, prompt 03 remains controlling:

```text
L_alpha(v) = sum_j D_alpha(p_obs,j || p_v,j)
```

with exact support semantics and no hidden pseudocounts.

Likewise, do not carry over mfapy goodness-of-fit tests or confidence machinery into this task.

---

# 5. Mandatory architecture decision after the audit

After MFAPY-REFERENCE reports, the lead agent and the relevant prompt-03 specialists must produce one short architecture decision record answering:

1. Which mfapy mechanisms are being retained conceptually?
2. Which existing FluxEMU components replace mfapy mechanisms directly?
3. Which mfapy mechanisms are deliberately rejected because of our KL/Rényi formulation?
4. Which useful mfapy capabilities are deferred to later tasks?
5. Are we missing any basic MFA workflow capability that mfapy has and this proposed core accidentally omitted?

MFA-GUARDIAN must approve this record before implementation.

This is specifically intended to stop token-wasting reinvention.

---

# 6. Implementation priority after approval

Once the guardian approves the mfapy transfer plan, continue with prompt 03 in full.

The implementation priority is:

```text
reuse mature MFA workflow ideas
        +
reuse existing FluxEMU native feasible geometry / complete states / EMU
        +
replace mfapy RSS with KL/Rényi MID divergence
        =
first production FluxEMU stationary MFA core
```

Do not redesign established mechanics for novelty.

The scientific innovation in this task is **not** inventing a new nonlinear optimizer or a new flux-coordinate system.

It is the combination of:

```text
native FluxEMU forward model
+ rigorous complete-flux feasibility
+ mature MFA optimization workflow
+ KL/Rényi divergence on MID distributions
```

---

# 7. Additional acceptance requirement

The final PR must include a short document or section titled conceptually:

```text
mfapy engineering comparison
```

It must state, with source references to the vendored snapshot:

- what MFA mechanics were adapted from mfapy;
- what FluxEMU already did better natively and therefore replaced;
- exactly what statistical machinery was intentionally not carried over;
- that `vendor/mfapy` remains read-only and is not a production dependency.

The final report must not say merely "inspired by mfapy". It must make the engineering lineage auditable.

MFA-GUARDIAN must explicitly confirm this before `FINAL APPROVE`.

---

# 8. Stop condition

The stop condition is unchanged from prompt 03:

```text
stationary MFA
+ exact KL/Rényi MID objective
+ constrained multistart optimization
+ complete fitted CanonicalFluxState
+ synthetic known-truth recovery
+ non-identifiability control
+ tests/docs/CI/public API
```

Do not proceed into observation-law hypothesis testing, uncertainty certificates, transient MFA, Bayesian inference, or experiment design.
