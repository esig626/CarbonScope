# Codex task: implement the first explicit stationary observation-law layer

Repository: `esig626/fluxemu-standalone`

Work on branch: `codex/stationary-observation-law-core`

Read `AGENTS.md` first and obey it exactly.

Then execute this prompt in full.

---

## 0. Goal and stopping boundary

Implement the first production-quality experimental observation-law layer on top of the merged stationary MFA core.

The scientific chain is now:

```text
complete feasible flux state v
        -> native stationary EMU
        -> predicted MID p_v
        -> explicit observation law P(Y | v)
```

This task must implement one mathematically exact baseline law:

```text
Y | v ~ Multinomial(n, p_v)
```

for observations that genuinely have isotopologue-count semantics.

This is deliberately a **baseline observation law**, not a claim that all mass-spectrometry MID measurements are multinomial counts. Normalised peak areas, percentages, arbitrary intensities, and mfapy-style noisy MDVs must never be silently reinterpreted as ion counts.

The purpose of this task is to establish a rigorous software boundary between:

```text
EMU probability vector p_v
```

and

```text
actual data-generating distribution P(Y | v).
```

It must also expose and verify the exact information identities of the multinomial law, because these identities explain when the MID-level KL/Rényi quantities have an operational statistical meaning.

### This task must implement

- a clean observation-law API that can support future laws without rewriting stationary EMU or MFA;
- an immutable fixed-total multinomial MID law;
- immutable count-observation records with strict integer/count validation;
- stable log-PMF evaluation with exact support semantics;
- deterministic sampling from the law using an explicit RNG/seed boundary;
- a stationary EMU bridge that maps a complete canonical flux state to one or more declared multinomial observation laws;
- exact KL and finite-order Rényi divergence of multinomial laws;
- exact independent-product aggregation across declared observation blocks/replicates;
- verification of the identities

```text
D_KL(Mult(n,p) || Mult(n,q)) = n D_KL(p || q)
```

and, for every finite real alpha > 0,

```text
D_alpha(Mult(n,p) || Mult(n,q)) = n D_alpha(p || q);
```

- verification of the multinomial likelihood/KL identity for an observed count vector k with n=sum(k):

```text
-log P_p(k) = n D_KL(k/n || p) + C(k)
```

where `C(k)` depends only on the observed counts and therefore does not affect optimisation over `p`;
- synthetic end-to-end tests using the existing stationary EMU engine;
- documentation that sharply distinguishes count data from arbitrary normalised MS intensities;
- focused CI and regression coverage.

### This task must NOT implement

- composite hypothesis testing;
- singleton hypothesis-test decision rules;
- finite-sample achievability/converse bounds;
- code from arXiv:2608.28068 or arXiv:2601.09550;
- p-values, goodness-of-fit tests, confidence regions, or test inversion;
- Bayesian priors or posterior inference;
- experiment design;
- a logistic-normal, Dirichlet, Gaussian, Poisson, gamma, or other second observation law;
- an inferred/effective sample size for normalised intensity data;
- conversion of arbitrary intensities into pseudo-counts;
- a new MFA optimiser;
- changes to the existing plain MID-divergence MFA objective unless required only for compatibility/exports and explicitly approved by the guardian;
- transient MFA;
- JAX;
- VFFVA performance work.

`AGENTS.md` explicitly keeps composite-hypothesis-testing research out of this standalone repository. Respect that boundary. The standalone package may expose an explicit observation law and its mathematical divergences; later hypothesis-testing research belongs in the separate research repository.

Stop after the multinomial observation-law core is implemented, tested, documented, committed, pushed, and opened as a PR.

---

# 1. Existing infrastructure to inspect and reuse

Before changing code, inspect the actual current `main` APIs. At minimum read:

```text
AGENTS.md
README.md
codex/docs/STATIONARY_MFA_RENYI_CORE.md
codex/docs/STATIONARY_MFA_ARCHITECTURE.md
codex/docs/MID_PREPROCESSING.md
codex/docs/MFAPY_ENGINEERING_COMPARISON.md
codex/src/fluxemu/mfa/divergence.py
codex/src/fluxemu/mfa/normalisation.py
codex/src/fluxemu/mfa/schema.py
codex/src/fluxemu/mfa/stationary.py
codex/src/fluxemu/emu/stationary.py
codex/src/fluxemu/execution.py
codex/src/fluxemu/model/schema.py
codex/src/fluxemu/model/validation.py
codex/tests/test_mfa_divergence.py
codex/tests/test_mfa_normalisation.py
codex/tests/test_mfa_schema.py
codex/tests/test_mfa_optimization.py
codex/tests/test_mfa_recovery.py
vendor/mfapy/mfapy/mdv.py
vendor/mfapy/mfapy/optimize.py
```

Important existing facts:

- stationary MFA and `normalise_mid(...)` are already merged into `main`;
- `normalise_mid(...)` explicitly closes non-negative intensity-like vectors to the simplex and deliberately discards total scale;
- raw total counts/intensity must therefore be retained separately when they are part of an observation model;
- the existing divergence kernel has exact support semantics and must be reused rather than reimplemented;
- native stationary EMU already maps complete `CanonicalFluxState` objects to ordered predicted MIDs;
- vendor is read-only;
- mfapy is a reference only and must not become a runtime dependency.

---

# 2. Mandatory persistent supervisor: OBSERVATION-LAW-GUARDIAN

Before spawning any implementation specialist, create one persistent supervisor agent named conceptually **OBSERVATION-LAW-GUARDIAN**.

It must remain active until final acceptance.

The guardian must explicitly issue one of:

```text
APPROVE
REJECT
BLOCKED
```

for:

1. the mfapy/data-semantics audit;
2. the observation-law API design;
3. the multinomial mathematics/numerics design;
4. the stationary EMU bridge design;
5. every implementation milestone;
6. the final PR state.

The lead may not override a `REJECT` without fixing the cited issue.

## OBSERVATION-LAW-GUARDIAN must reject any proposal that

- treats arbitrary normalised MS intensity as literal multinomial counts;
- rounds, rescales, or otherwise invents counts from percentages/intensities;
- invents an “effective n” without observed count semantics;
- silently feeds `normalise_mid(...)` output into a count law and pretends the discarded total is known;
- adds pseudocounts, epsilon floors, clipping, support filling, or hidden renormalisation;
- adopts mfapy's Gaussian-noise-plus-zero-clipping-plus-renormalisation procedure as if it were a rigorous probability density;
- silently changes the existing MFA objective to count-weighted loss;
- calls Rényi fitting a likelihood;
- says the multinomial model is the universal or default MS noise model;
- introduces hypothesis-test decision rules, p-values, confidence regions, finite-sample bounds, composite-class optimisation, or theorem code prohibited by `AGENTS.md`;
- duplicates the existing KL/Rényi implementation instead of reusing it;
- evaluates EMU on incomplete or independently assembled FVA coordinates;
- loses experiment/target/replicate ordering;
- leaves raw counts unidentifiable after normalisation;
- uses floating-point values for count observations without exact integer validation;
- treats booleans as counts or sample sizes;
- declares completion without exhaustive small-case probability checks and end-to-end stationary tests.

## Token discipline

The guardian must keep specialist scopes disjoint and stop redundant redesign. Require concise evidence-based reports and terminate agents once their deliverable is integrated or rejected.

---

# 3. Mandatory specialist agents

After the guardian is active, spawn specialists with non-overlapping responsibilities.

## Agent A — mfapy and measurement-semantics auditor

Own only the reference audit.

Inspect the vendored mfapy paths relevant to observed MDV handling, especially:

```text
vendor/mfapy/mfapy/mdv.py
vendor/mfapy/mfapy/optimize.py
vendor/mfapy/mfapy/metabolicmodel.py
```

Trace, with exact source locations where practical:

- how mfapy stores MDV ratios, standard deviations, and replicate data;
- `MdvData.add_gaussian_noise(...)` including absolute/relative noise, negative-value clipping, and optional renormalisation;
- how replicate data are averaged/stored;
- how standard deviations feed the weighted-RSS fitting path;
- whether mfapy ever has literal count semantics or a proper generative PMF/PDF for observed MDVs.

Produce a short transfer table:

```text
mfapy mechanism | reuse | adapt | reject | reason
```

Expected stance unless source evidence says otherwise:

- reuse its useful workflow concepts and explicit replicate handling ideas;
- do not copy its Gaussian-noise/clipping/renormalisation heuristic as the rigorous observation law;
- do not copy weighted RSS into this layer;
- do not infer a probability model that mfapy itself does not define.

Also inspect `codex/docs/MID_PREPROCESSING.md` and explain how explicit normalisation differs from retaining raw count totals.

No implementation until the guardian approves this audit.

## Agent B — multinomial mathematics and numerical referee

Own only probability-law mathematics and numerical semantics.

For a probability MID

```text
p = (p_0, ..., p_{m-1})
```

and a positive integer total count `n`, define the observation law on count vectors

```text
k_i >= 0 integer, sum_i k_i = n
```

by

```text
P_p(K=k) = n! / prod_i k_i! * prod_i p_i**k_i.
```

Specify exact support semantics:

- a positive count in a class with `p_i == 0` gives probability zero / log-PMF `-inf`;
- `k_i == 0` contributes no `log(p_i)` term, including when `p_i == 0`;
- no pseudocounts;
- no hidden probability repair;
- `n` and all counts are true non-negative integers, not floats or bools;
- count observations used as measurements must have positive total count.

Require a stable log-PMF based on `lgamma`/log-factorials and compensated sums where useful. Do not form factorials or products naively for large counts.

Prove and test in code, without implementing a hypothesis test:

### KL law identity

```text
D_KL(Mult(n,p) || Mult(n,q)) = n D_KL(p || q)
```

### Rényi law identity

For every finite real `alpha > 0`, including exact dispatch at `alpha == 1`:

```text
D_alpha(Mult(n,p) || Mult(n,q)) = n D_alpha(p || q).
```

The production implementation should reuse the existing MID divergence functions rather than enumerate the count support. Exhaustive count-support enumeration is required in tests for small dimensions/counts as an independent oracle.

### Count-likelihood / KL fitting identity

For an observed count vector `k`, with `n=sum(k)` and empirical MID `p_hat=k/n`, verify:

```text
-log P_p(k) = n D_KL(p_hat || p) + C(k)
```

where

```text
C(k) = -log(n!) + sum_i log(k_i!) - sum_{i:k_i>0} k_i log(k_i/n).
```

The documentation must state the consequence precisely:

- for genuine fixed-total multinomial count data, minimising `D_KL(p_hat || p)` for one fixed `n` is equivalent to maximum likelihood;
- across independent observations with different totals `n_j`, the joint likelihood corresponds to `sum_j n_j D_KL(p_hat_j || p_j)` plus a data-only constant;
- this does **not** mean the current unweighted multi-MID MFA objective is automatically the multinomial likelihood when totals differ;
- Rényi MID fitting is not to be relabelled as maximum likelihood.

No statistical decision rules in this task.

## Agent C — observation-law API architect

Own only scientific data structures and public API.

Design the smallest stable API that separates:

```text
predicted probability MID
```

from

```text
observed count vector and its sampling law.
```

Conceptual objects may include:

```text
MultinomialMIDLaw
MIDCountObservation
StationaryObservationSpecification
StationaryObservationLawResult
```

Exact names may differ if repository conventions justify them.

Required semantics:

- immutable records;
- explicit total count;
- count vector dimension exactly matches the predicted MID dimension;
- raw counts remain available and are never replaced by only `k/n`;
- empirical MID may be exposed as a derived property/helper, not the sole stored data;
- experiment ID, target ID, replicate ID, and declared mass-class order remain explicit where stationary bridge records are involved;
- deterministic fingerprints/provenance bind the law type, count total, experiment identity, target identity, model/experiment fingerprints, and probability vector where appropriate;
- ordinary package import must not require SciPy;
- future observation laws must be addable without modifying the EMU engine.

Do not create a sprawling generic probabilistic-programming framework. Do not redesign the stationary MFA problem unnecessarily. If a shared small internal identity structure can be reused safely, prefer that over duplicating a second model hierarchy.

The guardian must approve the API before implementation proceeds.

## Agent D — stationary EMU bridge and sampling engineer

Own only the bridge

```text
CanonicalFluxState v -> predicted MID p_v -> multinomial law -> synthetic count observation
```

Requirements:

- reuse native stationary experiment validation and compiled EMU plans;
- compile each EMU plan once per bridge operation/batch where practical;
- accept only complete canonical flux states in declared reaction order;
- preserve exact experiment/target/replicate order;
- require the caller to provide explicit count totals/specifications;
- never derive count totals from normalised MIDs or arbitrary intensities;
- deterministic sampling uses `numpy.random.Generator` with an explicit seed or caller-supplied generator contract;
- repeated sampling must not mutate the law;
- expose the predicted MID alongside generated counts for auditability;
- use the existing machine-simplex/validation boundaries for the probability vector rather than silently normalising EMU output in this layer.

Build an end-to-end stationary synthetic fixture from a known complete flux state and verify:

```text
v* -> EMU p_v* -> Multinomial(n,p_v*) -> sampled counts
```

with reproducible sample IDs/observation identities.

Do not fit the sampled data in this task unless a tiny acceptance demonstration can use an existing API without changing semantics. The required deliverable is the law layer, not another optimiser.

## Agent E — regression, exhaustive-oracle, docs, and CI reviewer

Own only validation and packaging.

Required tests include:

### Probability law

- log-PMF equals a direct small-number calculation;
- exhaustive support probabilities sum to one for small `(m,n)` cases;
- impossible support gives `-inf` exactly;
- all-zero observation is rejected as a measurement;
- negative, non-integer, boolean, NaN/infinite, and dimension-mismatched counts fail;
- very large counts remain numerically stable in log space;
- deterministic sampling is reproducible for a fixed seed and respects total count exactly.

### Divergence identities

Use independent exhaustive enumeration on small count spaces to verify the production identities for representative orders such as:

```text
alpha = 0.5, 0.73, 1.0, 1.3, 2.0
```

including support-mismatch cases.

Verify independent-product additivity across two blocks with different totals.

### KL likelihood identity

Verify numerically for several count vectors, including zeros, that

```text
-logpmf == n * KL(empirical_mid || predicted_mid) + C(counts)
```

within tight numerical tolerance.

### Stationary bridge

- complete-state EMU prediction is preserved exactly;
- declared order is preserved;
- explicit count totals are retained;
- sampled counts map back to the correct experiment/target/replicate;
- no FVA endpoint vectors are used;
- existing MFA, FBA/FVA, sampling, and stationary EMU regression suites remain green.

### Documentation

Add a concise user document, conceptually:

```text
codex/docs/STATIONARY_OBSERVATION_LAW.md
```

It must clearly say:

1. the multinomial law is appropriate only when the measurement has count semantics;
2. percentages, peak areas, and arbitrary intensities are not automatically counts;
3. `normalise_mid(...)` produces a probability composition but discards the total and therefore cannot manufacture `n`;
4. raw counts/totals must be retained separately;
5. the exact KL-likelihood equivalence holds under this specific count law;
6. law-level Rényi divergence scales as `n D_alpha(p||q)`, but this package does not yet implement hypothesis-test decision rules or finite-sample bounds;
7. more realistic MS intensity observation laws are future work and must be justified from actual measurement semantics/replicates rather than chosen aesthetically.

CI must exercise the new law in the base installation if no new dependency is needed.

---

# 4. Architecture requirements

Prefer a small package boundary that does not bury the observation model inside the optimizer. A reasonable conceptual layout is:

```text
fluxemu/observation/
    __init__.py
    multinomial.py
    stationary.py
```

or an equally clean repository-conventional alternative.

The important separation is:

```text
fluxemu.mfa          # MID-divergence fitting
fluxemu.observation  # explicit P(Y|v) data-generating laws
```

Do not make the observation-law package depend on the MFA optimizer.

The stationary runtime path should be auditable as:

```text
model + stationary experiment + complete v + explicit count specification
        |
        +--> validate native model/experiment/state
        +--> compile EMU plan
        +--> evaluate predicted MID p_v
        +--> construct MultinomialMIDLaw(n, p_v)
        +--> optional deterministic sample -> raw count observation k
```

The law object must also support direct `log_prob`/`log_pmf` of a supplied count observation.

---

# 5. Scientific invariants

These are non-negotiable.

## Counts are not intensities

Never do any of:

```text
normalised MID -> multiply by arbitrary N -> round -> pretend counts
percentages -> round -> pretend counts
peak areas -> cast to integers -> pretend counts
standard deviation -> infer effective N
```

If count semantics are absent, the multinomial law is not instantiated from those measurements.

## Support is exact

If predicted probability is zero for a class that has positive observed count:

```text
log P = -inf
```

No repair.

## Existing MID divergence semantics remain unchanged

Reuse:

```text
kl_divergence
renyi_divergence
```

Do not add a second implementation with different zero/support behaviour.

## Existing MFA remains a separate estimator

Do not silently turn the current multi-observation objective into

```text
sum_j n_j D_KL(...)
```

in this task. Document the likelihood identity and the distinction. A future explicit law-aware fitting mode can be considered separately if desired.

## No inference claims beyond the law

It is acceptable to expose exact law-level KL/Rényi divergence because it is a property of the observation distribution. Do not implement hypothesis-test thresholds, optimal errors, confidence sets, or composite projections here.

---

# 6. Required acceptance evidence

Before final approval, the guardian must receive concise evidence for all of the following:

1. exact current-main baseline SHA;
2. mfapy measurement/noise audit with source locations;
3. approved API map;
4. exhaustive PMF normalization checks for small laws;
5. exhaustive KL/Rényi law-divergence comparisons against the analytic `n * D_alpha` identity;
6. count-likelihood/KL identity checks;
7. support-mismatch tests;
8. deterministic sampling tests;
9. end-to-end native stationary EMU -> law -> counts fixture;
10. import/dependency isolation;
11. unchanged existing MFA semantics;
12. Stage 1/MFA regressions;
13. docs explicitly refusing pseudo-count conversion from arbitrary intensities;
14. clean working tree;
15. coherent commits pushed;
16. open non-draft PR with CI green.

The guardian must perform a final adversarial review and issue:

```text
FINAL APPROVE
```

only if every item is satisfied.

---

# 7. Final report format

Return only a concise completion summary containing:

- guardian final status;
- branch;
- final SHA;
- PR number/state;
- public API added;
- exact observation-law semantics;
- confirmation that arbitrary intensities are not reinterpreted as counts;
- PMF normalization evidence;
- KL/Rényi multinomial identity evidence;
- count-likelihood/KL identity evidence;
- end-to-end stationary synthetic evidence;
- test/CI totals;
- explicit deferred items.

Stop there.
