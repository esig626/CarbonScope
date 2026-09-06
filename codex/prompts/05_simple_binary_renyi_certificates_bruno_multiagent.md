# Codex task: simple binary finite-sample Rényi error certificates for stationary observation laws

Repository: `esig626/fluxemu-standalone`

Work on branch: `codex/simple-binary-renyi-certificates`

Read `AGENTS.md` first and obey it exactly.

Then execute this prompt in full.

---

## 0. Goal and stopping boundary

Implement the first **simple-vs-simple finite-sample hypothesis-testing certificate layer** on top of the already merged stationary multinomial observation-law infrastructure.

This is published-theorem software, not exploratory composite-testing research.

The required scientific path is:

```text
complete feasible null flux state v0
        -> stationary EMU
        -> explicit observation law P0 = P(Y | v0)

complete feasible alternative flux state v1
        -> stationary EMU
        -> explicit observation law P1 = P(Y | v1)

(P0, P1, epsilon)
        -> forward/reverse Rényi law divergences
        -> finite-sample lower certificate on optimal Type-II error
        -> transparent order-specific diagnostics
```

The theoretical reference is:

> Roberto Bruno, Adrien Vandenbroucque, Amedeo Roberto Esposito,
> **A Finite-Sample Strong Converse for Binary Hypothesis Testing via (Reverse) Rényi Divergence**, arXiv:2601.09550v2 (2026).

Use the paper as the source of truth. Audit the current arXiv v2/HTML/source before implementation if network access is available. The equations below are included to remove ambiguity if remote access is unavailable.

### This task must implement

- simple binary testing objects with explicit null and alternative roles;
- the Bruno et al. order-specific converse lower bounds for Type-II error under a Type-I constraint;
- full-law forward and reverse Rényi divergences using the existing `fluxemu.observation` layer;
- support for one multinomial law and ordered independent products of multinomial blocks;
- a stationary flux-state bridge that maps two complete feasible states into aligned null/alternative law families;
- exact mutual-absolute-continuity/support validation required by the theorem;
- stable log-domain numerical evaluation;
- log-likelihood-ratio evaluation for realised count observations;
- exact small-support deterministic-test oracles for validation only;
- synthetic flux-state discrimination examples showing how count total and MID separation change the finite-sample certificate;
- tests, docs, provenance/fingerprints, public exports, CI, commits, push, and PR.

### This task must NOT implement

- composite hypotheses or sets of flux states;
- test inversion or confidence/compatibility regions;
- the Vera-Sigüenza & Esposito composite theorem;
- least-favourable pairs;
- priors or Bayesian testing;
- p-values;
- chi-square/F tests;
- estimation uncertainty;
- additional observation laws beyond the already supported fixed-total multinomial law;
- transient MFA;
- JAX;
- experiment design;
- VFFVA performance work;
- a new MFA fitting objective;
- inferred effective sample sizes from intensities or normalised MIDs.

Stop once the simple binary certificate layer is implemented, independently validated, documented, committed, pushed, opened as a PR, and the guardian gives FINAL APPROVE.

---

# 1. Mandatory theorem semantics

The software must preserve the paper's asymmetric testing convention:

```text
H0 : observation law = P0
H1 : observation law = P1

Type I error  alpha = P0(decide H1)
Type II error beta  = P1(decide H0)

beta*(epsilon)
    = minimum achievable Type-II error
      subject to Type-I error <= epsilon.
```

Do not swap null/alternative roles and do not silently swap divergence directions.

## 1.1 Full-law form of Bruno et al. Theorem 1

Appendix A establishes the converse by data processing **before** i.i.d. tensorisation. Therefore implement the bound first in terms of the divergences of the complete observation laws.

For finite real `lambda > 1`, define

```text
Drev(lambda) = D_lambda(P1 || P0)
Dfwd(lambda) = D_lambda(P0 || P1)
```

and the two order-specific lower bounds

```text
B_reverse(lambda, epsilon)
    = 1 - (epsilon * exp(Drev(lambda)))^((lambda - 1) / lambda)

B_forward(lambda, epsilon)
    = (1 - epsilon)^(lambda / (lambda - 1))
      * exp(-Dfwd(lambda)).
```

Thus, for every admissible `lambda > 1`,

```text
beta*(epsilon) >= max(B_reverse(lambda, epsilon),
                      B_forward(lambda, epsilon)).
```

The paper's optimized envelope is

```text
beta*(epsilon) >= max {
    1 - inf_{lambda>1}
          (epsilon exp(D_lambda(P1||P0)))^((lambda-1)/lambda),

    sup_{lambda>1}
          (1-epsilon)^(lambda/(lambda-1))
          exp(-D_lambda(P0||P1))
}.
```

For the existing fixed-total multinomial law,

```text
D_lambda(Mult(n,p) || Mult(n,q)) = n D_lambda(p || q),
```

which was already established in the previous observation-law milestone.

For declared independent ordered blocks, use the existing product-law Rényi additivity. Different blocks may have different explicit count totals; never collapse them into a fabricated common sample size.

## 1.2 No fake global optimisation claim

The theorem optimizes over the full continuous interval `lambda > 1`.

The exact production primitive required in this task is therefore an **order-specific certificate** accepting every finite real `lambda > 1`.

Every such output is already a rigorous finite-sample lower certificate.

You may additionally implement a continuous numerical search helper only if all of the following hold:

- it does not expose a finite Rényi-order grid as the mathematical API;
- it searches a continuous transformed domain;
- the returned bound is labelled as the valid bound at the best order actually found;
- it explicitly states that numerical search does not certify the global supremum/infimum unless a genuine global certificate is proved;
- the mathematical full-order envelope remains documented exactly.

Do **not** call a local numerical optimum "the Bruno bound" without that qualification.

A grid-only optimizer, fixed list of orders, or hidden discretisation is grounds for guardian REJECT.

## 1.3 The theorem assumption is mandatory

Bruno et al. Theorem 1 assumes mutual absolute continuity of the null and alternative laws.

For `MultinomialMIDLaw`, this means the two laws must have the same positive-support mass classes for each corresponding fixed-total block.

For independent products, every corresponding block must satisfy this condition.

If this assumption fails:

- do not silently smooth;
- do not insert pseudocounts;
- do not clip zero probabilities;
- do not pretend the theorem applies because a divergence happens to be finite at one order;
- fail with a precise theorem-assumption error.

Exact support remains part of the scientific model.

---

# 2. Mandatory persistent supervisor: SIMPLE-TEST-GUARDIAN

Before spawning implementation specialists, create one persistent supervisor agent named conceptually **SIMPLE-TEST-GUARDIAN**.

It must remain active until final acceptance.

The guardian must issue one of

```text
APPROVE
REJECT
BLOCKED
```

for:

1. source/theorem audit;
2. testing API design;
3. numerical design;
4. stationary bridge design;
5. validation/oracle design;
6. each implementation milestone;
7. final PR state.

The lead may not override a `REJECT` without fixing the cited defect.

## Guardian must reject any proposal that

- swaps `P0` and `P1` or Type-I/Type-II semantics;
- uses MID divergence in place of observation-law divergence without the explicit multinomial-law identity;
- applies Bruno Theorem 1 when mutual absolute continuity fails;
- introduces pseudocounts, clipping, support repair, or hidden normalisation;
- treats arbitrary intensities or percentages as counts;
- infers an effective count total;
- hard-codes a finite Rényi-order grid as the theorem API;
- claims a numerical local search is the exact continuous-order envelope;
- adds composite hypotheses;
- adds p-values, chi-square logic, priors, or Bayesian decisions;
- changes the existing MFA objective;
- modifies vendor code;
- implements a duplicate observation-law stack instead of reusing `fluxemu.observation`;
- evaluates incomplete or infeasible flux states;
- uses independent FVA endpoints as flux hypotheses;
- omits failed theorem assumptions from diagnostics;
- leaves only happy-path tests without independent finite-support validation.

---

# 3. Mandatory specialist agents

After the guardian is active, spawn non-overlapping specialists.

## Agent A — published-theorem referee

Owns only the Bruno et al. source audit.

Must verify against arXiv:2601.09550v2:

- definitions of Type I, Type II, and `beta_n(epsilon)`;
- Theorem 1 and Appendix A;
- the forward and reverse divergence directions;
- the mutual-absolute-continuity assumption;
- the fact that Appendix A gives the full-law inequality before tensorisation;
- the exact exponents in both lower-bound terms;
- natural-log convention;
- what is and is not claimed about deterministic LLRTs and optimality.

Produce a concise theorem-transfer note with exact equation references before implementation proceeds.

If the paper differs from this prompt, stop and report the discrepancy to the guardian rather than silently choosing one.

## Agent B — testing API architect

Owns first-class simple-testing records and provenance.

Design objects conceptually equivalent to:

```text
SimpleBinaryLawPair
SimpleBinaryTestingConstraint
BrunoOrderCertificate
SimpleBinaryFluxHypotheses
StationarySimpleTestingResult
```

Exact names may follow repository conventions.

Required semantics:

- explicit `null` and `alternative` fields/labels, never positional ambiguity;
- one `epsilon` with strict `0 < epsilon < 1`;
- one finite real order `lambda > 1` for the exact core certificate;
- preserve law/block order;
- preserve experiment/target/replicate identities;
- store forward and reverse law divergences separately;
- store both raw Bruno lower-bound components and their maximum;
- bind null/alternative law fingerprints, epsilon, order, state fingerprints, and observation specification fingerprint where applicable;
- never call a lower certificate an observed p-value or confidence level;
- clear names distinguishing `Type I constraint epsilon` from `Rényi order lambda`.

## Agent C — law/numerical engineer

Owns the mathematical implementation over existing observation laws.

At minimum implement:

```text
bruno_converse_at_order(...)
```

for:

1. one pair of `MultinomialMIDLaw` objects;
2. corresponding ordered independent products of multinomial laws.

Reuse:

```text
kl_multinomial
renyi_multinomial
independent_product_kl
independent_product_renyi
```

Do not independently reimplement Rényi divergence.

### Numerical requirements

Evaluate in stable log space.

For the reverse term, work with

```text
x = ((lambda - 1) / lambda) * (log(epsilon) + Drev)
B_reverse = 1 - exp(x)
```

and use a stable formulation such as `-expm1(x)` where appropriate.

For the forward term, work with

```text
log_B_forward
    = lambda/(lambda-1) * log(1-epsilon) - Dfwd
```

using `log1p(-epsilon)`.

Requirements:

- no overflow merely because a mathematically vacuous reverse component is very negative;
- preserve raw component semantics rather than silently clipping them into `[0,1]`;
- the final maximum must be finite and interpretable when theorem assumptions hold;
- reject bool, NaN, infinity, `lambda <= 1`, malformed epsilon;
- identical laws must be handled correctly;
- near-one orders `nextafter(1, +inf)` must not be snapped to one;
- huge finite orders must either evaluate stably or fail with an explicit numerical-limit error;
- no arbitrary epsilon floors.

### Optional continuous-order search

If implemented, transform the unbounded domain rather than truncating it silently, for example

```text
t = (lambda - 1) / lambda in (0, 1)
lambda = 1 / (1 - t).
```

Any search result must retain the exact certificate at the returned order and a `global_envelope_certified=False`-style semantic unless a true global proof is supplied.

Do not make SciPy a base runtime dependency merely for this optional search.

## Agent D — stationary flux bridge engineer

Owns the mapping

```text
(v0, v1, StationaryObservationSpecification)
    -> aligned P0 and P1 law blocks
    -> simple binary testing certificate.
```

Reuse the merged API:

```text
evaluate_stationary_observation_laws(...)
```

Requirements:

- both hypotheses are complete `CanonicalFluxState` records;
- original-model feasibility must be independently validated by the existing observation bridge;
- compile/evaluate through the existing stationary observation-law path;
- preserve state, experiment, target, replicate, mass-class, and block order;
- corresponding null and alternative components must have identical declared identities and count totals;
- no biological FBA objective restriction unless already explicitly part of the observation specification/model constraints;
- no fitting occurs here;
- no FVA endpoints are hypotheses;
- expose the predicted MIDs and observation-law fingerprints so the error certificate is traceable back to the flux pair.

Add a simple log-likelihood-ratio utility for realised count samples:

```text
LLR(y) = log P1(y) - log P0(y)
```

for one block and aligned independent products.

Exact support semantics must produce the mathematically correct finite or infinite LLR; do not smooth.

## Agent E — independent oracle and regression reviewer

Owns validation independent of the production formulas.

### Required finite-support oracle

For small multinomial examples only, enumerate all count outcomes exactly enough to validate:

- PMFs sum to one within an explicit oracle tolerance;
- law Rényi values agree with direct finite-support summation;
- for many small cases, the Bruno order-specific lower certificate never exceeds the optimal Type-II error under the paper's deterministic binary-test convention.

The oracle must not call the production Bruno bound to derive its expected result.

Where the discrete Neyman-Pearson/randomisation convention is subtle, document it explicitly and validate against the exact convention actually stated in Bruno et al. Do not smuggle in a different randomized-test definition.

### Required flux-level programme

Use at least one deterministic synthetic stationary FluxEMU fixture with two distinct feasible flux states producing distinct MIDs.

For several genuine count totals, demonstrate that:

```text
same flux pair
-> same p0, p1
-> larger explicit count total
-> larger law Rényi separation
-> correspondingly changed finite-sample Type-II lower certificate.
```

Also include:

- identical-law control;
- matching-support but very close-law control;
- support-mismatch control that must reject Bruno certification;
- multi-block independent product with different declared count totals.

No random measurement model beyond the already explicit multinomial law is needed for acceptance.

---

# 4. Required package architecture

Do not bury this inside MFA fitting.

A reasonable structure is conceptually:

```text
fluxemu/testing/
    __init__.py
    simple.py          # simple law-pair schemas and common validation
    bruno.py           # published finite-sample Rényi certificates
    stationary.py      # flux-state -> observation-law -> certificate bridge
```

or another comparably clean structure consistent with the repository.

`fluxemu.observation` remains the owner of observation laws.
`fluxemu.mfa` remains the owner of estimation.
`fluxemu.testing` owns simple binary decision/error certificates.

The base package should remain importable without SciPy.

---

# 5. Required public scientific API behaviour

A user should be able to perform the conceptual equivalent of:

```python
laws = evaluate_stationary_simple_hypotheses(
    specification,
    null_state=v0,
    alternative_state=v1,
)

certificate = bruno_converse_at_order(
    laws,
    epsilon=0.05,
    order=1.7,
)

print(certificate.type_i_constraint)
print(certificate.reverse_renyi)
print(certificate.forward_renyi)
print(certificate.reverse_lower_bound)
print(certificate.forward_lower_bound)
print(certificate.type_ii_lower_bound)
```

Exact names may differ.

The result must make the interpretation explicit:

> Among tests satisfying Type-I error <= epsilon, the optimal Type-II error is at least this certified value for this null/alternative observation-law pair and this Rényi order.

Do not say that the certificate is the actual Type-II error.
Do not say that a particular realised sample has error probability equal to the bound.
Do not turn it into a p-value.

---

# 6. Operational interpretation to document

The documentation must connect the layers without overclaiming:

```text
flux state v
    -> EMU predicted MID p_v
    -> genuine-count law Mult(n, p_v)
    -> law Rényi divergence
    -> finite-sample binary-testing error certificate.
```

For a single fixed-total multinomial block,

```text
D_lambda(P1 || P0)
    = n D_lambda(p_v1 || p_v0).
```

Therefore, in this declared count model, the MID Rényi divergence has a direct operational role in finite-sample distinguishability of the two flux hypotheses.

Important wording:

- `n` is the **explicit genuine count total**, not a fitted tuning parameter;
- this interpretation does not apply to arbitrary peak areas, percentages, or normalised intensity vectors unless a valid observation law is separately declared;
- the Bruno certificate concerns discrimination between two **fixed** flux hypotheses;
- it is not yet a confidence region for an unknown flux;
- the next composite/testing-inversion layer is intentionally outside this task.

---

# 7. Exact theorem checks required

At minimum, tests must establish all of the following.

## 7.1 Formula checks

For fixed `lambda > 1`, independently verify the production result against direct formula evaluation for many finite-support law pairs.

Check both divergence directions separately.

## 7.2 Bound validity

For small finite outcome spaces where exact deterministic testing can be enumerated, verify

```text
certificate.type_ii_lower_bound <= exact_optimal_beta(epsilon)
```

up to a documented numerical tolerance.

Test many epsilon/order combinations, including difficult regimes where the reverse component is vacuous.

## 7.3 Tensorisation / multinomial bridge

Independently verify that the testing certificate obtained from full multinomial law divergences equals the corresponding expression using

```text
n * D_lambda(base MID distributions)
```

within numerical tolerance.

## 7.4 Product laws

For independent blocks, direct finite-support enumeration on tiny examples must agree with summed law Rényi divergence and the resulting order-specific certificate.

## 7.5 Support assumptions

Mutually absolutely continuous matching-support laws pass.
Support mismatch fails before certificate computation with an error naming the Bruno theorem assumption.
No smoothing is ever performed.

## 7.6 Directionality

Swapping null and alternative must generally change the reverse/forward divergence fields and can change the certificate.
Tests must catch an accidental direction swap.

## 7.7 Numerical extremes

Cover:

- `epsilon` near 0 and near 1 but strictly inside;
- order immediately above 1;
- moderate orders such as 1.1, 1.5, 2, 5;
- very large finite order where supported numerically;
- nearly identical laws;
- exact identical laws;
- large but valid count totals allowed by the observation layer.

---

# 8. Documentation and citation requirements

Add a focused document, for example:

```text
codex/docs/SIMPLE_BINARY_RENYI_CERTIFICATES.md
```

It must include:

- full bibliographic citation to Bruno, Vandenbroucque & Esposito (2026), arXiv:2601.09550v2;
- theorem/equation mapping to Theorem 1 and Appendix A;
- exact null/alternative and Type-I/Type-II convention;
- the full-law form before tensorisation;
- the multinomial identity already provided by FluxEMU observation laws;
- the order-specific API and why each order already yields a valid certificate;
- any numerical-order search limitations;
- mutual absolute continuity requirement;
- count-semantics boundary;
- one complete flux-state example;
- a statement that composite hypotheses/test inversion are deferred.

Update relevant API map, README, and known limitations without turning the standalone repository into a research manuscript.

Do not copy long prose passages from the paper. Cite and restate the mathematics in repository terminology.

---

# 9. Dependency and regression requirements

The published-bound core should require only the existing base numerical stack.

Do not add SciPy as a base dependency.

If an optional continuous-order optimizer truly needs SciPy, isolate it behind a clearly named optional extra or reuse an already optional boundary lazily; ordinary bound evaluation must remain base-only.

Run:

- all new simple-testing tests;
- all observation-law tests;
- all MFA tests in the appropriate environment;
- Stage 1 regression gates;
- import-isolation tests;
- `pip check` for base and relevant optional installs;
- compileall;
- `git diff --check`.

Vendor files must remain byte-for-byte unchanged.

---

# 10. Milestones and commits

Commit and push each coherent milestone before proceeding.

## Milestone A — source audit and API skeleton

- Bruno v2 theorem-transfer note;
- guardian approval;
- simple-testing schemas and theorem-assumption validation;
- tests for roles/order/epsilon/support assumptions.

Commit and push.

## Milestone B — order-specific published bound

- stable Bruno converse implementation;
- one-law and product-law support;
- forward/reverse diagnostics;
- finite-support independent oracle checks.

Guardian approval, then commit and push.

## Milestone C — stationary flux bridge

- null/alternative complete-state bridge;
- LLR utility for realised counts;
- identities/provenance/order preservation;
- synthetic flux discrimination fixtures.

Guardian approval, then commit and push.

## Milestone D — regression/docs/CI/final audit

- full regression;
- docs and citation;
- CI;
- adversarial audit;
- guardian FINAL APPROVE;
- clean push and PR.

---

# 11. Mandatory adversarial final audit

Before FINAL APPROVE, the guardian must answer all of these explicitly:

1. Are null and alternative roles preserved everywhere?
2. Is Type I `P0(decide H1)` and Type II `P1(decide H0)` everywhere?
3. Is reverse Rényi always `D_lambda(P1 || P0)`?
4. Is forward Rényi always `D_lambda(P0 || P1)`?
5. Does the exact core accept every finite real `lambda > 1` rather than a finite grid?
6. Are the two Bruno lower-bound terms implemented with the correct exponents?
7. Is the implementation based on full observation-law divergence before any multinomial simplification?
8. Is the multinomial `n * D_lambda` identity reused rather than rederived inconsistently?
9. Are independent product blocks aggregated only under explicit declared independence?
10. Is mutual absolute continuity checked before applying Theorem 1?
11. Is support mismatch rejected without smoothing?
12. Are arbitrary intensities/percentages never assigned pseudo-count semantics?
13. Does any numerical order search avoid claiming exact global optimization?
14. Are raw and combined lower-bound components exposed?
15. Are numerical formulas stable near lambda=1 and extreme epsilon?
16. Does an independent finite-support oracle verify bound validity?
17. Are complete flux hypotheses independently feasible and correctly ordered?
18. Are experiment/target/replicate identities preserved through the bridge?
19. Is LLR defined as log P1 - log P0?
20. Are MFA fitting and biological objectives unchanged?
21. Is there no composite hypothesis implementation in this task?
22. Are vendor files unchanged?
23. Are base imports free of accidental SciPy dependence?
24. Do all required CI/regression checks pass?
25. Does the documentation state exactly what the certificate means and what it does not mean?

Any `no` is a REJECT until corrected.

---

# 12. Final report

The final Codex response must report only substantive completion evidence:

- `SIMPLE-TEST-GUARDIAN: FINAL APPROVE` or failure reason;
- branch;
- final SHA;
- PR number/status;
- public API summary;
- exact Bruno equation mapping implemented;
- theorem-assumption handling;
- finite-support oracle evidence;
- stationary flux-state synthetic evidence;
- regression/CI counts;
- explicit deferred scope.

Do not proceed into composite testing or test inversion.
