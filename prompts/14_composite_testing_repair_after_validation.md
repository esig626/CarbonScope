# Repair composite testing after adversarial validation

Use repository `esig626/CarbonScope`.

Work only on branch `codex/composite-testing-repair`.

Read `AGENTS.md` first and obey it exactly.

Do not merge to `main`.
Do not edit manuscript material.
Do not change the runtime package name from `fluxemu`.

This is a repair task. The first implementation on `feature/composite-minimax-testing` failed adversarial validation. Your job is to reproduce the failures, repair the implementation, and leave a validated branch with evidence.

## Validation evidence already obtained

A prior independent swarm reported the following on a local branch that is not currently available on GitHub:

- Python 3.11: 1,476 tests passed after repair.
- Python 3.12: 1,476 tests passed after repair.
- General simulation campaign: 1,309 problems, 1,300 complete comparisons, nine explicit LP refusals.
- Numerical campaign: 10,000 problems, 7,777 accepted answers, 2,223 explicit refusals.
- Targeted numerical cases: 74, no unexpected outcomes.
- The original implementation contained 11 defect categories.
- Reported defects included false converse bounds, negative rejection probabilities, undetected LP optimality gaps, false moment certification, and inconsistent stationary provenance.
- The underlying minimax LP formulation and composite converse inequalities were judged mathematically correct.
- Documentation overclaims required correction.
- Worst achieved projected-test optimality gap observed: 0.87224148.
- General campaign converse gaps ranged from 1.87196e-8 to 0.98585635.
- A substantive numerical limitation remained: the LP necessarily refused fully supported binary count laws at n >= 40 and ternary laws at n >= 26 under the chosen HiGHS resolution policy.
- Continuous flux-family testing remained unsolved because rigorous optimisation or bounds over the complete continuous feasible family were not implemented.

Do not treat the summary above as proof that a particular fix is correct. Reproduce every relevant defect independently from the current branch state.

## Mathematical source of truth

For finite composite classes

    H0: P in C0
    H1: Q in C1

and a randomised test phi(y) in [0,1],

    alpha(phi; C0) = sup_{P in C0} E_P[phi(Y)]

and

    beta_star(epsilon; C0, C1)
      = inf_{phi: alpha(phi;C0) <= epsilon}
        sup_{Q in C1} E_Q[1-phi(Y)].

For an enumerable finite observation space, the exact minimax characterisation is the LP

    minimise t

    subject to
        sum_y P(y) phi(y) <= epsilon       for every P in C0
        sum_y Q(y) phi(y) + t >= 1         for every Q in C1
        0 <= phi(y) <= 1
        0 <= t <= 1.

The LP is unrestricted over all randomised tests on the finite observation space.

The projected Renyi route is different. For 0 < lambda < 1, select a pair minimising D_lambda(Q||P) over the explicitly supplied finite law classes, form

    h(y) = log(Q*(y)/P*(y)),

then verify the actual uniform exponential moments over every supplied law. A finite list is not automatically a convex class, so the convex-class theorem must never be silently invoked. Threshold calibration against the actual uniform Type I constraint gives an achieved test and therefore an upper bound on beta_star, not beta_star itself in general.

For lambda > 1, the composite converse is obtained pairwise and must satisfy

    converse <= beta_star.

Whenever all three calculations are applicable, the central invariant is

    composite converse <= beta_star <= achieved projected-test error.

## Required repair programme

### 1. Audit current implementation

Inspect line by line:

- `src/fluxemu/testing/composite.py`
- `src/fluxemu/testing/composite_stationary.py`
- `src/fluxemu/testing/__init__.py`
- simple testing code
- multinomial observation laws
- stationary observation bridge
- all current composite tests and docs

Before editing, write a private defect map linking each reported defect category to likely code paths and required regression tests.

### 2. Reproduce and fix all known defect categories

At minimum, actively reproduce and repair:

1. false composite converse bounds;
2. negative rejection probabilities returned by numerical LP solutions;
3. LP solutions accepted despite a material optimality gap or inconsistency;
4. false projected-moment certification;
5. inconsistent or lossy stationary provenance;
6. any incorrect clipping or sanitisation of solver decision variables;
7. any support loss caused by tiny positive probabilities;
8. any incorrect null/alternative or forward/reverse Renyi direction;
9. any result object that reports a mathematically stronger claim than was actually certified;
10. any documentation claim that confuses finite supplied classes with continuous flux families;
11. any numerical path where a plausible answer is returned although the programme should explicitly refuse.

Do not patch symptoms by clipping negative phi to zero or beta into [0,1]. If solver output is outside justified tolerance, fail explicitly or tighten the formulation/validation so the mathematical contract is restored.

### 3. Independent finite LP oracle

Build an independent audit implementation that does not call the production solver path. For small problems, use an independent LP formulation and direct recomputation of all constraints.

Mandatory controls:

- singleton simple-vs-simple recovers randomised Neyman-Pearson;
- identical hypotheses give beta_star = 1-epsilon;
- disjoint supports give the analytically expected optimum;
- duplicate laws do not alter beta_star;
- class permutation does not alter beta_star;
- increasing epsilon cannot increase beta_star;
- enlarging either uncertainty class cannot decrease beta_star;
- every returned phi is in [0,1] within the declared tolerance;
- direct recomputation agrees with reported Type I, Type II and beta_star;
- the solver objective, epigraph variable and recomputed worst Type II agree within justified tolerance.

### 4. LP optimality and numerical certification

Do not accept `kOptimal` from HiGHS as sufficient evidence by itself.

Inspect available HiGHS solution diagnostics and certify the numerical solution as far as practical. At minimum verify:

- primal feasibility;
- variable bounds;
- objective consistency;
- epigraph consistency;
- no materially negative rejection probabilities;
- no materially above-one rejection probabilities;
- probability coefficients below solver resolution are not silently dropped;
- any explicit refusal is deterministic and documented.

If the current HiGHS small-matrix policy makes a mathematically valid finite problem numerically uncertifiable, refuse explicitly rather than manufacture an answer.

Characterise the binary and ternary multinomial count thresholds at which this happens and test them directly.

### 5. Composite converse

Audit both Renyi directions and pair selection independently.

For every accepted simulation assert

    converse <= beta_star + justified_tolerance.

Search aggressively for violations near lambda=1+, moderate lambda and large lambda, including nearly identical laws and highly separated laws.

Never assume the pair minimising D_lambda(Q||P) also minimises D_lambda(P||Q).

### 6. Projected Renyi test

For every accepted result independently recompute

    sup_P E_P[phi]

and

    sup_Q E_Q[1-phi].

The returned achieved test must satisfy the Type I budget and its reported Type II error must equal the direct worst-case calculation.

Find stable examples where the projected test is strictly suboptimal relative to beta_star. Keep at least one as a regression test. The fact that a projected test is valid must never be upgraded into a claim of finite-sample minimax optimality.

Search for finite nonconvex classes where the stronger projected theorem moment inequalities fail. In those cases `projected_formula_certified` must be false.

### 7. Stationary flux-family bridge

Audit the complete path

    finite flux states -> EMU -> observation laws -> finite composite hypotheses.

Check:

- every state is independently validated;
- no MFA fitting is invoked;
- state order is preserved;
- experiment/target/replicate order is preserved;
- genuine count totals are preserved;
- multiple blocks require explicit independence;
- duplicate sample IDs do not collapse distinct states;
- provenance fingerprints differ when the underlying state differs;
- null and alternative provenance cannot be swapped or silently reused;
- structural zeros remain exact.

### 8. Simulation campaigns

Create or repair reproducible simulation scripts under `tools/composite_testing_validation/` or `benchmarks/composite_testing_validation/`.

Run at least:

- several hundred random categorical finite composite problems;
- at least 10,000 numerical stress problems if computationally practical;
- targeted full-support, structural-zero, near-zero and near-identical law cases;
- genuine multinomial count laws for increasing n;
- independent product blocks;
- finite grids of the affine ternary families already documented in the validation prompt.

For every applicable problem record and check

    converse <= beta_star <= projected achieved error.

Save compact machine-readable summaries and deterministic seeds. Do not commit huge raw datasets.

### 9. Documentation repair

Audit and correct:

- `README.md`
- `docs/COMPOSITE_TESTING.md`
- `docs/API_MAP.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/SCIENTIFIC_WORKFLOW.md`
- public docstrings

The documentation must clearly distinguish:

- mathematical exactness of the finite LP characterisation;
- floating-point numerical certification of a particular LP solution;
- an explicit finite list of laws;
- the complete continuous flux family;
- an achieved projected test;
- the unrestricted minimax optimum;
- a converse lower bound;
- genuine count observation semantics.

### 10. Required evidence report

Create or update

    results/composite_testing_validation/VALIDATION_REPORT.md

The report must state whether the original implementation failed, what was repaired, exact commit tested, test counts, simulation counts, all defect categories, worst projected-test gap, tightest/loosest converse gaps, explicit solver-refusal regimes, remaining continuous-family limitation, and anything still not validated.

## Commit discipline

For each defect category:

1. add the smallest failing regression test first;
2. commit the test where practical;
3. implement the smallest coherent fix;
4. run focused tests;
5. commit the fix;
6. periodically run the full suite.

Do not perform unrelated refactors.

## Final gates

Do not declare success unless:

- Python 3.11 full suite passes;
- Python 3.12 full suite passes;
- independent singleton controls recover NP;
- identical-hypothesis control gives 1-epsilon;
- all accepted LP solutions pass independent recomputation;
- no accepted simulation violates converse <= beta_star <= achieved projected error;
- at least one stable case demonstrates projected-test suboptimality;
- false moment certification is impossible under tested cases;
- stationary provenance tests pass;
- numerical refusal boundaries are explicit and documented;
- no continuous-family optimality claim is made;
- no ordinary MID/peak-area data are treated as genuine multinomial counts.

## Final response

Report:

- branch;
- final commit SHA;
- exact test counts on Python 3.11 and 3.12;
- simulation counts;
- all defect categories found and repaired;
- any defects from the prior report that could not be reproduced;
- worst projected-test optimality gap;
- tightest and loosest converse gaps;
- numerical refusal thresholds;
- files changed;
- unresolved mathematical and numerical limitations.

Do not merge to main. Do not claim the continuous flux-family problem is solved.