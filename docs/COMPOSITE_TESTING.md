# Finite composite binary testing

FluxEMU implements explicit finite composite-testing layers for **complete observable laws**. A genuine-count class member is an ordered product of multinomial MID blocks with explicitly declared independence. A parallel continuous implementation accepts externally corrected Dirichlet MID blocks. A finite H0/H1 family can be supplied directly or generated from complete feasible flux states through native stationary EMU.

The represented family is exactly the statistical class passed to the solver. FluxEMU does not silently convexify it, treat member frequency as a prior, or claim that a finite sample exhausts a larger biological mechanism class.

## Statistical problem

For a randomised decision rule `phi(y)` equal to the probability of deciding H1,

```text
alpha(phi; H0) = max_{P in H0} E_P[phi(Y)]
beta(phi; H1)  = max_{Q in H1} E_Q[1 - phi(Y)].
```

At Type-I budget `epsilon`, the represented finite minimax value is

```text
beta*(epsilon)
  = min_phi max_{Q in H1} E_Q[1-phi]
    subject to max_{P in H0} E_P[phi] <= epsilon.
```

The decision rule consumes observable data only. Flux coordinates, state IDs, sampling frequencies, inverse-MFA estimates, priors and class averages do not enter the statistic.

## Complete product observation laws

`IndependentMIDProductLaw` contains one or more ordered `MultinomialMIDLaw` blocks and their experiment/target/replicate identities. Corresponding H0/H1 members must have the same block order, count totals and mass-class spaces. Count totals may differ between blocks.

For explicitly independent blocks,

```text
D_lambda(Q || P)
  = sum_b D_lambda(Q_b || P_b).
```

This lets the composite layer operate directly on a joint panel of MID measurements without pretending the blocks are independent decisions.

The parallel `IndependentDirichletMIDProductLaw` applies the same explicit
product semantics to continuous blocks and conditionally independent
replicates. Corresponding laws must have identical ordered block identities,
replicate structures, mass classes and active simplex faces. Its directed
Rényi divergence adds across blocks and multiplies each block contribution by
its declared replicate count. A biological replicate count remains an
observation-design declaration and is never a multinomial total.

## Order-specific composite Rényi converse

`composite_renyi_converse_at_order(...)` accepts a supplied finite `lambda > 1` and computes

```text
D_lambda(H1 || H0)
  = min_{Q in H1, P in H0} D_lambda(Q || P)
```

for the **full product observation laws**. The order-specific pairwise composite lower bound is

```text
beta*(epsilon)
  >= max(
       0,
       1 - exp((lambda-1)/lambda * [log(epsilon) + D_lambda(H1||H0)])
     ).
```

This API uses the reverse direction `D_lambda(Q||P)`. Minimising the forward direction `D_lambda(P||Q)` is a different optimisation and may select another pair; a reverse minimiser cannot be reused as an asserted forward minimiser. No convexity, ordering or least-favourable-pair assumption is needed for the displayed reverse converse. The supplied order is used unchanged; FluxEMU does not replace a continuous-order optimisation by a finite grid.

## Finite minimax LP and numerical validation

`exact_finite_composite_minimax(...)` enumerates the Cartesian product of all MID-block count spaces and constructs the unrestricted randomised minimax linear programme:

```text
minimise t
subject to sum_y P(y) phi(y) <= epsilon   for every P in H0
           sum_y Q(y) phi(y) + t >= 1    for every Q in H1
           0 <= phi(y) <= 1,  0 <= t <= 1.
```

This LP is a mathematically exact characterisation of `beta_star` for the supplied finite classes and complete finite observation space. The implementation uses floating-point arithmetic, so a returned result is a numerically validated solution under its stated tolerances, not an exact-arithmetic certificate. HiGHS reporting an optimal status alone is insufficient.

This is a small-problem/discretised oracle, not the scalable path for a large multi-block panel. The joint space is protected by `max_outcomes`; exceeding the cap raises `CompositeEnumerationLimitError` and does not trigger a reduced-space, Monte Carlo or asymptotic substitute.

The Type-I constraints are scaled by `1/epsilon` before HiGHS optimisation. The numerical checks verify decision-variable bounds, all original constraints, the solver objective, the Type-II epigraph variable, and a dual lower bound. Worst-case errors are recomputed directly as `E_P[phi]` and `E_Q[1-phi]` with accurate summation. The reported objective, epigraph and recomputed Type-II error must agree, and the primal–dual gap must pass the declared tolerance. Decision probabilities are never repaired by clipping.

The result exposes `solver_objective`, `epigraph_variable`, `dual_lower_bound`, `optimality_gap`, `numerical_tolerance` and `minimum_nonzero_coefficient` so callers can inspect the numerical evidence. This is floating-point validation, not an interval-arithmetic or symbolic optimality proof.

The following limits cause explicit refusal:

- Budgets below `MIN_EXACT_COMPOSITE_EPSILON = 1e-12` are rejected, with no budget enlargement.
- HiGHS is configured with `small_matrix_value=1e-12`. Any nonzero coefficient of magnitude at or below that floor in the scaled LP causes refusal before solving, because HiGHS could otherwise discard it. Positive support is never replaced by zero.
- An unrepresentable positive probability, invalid solver output, inconsistent diagnostics or a material optimality gap causes refusal. An optimal solver status does not override these checks.

For a full-support binary count block, some category probability is at most `1/2`; for a full-support ternary block, some category probability is at most `1/3`. The corresponding extreme outcome has positive probability at most `2^-n` or `3^-n`. Since alternative-law coefficients are unscaled, this policy necessarily refuses full-support binary count laws at `n >= 40` and ternary laws at `n >= 26`. These are upper ceilings, not promises of acceptance below them: unbalanced probabilities and independent products can cross the coefficient floor earlier, and the Type-I budget also affects scaled null coefficients. Structural zeros are handled as exact zeros rather than as tiny positive coefficients. Measured boundary controls are recorded in [the validation report](../results/composite_testing_validation/VALIDATION_REPORT.md).

The LP requires the optional testing dependency:

```bash
python -m pip install '.[testing]'
```

This entire LP section applies only to genuine-count finite observation
spaces. A Dirichlet law is continuous. `exact_dirichlet_composite_minimax(...)`
therefore returns the explicit reason
`unsupported_for_continuous_observation_space`; it does not discretise the
simplex and call the result exact.

## Finite-family Rényi candidate score

For `0 < lambda < 1`, minimising Rényi divergence over a finite non-convex family selects a **vertex-pair candidate score**. It is not automatically a joint Rényi projection of convex classes and not automatically a finite-blocklength least-favourable pair.

`composite_renyi_score_candidate(...)` returns the minimum-divergence represented pair together with direct support and uniform-moment diagnostics. If `(P*,Q*)` is selected, its full product log-likelihood-ratio score is

```text
h(y) = log Q*(y) - log P*(y).
```

The candidate's `uniform_moment_bounds_verified` field is true only when the numerical moment checks over the complete represented families establish

```text
max_{P in H0} log E_P[exp(lambda h)]
  <= (lambda-1) D_lambda(Q*||P*)

max_{Q in H1} log E_Q[exp((lambda-1) h)]
  <= (lambda-1) D_lambda(Q*||P*).
```

The moment calculations factor over the independent multinomial blocks, so they do not require joint count-space enumeration. These are direct finite-family checks; the list is not assumed to be convex. The caller's candidate tie tolerance cannot relax the moment inequalities or certify a failed projected formula. The implementation checks the moments at higher precision and refuses unresolved near-equalities except for algebraically identical selected laws. This remains numerical evidence rather than a proof over an unspecified continuous class. A false `uniform_moment_bounds_verified` value prevents use of the analytical projected formula; it does not by itself prevent direct finite-space calibration of a well-defined score.

Structural zeros are retained. Infinite score coordinates are permitted when the uniform support conditions make them harmless. A category with `P*=Q*=0` may be ignored only if every represented class member also assigns zero mass there. Otherwise the candidate fails verification.

`verified_composite_renyi_score(...)` raises `CompositeScoreVerificationError` when the finite-family minimum is only pairwise and the uniform composite gates fail.

For Dirichlet products, the parallel
`composite_dirichlet_renyi_score_candidate(...)` evaluates every required
moment analytically with multivariate log-beta functions. If
`gamma+t(beta*-alpha*)` is not componentwise positive, that moment is
mathematically infinite and verification fails where required. No continuous
score is approximated as Gaussian. The verified wrapper is
`verified_composite_dirichlet_renyi_score(...)`.

## Analytical score bound

For a verified candidate and Type-I budget `epsilon`, `composite_score_bound_at_order(...)` returns the analytical threshold

```text
tau = [-log(epsilon) - (1-lambda) D_lambda(Q*||P*)] / lambda
```

and the projected-score Type-II exponential guarantee

```text
exp(-(1-lambda)/lambda * [D_lambda(Q*||P*) + log(epsilon)]).
```

These formulas describe the mathematical quantities. The implementation uses the directly evaluated log Hellinger integral, with conservative upward rounding of the common log-moment value, threshold and Type-II bound. It refuses an unrepresentable threshold or a positive bound that underflows. For verified disjoint supports, a finite separating threshold gives zero Type-II error.

The function also reports the separate constant-randomised-test guarantee `1-epsilon` and their minimum as a bound on the represented minimax value. It does not require enumeration of the joint count space. A failed uniform-moment certificate cannot be bypassed to obtain this analytical formula.

`evaluate_composite_score_test(...)` is the optional small-space oracle that enumerates the deterministic threshold rule and reports its actual worst-case Type-I and Type-II errors.

`composite_dirichlet_score_bound_at_order(...)` supplies the analogous
projected analytical bound after finite-family uniform moment verification.
Exact evaluation and calibration of that deterministic continuous score are
not implemented: a weighted sum of `log(Y_i)` under a Dirichlet law has no
implemented certified exact CDF. The corresponding functions return explicit
continuous-observation refusals rather than using a normal approximation,
Monte Carlo as a guarantee, or finite simplex discretisation.

## Calibration within the score family

`calibrate_composite_score_test(...)` keeps a candidate score fixed, enumerates the represented joint count space, orders outcomes by that score and randomises at the boundary required to exhaust the Type-I budget. It calibrates against the actual worst-case Type-I constraint over every supplied null law and evaluates the actual Type-II errors over every supplied alternative law.

The score must be defined on the represented support. Calibration may proceed when the stronger uniform projected-moment inequalities fail, because direct finite-space calibration does not use those inequalities. Such a result is an achieved test and does not change the candidate's failed analytical certificate. Undefined scores and unrepresentable positive masses still cause explicit refusal.

Calibration returns an achieved test and its worst-case errors over all supplied laws. It is optimal only within that fixed upper-score threshold family; numerical equality with an LP result is evidence only to their validation tolerances. In general,

```text
composite converse <= represented minimax beta*
    <= calibrated score-family achieved beta
    <= deterministic analytical-score achieved beta.
```

The final inequality applies when the deterministic analytical threshold is feasible at the same Type-I budget. Equality with the unrestricted minimax value is not assumed.

## Flux-state family bridge

`evaluate_stationary_composite_hypotheses(...)` accepts finite tuples of complete `CanonicalFluxState` records for H0 and H1. Every state passes the existing original-model feasibility and stationary-EMU validation before its count blocks are grouped into an `IndependentMIDProductLaw`.

`evaluate_stationary_dirichlet_composite_hypotheses(...)` is the separate
continuous bridge. It evaluates H0 and H1 jointly so that structural zeros
common to all represented states in one block define an explicit simplex face.
State-dependent active faces are refused. Each state is then represented by an
`IndependentDirichletMIDProductLaw` with precision, correction, replicate and
support provenance retained.

Multiple experiment/target/replicate blocks require explicit `independent_blocks=True`. State order, experiment/target/replicate order and genuine count totals are preserved. Provenance binds each result to its complete underlying state and hypothesis role. Duplicate sample IDs within one family are rejected; equal labels across H0 and H1 remain distinct through role-bound state IDs. The decision layer sees the generated observable laws, not the hidden flux state or its sampling frequency. No MFA fitting is invoked by this bridge.

This is the intended finite represented-class bridge for comparing mechanistically generated MID-law families. A sampled state family remains a discretisation of any larger continuous mechanism class unless a separate argument establishes otherwise.

## Current non-goals

The current finite engine does not implement:

- automatic convex-hull closure of finite flux families;
- continuous or implicitly parameterised mechanism classes;
- numerical optimisation over a continuum of nuisance/flux parameters;
- a claim that a vertex-pair Rényi minimum is a joint convex projection;
- a claim that a candidate score pair is finite-sample least favourable without separate ordering/optimality evidence;
- scalable exact minimax optimisation over enormous multi-block count spaces;
- biological-replicate random effects or culture-varying nuisance products;
- a generic composite p-value;
- test inversion into flux compatibility/confidence regions;
- Bayesian nuisance integration.

Those require separate statistical specifications and, for a continuous flux family, rigorous optimisation or bounds over its complete feasible set. Validation evidence for the count layer is recorded in [the count validation report](../results/composite_testing_validation/VALIDATION_REPORT.md). The continuous law, formulas, support rules and distinct validation campaign are documented in [DIRICHLET_MID_OBSERVATION.md](DIRICHLET_MID_OBSERVATION.md) and the [Dirichlet validation report](../results/dirichlet_mid_validation/VALIDATION_REPORT.md).
