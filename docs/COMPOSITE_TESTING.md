# Finite composite binary testing

FluxEMU implements an explicit finite composite-testing layer for **complete observable laws**. A class member is an ordered product of genuine-count multinomial MID blocks with explicitly declared independence. A finite H0/H1 family can be supplied directly or generated from complete feasible flux states through native stationary EMU.

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

No convexity, ordering or least-favourable-pair assumption is needed for this converse. The supplied order is used unchanged; FluxEMU does not replace a continuous-order optimisation by a finite grid.

## Exact represented finite minimax oracle

`exact_finite_composite_minimax(...)` enumerates the Cartesian product of all MID-block count spaces and solves the complete randomised minimax problem as a linear programme. There is one decision variable in `[0,1]` for every joint count outcome, one Type-I constraint for every null member and one Type-II worst-case constraint for every alternative member.

This is a small-problem/discretised oracle, not the scalable path for a large multi-block panel. The joint space is protected by `max_outcomes`; exceeding the cap raises `CompositeEnumerationLimitError` and does not trigger a reduced-space, Monte Carlo or asymptotic substitute.

The Type-I constraints are scaled before HiGHS optimisation and worst-case errors are recomputed afterwards with accurate summation. Budgets below `MIN_EXACT_COMPOSITE_EPSILON = 1e-12` are rejected as an explicit LP numerical-limit condition rather than silently enlarged.

The LP requires the optional testing dependency:

```bash
python -m pip install '.[testing]'
```

## Finite-family Rényi candidate score

For `0 < lambda < 1`, minimising Rényi divergence over a finite non-convex family selects a **vertex-pair candidate score**. It is not automatically a joint Rényi projection of convex classes and not automatically a finite-blocklength least-favourable pair.

`composite_renyi_score_candidate(...)` returns the minimum-divergence represented pair together with direct support and uniform-moment diagnostics. If `(P*,Q*)` is selected, its full product log-likelihood-ratio score is

```text
h(y) = log Q*(y) - log P*(y).
```

The candidate is marked uniformly verified only when the complete represented families satisfy

```text
max_{P in H0} log E_P[exp(lambda h)]
  <= (lambda-1) D_lambda(Q*||P*)

max_{Q in H1} log E_Q[exp((lambda-1) h)]
  <= (lambda-1) D_lambda(Q*||P*).
```

The moment calculations factor over the independent multinomial blocks, so they do not require joint count-space enumeration.

Structural zeros are retained. Infinite score coordinates are permitted when the uniform support conditions make them harmless. A category with `P*=Q*=0` may be ignored only if every represented class member also assigns zero mass there. Otherwise the candidate fails verification.

`verified_composite_renyi_score(...)` raises `CompositeScoreVerificationError` when the finite-family minimum is only pairwise and the uniform composite gates fail.

## Analytical score bound

For a verified candidate and Type-I budget `epsilon`, `composite_score_bound_at_order(...)` returns the analytical threshold

```text
tau = [-log(epsilon) - (1-lambda) D_lambda(Q*||P*)] / lambda
```

and the projected-score Type-II exponential guarantee

```text
exp(-(1-lambda)/lambda * [D_lambda(Q*||P*) + log(epsilon)]).
```

The function also reports the separate constant-randomised-test guarantee `1-epsilon` and their minimum as a bound on the represented minimax value. It does not require enumeration of the joint count space.

`evaluate_composite_score_test(...)` is the optional small-space oracle that enumerates the deterministic threshold rule and reports its actual worst-case Type-I and Type-II errors.

## Calibration within the score family

`calibrate_composite_score_test(...)` keeps the verified score fixed, enumerates the represented joint count space, orders outcomes by that score and randomises at the boundary required to exhaust the Type-I budget.

This is an optimum only within that fixed upper-score threshold family. In general,

```text
exact represented minimax beta*
    <= calibrated score-family beta
    <= deterministic analytical-score beta.
```

Equality is not assumed.

## Flux-state family bridge

`evaluate_stationary_composite_hypotheses(...)` accepts finite tuples of complete `CanonicalFluxState` records for H0 and H1. Every state passes the existing original-model feasibility and stationary-EMU validation before its count blocks are grouped into an `IndependentMIDProductLaw`.

Multiple experiment/target/replicate blocks require explicit `independent_blocks=True`. The state/member IDs are retained only for provenance. The decision layer sees the generated observable laws, not the hidden flux state or its sampling frequency.

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

Those require separate statistical specifications. The finite product-law layer is the validated core/oracle on which those later mechanisms can be built.
