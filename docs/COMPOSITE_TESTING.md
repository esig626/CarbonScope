# Finite composite binary testing

FluxEMU implements a finite, explicit composite binary testing problem for genuine isotopologue counts. The implementation is deliberately narrower than a general uncertainty-class framework: the null and alternative are declared finite families of fixed categorical MID laws, or finite families of complete feasible flux states that map to those laws.

No finite family is silently replaced by its convex hull and no continuum of mechanisms is inferred.

## Statistical problem

Let

```text
H0: p is one of P = {p_1, ..., p_J}
H1: q is one of Q = {q_1, ..., q_K}
```

where every member uses the same ordered mass classes and the same genuine count total `n`. The observation is

```text
Y ~ Multinomial(n, p)   under H0
Y ~ Multinomial(n, q)   under H1.
```

A randomised test `phi(y)` is the probability of deciding H1. FluxEMU uses the minimax errors

```text
alpha(phi; P) = max_j E_{P_j}[phi(Y)]
beta(phi; Q)  = max_k E_{Q_k}[1 - phi(Y)].
```

For a Type-I budget `epsilon`, the exact finite-family minimax value is

```text
beta*(epsilon; P, Q)
  = min_phi max_k E_{Q_k}[1 - phi(Y)]
    subject to max_j E_{P_j}[phi(Y)] <= epsilon.
```

The roles remain fixed: H0 is null, H1 is alternative, Type I means deciding H1 under H0, and Type II means deciding H0 under H1.

## Finite-family Rényi converse

For a supplied finite real order `lambda > 1`, FluxEMU computes the directed composite single-draw separation

```text
D_lambda(Q || P)
  = min_{q in Q, p in P} D_lambda(q || p).
```

Writing

```text
r = -log(epsilon) / n,
```

the implemented order-specific lower bound is

```text
beta*(epsilon; P, Q)
  >= 1 - exp(
       -n * (lambda - 1)/lambda
          * [r - D_lambda(Q || P)]_+
     ).
```

Use `composite_renyi_converse_at_order(...)`.

This calculation requires no convexity, projection, ordering or least-favourable-pair assumption. It does not search over Rényi orders and does not claim to evaluate the continuous-order envelope.

## Exact finite-sample minimax test

`exact_finite_composite_minimax(...)` enumerates the complete count space and solves the randomised minimax problem as a linear programme. There is one decision variable `phi(y) in [0,1]` for every count outcome, one Type-I constraint for every null member, and one Type-II worst-case constraint for every alternative member.

This is an exact optimisation of the represented finite probability model up to numerical LP precision. It is not a deterministic-only approximation.

Enumeration has an explicit `max_outcomes` cap. If the complete count space is larger, FluxEMU raises `CompositeEnumerationLimitError`. It does not switch to Monte Carlo, asymptotics or a reduced outcome set.

The LP requires SciPy and is available through

```bash
python -m pip install '.[testing]'
```

## Verified order-below-one projected test

For `0 < lambda < 1`, a pair that minimises Rényi divergence over an arbitrary finite family does **not** automatically define a valid composite test. In particular, pairwise optimality alone does not guarantee uniform error control.

`verified_composite_renyi_projection(...)` therefore performs two separate steps:

1. find a Rényi-minimising pair among the explicitly declared members;
2. directly verify the two uniform single-draw moment inequalities over every null and alternative member.

For a selected full-support pair `(p*, q*)`, set

```text
h(x) = log(q*(x) / p*(x))
z     = sum_x q*(x)^lambda p*(x)^(1-lambda).
```

The pair is accepted for composite projected testing only if

```text
max_{p in P} E_p[exp(lambda h)]         <= z
max_{q in Q} E_q[exp((lambda-1) h)]     <= z
```

within the explicit numerical tolerance. If either inequality fails, FluxEMU raises `CompositeProjectionError` rather than presenting a pairwise likelihood ratio as a composite procedure.

The current projected path requires full support for every declared family member. Structural zeros remain supported by the exact minimax and converse paths.

## Closed-form projected bound

For a verified pair, define the categorical projected divergence

```text
D* = D_lambda(q* || p*)
```

and the threshold

```text
tau_min = n * [r - (1-lambda) D*] / lambda.
```

The deterministic upper-threshold test based on the count-weighted score is evaluated by `projected_composite_bound_at_order(...)`. The returned object reports separately:

- the analytical Type-II upper bound;
- the actual worst-case Type-I error over the declared finite null family;
- the actual worst-case Type-II error over the declared finite alternative family.

The global analytical upper bound is the smaller of the projected exponential expression and the constant randomised-test value `1 - epsilon`.

## Calibration within the projected score family

`calibrate_composite_projected_test(...)` keeps the verified score fixed and calibrates a randomised upper-threshold rule directly against the finite composite Type-I constraint. It enumerates all count outcomes, groups equal score values exactly at the represented log-ratio precision, and randomises at the first boundary needed to exhaust the Type-I budget.

This gives the restricted optimum within that ordered threshold family. In general,

```text
exact minimax beta*
    <= calibrated projected beta
    <= deterministic projected beta.
```

Equality is not assumed.

## No automatic least-favourable-pair claim

A Rényi-minimising pair, even one satisfying the two uniform moment inequalities, is not automatically a finite-blocklength least-favourable pair. The public projection record therefore explicitly reports that no finite-`n` least-favourable claim is being made.

Exact reduction to a simple pair requires an additional ordering/optimality result. FluxEMU does not infer such a result from numerical pair minimisation.

## Flux-state families

`evaluate_stationary_composite_hypotheses(...)` accepts explicit finite tuples of complete `CanonicalFluxState` records for H0 and H1. Every state is validated by the existing original-model feasibility layer and mapped through native stationary EMU to a genuine-count law.

The bridge currently requires exactly one experiment/target/replicate count block. This is intentional: the implemented finite composite theory is the common i.i.d. categorical problem. FluxEMU does not infer a composite theorem for non-identical independent product blocks from the simple-testing product API.

## Current non-goals

The current release does not implement:

- automatic convex-hull closure of finite flux families;
- continuous or implicitly parameterised composite classes;
- numerical optimisation over a continuum of flux mechanisms to locate a joint Rényi projection;
- a claim that a projected pair is finite-sample least favourable without separate ordering evidence;
- general multi-block composite testing with unequal genuine count totals;
- a generic composite p-value;
- test inversion into flux compatibility or confidence regions;
- Bayesian nuisance integration or random-effects models.

These are separate statistical specifications, not UI options to be guessed by software.
