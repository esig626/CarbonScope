# Composite testing in CarbonScope

CarbonScope separates two different objects that should not be conflated.

## From finite flux families to observation law classes

`evaluate_stationary_finite_composite_hypotheses(...)` takes two explicitly supplied tuples of complete feasible flux states, maps every state through the existing stationary EMU and genuine count observation layer, and returns aligned finite null and alternative law classes.

This is an integration bridge, not a mathematical sleight of hand. If the supplied states came from a finite sampler, the resulting law classes remain finite numerical approximations to the underlying continuous flux families.

## 1. A principled explicit test from Rényi projection

For declared null and alternative law classes

```text
H0: P in C0
H1: Q in C1
```

with uniform Type I constraint

```text
sup_P P(decide H1) <= epsilon,
```

`projected_renyi_test(...)` selects, at one supplied order `0 < lambda < 1`, the pair in the explicitly supplied finite classes minimising

```text
D_lambda(Q || P).
```

That pair defines the score

```text
h(y) = log(Q*(y) / P*(y)).
```

CarbonScope then does two things.

First, it evaluates the two exponential moments of this score over every supplied null and alternative law. This gives a direct finite sample uniform score guarantee without pretending that a finite list is automatically a convex class.

Second, because the observation space is finite, CarbonScope calibrates the threshold and boundary randomisation directly against the uniform Type I constraint. The resulting threshold test is an achieved test, so its worst Type II error is an upper bound on the unrestricted optimum.

The result also reports whether the stronger projected Rényi formula is certified by the supplied finite classes. For genuinely convex classes, a joint Rényi projection can guarantee these uniform moment inequalities under the theorem assumptions. A finite list of sampled flux states is not silently treated as such a convex class.

`composite_renyi_converse_at_order(...)` provides the complementary order specific lower bound. It applies the pairwise Rényi converse over all declared null and alternative law pairs and retains the strongest resulting class lower bound. The current complete observation laws already include their genuine count totals, so no additional sample size multiplier is inserted.

Together these give

```text
Rényi converse lower bound <= beta* <= achieved projected test error.
```

The gap is informative. A small gap means the explicit projected test is close to the best possible test. A large gap means the theory has not localised finite sample performance tightly enough.

## 2. The actual finite minimax optimum

For an explicitly finite class and an enumerable finite observation space, the unrestricted minimax problem can be solved directly.

Let `phi(y)` be the probability of deciding `H1` after outcome `y`. CarbonScope solves

```text
minimise t

subject to
    sum_y P(y) phi(y) <= epsilon       for every P in C0
    sum_y Q(y) phi(y) + t >= 1         for every Q in C1
    0 <= phi(y) <= 1
    0 <= t <= 1
```

through `solve_finite_minimax_test(...)`.

The optimum is

```text
beta*(epsilon; C0, C1)
```

for the explicitly supplied finite law classes. This optimisation is over all randomised tests on the enumerated observation space. It is not restricted to likelihood ratio tests, generalised likelihood ratio tests, projected scores or any other chosen statistic.

The LP characterisation is exact. The numerical solution uses HiGHS and is independently checked against explicit feasibility tolerances. CarbonScope fails rather than dropping positive support below the declared solver resolution.

For a singleton null and singleton alternative, this finite LP reduces numerically to the randomised Neyman Pearson optimum. For genuinely composite finite classes it solves the minimax problem directly, so no composite analogue of the Neyman Pearson lemma is assumed.

## Why both are useful

The minimax LP answers the finite problem exactly when the problem is small enough to enumerate. It is therefore a benchmark and, for small experiments, the actual optimal test.

The Rényi construction connects directly to finite sample achievability, converse bounds, error exponents and experimental design. It gives a single interpretable statistic rather than an arbitrary table of optimal decision probabilities.

The two should be used together when possible:

```text
explicit projected test
        |
        v
achievable worst Type II error
        |
        v
compare with exact minimax beta* and Rényi converse
```

This reveals whether the proposed Rényi test is merely valid or actually close to optimal.

## Critical boundary: a sampled flux ensemble is not the full hypothesis

If CarbonScope samples 100 or 1000 feasible flux states and converts them to observation laws, `solve_finite_minimax_test(...)` gives the exact minimax optimum for those supplied laws only.

It does **not** prove that the same value is the optimum over the complete continuous feasible flux region.

Likewise, minimising Rényi divergence over the sampled laws does not certify the joint projection over the full biological hypothesis family.

To make a statement about the full continuous family, the corresponding worst case or projection optimisation over that family must itself be solved or rigorously bounded. Sampling is useful for discovery and numerical approximation, not for quietly manufacturing a theorem.

## Current observation semantics and the meaning of sample size

The current implementation operates on `MultinomialMIDLaw` blocks with genuine fixed total count semantics. Multiple blocks require an explicit independence declaration. Percentages, peak areas, normalised MIDs and arbitrary intensities are not converted into pseudo counts.

A law `MultinomialMIDLaw(n, p)` is already the complete distribution of a count vector with total `n`. Consequently, its Rényi divergence already contains the factor `n`. The composite routines operate on that complete law and do not multiply by `n` again.

This is equivalent to using the count vector as the sufficient statistic for `n` genuinely i.i.d. categorical observations. It is **not** permission to reinterpret the number of biological replicates, a normalised MID, or an instrument intensity as a multinomial count total. Those require an observation law matching their actual measurement semantics.

Exact structural zeros are retained. No pseudocounts, clipping, hidden normalisation or inferred effective sample sizes are introduced.

## Public API

```python
from fluxemu.testing import (
    FiniteObservationLaw,
    FiniteCompositeHypotheses,
    evaluate_stationary_finite_composite_hypotheses,
    projected_renyi_test,
    composite_renyi_converse_at_order,
    solve_finite_minimax_test,
)
```

The current implementation is deliberately finite class first. Extending certification from sampled law families to full continuous flux families is a separate optimisation problem and should remain explicit in the software design.
