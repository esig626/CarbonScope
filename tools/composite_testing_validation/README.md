# Independent composite validation

Run from the repository root with the testing dependencies installed:

```bash
PYTHONPATH=src:. python -m tools.composite_testing_validation.run_campaign --label repaired --random 400 --stress 10000
```

The committed baseline summary audits source exported from main commit
`4f457d2afd75ad1bdffdbe0e9eea25fcaf8a9ad4`, placed first on `PYTHONPATH`.
For example, export that commit's `src/` directory into a separate temporary
directory, then run the same command with `PYTHONPATH=/temporary/export/src:.`
and `--label baseline`. No code from the historical repair branch is used.

`oracle.py` implements three references independently of the production solver
and enumeration helpers:

- Integer multinomial coefficients and 80-digit Decimal probability products
  enumerate the complete observation space without hidden normalisation.
- Singletons use the randomised Neyman–Pearson fractional-knapsack solution.
- Tiny composite problems enumerate every active-set vertex using Decimal
  Gaussian elimination. The first 40 random cases also cross-check vertices
  against the independent dual LP.
- Larger composite problems solve the separately derived dual linear programme
  and recompute a feasible dual lower bound using the original Decimal masses.
  The optimiser's alternative weights are explicitly rescaled into a simplex;
  the dual slack is constructed as the positive part of the dual residual.
  These operations define feasible **dual variables**; they do not alter any
  observable probability or rejection rule.

An accepted production test supplies a directly recomputed feasible upper bound.
Agreement with the independent dual lower bound certifies optimality up to the
declared tolerance. Tiny cases fall back to exhaustive vertices when the dual
solver refuses or its bound is insufficiently tight. Any accepted production
result without an adequate independent certificate is a campaign failure.
The comparison tolerance is `5e-10` for Decimal singleton/vertex references and
`2e-9` for independent floating-point dual certificates and central inequalities.
Direct Type-I feasibility is checked at a relative `5e-9` tolerance; this does
not enlarge the budget used by the optimiser. The oracle tests additionally
compare all returned errors at absolute `2e-12` tolerance.

The main campaign uses seed `20260909`, 400 random finite categorical classes,
10,000 numerical stress problems, and 121 targeted cases. It includes sparse
support, positive probabilities down to approximately `1e-300`, nearly equal
laws, small budgets, supplied Rényi orders near one and as large as 1000,
binary counts `n=1..64`, ternary counts `n=1..40`, and independent product blocks.
The count sweep's acceptance pattern is specific to its stated probabilities,
epsilon and numerical policy; it is not a universal cutoff in the count total.

No historical affine ternary parameterisation was present in the available
repository, issue or repair specification. The reproducible replacement control
is explicitly `p(t)=(0.6-0.2t, 0.3-0.1t, 0.1+0.3t)`, with finite H0 grid
`t=(0, 0.25, 0.5)` and H1 grid `t=(0.6, 0.8, 1)`, at count totals
`n=(1, 2, 3, 5, 8)`. This does not validate a continuous affine family.

The compact JSON summaries record attempted and accepted cases, explicit
refusals, independent certification counts, central-invariant comparisons,
numerical gaps, small reproductions for failures, and the complete count-sweep
acceptance pattern. An expected explicit refusal is distinguished from an
unexpected programming exception, which is recorded as a failure.

`tests/test_composite_validation_oracle.py` supplies the mandatory deterministic
controls: singleton Neyman–Pearson, identical hypotheses, disjoint and partial
support, duplicate/permuted classes, budget monotonicity, uncertainty-class
enlargement, direct error recomputation, and objective/epigraph certificates.
The supplemental `converse_stress.py` has a separate deterministic seed and
summary; its 8,000 cases are not counted as part of the main 10,000 stress cases.
