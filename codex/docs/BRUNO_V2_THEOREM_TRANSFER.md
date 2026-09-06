# Bruno v2 theorem-transfer audit

Source: Roberto Bruno, Adrien Vandenbroucque, and Amedeo Roberto Esposito,
*A Finite-Sample Strong Converse for Binary Hypothesis Testing via (Reverse)
Rényi Divergence* (2026), **arXiv:2601.09550v2**.
[Versioned primary source](https://arxiv.org/html/2601.09550v2).
Audited against its HTML on 2026-09-06 before implementation.

## Published definitions and equation mapping

The paper uses natural logarithms (Section II, immediately after Eq. (3)).
Its test is deterministic: Eq. (1) has `phi` taking values in `{0, 1}`.
The set `A = {phi = 1}` accepts the alternative.

| Source location | Required interpretation |
| --- | --- |
| Section II, Eq. (1) and ensuing error definitions | `H0 = P0`, `H1 = P1`; Type I is `P0(A)`, Type II is `P1(A^c)`. |
| Eq. (2) | `beta_n(epsilon)` is the infimum of Type II over deterministic tests satisfying Type I `<= epsilon`. |
| Definition 1, Eq. (4) | `D_lambda(P || Q) = log(integral p^lambda q^(1-lambda))/(lambda-1)`. |
| Theorem 1, Eq. (5) | Assumes mutually absolutely continuous complete observation laws; optimizes over all real `lambda > 1`. |
| Appendix A, Eqs. (14)–(17) | Reverse divergence is `D_lambda(P1 || P0)`; full-law reverse bound precedes tensorization. |
| Appendix A, unnumbered equation after Eq. (17) | Forward divergence is `D_lambda(P0 || P1)`; full-law forward bound. |
| Appendix A, final displayed equation | Only here are i.i.d. full-law divergences replaced by `n` times single-observation divergences. |

## Exact full-law transfer

Here `P0` and `P1` denote the **complete observation laws**; the paper writes
these as `P0^n` and `P1^n`. Let

```text
Drev = D_lambda(P1 || P0)
Dfwd = D_lambda(P0 || P1)
0 < epsilon < 1, finite real lambda > 1

B_reverse = 1 - exp(((lambda - 1) / lambda) * (log(epsilon) + Drev))
B_forward = exp((lambda / (lambda - 1)) * log(1 - epsilon) - Dfwd)

beta*(epsilon) >= max(B_reverse, B_forward)
```

The continuous-order envelope corresponding to Eq. (5) is

```text
beta*(epsilon) >= max {
    1 - inf_{lambda > 1} exp(((lambda - 1) / lambda)
                             * (log(epsilon) + Drev(lambda))),
    sup_{lambda > 1} exp((lambda / (lambda - 1))
                         * log(1 - epsilon) - Dfwd(lambda))
}.
```

No exponent or divergence direction in the task differs from v2.

## Implementation consequences

Apply Theorem 1 only after validating mutual absolute continuity. Enforce the
prompt's exact corresponding-block positive-support requirement; retain
structural zeros without pseudocounts, clipping, or support repair.

Reuse the existing observation-law identity
`D_lambda(Mult(n,p) || Mult(n,q)) = n D_lambda(p || q)` and existing independent
product additivity. Each block keeps its declared genuine count total and
identity. The returned full-law divergences already include these totals:
do not multiply by another `n` or fabricate one for unequal-depth blocks.

Each finite admissible order supplies a certificate. A sampled list or local
numerical search does not certify the full continuous envelope. Retain the raw
reverse component even when negative; it is then vacuous, not an invalid
probability estimate. The certificate is a lower bound on optimal Type II,
not the actual error of a selected test or realised observation.

## Deterministic finite-support oracle caveat

Section II introduces a deterministic LLRT in Eq. (3), rejecting when
`log(P1/P0) >= tau`, and broadly states LLRT optimality for Eq. (2). It does
not introduce randomized boundary decisions there.

For this repository's finite atomic oracle, evaluate **every deterministic
rejection subset** `A`, keep those satisfying `P0(A) <= epsilon`, and minimize
`P1(A^c)`. Whole-atom likelihood-ratio thresholds are not a general replacement
for that constrained enumeration: discrete budgets need not be attainable by
a threshold, and the deterministic feasible set must remain explicit. Do not
interpolate boundary atoms or silently compare with a randomized optimum.

This implementation caveat does not change the converse: Appendix A's
data-processing step applies to any deterministic rejection region, and a
minimum exists over the oracle's finite collection of subsets. LLRT
achievability (Theorem 2, Eq. (8), orders in `(0,1)`) is outside this task.
