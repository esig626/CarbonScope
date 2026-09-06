# Simple binary Rényi bounds and exact likelihood-ratio p-values

`fluxemu.testing` operates on two fixed complete observation laws,

\[
H_0:P_0=P(Y\mid v_0),\qquad H_1:P_1=P(Y\mid v_1),
\]

with the fixed convention

\[
\text{Type I}=P_0(\text{decide }H_1),\qquad
\text{Type II}=P_1(\text{decide }H_0).
\]

This layer is separate from MFA fitting. It does not turn percentages,
normalised MIDs, peak areas, or arbitrary intensities into counts. The
observation laws must come from explicit genuine-count specifications.

## Law-level Rényi quantities and Type-II lower bounds

`bruno_converse_at_order(...)` evaluates the finite-order Bruno,
Vandenbroucque & Esposito (2026) lower bound on the optimal Type-II error under
a declared Type-I budget `epsilon` and a supplied finite real Rényi order
`lambda > 1`.

At each supplied order it evaluates both directions

\[
D_\lambda(P_1\|P_0),\qquad D_\lambda(P_0\|P_1),
\]

using the existing full observation-law divergence implementation. The returned
`type_ii_lower_bound` is a lower bound on the best achievable Type-II error at
that order. It is not the error probability of a particular observed sample.

The Bruno theorem requires mutual absolute continuity of the complete laws.
FluxEMU checks matching positive support exactly before evaluating the bound.
It does not add pseudo-counts, clip probabilities, smooth structural zeros, or
repair support mismatches.

Supplying one or several orders does not establish the global continuous-order
optimum over every real `lambda > 1`. FluxEMU does not replace that continuum
with a finite order grid and does not label a sampled optimum as exact.

The current-facing result type is `BrunoOrderBound`.

## Realised-data likelihood ratio

For a realised genuine-count observation `y`, FluxEMU computes

\[
L(y)=\log\frac{P_1(y)}{P_0(y)}
\]

with `log_likelihood_ratio(...)`.

A positive value favours the fixed alternative relative to the fixed null; a
negative value favours the null. Exact support is retained. If only `P0(y)` is
zero the result is `+inf`; if only `P1(y)` is zero it is `-inf`; if both complete
laws assign zero probability the likelihood ratio is undefined and FluxEMU
raises an explicit error.

## Exact simple-null p-value

For the same fixed pair and realised sample,
`likelihood_ratio_p_value(...)` reports

\[
p(y)=P_0\!\left\{L(Y)\geq L(y)\right\}.
\]

This is the null probability of observing likelihood-ratio evidence in favour
of the declared alternative at least as strong as the evidence actually
observed. It is therefore both sample-specific and alternative-specific.

The p-value is not a divergence. It is also not the Type-II lower bound. Those
quantities answer different questions:

| Quantity | Question |
| --- | --- |
| `D_lambda(P1 || P0)`, `D_lambda(P0 || P1)` | How separated are the two complete observation laws? |
| `log_likelihood_ratio(...)` | Which fixed hypothesis does this realised data set favour, and by how much on the log-likelihood scale? |
| `likelihood_ratio_p_value(...)` | If the null were true, how often would evidence for this fixed alternative be at least this strong? |
| `type_ii_lower_bound` | At the declared Type-I budget, how small can the optimal Type-II error possibly be according to the finite-order Bruno bound? |

Because the multinomial sample space is discrete, the exact tail p-value is in
general conservative/super-uniform under the null rather than continuously
uniform.

### Exact enumeration boundary

The implementation enumerates every positive-probability null count outcome,
including Cartesian products of explicitly independent observation blocks.
The default maximum is

```python
DEFAULT_EXACT_P_VALUE_MAX_OUTCOMES = 1_000_000
```

and callers may supply another positive integer through `max_outcomes`.

If the null sample space is larger than that limit,
`ExactPValueEnumerationLimitError` is raised. FluxEMU does **not** silently
switch to a chi-square approximation, Monte Carlo p-value, saddlepoint
approximation, or another inferential procedure.

## Example

For the fixed mixer example

\[
p_0=(0.25,0.75),\qquad p_1=(0.5,0.5),\qquad n=4,
\]

and realised counts

\[
y=(3,1),
\]

FluxEMU gives approximately

\[
L(y)=1.6739764,
\]

and the exact simple-null likelihood-ratio p-value is

\[
p(y)=0.05078125.
\]

At Rényi order `lambda=2`, the complete multinomial laws have

\[
D_2(P_1\|P_0)\approx1.15072829,
\qquad
D_2(P_0\|P_1)\approx0.89257421.
\]

For `epsilon=0.05`, the Bruno Type-II lower bound at that order is
approximately `0.602476804`.

These values are deliberately reported together because they describe distinct
parts of the same testing problem: law separation, realised evidence, null-tail
surprisingness, and finite-sample error limitations.

## Relationship to KL deviance

For a fully specified multinomial null `p0`, observed counts `k`, total `n`, and
empirical proportions `p_hat=k/n`, the likelihood-ratio deviance against the
saturated multinomial model satisfies the exact identity

\[
G^2=2nD_{\mathrm{KL}}(\widehat p\|p_0).
\]

Under the usual regular large-sample conditions, this deviance has a chi-square
limit. That classical result explains one direct route from KL divergence to a
conventional asymptotic p-value. FluxEMU's simple-vs-simple exact p-value above
is different: it uses the finite null distribution of `log(P1/P0)` directly and
does not invoke the saturated model or Wilks' approximation.

See
[`technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex`](technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex)
for the detailed derivation and interpretation.

## Public API

```python
from fluxemu.testing import (
    BrunoOrderBound,
    ExactPValueEnumerationLimitError,
    LikelihoodRatioPValue,
    SimpleBinaryLawPair,
    bruno_converse_at_order,
    likelihood_ratio_p_value,
    log_likelihood_ratio,
)
```

Composite hypotheses, test inversion, flux regions, Bayesian inference and
p-values for composite nulls remain outside this layer.

## Reference

Roberto Bruno, Adrien Vandenbroucque, and Amedeo Roberto Esposito,
*A Finite-Sample Strong Converse for Binary Hypothesis Testing via (Reverse)
Rényi Divergence*, arXiv:2601.09550v2 (2026).
