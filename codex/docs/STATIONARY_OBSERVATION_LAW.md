# Stationary observation laws for genuine isotopologue counts

The `fluxemu.observation` package supplies an explicit fixed-total observation
law on top of native stationary EMU:

```text
complete feasible flux state v -> stationary EMU MID p_v -> Multinomial(n, p_v)
```

Use this law only when the measurement actually records counts in mutually
exclusive isotopologue classes, with a declared fixed total and multinomial
sampling assumptions. It is a baseline count law, not a universal or default
mass-spectrometry noise model. Numeric integer validation cannot establish the
physical meaning of a measurement; the caller must establish count semantics.

Percentages, peak areas, normalised MIDs, arbitrary intensities, and
mfapy-style noisy MDVs are not automatically counts. Do not multiply a MID by
an arbitrary total, round percentages, cast intensities, or infer an effective
sample size from standard deviations. `normalise_mid(...)` explicitly creates
a probability composition and discards the original total. Its output cannot
manufacture `n`. Retain genuinely observed raw counts and their total
separately; an empirical MID is a derived view of those counts.

The [measurement-semantics audit](MFAPY_OBSERVATION_SEMANTICS_AUDIT.md) records
the inspected vendored source. Its Gaussian perturbation, negative-value
clipping, optional closure, replicate averaging, and weighted RSS do not
define the count law implemented here. Vendor source remains read-only and
is never an observation-layer runtime dependency.

## Installation and direct law API

The ordinary base installation is sufficient:

```bash
python -m pip install ./codex
```

No new dependencies or MFA optimizer are required. Importing and executing
the direct law and stationary bridge work without SciPy, COBRApy, optlang,
mfapy, or matplotlib. Future observation laws can use this separate package
boundary without modifying the EMU engine or the existing MFA optimizer.

| Public API in `fluxemu.observation` | Contract |
| --- | --- |
| `MIDCountObservation(counts, n)` | Immutable raw integer counts and explicit positive total; `sum(counts) == n`. |
| `MultinomialMIDLaw(n, probabilities)` | Immutable fixed-total law with the predicted MID in declared M+0, M+1, … order. |
| `law.log_pmf(counts)` / `law.log_prob(counts)` | Log PMF of a count vector or `MIDCountObservation`, using natural logarithms. |
| `law.sample(seed=...)` or `law.sample(rng=...)` | One raw count observation; exactly one seed or NumPy `Generator` is required. |
| `kl_multinomial(p_law, q_law)` | Directed law-level KL, using the existing MID divergence kernel. |
| `renyi_multinomial(p_law, q_law, alpha)` | Directed law-level Rényi for every finite real `alpha > 0`, with exact KL dispatch at one. |
| `independent_product_kl(p_laws, q_laws)` | Sum of corresponding ordered block KL divergences under declared independence. |
| `independent_product_renyi(p_laws, q_laws, alpha)` | The corresponding independent-product Rényi divergence. |
| `multinomial_count_constant(observation)` | Data-only constant in the count-likelihood/KL identity below. |

For a hypothetical raw count measurement with three declared mass classes:

```python
import math
from fluxemu.observation import MIDCountObservation, MultinomialMIDLaw

observed = MIDCountObservation(counts=(12, 8, 0), n=20)
law = MultinomialMIDLaw(n=20, probabilities=(0.5, 0.5, 0.0))
log_likelihood = law.log_pmf(observed)
assert math.isfinite(log_likelihood)
assert observed.counts == (12, 8, 0)
assert observed.empirical_mid == (0.6, 0.4, 0.0)
assert observed.mass_classes == law.mass_classes == (0, 1, 2)
assert law.log_pmf((12, 0, 8)) == -math.inf

synthetic = law.sample(seed=626)
assert synthetic == law.sample(seed=626)
assert sum(synthetic.counts) == synthetic.n == 20
assert synthetic.counts[2] == 0
```

Counts and totals must be integers, including supported NumPy integer
scalars. Floats are rejected even when integral, such as `12.0`; booleans,
negative counts, nonfinite values, and dimension mismatches also fail. Count
records require a nonempty immutable tuple and a positive matching total, so
an all-zero vector cannot be a measurement. A well-formed count vector passed
directly to `log_pmf` with a different total, including an all-zero vector,
lies outside the law's support and returns exactly `-inf`.

## Exact support and numerical boundaries

For `k_i >= 0` integer and `sum(k) = n`, the mathematical PMF is

\[
P_p(K=k)=\frac{n!}{\prod_i k_i!}\prod_i p_i^{k_i}.
\]

A positive count at `p_i == 0` gives probability zero and log PMF `-inf`.
A zero count contributes no logarithm, including when its predicted
probability is zero. No pseudocounts, epsilon floors, clipping, support filling,
or hidden normalisation occur.

Probability inputs reuse the existing strict MID validator and must satisfy
the machine-simplex boundary of 16 ulps at one (about `3.55e-15`). Their values
and order are retained. A source probability outside `[0, 1]`, or a positive
source value that would become zero on conversion to float, is rejected.

Totals are explicitly supported from `1` through `2**63 - 1`, matching the
NumPy sampler's signed-int64 output range. A second numerical guard prevents
a large total from amplifying a small represented simplex residual:

\[
\left|n\log\left(\sum_i p_i\right)\right|\leq 10^{-9},
\]

where the sum in this guard is evaluated using the exact binary floating-point
values in higher precision. This bounds the represented PMF's log total-mass
error. Inputs that fail are rejected explicitly; neither probabilities nor
the total are changed. At very large totals, even ordinary non-dyadic decimal
inputs can fail this guard. The identities below are exact mathematical
identities on probability laws, evaluated subject to these explicit numerical
representation limits.

Log PMFs and count constants use 60-digit Decimal arithmetic to retain
cancellation accuracy for large counts. Small log factorials use bounded
exact factorials; larger ones use the log-factorial Stirling expansion through
the inverse fifteenth power, whose remainder is below `4e-27` on that branch.
Large factorials and probability products are never formed in production.

Sampling uses only positive support, placing the largest positive probability
in NumPy's residual category internally and restoring the original order in
the returned counts. This prevents the residual category from assigning
counts to an exact-zero class. The law and stored MID remain unchanged.
A caller-supplied `numpy.random.Generator` advances; a seed creates a local
generator without touching global RNG state. Fixed-seed replay is scoped to
the same algorithm and NumPy environment, not guaranteed across versions.

## Native stationary bridge and provenance

| Public record or function | Contract |
| --- | --- |
| `StationaryCountSpecification(target_id, total_count, replicate_id="0")` | An explicit count total for one declared target and replicate. |
| `StationaryObservationExperiment(experiment_id, experiment, specifications)` | One native stationary experiment with ordered count specifications. |
| `StationaryObservationSpecification(model, experiments)` | One canonical model and ordered experiment blocks. |
| `validate_stationary_observation_specification(specification)` | Reuses native model/experiment validation and validates declared identities. |
| `evaluate_stationary_observation_laws(specification, states)` | Validates a tuple of complete canonical states and returns native predicted MIDs with their count laws. |
| `sample_stationary_observations(result, seed=...)` / `rng=...` | Returns ordered `StationaryCountSample` records pairing raw observations with source components. |
| `StationaryObservationLawResult` / `StationaryObservationLawComponent` | The complete states, native validation, ordered laws, scientific identities, and deterministic provenance. |
| `stationary_observation_specification_fingerprint(specification)` | Binds the model, ordered experiment science, identities, law type, and explicit totals. |

The following runnable example uses the existing synthetic stationary fixture
from the repository root. It evaluates its complete known state and never
invokes fitting:

```python
import runpy
from fluxemu.observation import (
    StationaryCountSpecification,
    StationaryObservationExperiment,
    StationaryObservationSpecification,
    evaluate_stationary_observation_laws,
    sample_stationary_observations,
)

fixture = runpy.run_path("codex/examples/stationary_mfa_recovery.py")
problem, truth = fixture["build_mixture_problem"]()
specification = StationaryObservationSpecification(
    problem.model,
    (StationaryObservationExperiment(
        "count-experiment",
        problem.experiments[0].experiment,
        (StationaryCountSpecification("O-mid", 1000, "replicate-1"),
         StationaryCountSpecification("O-mid", 600, "replicate-2")),
    ),),
)
result = evaluate_stationary_observation_laws(specification, (truth,))
samples = sample_stationary_observations(result, seed=626)
assert result.validation.valid
assert tuple(item.observation.n for item in samples) == (1000, 600)
assert samples[0].component.predicted_mid == result.components[0].law.probabilities
assert samples[0].component.sample_id == truth.sample_id
assert samples[0].component.replicate_id == "replicate-1"
assert samples == sample_stationary_observations(result, seed=626)
```

The known state is `(Z_IN=3, A_IN=7, M_OUT=10)` in canonical reaction order.
The native prediction is approximately `(0.3, 0.7)`; the exact floating-point
tuple emitted by EMU remains available as each component's `predicted_mid`.
The explicit totals above are declared synthetic counting totals, not derived
from a normalised experimental MID or invented to reinterpret intensity data.

The bridge validates complete states against original-model bounds and mass
balance before EMU evaluation. Incomplete, misordered, or independently
assembled FVA coordinates are not a state-construction mechanism. Native
stationary restrictions, including explicit atom maps and `correction: no`,
remain in force. One EMU plan is compiled per experiment for an entire batch
of states, and each native prediction is reused across specified replicates.
Result order is state order, then experiment order, then each experiment's
specification order. Mass classes retain native M+0 through M+c order.

Each sampled record retains its source component, raw counts, and total.
Components retain state/sample, experiment, target, and replicate identities;
no replicates are silently averaged. Fingerprints bind the declared law type,
probabilities, total, mass order, complete state, scientific identities,
model/experiment fingerprints, and specification. Changing any of those
inputs changes the relevant provenance rather than disguising a new law.

## Exact law-divergence identities

For two laws with the same total `n` and ordered mass-class space,

\[
D_{\mathrm{KL}}(\operatorname{Mult}(n,p)\Vert\operatorname{Mult}(n,q))
=nD_{\mathrm{KL}}(p\Vert q),
\qquad
D_{\alpha}(\operatorname{Mult}(n,p)\Vert\operatorname{Mult}(n,q))
=nD_{\alpha}(p\Vert q).
\]

The Rényi identity holds for every finite real `alpha > 0`, with exact KL at
one. The multinomial coefficients cancel in the log likelihood ratio, giving
`sum_i k_i log(p_i/q_i)`; its expectation uses `E_p[K_i] = n p_i`, proving KL.
For Rényi at an order other than one, the multinomial theorem gives

\[
\sum_k P_p(k)^\alpha P_q(k)^{1-\alpha}
=\left(\sum_i p_i^\alpha q_i^{1-\alpha}\right)^n.
\]

Taking the logarithm and dividing by `alpha - 1` proves the identity.
For `alpha >= 1`, positive `p` mass outside `q` support gives infinity.
For `0 < alpha < 1`, only shared support contributes: partial overlap can
give a finite divergence, while disjoint support gives infinity. Production
uses `kl_divergence` and `renyi_divergence` from the existing MID kernel,
preserving these exact support conventions; it does not enumerate counts.
Comparisons reject mismatched totals or mass-class spaces instead of applying
the common-total identity to incompatible laws.

For explicitly independent ordered observation blocks, divergences add:

\[
D_\alpha\!\left(\bigotimes_j P_j\,\middle\Vert\,\bigotimes_j Q_j\right)
=\sum_j D_\alpha(P_j\Vert Q_j)
=\sum_j n_j D_\alpha(p_j\Vert q_j).
\]

The totals may differ across blocks but must agree within each compared pair.
Product independence is a declared measurement assumption; separate replicate
labels alone do not establish it. The API aggregates the supplied ordered
blocks and does not infer an experimental dependence model.

## Count likelihood and the existing MFA criterion

For observed raw counts `k`, `n = sum(k) > 0`, and empirical MID `p_hat = k/n`,

\[
-\log P_p(k)=nD_{\mathrm{KL}}(\hat p\Vert p)+C(k),
\quad
C(k)=-\log(n!)+\sum_i\log(k_i!)
-\sum_{i:k_i>0}k_i\log(k_i/n).
\]

This follows by adding and subtracting `sum_i k_i log(k_i/n)` in the negative
log PMF. Terms with zero counts vanish. An impossible positive count has
infinite negative log likelihood and infinite KL, with the same finite
data-only constant.

The identity uses the exact rational composition `k/n`. The exposed
`empirical_mid` uses binary64 values; at extreme totals, multiplying its
roundoff-sensitive MID KL by `n` can amplify representation error substantially.
Use `law.log_pmf(raw_observation)` for the numerical count likelihood: it works
directly from retained integers. Tight numerical reconstruction of that value
from the floating-point empirical MID is verified for the small/moderate
fixtures, not promised throughout the full signed-int64 count range. This
distinction does not change the mathematical identity or repair either input.
For example, at `n = 10**18`, counts `(n//2 + 64, n//2 - 64)`, and prediction
`(0.5, 0.5)`, direct negative log PMF is approximately `20.94905718959`, while
reconstruction from the binary64 empirical MID is about `76.46020842085`.
Raw-count evaluation preserves the required numerical likelihood in this case.

For one genuine fixed-total multinomial observation, minimising
`KL(p_hat || p)` is equivalent to maximum likelihood because `n` is fixed and
`C(k)` does not depend on the prediction. Across independent observations with
different observed totals, the joint negative log likelihood is
`sum_j n_j KL(p_hat_j || p_j) + sum_j C(k_j)`.

The current stationary MFA objective remains the **plain unweighted sum** of
MID divergences. No objective, weights, optimizer, or fitting behavior changes
in this observation-law layer. That existing objective is not automatically
the multinomial likelihood when totals differ. Rényi MID fitting is not
relabelled as maximum likelihood.

## Validation and scope

Independent tests construct exact rational PMFs using integer factorials and
enumerate 671 count vectors across 48 small laws with one through four mass
classes and totals one through six. Each full support sums to one; impossible
mass is exactly zero. Whole-law enumeration verifies 120 KL/Rényi comparisons
at orders `0.5, 0.73, 1.0, 1.3, 2.0`, including zeros and support mismatch,
plus 15 Cartesian-product cases with block totals two and three. Seven
single/joint likelihood checks verify the data-only constant and the role of
heterogeneous totals.

Focused tests also cover strict count validation, large-count numerics,
deterministic sampling, immutable records, exact native predictions, complete
feasible states, ordering, provenance, and a native EMU-to-law-to-counts
fixture. Fresh-process import guards and a base-installation CI matrix run
without optional scientific stacks. Existing MFA and native Stage 1
FBA/FVA/sampling/stationary-EMU regressions remain acceptance gates.

More realistic MS intensity observation laws are future work. They require
actual measurement semantics and replicate evidence, not aesthetic selection
of a distribution or an invented effective count. No second law, new MFA
optimizer, hypothesis-test decision rule, p-value, confidence region,
finite-sample bound, composite testing, Bayesian inference, experiment design,
transient MFA, JAX, or VFFVA performance work is included here.
