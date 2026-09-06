# Simple binary finite-sample Rényi certificates

`fluxemu.testing` evaluates a published lower bound on optimal Type-II error
for two **fixed** observation laws. Its stationary bridge connects two complete
feasible flux states to those laws through the existing EMU observation path:

```text
fixed flux state v -> stationary predicted MID p_v -> declared Mult(n, p_v)
fixed null/alternative laws -> full-law Rényi divergences -> Type-II lower certificate
```

Install the ordinary base package with `python -m pip install ./codex`.
Certification and likelihood-ratio evaluation require no SciPy or other optional
scientific stack. `fluxemu.observation` owns the count laws; `fluxemu.mfa` retains
its existing fitting objective.

## Published source and testing convention

Roberto Bruno, Adrien Vandenbroucque, and Amedeo Roberto Esposito (2026),
*A Finite-Sample Strong Converse for Binary Hypothesis Testing via (Reverse)
Rényi Divergence*, **arXiv:2601.09550v2**.
[Versioned article](https://arxiv.org/html/2601.09550v2).
The [theorem-transfer audit](BRUNO_V2_THEOREM_TRANSFER.md) records the source
check made before implementation.

| Quantity | Meaning |
| --- | --- |
| `null`, `H0`, `P0` | Fixed null observation law. |
| `alternative`, `H1`, `P1` | Fixed alternative observation law. |
| Type I | `P0(decide H1)`. |
| Type II | `P1(decide H0)`. |
| `epsilon` | Type-I constraint, strictly `0 < epsilon < 1`. |
| `beta*(epsilon)` | Minimum Type II over the paper's deterministic binary tests satisfying Type I `<= epsilon`. |
| `order`, `lambda` | Finite real Rényi order, strictly `lambda > 1`. |
| `reverse_renyi` | `D_lambda(P1 || P0)` of the complete laws. |
| `forward_renyi` | `D_lambda(P0 || P1)` of the complete laws. |

All logarithms are natural. Theorem 1 assumes mutual absolute continuity.
For every corresponding fixed-total block, FluxEMU requires exactly the same
positive-support mass classes under both hypotheses. Matching structural zeros
are retained. A mismatch raises `BrunoTheoremAssumptionError` naming Theorem 1,
the block identity, and both positive supports **before either divergence is
computed**. A finite divergence in only one direction cannot waive this check.
There are no pseudocounts, clipped probabilities, or support repairs.

| Source location | Implemented transfer |
| --- | --- |
| Eqs. (1)–(2) | Deterministic decisions and constrained optimal Type II. |
| Theorem 1, Eq. (5) | Two converse terms and the full continuous-order envelope. |
| Appendix A, Eqs. (14)–(17) | Reverse full-law inequality before tensorisation. |
| Appendix A, unnumbered inequality after Eq. (17) | Forward full-law inequality before tensorisation. |
| Appendix A, final display | I.i.d. tensorisation; the implementation instead accepts the complete law directly. |

## Full-law formula and the continuous envelope

For one admissible order, define `Drev = D_lambda(P1 || P0)` and
`Dfwd = D_lambda(P0 || P1)`. The two components are

```math
B_{\rm reverse}=1-\exp\!\left[\frac{\lambda-1}{\lambda}
  (\log\epsilon+D_{\rm rev})\right],\qquad
B_{\rm forward}=\exp\!\left[\frac{\lambda}{\lambda-1}
  \log(1-\epsilon)-D_{\rm fwd}\right].
```

Thus `beta*(epsilon) >= max(B_reverse, B_forward)`. Each order supplies a
lower certificate. The mathematical envelope remains the full continuous one:

```math
\beta^*(\epsilon)\ge\max\left\{
1-\inf_{\lambda>1}
  \left(\epsilon e^{D_\lambda(P_1\Vert P_0)}\right)^{(\lambda-1)/\lambda},
\sup_{\lambda>1}(1-\epsilon)^{\lambda/(\lambda-1)}
  e^{-D_\lambda(P_0\Vert P_1)}\right\}.
```

The package implements the order-specific primitive. It supplies no order
search, finite-order grid API, or global-envelope optimization. A returned
certificate has `global_envelope_certified == False`. The finite sets of orders
in validation tests are test coverage only.

For the existing multinomial law,
`D_lambda(Mult(n,p) || Mult(n,q)) = n D_lambda(p || q)`. The production
implementation calls `renyi_multinomial` or `independent_product_renyi` in each
direction. It does not implement a second MID or law divergence kernel.
Under declared independence, full-law divergences add over ordered blocks;
corresponding blocks share a total, while different blocks may have different
totals. No additional factor of `n` is applied to an already full-law value.

## Public API

| API | Contract |
| --- | --- |
| `SimpleBinaryLawPair(null=..., alternative=...)` | Immutable explicit roles for one law pair. |
| `SimpleBinaryLawPair(null=(...), alternative=(...), independent=True)` | Explicit independent products, preserving block order and optional experiment/target/replicate identities. |
| `SimpleBinaryTestingConstraint(epsilon=...)` | Strict Type-I budget validation and fingerprint. |
| `bruno_converse_at_order(laws, epsilon=..., order=...)` | Accepts a law pair or stationary result and returns `BrunoOrderCertificate`. |
| `SimpleBinaryFluxHypotheses(null_state=..., alternative_state=...)` | Two fixed complete canonical state records with distinct roles. |
| `evaluate_stationary_simple_hypotheses(specification, null_state=..., alternative_state=..., independent_blocks=False)` | Existing native observation evaluation once per role; multiple blocks require `independent_blocks=True`. |
| `StationarySimpleTestingResult` | Both source law results, both native feasibility reports, predicted MIDs, components, hypotheses, and `law_pair`. |
| `log_likelihood_ratio(laws, observations)` | `log P1(y) - log P0(y)` for one count vector/record or an ordered tuple of product-block observations. |
| `validate_bruno_assumptions(pair)` | Exact corresponding-support theorem gate. |

For a law-only calculation:

```python
from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import SimpleBinaryLawPair, bruno_converse_at_order

pair = SimpleBinaryLawPair(
    null=MultinomialMIDLaw(4, (0.25, 0.75)),
    alternative=MultinomialMIDLaw(4, (0.5, 0.5)),
)
certificate = bruno_converse_at_order(pair, epsilon=0.05, order=1.7)
print(certificate.type_i_constraint)
print(certificate.reverse_renyi, certificate.forward_renyi)
print(certificate.reverse_lower_bound, certificate.forward_lower_bound)
print(certificate.type_ii_lower_bound)
```

Interpretation: among tests with Type I at most `0.05`, optimal Type II is at
least the returned lower certificate for this fixed pair and this order.
The value is not the actual Type-II error of a chosen test or an observed
sample, a p-value, or a confidence level. Even for identical laws, a finite
order need not attain the full-envelope value `1 - epsilon`; the discrete
deterministic optimum can also exceed that value.

Certificate fingerprints bind the ordered null and alternative laws,
independence declaration, identities, epsilon, order, both divergences, and raw
and combined components. Stationary certificates additionally bind both complete
state fingerprints and the observation-specification fingerprint. Source
components expose model/experiment fingerprints and unchanged predicted MIDs.

## Complete stationary flux example

Run the [self-contained native example](../examples/simple_binary_flux_discrimination.py):

```bash
python codex/examples/simple_binary_flux_discrimination.py
```

It declares a balanced three-reaction, one-carbon mixer, explicit atom mappings,
and unlabelled/fully labelled tracers. Reaction order is `Z_IN, A_IN, M_OUT`.
The fixed states are `v0=(2.5,7.5,10)` and `v1=(5,5,10)`. Their stationary MIDs
are `p0=(0.25,0.75)` and `p1=(0.5,0.5)`.

This complete repository-root snippet reuses only the example's model fixture:

```python
import runpy
from fluxemu.execution import CanonicalFluxState
from fluxemu.observation import (
    MIDCountObservation, StationaryCountSpecification,
    StationaryObservationExperiment, StationaryObservationSpecification,
)
from fluxemu.testing import (
    bruno_converse_at_order, evaluate_stationary_simple_hypotheses,
    log_likelihood_ratio,
)

fixture = runpy.run_path("codex/examples/simple_binary_flux_discrimination.py")
model, experiment = fixture["_model_and_experiment"]()
specification = StationaryObservationSpecification(model, (
    StationaryObservationExperiment("fixed-tracer", experiment, (
        StationaryCountSpecification("O-mid", 4, "counts-1"),
    )),
))
v0 = CanonicalFluxState("v0", (("Z_IN", 2.5), ("A_IN", 7.5), ("M_OUT", 10.0)))
v1 = CanonicalFluxState("v1", (("Z_IN", 5.0), ("A_IN", 5.0), ("M_OUT", 10.0)))
laws = evaluate_stationary_simple_hypotheses(
    specification, null_state=v0, alternative_state=v1,
)
certificate = bruno_converse_at_order(laws, epsilon=0.05, order=2.0)
assert laws.null_observation_laws.validation.valid
assert laws.alternative_observation_laws.validation.valid
print(laws.null_predicted_mids, laws.alternative_predicted_mids)
print(certificate.type_ii_lower_bound)
print(log_likelihood_ratio(laws, MIDCountObservation((1, 3), 4)))
```

At `epsilon=0.05`, `lambda=2`, the example produces:

| Genuine count total | Reverse law Rényi | Forward law Rényi | Type-II lower certificate |
| ---: | ---: | ---: | ---: |
| 1 | 0.28768207 | 0.22314355 | 0.74180111 |
| 4 | 1.15072829 | 0.89257421 | 0.60247680 |
| 16 | 4.60291316 | 3.57029682 | 0.02540312 |
| 64 | 18.41165264 | 14.28118728 | 0.0000005665084 |

The same flux pair has the same MIDs at every total. Each complete-law
divergence scales with its explicitly declared count total. The smaller lower
certificates at larger totals permit smaller errors but do not establish that
a particular error is achievable. The example also varies MID separation at
fixed total and includes identical-state, unequal-total independent-product,
and support-mismatch controls.

Both hypotheses are independently validated against the original model before
their stationary EMU evaluation. Incomplete, reordered, unbalanced, or
out-of-bounds states fail. A model's FBA objective adds no optimality-fraction
constraint here; tests include feasible states below its optimum. Identical
states and distinct states sharing a source sample ID retain those IDs, because
the two roles use separate observation evaluations. Hypotheses are complete
fixed states, never a combination of independently chosen FVA interval endpoints.

## Count semantics, numerical limits, and LLR support

`n` is the explicit **genuine count total**, within the existing observation
layer's supported range. It is not fitted or inferred. Arbitrary intensities,
peak areas, percentages, and normalized MIDs do not acquire count semantics
through this API. Calling MID normalization does not create a count model.
The declared multinomial model is what gives MID Rényi divergence its direct
operational role in finite-sample discrimination of these two flux states.

The reverse term uses `-expm1(x)` with
`x=((lambda-1)/lambda)*(log(epsilon)+Drev)`. The forward term retains
`log_forward_lower_bound=lambda/(lambda-1)*log1p(-epsilon)-Dfwd` and exponentiates
that logarithm. There are no epsilon floors or probability clamps.

- Orders immediately above one are used unchanged. Finite real input scalars
  that cannot be represented as a finite binary64 order above one raise
  `NumericalLimitError`; there is no endpoint substitution or grid.
- Bool, NaN, infinite or out-of-domain order/epsilon inputs fail validation.
  Interior epsilon values that round to an endpoint also fail explicitly.
- A negative raw reverse component remains negative. If its magnitude exceeds
  floating-point range, the raw field is `-inf` and `reverse_log_power` remains
  finite. `reverse_component_overflowed` reports this case.
- Forward underflow produces zero with a retained finite log bound and
  `forward_component_underflowed=True`. The combined field stays finite.
- The existing law layer's probability-mass representability boundary remains
  in force. Inherited negative/nonfinite divergence results or failed numerical
  evaluations raise `NumericalLimitError` with their cause; they are not clipped.

LLR evaluation first validates count vectors through both existing laws.
It returns `+inf` when only `P0(y)=0`, and `-inf` when only `P1(y)=0`.
When both **complete** laws assign zero mass, it raises
`UndefinedLikelihoodRatioError`, including product samples with opposite
off-support blocks. It never adds `+inf` and `-inf` into a silent NaN.
On common support, common multinomial coefficients cancel and 60-digit Decimal
log-ratio accumulation preserves small signals lost by subtracting two rounded
large log PMFs. No decision threshold or test is selected.

## Independent validation and scope

The test-only oracle independently enumerates factorial/Fraction multinomial
PMFs and complete product spaces. It checks exact rational normalization,
compares both full-law Rényi directions using 100-digit Decimal summation, and
checks both formula components using 400-digit arithmetic. Its optimal Type II
comes from every deterministic rejection subset satisfying the exact rational
Type-I budget. It never derives that optimum from the production certificate,
randomizes boundary atoms, or assumes a whole-atom likelihood threshold solves
the discrete budget problem.

The matrix contains 27 law pairs, 1,323 epsilon/order cases, and 7,032 exhaustive
rejection regions. Float comparisons use PMF absolute tolerance `3e-14`, formula
relative/absolute tolerances `3e-13`/`3e-14`, and bound-validity absolute tolerance
`1e-12`. These are validation tolerances, not permission to alter support or
normalize probabilities. Additional tests cover arbitrary non-grid orders,
maximal finite order, large allowed counts, native feasibility, provenance,
and fresh-process optional-import isolation. See
[acceptance evidence](SIMPLE_BINARY_ACCEPTANCE.md) and the
[guardian audit](SIMPLE_BINARY_GUARDIAN_AUDIT.md).

This layer concerns two fixed hypotheses. Composite hypotheses, least-favourable
pairs, testing inversion, confidence/compatibility regions, p-values, Bayesian
inference, uncertainty estimation, other observation laws, transient MFA, JAX,
experiment design, and FVA performance changes are deferred. The existing MFA
fitting objective is unchanged.
