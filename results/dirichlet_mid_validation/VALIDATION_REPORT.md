# Dirichlet MID observation V1 validation

## Scope and conclusion

The deterministic synthetic campaign passed under Python 3.11 and 3.12. It
independently checks the continuous `Dirichlet(kappa * p)` distribution
identities, directed Renyi formula, score exponential moments, replicate and
block composition, projected-test Type-I behaviour, and model diagnostics.
The two runtimes produced identical scientific metrics.

This is validation of the finite represented-family Dirichlet V1 slice. It is
not evidence that Dirichlet noise is universally adequate, that `kappa` is an
effective count, that a same-data plug-in precision has a known-precision
guarantee, or that continuous feasible flux-family testing has been solved.

No suitable real public replicate MID dataset was found in this repository.
The private motivating workbook was not used. External empirical validation on
a public biological dataset therefore remains pending, and issue #25 should
remain open for broader observation models and empirical/model extensions.

## Reproduction

From the repository root with the `testing` extra installed:

```bash
python tools/dirichlet_mid_validation/run_campaign.py \
  --output results/dirichlet_mid_validation/summary.json
```

The committed records are
`summary_py311.json` and `summary_py312.json`. Both use seed `20260909`, scale
`1.0`, and contain SHA-256 hashes of the four production implementation files.
The campaign uses SciPy formulas, 80-digit mpmath controls, and direct NumPy
simulation rather than evaluating the production formula twice.

## Distribution and numerical identities

- 300,000 Dirichlet draws covered five configurations, dimensions 2 through 7
  in the distribution checks, and precision from 0.5 through 10,000.
- Maximum mean deviation was 1.627 standard errors; maximum relative covariance
  Frobenius error was 0.00803; maximum simplex residual was
  `4.440892098500626e-16`.
- Analytic covariance rows and columns summed to zero, and increasing precision
  reduced every marginal variance in the control.
- 3,000 adversarial Renyi cases covered dimensions 2 through 12, precision
  approximately 0.05 through 10,000,000, orders 0.01 through 50, boundary-near
  probabilities, near-one orders, and integrability failures.
- 2,479 Renyi evaluations were accepted; 521 were explicitly refused by the
  binary64 conditioning policy. Of accepted finite results, 1,706 agreed with
  independent SciPy calculations to maximum relative error
  `2.051511298174527e-10`; 51 high-precision controls agreed with mpmath to
  `3.078716726603991e-11`.
- 773 mixed-parameter cases correctly returned infinite divergence. Identical
  laws, order-one-half symmetry, asymmetric direction controls, KL-near-one
  controls, and seven-replicate additivity passed. Orders within the fixed
  near-one exclusion and cases exceeding the conservative roundoff budget were
  refused without clipping.

## Score moments and projected tests

The campaign checked 1,500 analytic score moments against SciPy: 735 were
finite, 765 correctly identified nonintegrability, and the maximum relative
error was `2.2818998752187732e-11`. Three independent Monte Carlo controls used
150,000 draws each and differed from their analytic log moments by at most
1.255 estimated Monte Carlo standard errors.

Three analytically certified projected score constructions were simulated with
80,000 draws for every represented member. All five represented H0 members
were checked. At declared epsilon 0.05, observed Type-I rates were:

| Campaign | H0 member rates | 95% Wilson intervals | Analytic Type-II upper bound | Observed H1 Type-II rates |
| --- | --- | --- | ---: | --- |
| ordered one block, 2 replicates | 0.000625, 0.007425 | [0.000474, 0.000824], [0.006853, 0.008044] | 0.068471 | 0.000838, 0.010638 |
| singleton one block | 0.007175 | [0.006613, 0.007784] | 0.120053 | 0.019363 |
| ordered two blocks | 0.000625, 0.005650 | [0.000474, 0.000824], [0.005154, 0.006194] | 0.692273 | 0.030613, 0.134575 |

Simulation is a sanity check only. The guarantee comes from the analytically
verified uniform exponential-moment inequalities, not from Monte Carlo.

## Model-adequacy diagnostics

Each synthetic diagnostic dataset contained 1,200 replicate vectors. For a
true Dirichlet with generating precision 80, the scalar covariance-shape fit
estimated 77.4583, had relative covariance error 0.05046, and bootstrap
diagnostic p-value 0.0861. A deliberately anisotropic logistic-normal dataset
had relative covariance error 0.62112 and bootstrap p-value 0.00662. It was
marked unsuitable for rigorous testing because its precision was estimated
from the same data and analytical and biological contributions were not
separable.

The bootstrap uses a fixed seed and is a goodness-of-fit diagnostic, not proof
of the model. No universal scientific pass/fail threshold is imposed by the
public diagnostic API.

## Deliberate refusals and open work

- The finite-outcome exact minimax LP, exact score CDF evaluation, and exact
  count-space score calibration are unavailable for a continuous simplex.
- Exact zeros in active observed coordinates require a censoring or
  detection-limit model. State-dependent active faces are refused. Structural
  zeros common to every represented law are retained in provenance and removed
  from the active face without pseudocounts.
- Raw or correction-unknown MIDs are refused; V1 accepts only explicitly
  externally corrected compositions with method and source provenance.
- Correlated blocks/replicates, unknown-precision finite-sample theory,
  biological random effects, zero inflation/censoring, non-Dirichlet noise,
  and continuous feasible flux-family testing remain open.
