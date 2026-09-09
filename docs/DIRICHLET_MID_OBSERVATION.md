# Continuous Dirichlet MID observation V1

## Scope

CarbonScope can represent a properly QC'd, externally corrected mass-
isotopomer distribution as a continuous composition:

```text
Y | p, kappa ~ Dirichlet(kappa * p).
```

Here `p` is the MID predicted by native stationary EMU and `kappa > 0` is an
explicit concentration (precision) parameter. **Neither `kappa` nor the number
of replicate MIDs is a multinomial count.** V1 never derives an effective count
from fractions, percentages, uncertainty, peak area, signal intensity, or
replicate number.

The model implies

```text
E[Y_i]          = p_i
Var[Y_i]        = p_i (1-p_i) / (kappa+1)
Cov[Y_i, Y_j]   = -p_i p_j / (kappa+1), i != j.
```

This scalar covariance shape is a testable modelling assumption. It is not
made true by convenient algebra, and the diagnostic APIs below are intended to
expose incompatible replicate covariance.

## Public law and support contract

`fluxemu.observation.DirichletMIDLaw` is an immutable record containing ordered
mass classes, the full predicted MID, an explicit active face, precision and
its source, observation identity, correction provenance, and replicate
semantics. It provides `mean`, `covariance`, `parameters`, `log_density(...)`,
`sample(...)`, `sample_replicates(...)`, `renyi_divergence(...)`,
`kl_divergence(...)`, and `pairwise_score(...)`.

```python
from fluxemu.observation import DirichletMIDLaw, MIDCorrectionProvenance

correction = MIDCorrectionProvenance(
    status="externally_corrected",
    method="declared-laboratory-method",
    provenance="calibration-record-sha256:...",
)
law = DirichletMIDLaw(
    mass_classes=(0, 1, 2, 3),
    active_support=(0, 2, 3),
    predicted_mid=(0.2, 0.0, 0.3, 0.5),
    precision=80.0,
    precision_source="independent_calibration",
    precision_provenance="independent QC batch 2026-09",
    observation_identity=("experiment-1", "fragment-a", "technical-series-1"),
    correction=correction,
    replicate_count=3,
    replicate_semantics="technical_measurement_variability",
    independent_replicates=True,
)
```

Inputs must already lie on the simplex within
`DIRICHLET_SIMPLEX_TOLERANCE = 1e-12`; they are not renormalised. All active
coordinates must be strictly positive. A zero coordinate is allowed only as an
explicit structural zero. The stationary H0/H1 bridge computes the common
support of every represented law in each block, removes zeros common to all
states from the active face, and retains their identities in the full MID and
report. State-dependent active support is refused rather than repaired with
epsilon.

An observed exact zero on an active, continuously positive coordinate is
outside this law and `log_density(...)` refuses it. Such observations need a
censoring, detection-limit, zero-inflated, or other support-aware model.

## Correction and precision provenance

V1 does not implement natural-abundance correction. It requires
`MIDCorrectionProvenance(status="externally_corrected", ...)` with nonempty
method and source text. Unknown, uncorrected, or raw MIDs are refused as inputs
to the rigorous workflow. A block can override the workflow-level correction
record when fragments used different external methods.

Supported precision sources are:

| `precision_source` | Meaning in the testing layer |
| --- | --- |
| `fixed_external` | Positive finite concentration supplied with provenance; treated as fixed for the declared law. |
| `external_calibration` | Concentration supplied by a separately identified calibration result. |
| `independent_calibration` | Concentration calibrated from data independent of the tested observations. |
| `same_data_plugin` | Same observations supplied the estimate; diagnostic/model-conditional only and refused for known-precision finite-sample guarantees. |

Every law also declares one of
`technical_measurement_variability`, `biological_replicate_variability`, or
`total_replicate_variability`. Biological variability is never silently called
instrument noise. If analytical and biological contributions are inseparable,
use `total_replicate_variability` and report that limitation.

For `replicate_count > 1`, `independent_replicates` must literally be `true`.
The complete block law is then the declared conditional product. Rényi
divergences and log score moments multiply by the replicate count in log space.
This declaration does not turn the replicate count into an isotopologue count.

## Precision and adequacy diagnostics

`implied_dirichlet_precision_from_uncertainty(...)` requires a declared
uncertainty convention and replicate count. For an SD it reports each

```text
kappa_i = p_i(1-p_i) / SD_i^2 - 1,
```

and for an SE of an independent `R`-replicate mean it reports

```text
kappa_i = p_i(1-p_i) / (R * SE_i^2) - 1.
```

The structured `ImpliedPrecisionDiagnostic` retains every component estimate,
nonpositive/nonfinite status, range, median, maximum/minimum ratio, median
absolute deviation, assumptions, and warnings. It deliberately has no pooled
precision and is never suitable by itself for rigorous testing. The API does
not guess SD versus SE or impose a universal agreement threshold.

`diagnose_dirichlet_covariance(...)` compares replicate means and empirical
covariance against a declared centre and precision. Its structured result
contains covariance residuals, relative Frobenius error, maximum absolute
residual, centre RMSE, component-wise precision diagnostics, assumptions, and
semantic warnings.

`estimate_dirichlet_precision_from_replicates(...)` fits scalar
`c = 1/(kappa+1)` by Frobenius least-squares projection of empirical covariance
onto `diag(p)-p p.T`. A supplied centre is treated as fixed and covariance uses
divisor `R`; when the centre is omitted, the replicate mean is explicitly used
and ordinary sample covariance is used. The result records the centre source,
estimate, empirical and fitted covariance, residual quality, semantics,
same-data/independent status, assumptions, and warnings. An optional
deterministic parametric bootstrap refits the same statistic and supplies a
goodness-of-fit diagnostic p-value. The bootstrap is a model check, not proof.

Only an estimate explicitly declared independent of the tested observations is
marked potentially suitable for downstream testing. Suitability still depends
on the scientific calibration design and model adequacy; the API does not
certify those facts merely from a p-value.

## Rényi divergence and score moments

For common active support, `renyi_dirichlet(P, Q, lambda)` preserves the
directed order `D_lambda(P || Q)` and evaluates

```text
[log B(lambda*alpha + (1-lambda)*beta)
 - lambda log B(alpha) - (1-lambda) log B(beta)] / (lambda-1).
```

Every beta function is evaluated in log space. At `lambda = 1`,
`kl_dirichlet(...)` evaluates the analytic KL formula. A nonpositive mixed
parameter gives mathematical infinity; it is never clipped. Non-unit orders
within `DIRICHLET_NEAR_ONE_REFUSAL_RADIUS = 1e-8`, and additional cases whose
log-beta subtraction exceeds
`DIRICHLET_RENYI_ROUNDOFF_RELATIVE_BUDGET = 1e-8`, raise
`DirichletNumericalError`. This conservative binary64 conditioning check is a
refusal policy, not an interval-arithmetic certificate.

For selected `P*=Dir(alpha*)`, `Q*=Dir(beta*)`, the public score is

```text
s(y) = log B(alpha*) - log B(beta*)
       + sum_i (beta*_i-alpha*_i) log(y_i).
```

`DirichletLogLikelihoodScore.log_moment(R, t)` evaluates its exponential
moment in log space using `B(gamma+t(beta*-alpha*)) / B(gamma)`. A nonpositive
shifted parameter yields mathematical infinity; no parameter or score is
clipped.

## Finite represented-family testing

The parallel continuous classes in `fluxemu.testing` are:

- `IndependentDirichletMIDProductLaw` for explicitly independent blocks and
  replicates;
- `DirichletCompositeMIDLawFamily` and
  `DirichletCompositeBinaryTestingProblem` for explicit finite H0/H1 lists;
- `composite_dirichlet_renyi_converse_at_order(...)` for an order-specific
  lower bound on represented minimax Type-II error;
- `composite_dirichlet_renyi_score_candidate(...)` and
  `verified_composite_dirichlet_renyi_score(...)` for finite-pair selection and
  exact analytic uniform moment checks;
- `composite_dirichlet_score_bound_at_order(...)` for the verified projected
  achievable upper bound.

When both sides are available, the justified statement is

```text
order-specific composite converse <= represented beta*
    <= verified projected achievable upper bound.
```

The value of `beta*` is not fabricated. The existing exact minimax LP enumerates
a finite count space and is inapplicable to the continuous simplex.
`exact_dirichlet_composite_minimax(...)` therefore raises a structured
`unsupported_for_continuous_observation_space` refusal. Exact deterministic
score error and exact score calibration likewise refuse because V1 implements
no certified CDF for a weighted sum of log-Dirichlet components. It does not
substitute a Gaussian CDF, Monte Carlo guarantee, or simplex discretisation.

These results cover only the explicit finite list of represented flux states.
They do not establish a guarantee over the complete continuous feasible flux
family.

## Declarative workflow

The existing schema-version-1 hypothesis workflow selects the observation
branch through `observations.semantics`. Dirichlet experiments omit `counts`;
the continuous block declarations live under `observations.blocks`:

```yaml
experiments:
  - experiment_id: experiment-1
    specification: experiment.yaml

observations:
  semantics: corrected_mid_dirichlet
  correction:
    status: externally_corrected
    method: declared-method
    provenance: declared-source-or-fingerprint
  blocks:
    - experiment_id: experiment-1
      target_id: fragment-a
      replicate_id: technical-series-1
      replicate_count: 3
      replicate_semantics: technical_measurement_variability
      independent_replicates: true
      noise_model:
        distribution: dirichlet
        precision: 250.0
        precision_source: independent_calibration
        precision_provenance: independent-QC-record
  independent_blocks: false
```

Multiple blocks require literal `independent_blocks: true`. Each block has its
own precision, replicate design, active face, and optional correction override.
Strings are not coerced to numbers, percentages are not converted to fractions,
and SD/SE has no place in the testing schema: uncertainty first passes through
an explicit diagnostic/calibration workflow.

`fluxemu test-hypotheses` accepts this YAML without changing the genuine-count
branch. Unsupported optional continuous procedures appear as explicit refusals
and the valid workflow exits 0 with `completed_with_refusals`; invalid support,
provenance, precision, or independence declarations still exit 2.

The JSON report identifies continuous observation semantics, block support,
all state-centred parameter vectors, precision and source, replicate and
correction provenance, law fingerprints, supported results, refusals, and the
distinction between finite represented and continuous flux families. It never
reuses a genuine-count total field for concentration.

## Validation and open limitations

See the [Dirichlet V1 validation report](../results/dirichlet_mid_validation/VALIDATION_REPORT.md)
for independent SciPy/mpmath checks, deterministic simulations, model-
adequacy controls, exact stress counts, and the numerical refusal profile.

No suitable real public replicate MID dataset was available in this repository,
so validation on an external biological dataset remains pending. Correlated
replicates/blocks, biological random effects, unknown-precision finite-sample
theory, detection limits/censoring, zero-inflated or non-Dirichlet observation
models, exact continuous minimax computation, generic composite p-values, and
continuous feasible flux-family testing remain outside V1. Issue #25 therefore
remains open.
