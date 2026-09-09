# Composite testing repair v2: independent validation report

Repository: `esig626/CarbonScope`; issue [#22](https://github.com/esig626/CarbonScope/issues/22).
Branch: `codex/composite-testing-repair-v2`. No merge to main was performed.

## Audited source and evidence identity

- Exact base main SHA: `4f457d2afd75ad1bdffdbe0e9eea25fcaf8a9ad4`.
- Final published runtime repair commit: `e1421bf34a151f37e5b030b7f6a6feb73039f102` (identical tree to local `953c2570fcb16dd25c03bf9aa0d5c29f4e4da6f6`).
- Complete published regression/tooling commit: `55df38ef40ad88c85072baf46ca11c417202336b`, identical to local full-suite validation commit `efa13222b2e24e7ceb021df8d4badd8b4ec5f511` (subsequent changes are report/evidence and wording only).
- The publication commit containing this report is identified by the branch head and the final GitHub Actions run; the final pushed SHA/run/job conclusions are reported with delivery. This report does not claim CI had run before publication.
- Both campaign JSONs retain SHA-256 hashes of the actual audited production source. Their `executed_head` records the surrounding working checkout, whereas `audited_source_sha256` identifies which exported or repaired runtime was imported.

`AGENTS.md`, the entire issue and its latest comment, and the historical repair specification were read first. Only the specification/history was read from `codex/composite-testing-repair`; its code was never used as the base. Original main source was exported for independent baseline reruns. Private defect maps and red-test output were prepared before the corresponding production repairs.

The original implementation failed this audit. The historical report was not uniformly reproducible: the reverse converse algebra and strict LP decision bounds were already correct on current main.


### Publication identity mapping

Direct Git transport lacked write credentials. Publication therefore used the connected GitHub app. Each remote repair/documentation/test commit below has exactly the same Git tree as its locally validated counterpart; only commit metadata and parent IDs differ. The final evidence commit adds this mapping. Production source and regression logic are unchanged.

| Local commit | Remote commit | Identical Git tree |
|---|---|---|
| `9f844b9b481d218d7db4abb890f7cf670addffc7` | `bf5605c60f31f0fc4c2dc60bef36bd294673688f` | `a33b3e0f1b3981935596f75cc8843c11b53b53ed` |
| `69fb55475aea154ec1f649220ade20d38324bfb6` | `cbac1b3467dd881aaeb738bc7191a3ebfb077099` | `46c0a603a0c7a3445418a95d3c284588c0e15610` |
| `d416f2be0714860ebe5136cd4c84cb2fdc2f21da` | `75a196f909b7bf07e81b0139483e35a7b1de72d3` | `1a77ef4d7f8bc856427724a1fe9205bb30ebf294` |
| `953c2570fcb16dd25c03bf9aa0d5c29f4e4da6f6` | `e1421bf34a151f37e5b030b7f6a6feb73039f102` | `f4eecb5ab066128e5002eb5f364e32e6331febce` |
| `fc330cc55c05d8c90d06cd4106a296f5ee02e3c0` | `a3fe2810159d4a83abf602b48d53259c528f0c3d` | `c6ec04dfd75a568b4ed0d83301091cc8aa840463` |
| `efa13222b2e24e7ceb021df8d4badd8b4ec5f511` | `55df38ef40ad88c85072baf46ca11c417202336b` | `6e55214472dc45aef786256929858c6956d1c8c7` |

## Reproduced defects and historical claims

| Historical category | Independent result on current main | Repair/evidence |
|---|---|---|
| False composite Rényi converse | No material false converse reproduced. Separate 8,000-case Decimal NP audit found maximum excess only `2.22e-16`. | Retained original algebra; 30 independent converse/direction/support controls. |
| Negative returned LP rejection probabilities | Not reproduced. Original code already rejected every negative or above-one decision; adversarial injected values also refused. | Retained strict bounds and regression coverage; no clipping introduced. |
| Accepted LP objective/optimality inconsistency | Confirmed: an inconsistent `result.fun` and a feasible but suboptimal success result were accepted. Two real baseline singleton cases had reference gaps of `7.583416117e-10` and `5.08993180937e-10`, exceeding the `5e-10` allowance. A gross naturally occurring optimality error was not found. | Recompute original/scaled feasibility, objective and epigraph, and a conservative Lagrangian dual lower bound; reject uncertifiable gaps. Retain both actual fixtures plus injected-result regressions. |
| False projected moment certification | Confirmed at the default tolerance near order zero/one and with a loose caller tolerance. Three natural campaign examples falsely certified uniform moments. Opposing infinite block scores could also receive certification. | 80-digit direct moments, algebraic selected-law equalities, refusal of unresolved other-law equalities, and explicit undefined-product-support checks. Pair tie tolerance cannot relax moment inequalities. |
| Inconsistent/lossy stationary provenance | Confirmed: swapped H0/H1 laws with old member IDs, altered component provenance/state order/totals/laws, missing blocks and duplicate experiment metadata were accepted; fingerprints omitted source provenance. | Exact ordered source/state/component/law binding; complete source fingerprint inputs; explicit specification validation. Eleven original failing regressions, with positive native controls. |
| Clipping or sanitisation | A calibration `min(1,max(0,eta))` path existed and was removed. A naturally generated materially invalid LP decision being clipped was not reproduced. | Calibration now rejects any invalid boundary randomisation. Mathematical minimum/maximum operations defining dual endpoint bounds and trivial statistical bounds are not probability repairs. |
| Loss of tiny positive support/error | Confirmed in score ratio overflow, subtraction of power from one, and underflow of a positive expectation contribution. Source `MultinomialMIDLaw` already retained subnormal support and refused conversion that erased it. | Log differences replace ratios; directly sum `Q*(1-phi)`; explicitly refuse a nonzero product that underflows. A true achieved error near `1e-330` is refused rather than reported as zero. |
| H0/H1 or Rényi direction error | No algebraic direction error reproduced in simple or composite converse. Reverse and forward minimisers demonstrably differ. Provenance role swapping was a separate confirmed defect above. | Retain the documented reverse-only composite API and separate simple forward component. Independent directional-pair fixture prevents assumed reuse. |
| Result claims stronger than evidence | The LP result/docstrings used unqualified exact-solution language without objective/dual evidence. An incorrect minus-infinity disjoint-support threshold caused evaluation to refuse before returning an achieved result. | Expose numerical objective/epigraph/dual-gap diagnostics and qualify exactness. Use a finite disjoint-support threshold. Achieved score calibration remains distinct from unrestricted minimax. |
| Documentation and continuous families | Stale statements that composite testing was unimplemented were reproduced. Much of main already correctly excluded continuous families; an explicit claim that the complete continuous flux problem was solved was not found. | Correct stale status, numerical exactness language, achieved-versus-minimax distinctions, and refusal limits in all five requested documents. |
| Plausible answers where refusal is needed | Confirmed: tiny positive matrix coefficients could be discarded by HiGHS; projected errors could underflow to zero. Malformed solver vectors also leaked raw exceptions. | Explicit coefficient-resolution, expectation-underflow, vector-shape and optimality refusals. |

Additional corrected behavior: direct finite calibration can now use a well-defined candidate score even when stronger uniform moment conditions fail. The analytical projected formula still refuses such certification. This is a directly evaluated achieved test, not an application of a convex-class theorem.

### Regression-first evidence

The initial LP red run produced **3 failed, 2 passed**. Stationary red controls produced **11 failed, 73 passed**. The projected red run produced **6 failed, 1 passed**, and a separate calibration-gate regression failed. The later underflow/vector edge run produced **3 failed, 13 passed** before its repair. The two naturally occurring baseline LP-resolution fixtures each failed against exported main and pass after repair. The passing projected control was retained specifically to demonstrate strict suboptimality. Compact campaign counterexamples and all regression sources are committed; large raw logs are not.

## Mathematical and numerical checks

The finite LP is unrestricted over every randomised test on the complete declared observation space. Enumeration is never truncated or replaced by a product-test approximation. Its mathematical characterisation is exact; any returned floating-point solution is validated only to the documented tolerances.

For `A*x <= b`, `0 <= x <= 1`, and nonnegative multipliers `u`, the independent lower-bound identity used to check production optimality is

```text
min c*x >= -u*b + sum_j min(0, c_j + (A^T*u)_j).
```

Endpoint contributions are recomputed with accurate summation and conservative floating-point roundoff allowances. Acceptance checks the gap against `5e-10` plus recorded final summation allowance. Large uncertain certificates, invalid dual multipliers, invalid decisions, inconsistent objective/epigraph values and Type-I violations refuse explicitly. No solver probability is clipped.

The composite converse follows from Hölder applied to `phi`:

```text
E_Q phi <= epsilon^((lambda-1)/lambda)
           * exp((lambda-1)*D_lambda(Q||P)/lambda).
```

Thus minimising the reverse divergence over supplied pairs gives the documented lower bound. The simple forward component applies Hölder to `1-phi` and uses `D_lambda(P||Q)`. Their minimisers are not interchangeable. Every order is supplied explicitly; no continuous-order optimum is asserted.

Independent validation uses integer multinomial coefficients and 80-digit Decimal masses, randomised Neyman–Pearson for singletons, exhaustive Decimal active-set vertices for tiny composite spaces, and a separately formulated dual LP with independently recomputed feasible multipliers. The first 40 random problems cross-check both independent composite oracles. Oracle solver output alone is not treated as proof. Tiny cases use vertex fallback when a dual bound is insufficiently tight.

All accepted production LP cases receive direct recomputation and an independent optimality comparison. Reference allowances are `5e-10` for Decimal NP/vertex comparisons and `2e-9` for the separate floating-point dual/central comparisons. Type-I errors are recomputed from the supplied laws, with the production relative tolerance `5e-10` and independent campaign check `5e-9`; no input budget is increased. Signed near-zero gaps are reported without clipping.

## Test and campaign results

| Gate | Result |
|---|---:|
| Full Python 3.11.16 suite | **1324 passed** |
| Full Python 3.12.14 suite | **1324 passed** |
| Public examples on each Python version | All three completed successfully |
| Independent mandatory-control/regression file | 32 passed |
| Independent converse/direction/support controls | 30 passed on each Python version |
| Random categorical problems per main campaign | 400 |
| Numerical stress problems per main campaign | 10,000 |
| Targeted problems per main campaign | 121 |
| Total per main campaign | 10,521 |

The baseline and repaired campaigns both used Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1 and seed `20260909`. The Python 3.11 full suite additionally tested NumPy 2.4.6 and SciPy 1.17.1. Native `highspy` was 1.12.0; the composite LP uses SciPy's bundled HiGHS.

| Campaign outcome | Original main | Repaired |
|---|---:|---:|
| LP accepted | 10,014 | 7,457 |
| LP explicitly refused | 507 | 3,064 |
| Accepted LP results independently certified | 10,012 | **7,457 / 7,457** |
| False moment certificates | 3 | 0 |
| Accepted reference-gap failures | 2 | 0 |
| Achieved projected tests evaluated | 1,091 | 1,407 |
| Full converse/minimax/achieved comparisons | 945 | 1,091 |
| Converse comparisons with reference evidence | 9,250 | 8,890 |

Every applicable repaired accepted comparison satisfied `converse <= beta_star <= achieved projected error` within the declared numerical allowance. No unexpected programming exceptions were counted as numerical refusals. Repaired maximum independent optimality discrepancy was `4.725968505425726e-10`; maximum directly recomputed error discrepancy was `2.220446049250313e-16`.

A separate **8,000-problem** converse campaign (seed `5302026`) checked orders from the next float above one to `1e300`, binary counts up to 100, tiny positive support and nearly identical laws. The composite API accepted 6,948 and refused 1,052; simple Bruno accepted 5,809 and refused 2,191. Both APIs were independently evaluated whenever accepted. No material bound violation occurred; the maximum excess was `2.22e-16` against a `2e-12` allowance. These 8,000 cases are additional to the main 10,000 stress cases.

### Projected suboptimality and converse gaps

- Worst calibrated-score gap in the repaired campaign: **`0.8654864295286583`**, case `random-253`: minimax `0.08426845939221521`, achieved `0.9497548889208735`, epsilon `0.8277874168104449`, score order `0.9`. Its stronger moment conditions fail; the actual calibrated rule is nevertheless valid. The complete witness is in `repaired_summary.json`.
- Stable **verified-moment** regression: H0 categorical laws `(0.606,0.010,0.384)` and `(0.425,0.171,0.404)`; H1 `(0.948,0.050,0.002)` and `(0.517,0.419,0.064)`; epsilon `0.2`, order `0.5`. Achieved error `0.8853129411764706`, minimax `0.6778133486027607`, gap **`0.2074995925737099`**.
- Main repaired signed converse gap range (reference beta minus converse): **`[-3.5828007227678427e-12, 0.9999999999989921]`**. The tiny negative endpoint is within comparison resolution; it is not silently reset to zero or reported as a rigorous negative mathematical gap.
- Supplemental composite NP-minus-converse range: `[-2.220446049250313e-16, 1.0000000000000022]`. The few-ulps excess above one comes from the accepted represented-law mass residual; no renormalisation was performed.

## Explicit refusal regimes

HiGHS is configured with `small_matrix_value=1e-12`. Every nonzero scaled LP coefficient at or below that magnitude causes refusal before solving. This protects positive support from the solver's small-matrix policy. Budgets below `1e-12` and positive masses/errors below representable resolution also refuse.

| Declared control at epsilon 0.05 | Last accepted adjacent total | First coefficient refusal |
|---|---:|---:|
| Identical uniform binary laws `(1/2,1/2)` | 39 | 40 |
| Identical uniform ternary laws `(1/3,1/3,1/3)` | 25 | 26 |
| Binary H0 `(.8,.2),(.7,.3)`; H1 `(.3,.7),(.2,.8)` | 17 | 18 |
| Ternary H0 `(.6,.3,.1),(.5,.3,.2)`; H1 `(.2,.3,.5),(.1,.3,.6)` | 12 | 13 |

Complete accepted/refused sweeps for binary `n=1..64`, ternary `n=1..40`, and the eight uniform boundary cases are recorded in JSON. Some smaller problems also refuse for feasibility, invalid duals or optimality gaps; count total alone is not a sufficient acceptance condition.

The uniform limits also provide universal ceilings under this policy: at least one category of a full-support binary/ternary law is at most `1/2` or `1/3`, so its extreme-count outcome has positive probability at most `2^-n` or `3^-n`. Alternative coefficients are unscaled. Hence every such full-support block reaches the policy ceiling by 40/26; asymmetric probabilities and independent products can reach it earlier. Structural zeros remain exactly zero.

The historical affine ternary parameters were not supplied in the available repository, issue or specification. This missing historical control could not be reproduced literally. The documented replacement is `p(t)=(.6-.2t,.3-.1t,.1+.3t)`, finite H0 grid `(0,.25,.5)`, H1 grid `(.6,.8,1)`, and count totals `(1,2,3,5,8)`. It is a finite-grid control only.

## Stationary and count semantics

Every native input state is independently checked for complete canonical reaction order, original-model feasibility and native constraints before stationary EMU evaluation. State, experiment, target and replicate order and genuine count totals are retained. Duplicate sample labels cannot collapse distinct states. Multiple blocks require explicit independence. Structural zeros remain exact and no inverse-MFA fitting is invoked by the bridge.

Ordinary MID fractions, percentages, peak areas and intensities are never converted into counts. Count totals must be explicitly declared using genuine-count specifications; Boolean/floating totals and malformed counts are rejected by the existing observation validators and tests.

Result construction now verifies consistency of all supplied native provenance records and exact source laws, including H0/H1 bindings. Fingerprints describe those records; they do not authenticate a coherently fabricated collection of records supplied by a caller. Native evaluation is the path that performs actual model/state validation.

## Remaining limitations and publication gate

This work validates a **supplied finite family**, not the complete continuous feasible flux family. Rigorous optimisation or bounds over that continuous family remain unimplemented. Neither a finite sample nor an affine grid proves coverage of a continuum. A candidate score is not automatically a least-favourable pair, and achieved calibration need not be unrestricted finite-sample minimax.

The implementation and its Decimal audit are numerical, not symbolic or interval-arithmetic proofs. Machine-simplex mass residuals remain part of the existing law contract. Extreme orders, tiny budgets, unresolved moments, underflow, coefficient resolution and ill-conditioned LP evidence can legitimately refuse valid mathematical problems. The complete observation-space cap and dense LP also limit scalability. No continuous-order optimisation, generic composite p-value or continuous-family optimality claim is made.

The branch is ready for publication only after the local gates above. Final completion additionally requires every existing Python 3.11/3.12 GitHub Actions job to pass on the exact pushed publication SHA. That SHA and its completed run/job conclusions are verified after pushing and reported in the delivery message; no merge is authorised or performed.

## Files changed

- `README.md`
- `docs/API_MAP.md`
- `docs/COMPOSITE_TESTING.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/SCIENTIFIC_WORKFLOW.md`
- `results/composite_testing_validation/VALIDATION_REPORT.md`
- `results/composite_testing_validation/baseline_converse_stress.json`
- `results/composite_testing_validation/baseline_summary.json`
- `results/composite_testing_validation/repaired_summary.json`
- `src/fluxemu/testing/composite.py`
- `src/fluxemu/testing/composite_stationary.py`
- `tests/test_composite_converse_repair.py`
- `tests/test_composite_lp_repair.py`
- `tests/test_composite_projected_repair.py`
- `tests/test_composite_stationary_repair.py`
- `tests/test_composite_validation_oracle.py`
- `tools/composite_testing_validation/README.md`
- `tools/composite_testing_validation/__init__.py`
- `tools/composite_testing_validation/converse_stress.py`
- `tools/composite_testing_validation/oracle.py`
- `tools/composite_testing_validation/run_campaign.py`
