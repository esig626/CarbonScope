# SIMPLE-TEST-GUARDIAN audit

This independent review concerns only
`codex/prompts/05_simple_binary_renyi_certificates_bruno_multiagent.md`
on `codex/simple-binary-renyi-certificates`. The guardian was created before
the implementation specialists and remains available through final acceptance.
An approval of a design is separate from acceptance of its implementation.

## Gate record

| Gate | Decision | Evidence or remaining requirement |
| --- | --- | --- |
| Source/theorem audit | APPROVE | Independently checked [Bruno et al., arXiv:2601.09550v2](https://arxiv.org/html/2601.09550v2), version dated 17 January 2026: definitions (1)–(4), Theorem 1 (5), and Appendix A (14)–(17) plus its unnumbered forward inequality before tensorisation. The published-theorem referee agrees. |
| Testing API design | APPROVE | Keyword-only null/alternative records; explicit product independence; immutable identities/fingerprints; separate exact support theorem gate; strict epsilon/order validation; raw components and finite combined result. |
| Numerical design | APPROVE | Existing observation-law kernels in the required directions; unchanged finite orders; MAC before computation; `expm1`/`log1p` formulas; explicit overflow/underflow and inherited divergence numerical-limit diagnostics; no order search. |
| Stationary bridge design | APPROVE | Revised design evaluates each role separately through the merged observation bridge, preserving identical/shared sample IDs and both independent native validations. Alignment and source provenance remain explicit; LLR validates through the laws before stable coefficient cancellation and joint-support handling. |
| Independent oracle design | APPROVE | Exact Fraction count PMFs and exhaustive deterministic rejection subsets; independent Decimal full-law Rényi and formula evaluation; explicit tolerances and unequal-total independent products. |
| Milestone A | APPROVE | Source note, schemas, exports, and tests reviewed. Guardian rerun: 108 focused tests passed; lead combined testing/observation run: 446 passed. Base imports exclude SciPy. Committed/pushed as `a74a2387a73233901151cf911d8e3fa4346ee630`; all eight triggered remote workflows succeeded. |
| Milestone B | APPROVE | Full-law formulas, support gate order, diagnostics, and oracle reviewed. Guardian reruns: 246 schemas/numerics tests and 1,359 oracle tests passed. Oracle covers 27 pairs, 1,323 epsilon/order comparisons, 7,032 deterministic rejection subsets, and 49,224 exact budget checks. Committed/pushed as `077379bea5afc5ce8cd8eb14c026fc22f7d74364`; all eight triggered workflows succeeded. |
| Milestone C | APPROVE | Bridge, stable LLR, exports, tests, and synthetic example reviewed. Guardian reran 57 tests and the example successfully; lead aggregate: 2,000 testing/observation tests passed. Fixed MIDs at totals 1, 4, 16, 64 produce scaled divergences and lower certificates 0.7418011, 0.6024768, 0.02540312, 5.665084e-7. Shared IDs, controls, support rejection, unequal totals, suboptimal feasible states, and invalid states are covered. Committed/pushed as `2cab2f7f9582bded66d0610fce0e7a6c934b1c2c`. |
| Milestone D implementation/documentation | APPROVE | Reviewed scientific guide, acceptance commands/evidence, README/API/limitations/crosslinks, Python 3.11/3.12 base-wheel CI, and optional-import isolation. Guardian reran both new isolation tests and both `pip check` commands successfully and verified baseline vendor/MFA/observation/dependency diffs are empty. Final installed-base-wheel aggregate: 2,002 passed; recorded native/MFA groups: 572 passed with one expected optional skip, and 525 passed with the MFA extra. Compileall, snippets, example, links, and whitespace pass. Lead may commit/push and open the PR. |
| Final PR implementation audit | APPROVE | Independently reviewed open, nondraft, mergeable PR #18 at `a2d6c01bdc60baae6b20b54395300d9e56361e3d`, its 21 scoped files, and accurate PR description. All ten PR-triggered workflows succeeded, including both new Python matrix jobs; the lead additionally verified all 19 push/PR runs. All 25 adversarial checks below are affirmative. |

## Mandatory review boundaries

- Null is `P0`; alternative is `P1`. Type I is `P0(decide H1)`;
  Type II is `P1(decide H0)`.
- Reverse is `D_lambda(P1 || P0)`; forward is `D_lambda(P0 || P1)`.
- The theorem requires mutual absolute continuity. Exact support is checked
  before certification, without smoothing, normalization, or count inference.
- The production core accepts finite real orders greater than one. It does
  not substitute a grid or call a numerical local optimum the continuous
  global envelope.
- Observation-law ownership remains in `fluxemu.observation`. Corresponding
  ordered blocks retain identities and explicit totals; aggregation requires
  declared independence.
- Complete flux states are independently validated through the existing
  stationary observation path. No fitting or biological objective is added.
- Raw reverse and forward components remain observable. A vacuous negative
  reverse component is not silently clipped; the combined result is finite.
- The validation oracle exhausts deterministic rejection subsets on tiny
  count spaces. It neither randomizes boundary atoms nor assumes every
  deterministic constrained optimum is a whole-atom likelihood threshold.
- Every implementation milestone requires focused tests, guardian approval,
  a coherent commit, and a push before work proceeds to the next milestone.
- Final acceptance requires explicit answers to all 25 prompt audit questions,
  all mandated checks, unchanged vendor/MFA objectives, and reviewed PR state.

The initial bridge design received BLOCKED because batching both hypotheses
would reject valid identical/shared sample IDs in the existing native batch
validator. Separate role-specific evaluation resolved the issue before
implementation; no scientific identifier was renamed.

## Mandatory adversarial final audit

**Decision: APPROVE — all 25 checks are affirmative.**

Reviewed implementation/documentation commit:
`a2d6c01bdc60baae6b20b54395300d9e56361e3d`.
[PR #18](https://github.com/esig626/fluxemu-standalone/pull/18) is open,
nondraft, and mergeable against the unchanged `main` baseline
`217244a85b3c91e60e396842c0cde5e8b8f8c350`.
The guardian independently verified the PR state and CI on 2026-09-06.

| # | Required question | Answer and evidence |
| ---: | --- | --- |
| 1 | Are null and alternative roles preserved everywhere? | **Yes.** Keyword-only role records, role-labelled fingerprints, separate stationary source results, and direction-swap tests preserve H0/P0 and H1/P1. |
| 2 | Is Type I `P0(decide H1)` and Type II `P1(decide H0)` everywhere? | **Yes.** Runtime docstrings, source transfer, user guide, example, oracle rejection regions, and PR description use this convention. |
| 3 | Is reverse Rényi always `D_lambda(P1 || P0)`? | **Yes.** `bruno_converse_at_order` calls the existing full-law routine with alternative first and null second; asymmetric independent checks verify it. |
| 4 | Is forward Rényi always `D_lambda(P0 || P1)`? | **Yes.** The second full-law call has null first and alternative second, with distinct forward diagnostics and swap checks. |
| 5 | Does the exact core accept every finite real `lambda > 1` rather than a finite grid? | **Yes.** The API takes a supplied continuous real order, preserving every admissible binary64 order, including the successor of one and maximum finite float. Unrepresentable real inputs raise an explicit numerical-limit error; no grid or endpoint substitution exists. |
| 6 | Are the two Bruno lower-bound terms implemented with the correct exponents? | **Yes.** Reverse uses `(lambda-1)/lambda` on `log(epsilon)+Drev`; forward uses `lambda/(lambda-1)` on `log(1-epsilon)` and subtracts `Dfwd`. Independent complete-law formulas agree. |
| 7 | Is the implementation based on full observation-law divergence before any multinomial simplification? | **Yes.** Production consumes `MultinomialMIDLaw` pairs and calls observation-owned full-law APIs. Source mapping identifies Appendix A before tensorisation. |
| 8 | Is the multinomial `n * D_lambda` identity reused rather than rederived inconsistently? | **Yes.** The existing `renyi_multinomial` identity is unchanged. Independent factorial count-space checks verify tensorisation without deriving expected values from production kernels. |
| 9 | Are independent product blocks aggregated only under explicit declared independence? | **Yes.** Law tuples require literal `independent=True`; multiblock stationary evaluation requires `independent_blocks=True`. Order and different per-block totals are retained. |
| 10 | Is mutual absolute continuity checked before applying Theorem 1? | **Yes.** Exact corresponding positive supports are checked before either divergence, with monkeypatch tests proving failure precedes evaluation. Certificate records also enforce the theorem gate. |
| 11 | Is support mismatch rejected without smoothing? | **Yes.** `BrunoTheoremAssumptionError` names Theorem 1, block identity, and both supports. Shared zeros remain exact; mismatches are never repaired. |
| 12 | Are arbitrary intensities/percentages never assigned pseudo-count semantics? | **Yes.** Genuine totals and integer observations reuse the existing count validators. No conversion or effective-total inference is present; the documented measurement boundary is explicit. |
| 13 | Does any numerical order search avoid claiming exact global optimization? | **Yes.** No order search is implemented. Every certificate reports `global_envelope_certified=False`; the full continuous envelope remains documented separately. |
| 14 | Are raw and combined lower-bound components exposed? | **Yes.** Reverse, forward, and their maximum are separate fields, with retained log components and overflow/underflow diagnostics. Negative reverse values are not clipped. |
| 15 | Are numerical formulas stable near lambda=1 and extreme epsilon? | **Yes.** `expm1`, `log1p`, unchanged near-one orders, stable inherited law kernels, and explicit numerical limits are covered at interior endpoint neighbours, huge orders, and maximum supported counts. |
| 16 | Does an independent finite-support oracle verify bound validity? | **Yes.** Exact Fraction PMFs, independent Decimal full-law sums, and exhaustive deterministic subsets verify 1,323 epsilon/order cases across 27 pairs. All 1,359 oracle tests passed in the guardian rerun. Randomized interpolation and LR-prefix substitution are explicitly excluded. |
| 17 | Are complete flux hypotheses independently feasible and correctly ordered? | **Yes.** Each role passes the existing original-model and native checks before its EMU evaluation. Tests reject incomplete, reordered, unbalanced, and out-of-bounds states and accept feasible states below the FBA objective optimum. |
| 18 | Are experiment/target/replicate identities preserved through the bridge? | **Yes.** Declaration tuples are checked against both native sources, source fingerprints, and the law pair. Tests retain deliberately nonalphabetic experiment/target/replicate order and unequal totals. Shared state IDs are preserved through separate role evaluations. |
| 19 | Is LLR defined as log P1 - log P0? | **Yes.** Count-weighted `log(p1)-log(p0)` follows existing law count/support validation. Independent tests verify sign, infinities, joint zero/zero errors, and small signals at total `10**15`. |
| 20 | Are MFA fitting and biological objectives unchanged? | **Yes.** MFA runtime and dependency declarations match baseline; the bridge performs no fitting and passes no biological objective fraction to the existing observation path. |
| 21 | Is there no composite hypothesis implementation in this task? | **Yes.** Production objects contain exactly two fixed roles. No hypothesis sets, least-favourable pairs, inversion, confidence regions, or composite optimization were added. |
| 22 | Are vendor files unchanged? | **Yes.** Baseline Git diffs are empty and all 95 original vendor/MFA/observation blobs were hash-checked unchanged. PR files are confined to the new testing layer, tests, example, focused documentation, and CI. |
| 23 | Are base imports free of accidental SciPy dependence? | **Yes.** Fresh processes block all optional stacks while executing certificates, LLR, and the complete native example. Base-wheel CI separately verifies SciPy, COBRApy, optlang, mfapy, Matplotlib, and JAX are absent. |
| 24 | Do all required CI/regression checks pass? | **Yes.** Final installed-base-wheel group: 2,002 passed; base MFA/Stage 1: 572 passed with one expected optional-SciPy skip; MFA-extra group: 525 passed. Both pip checks, compileall, examples/snippets, import isolation, and whitespace checks pass. All ten exact-head PR workflows and both new Python 3.11/3.12 jobs succeeded; the lead verified all 19 push/PR runs. |
| 25 | Does the documentation state exactly what the certificate means and what it does not mean? | **Yes.** The guide states a lower bound on optimal deterministic Type II under the explicit Type-I budget for a fixed pair/order; it excludes actual or sample-specific errors, p-values, confidence levels, achievable-error claims, and uncertified global-envelope claims. Count semantics and deferred scope are explicit. |

The independent CI inspection includes the successful
[new certificate workflow](https://github.com/esig626/fluxemu-standalone/actions/runs/34024988179),
whose Python 3.11 and 3.12 jobs both passed wheel installation, optional-stack
absence, whitespace, compilation, all testing/observation tests, and the native
example. Existing observation, MFA, Stage 1, compatibility, and forward gates
also succeeded. Their execution does not add any deferred implementation.

This audit is recorded separately from runtime implementation. Final task
closure verifies the resulting audit-only successor commit, its clean pushed
state, the PR head, and its CI; the guardian then emits
`SIMPLE-TEST-GUARDIAN: FINAL APPROVE` for that exact SHA without another
repository edit. A runtime or scientific-documentation change would require
renewed review rather than inheriting this approval.
