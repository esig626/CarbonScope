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
| Final PR | BLOCKED | Awaiting committed D PR/head CI, the explicit 25-point audit, and verification of the final audit-only commit and PR state. |

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

No final approval has been issued.

The initial bridge design received BLOCKED because batching both hypotheses
would reject valid identical/shared sample IDs in the existing native batch
validator. Separate role-specific evaluation resolved the issue before
implementation; no scientific identifier was renamed.
