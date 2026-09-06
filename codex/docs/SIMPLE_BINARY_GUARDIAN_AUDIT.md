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
| Milestone B | APPROVE | Concrete full-law formulas, directionality, support gate order, numerical diagnostics, and independent oracle reviewed. Guardian reran schemas/numerics: 246 passed; final oracle: 1,359 passed in 44.48 s. Oracle covers 27 law pairs, 1,323 epsilon/order comparisons, 7,032 deterministic rejection subsets, and 49,224 exact budget checks. Whitespace check is clean; vendor, MFA, and observation source are unchanged. Lead may commit and push this milestone. |
| Milestone C | BLOCKED | Awaiting stationary bridge and synthetic flux-pair evidence. |
| Milestone D / final PR | BLOCKED | Awaiting full regression, documentation, CI, final committed/pushed head, PR, and the explicit final audit. |

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
