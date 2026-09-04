# Codex task: finish FluxEMU Stage 1 native engine

Repository: `esig626/fluxemu-standalone`

Work on branch: `codex/stage1-native-completion`

This branch starts from `codex/milestone-1-5-native-vffva` and therefore already contains the unmerged native VFFVA-style HiGHS implementation from PR #14. **Do not restart or reimplement that milestone from scratch.** Audit it, finish its performance acceptance, then continue through native feasible-state sampling and sample-to-EMU integration until Stage 1 is complete.

Read `AGENTS.md` first and obey it exactly.

Then execute this prompt in full.

---

## 0. Scope and stopping condition

This task ends at **Stage 1 only**.

The required final native pipeline is:

```text
CBM / canonical FluxModel
        -> native FBA
        -> VFFVA-class fast native FVA
        -> complete jointly feasible flux-state sampling under the retained biological objective
        -> native stationary EMU
        -> MID ensemble
```

Stage 1 is complete only when the repository has a tested, documented, public native path that can execute that sequence without requiring COBRApy or mfapy at runtime.

Do **not** implement:

- inverse MFA;
- parameter fitting;
- profile likelihoods;
- confidence intervals for MFA;
- Monte-Carlo MFA uncertainty;
- information theory;
- KL divergence;
- Rényi divergence;
- hypothesis testing;
- paper analysis;
- biological interpretation beyond what is needed for software fixtures.

Do not create exploratory research in this standalone repository.

---

## 1. Non-negotiable scientific invariants

Preserve all existing repository rules and especially these invariants:

1. FVA bounds are independent endpoint optima. Never assemble minima/maxima into a putative complete flux state.
2. Sampling must return **complete jointly feasible flux vectors**.
3. Never sample each reaction independently from its FVA interval.
4. Every sampled state must preserve canonical reaction order exactly.
5. Every sampled state must be independently validated for:
   - finite values;
   - reaction bounds;
   - balanced-metabolite steady-state constraints `S v = 0`;
   - retained biological objective constraint;
   - exact reaction membership/order.
6. The CBM biological objective remains the linear objective in `FluxModel`. Do not repurpose it as a statistical fitting objective.
7. Atom mappings remain explicit and authoritative. Do not infer them from stoichiometry.
8. The standard Stage 1 runtime must remain native FluxEMU + HiGHS. COBRApy may be used only as a compatibility/parity oracle in tests or benchmarks.
9. Do not add CPLEX, GLPK, MPI/OpenMPI, the VFFVA binary, or the original VFFVA C code as production dependencies.
10. Preserve declared scientific ordering everywhere.
11. Do not silently clip, repair, or normalize invalid flux states.
12. Do not claim that a sampler is uniform, representative, or biologically probabilistic unless the implementation and evidence justify that statement. Record the actual sampling algorithm and its provenance instead.

---

## 2. Existing state you must reuse

Before editing, inspect the current branch and verify these facts rather than assuming them:

- `main` contains the cold correctness-reference native HiGHS FVA path.
- this branch contains `run_highs_vffva(...)` with worker-local reusable HiGHS models, objective switching, simplex reuse, and dynamic process scheduling;
- the cold reference must remain available as a permanent correctness oracle;
- PR #14 already has correctness/structural VFFVA tests and green CI;
- native stationary EMU already accepts a batch of `CanonicalFluxState` records;
- the older COBRA compatibility layer already demonstrates the intended semantics of complete feasible-state sampling, but production Stage 1 must not depend on COBRApy;
- the deterministic native stationary orchestration currently feeds one complete FBA optimum into EMU and does not yet expose a native sampled ensemble path.

Read at minimum:

```text
AGENTS.md
codex/docs/PHASE3B_NATIVE_HIGHS_FLUX_ANALYSIS.md
codex/docs/PHASE3B_NATIVE_FLUX_TO_MID.md
codex/docs/KNOWN_LIMITATIONS.md
codex/docs/API_MAP.md
codex/src/fluxemu/flux_analysis/highs.py
codex/src/fluxemu/flux_analysis/results.py
codex/src/fluxemu/analysis/stationary.py
codex/src/fluxemu/emu/stationary.py
codex/src/fluxemu/execution.py
codex/src/fluxemu/cobra_analysis.py
codex/tests/test_native_highs_flux_analysis.py
codex/tests/test_native_highs_vffva.py
codex/tests/test_native_highs_cobra_parity.py
codex/tests/test_native_stationary_analysis.py
codex/tests/test_cobra_analysis.py
codex/benchmarks/benchmark_highs_fva_reference.py
```

Also inspect upstream VFFVA conceptually, especially `marouenbg/VFFVA/lib/veryfastFVA.c` and its documented dynamic scheduling strategy. Use the **algorithmic ideas**, not copied source code.

---

# 3. Mandatory multi-agent execution

Use multiple agents. Do not perform this as a single monolithic pass.

The lead agent owns integration, branch hygiene, commits, and final acceptance. Spawn specialist agents with non-overlapping primary responsibilities whenever possible.

At the beginning, create at least the following agents.

## Agent A — VFFVA correctness/performance auditor

Responsibilities:

- audit the existing `run_highs_vffva(...)` implementation rather than rewriting it reflexively;
- compare it against the cold reference and COBRA parity oracle;
- inspect whether the production orchestration redundantly recompiles the LP or re-solves the biological FBA optimum;
- inspect dynamic scheduling granularity, objective switching, worker-local solver reuse, and error propagation;
- identify any correctness or performance defects that must be fixed before acceptance;
- design the performance benchmark and return concrete recommendations plus tests.

This agent must distinguish hard correctness gates from noisy timing measurements.

## Agent B — native sampling scientist/architect

Responsibilities:

- define the mathematically correct feasible region under the retained objective;
- review the older COBRA ACHR/OptGP path only as a semantic oracle;
- propose the simplest native sampling algorithm that produces complete feasible states without adding COBRApy/mfapy runtime dependencies;
- explicitly handle:
  - `fraction_of_optimum == 1` optimal-face sampling;
  - `fraction_of_optimum < 1` objective-inequality sampling;
  - maximization and minimization objectives;
  - fixed/blocked reactions;
  - reversible signed reactions supported by the canonical model;
  - zero-dimensional feasible regions;
  - unbounded sampled directions/regions;
  - deterministic seeds and provenance;
- assess whether to use native ACHR-style sampling, hit-and-run, or another justified Markov-chain method;
- explain what can and cannot be claimed about its sampling law/mixing.

The goal is not theoretical novelty. The goal is a robust, explicit native sampler suitable for synthetic experiments.

## Agent C — Stage 1 pipeline/integration architect

Responsibilities:

- design the clean public API for:

```text
FBA -> fast FVA -> sampled CanonicalFluxState batch -> stationary EMU -> MID ensemble
```

- preserve the existing deterministic one-FBA-state path for compatibility unless there is a compelling reason not to;
- avoid unnecessary repeated model compilation and repeated biological-objective solves in the composed Stage 1 path;
- define result/provenance records and public exports;
- ensure the native sampler feeds `evaluate_stationary(...)` through complete canonical states, never through FVA intervals.

## Agent D — acceptance/benchmark engineer

Responsibilities:

- design hard correctness gates for FVA and sampling;
- design end-to-end Stage 1 tests;
- extend benchmarks to compare cold native FVA vs reusable native FVA and, where useful, COBRApy as a non-production comparator;
- make performance evidence reproducible without making CI brittle;
- identify representative small, medium, and real-model fixtures already present in the repository.

## Agent E — documentation/portability reviewer

Responsibilities:

- audit dependency boundaries and clean-install behavior;
- ensure documentation clearly separates:
  - FVA bounds;
  - complete sampled flux states;
  - the sampler's algorithm/provenance;
  - one deterministic FBA-state MID from a sampled MID ensemble;
- ensure no stale docs still claim that the VFFVA layer or native sampling are future work once they are complete;
- inspect CI and public exports.

---

# 4. Mandatory HYPE / FINISHER agent

Create one persistent agent named conceptually **HYPE-FINISHER**, **Completion Marshal**, or equivalent.

This agent does not merely summarize progress. Its job is to **push every other agent to finish adequately**.

The Completion Marshal must:

1. maintain a live acceptance checklist for all Stage 1 requirements;
2. read every specialist-agent report critically;
3. reject vague statements such as "mostly done", "appears correct", "should work", or "future work" when the item is required for Stage 1;
4. send incomplete tasks back for additional work through the lead agent;
5. look specifically for:
   - TODOs left in required code paths;
   - missing tests;
   - skipped acceptance cases;
   - unsupported performance claims;
   - accidental COBRA/mfapy runtime dependencies;
   - FVA endpoints being confused with complete flux states;
   - redundant FBA solves or LP rebuilds;
   - sampler states that are not independently validated;
   - missing provenance;
   - docs that overclaim uniformity or mixing;
   - missing CI coverage;
6. insist that agents produce evidence: tests, benchmark output, code references, and failure-mode coverage;
7. keep the team focused on **finishing Stage 1**, not expanding into Stage 2 or research ideas;
8. perform a final adversarial audit before the lead agent may declare completion.

The Completion Marshal is allowed to identify and patch small integration defects itself if that is faster than reassigning them, but its primary role is completion pressure and quality control.

The lead agent may not declare Stage 1 complete until the Completion Marshal explicitly reports that every hard acceptance item is satisfied or clearly documents a genuine blocker. Do not downgrade acceptance criteria merely to obtain a green report.

---

# 5. Milestone 1 — finish native VFFVA/FastFVA acceptance

Start by auditing the existing branch implementation.

## Required behavior

The production fast FVA implementation must preserve the same mathematical problem as the cold reference:

For a maximizing biological objective with optimum `z*` and retained fraction `f`, every endpoint solve must respect

```text
c^T v >= f z*
```

For minimization, preserve the corresponding upper retained-objective constraint.

Every endpoint must be independently validated.

The fast implementation must:

- compile the canonical LP once per top-level analysis where practical;
- solve the biological optimum once per composed Stage 1 run where practical;
- create no more than one reusable FVA solver/model per worker;
- change endpoint objective coefficients/sense cheaply;
- reuse HiGHS simplex state naturally within each worker;
- use HiGHS internal threads = 1 when using process-level parallelism;
- dynamically distribute work so slow endpoints do not strand other workers;
- restore results to exact canonical reaction order independent of completion order;
- expose contextual failures containing reaction and min/max direction.

Do not require explicit basis serialization if retaining the same `Highs` object already preserves the basis correctly.

## Correctness acceptance

At minimum, fast FVA must agree with the cold native reference within existing solver tolerances on:

- the dedicated synthetic VFFVA fixture;
- all existing repository SBML parity models that can be projected to the canonical flux model;
- fractional optimum cases;
- maximization;
- minimization;
- multi-term biological objectives;
- fixed reactions;
- blocked reactions;
- signed/reversible reactions where supported;
- serial and parallel execution;
- deterministic output ordering.

Retain COBRApy FVA parity tests as an optional compatibility oracle where already used.

## Performance acceptance

Create or extend a benchmark that records, after warm-up and with repeated measurements:

```text
cold native FVA
fast native FVA, workers=1
fast native FVA, workers>1 where available
optional COBRA processes=1 comparator
```

Record model size, worker count, Python/HiGHS version where practical, median wall time, and speedup ratios.

Do not make a fragile wall-time ratio a CI hard gate.

However, Stage 1 must not be declared complete if the "fast" implementation is not materially faster on at least one representative nontrivial model. As a practical target, demonstrate at least about a 2x median speedup of reusable serial FVA over the cold rebuild path on a model large enough that cold FVA timing is meaningful, and demonstrate that process-level dynamic scheduling does not break correctness. If the initial implementation fails to produce a material speedup, profile and fix it rather than documenting the failure as completion.

External VFFVA+CPLEX may be benchmarked if the environment already supports it, but it is not required and must not become a runtime or CI dependency.

## Milestone commit

Run focused tests. Commit and push a coherent milestone immediately once FastFVA correctness and benchmark infrastructure are complete.

Suggested commit theme:

```text
Finish native VFFVA-class FVA acceptance
```

Do not begin substantial sampling implementation with this work left uncommitted.

---

# 6. Milestone 2 — implement native complete feasible-state sampling

This is the main missing Stage 1 capability.

## Feasible region

For a maximizing biological objective, define the sampled region from the canonical flux model as

```text
S v = 0
l <= v <= u
c^T v >= f z*
```

with the appropriate reversed objective inequality for minimization.

At `f = 1`, the admissible region is generally an optimal face. Handle that face deliberately and numerically robustly rather than pretending it is a full-dimensional interior polytope.

## Required public behavior

Implement a native sampler API under the native flux-analysis package.

The exact naming is up to the implementation, but the result must include at least:

- ordered complete `CanonicalFluxState` records or a lossless equivalent readily converted to them;
- sample count;
- algorithm name;
- seed;
- retained fraction;
- biological optimum;
- retained objective bound;
- objective direction;
- model/reaction order identity or fingerprint;
- validation diagnostics/provenance;
- algorithm parameters required to reproduce the run, such as burn-in/thinning where applicable.

If a warm-up construction is required, reuse existing native LP/FVA infrastructure where sensible rather than rebuilding redundant solver state blindly.

## Sampling semantics

The algorithm must sample **jointly feasible complete states**.

It may use ACHR-style sampling, hit-and-run, or another justified method, but:

- document the actual stationary target/heuristic honestly;
- do not call it exact-uniform if it is not exact-uniform;
- do not claim mixing quality that is not tested;
- preserve deterministic reproducibility for a fixed seed where the algorithm/environment allows it;
- never create states by independently drawing reaction coordinates from FVA ranges.

## Edge cases

Handle explicitly and test:

- zero-dimensional feasible region / unique admissible flux state;
- optimal-face sampling at fraction 1;
- fractional objective region;
- maximizing objective;
- minimizing objective;
- blocked/fixed variables;
- signed bounded reactions;
- invalid count/seed/fraction;
- unbounded sampling regions or directions: raise a clear error rather than inventing an arbitrary truncation;
- numerically degenerate intervals;
- inability to generate a valid next state.

## Independent validation

Every returned sample must pass a FluxEMU-owned validator independent of the sampler's own internal assumptions.

The validator must report enough diagnostics to locate:

- lower-bound violation;
- upper-bound violation;
- maximum mass-balance residual;
- retained-objective violation;
- malformed reaction membership/order;
- non-finite values.

Reuse existing validation logic where appropriate, but keep the native path independent of COBRApy.

## Sampling acceptance tests

At minimum require:

1. all samples independently valid on a low-dimensional analytical toy;
2. deterministic replay for a fixed seed where promised;
3. distinct samples when the feasible region has positive dimension;
4. correct unique-state behavior on a zero-dimensional region;
5. correct fraction-1 optimal-face behavior;
6. correct fraction-below-1 objective inequality behavior;
7. correct minimization semantics;
8. exact canonical reaction order;
9. no COBRA/mfapy import on the native runtime path;
10. successful sampling on at least one nontrivial repository canonical model.

Statistical quality checks may be added, but do not substitute them for hard feasibility validation.

## Milestone commit

Run focused tests. Commit and push immediately once the native sampling engine and its direct tests are coherent.

Suggested commit theme:

```text
Add native feasible flux-state sampling
```

---

# 7. Milestone 3 — integrate sampled states with native stationary EMU

Build a clear public Stage 1 orchestration path.

The required user-level sequence is:

```text
canonical model
-> FBA result
-> fast FVA result
-> sampled complete flux states
-> one compiled stationary EMU plan
-> MID predictions for every sampled state
```

Do not compile the EMU topology separately for every sample.

Do not rebuild the canonical model per sample.

Do not transform FVA bounds into flux states.

Use the existing batch-capable native stationary EMU evaluator.

## API requirements

Preserve the existing deterministic single-FBA-state stationary analysis unless breaking it is demonstrably necessary.

Prefer adding an explicit ensemble/sampling path or an optional sampling mode whose semantics are unmistakable.

The returned Stage 1 result should make it impossible to confuse:

- the biological FBA optimum;
- FVA endpoint ranges;
- the sampled complete flux states;
- MID predictions conditional on each sampled state.

Sample IDs must propagate from flux states to MID results exactly.

## Efficiency requirement

Audit the composed pipeline for redundant work.

In particular, avoid unnecessarily:

- compiling the same canonical LP repeatedly;
- solving the biological objective twice just because FBA and FVA are separate convenience APIs;
- constructing the retained-objective geometry twice when FVA and sampling can safely share prepared information;
- recompiling EMU topology per sample.

Do not over-engineer a large framework merely to remove negligible work, but eliminate clear repeated expensive work.

## End-to-end acceptance

Add a deterministic synthetic test that executes:

```text
FluxModel / CanonicalModel
-> native FBA
-> native fast FVA
-> N complete feasible native samples
-> native stationary EMU
-> N sample-indexed MID predictions
```

Require:

- all samples pass native feasibility validation;
- all MIDs pass the existing MID normalization/nonnegativity checks;
- sample IDs are preserved;
- number of MID records matches sample count x requested targets;
- rerunning with the same seed reproduces the expected deterministic outputs where promised;
- no COBRA or mfapy runtime import is required;
- FVA ranges are not used as EMU inputs.

Also run an end-to-end acceptance on at least one existing nontrivial repository model/experiment for which native stationary EMU already works.

## Milestone commit

Run focused tests. Commit and push immediately.

Suggested commit theme:

```text
Integrate native flux ensembles with stationary EMU
```

---

# 8. Milestone 4 — Stage 1 acceptance, docs, CI, cleanup

After the three implementation milestones are pushed, perform a dedicated acceptance pass.

## CI

Add/extend CI so that Stage 1 native coverage includes:

- cold native FVA oracle tests;
- fast native FVA tests;
- fast-vs-reference parity;
- native sampling tests;
- native sample-to-MID end-to-end tests;
- native import-isolation/clean-install checks;
- existing stationary and real-model acceptance gates.

Do not add performance timing thresholds that will create flaky CI.

## Documentation

Create/update a Stage 1 document that explains the final supported native workflow and exact guarantees.

It must state clearly:

```text
FVA = reaction-wise extrema, not a joint state
sampling = complete jointly feasible states under a declared algorithm
EMU = maps each complete flux state to MIDs
```

Document:

- FastFVA architecture;
- correctness oracle;
- retained objective semantics;
- parallel worker model;
- sampling algorithm and provenance;
- validation tolerances;
- fraction-1 optimal-face behavior;
- limitations on claims about mixing/representativeness;
- public APIs;
- dependency boundary;
- benchmark invocation;
- Stage 1 end-to-end example.

Remove or update stale text saying that the VFFVA layer or native feasible sampling are future milestones.

## Benchmark evidence

Run the performance benchmark and preserve a concise machine-readable result or documented terminal result sufficient to show that the reusable implementation is materially faster than the cold reference in the execution environment.

Do not fabricate a speedup if the environment cannot establish one. If performance is unexpectedly poor, investigate before declaring completion.

## Full regression

Run all relevant focused suites and then the complete repository test suite that is feasible in the environment.

If optional compatibility dependencies are unavailable, native Stage 1 tests must still run. Clearly separate genuinely optional skipped parity tests from required native tests.

---

# 9. Completion Marshal final audit

Before finalizing, the Completion Marshal must independently inspect the branch and answer all of the following with evidence:

```text
[ ] Fast FVA gives the same bounds as the cold trusted native reference.
[ ] Existing COBRA parity remains intact where applicable.
[ ] Fast FVA reuses worker-local LPs rather than rebuilding every endpoint.
[ ] Dynamic parallel scheduling is implemented and tested.
[ ] Fast FVA is materially faster than cold FVA on a meaningful benchmark.
[ ] Native complete feasible-state sampling exists.
[ ] No reaction-wise independent FVA interval sampling occurs.
[ ] Every sampled complete state is independently validated.
[ ] Fraction-of-optimum semantics are correct for max and min objectives.
[ ] Fraction 1 / optimal-face behavior is handled deliberately.
[ ] Sampling provenance is returned and documented.
[ ] Native sampling does not require COBRApy or mfapy.
[ ] Sampled CanonicalFluxState records feed the existing native stationary EMU evaluator.
[ ] EMU topology is compiled once for a sampled batch.
[ ] Sample IDs propagate to MIDs.
[ ] End-to-end CBM -> FBA -> fast FVA -> feasible states -> EMU -> MIDs passes.
[ ] Clean native dependency boundaries remain intact.
[ ] No inverse MFA has been introduced.
[ ] No information-theory research has been introduced.
[ ] Documentation reflects the final implementation rather than future-work language.
[ ] Required tests and CI are green.
[ ] All coherent milestones are committed and pushed.
```

If any required box is false, Stage 1 is not complete. Continue working.

---

# 10. Git and execution discipline

Follow `AGENTS.md` exactly.

Additionally:

- work only on `codex/stage1-native-completion`;
- inspect the branch history before editing;
- preserve the existing VFFVA work rather than rebuilding it;
- do not rewrite unrelated code;
- keep implementation under `codex/src/fluxemu/`;
- keep `vendor/` read-only;
- add focused tests before or with each implementation milestone;
- run focused tests before every implementation commit;
- commit and push each coherent milestone immediately;
- do not leave substantial work only in the working tree;
- do not squash the entire Stage 1 implementation into one final commit;
- if a test exposes a scientific-semantic defect, fix the implementation rather than weakening the test;
- if an agent proposes expanding into MFA or information theory, stop that expansion and return to Stage 1 acceptance.

---

# 11. Final response required from Codex

When—and only when—the Completion Marshal confirms all hard Stage 1 acceptance items, provide a concise final report containing:

1. branch and final commit SHA;
2. commits created during this task;
3. exact public Stage 1 API/path now supported;
4. FastFVA correctness evidence;
5. performance benchmark table or concise results;
6. native sampling algorithm and guarantees/limitations;
7. sample-to-EMU end-to-end acceptance evidence;
8. test/CI summary;
9. any remaining limitations that do **not** prevent Stage 1 completion;
10. explicit statement that MFA and information-theoretic work were not started.

The final statement should be equivalent to:

```text
Stage 1 is complete: FluxEMU can natively execute
CBM -> FBA -> fast FVA -> complete feasible flux-state sampling -> stationary EMU -> MID ensembles.
```

Do not make that statement if the performance acceptance, native sampler, or end-to-end sampled MID path is incomplete.
