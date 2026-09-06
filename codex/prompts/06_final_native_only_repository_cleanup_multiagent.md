# Task 06 — Final native-only FluxEMU repository clean-up and release hardening

## Mission

Turn `esig626/fluxemu-standalone` into a clean, native-only, publication-ready software repository.

This is a **repository finalisation and hardening task**, not a new scientific-method task.

The repository already contains the scientific capabilities that must survive this task:

1. canonical model / SBML ingestion;
2. native HiGHS FBA;
3. native FVA and the VFFVA-style reusable/shared-memory FastFVA architecture;
4. complete feasible-state sampling (never independent FVA-coordinate sampling);
5. native stationary EMU;
6. native transient forward EMU;
7. stationary MFA with exact KL / finite positive-order Rényi objectives and explicit `normalise_mid(...)` preprocessing;
8. genuine-count multinomial observation laws;
9. simple-vs-simple likelihood-ratio evaluation;
10. exact simple-null likelihood-ratio p-values for enumerable multinomial spaces;
11. finite-order Bruno Type-II lower bounds with the fixed null/alternative and Rényi-direction conventions;
12. the native E. coli real-model acceptance programme;
13. the authoritative carbon-transition library and all scientifically required atom-mapping data.

The target end-state is a repository that no longer looks like a migration workspace and no longer carries mfapy/COBRApy compatibility baggage.

---

## Repository and branch

Repository:

`esig626/fluxemu-standalone`

Work only on:

`codex/final-native-only-repository-cleanup`

This branch was created from the merged `main` baseline after PR #19. Baseline scientific main commit:

`b72622d350c06443e9b3f23811919b5283fe8008`

Read `AGENTS.md` first and obey all scientific invariants in it. If this task deliberately changes the repository layout, update `AGENTS.md` in the same coherent commit so its path rules remain correct.

Do not modify `main` directly.

---

## Scope boundaries

### This task MAY

- remove obsolete files and directories;
- remove superseded compatibility APIs;
- remove vendored third-party source trees;
- remove obsolete tests, fixtures, workflows, migration archives, prompts and internal audit artefacts;
- reorganise the repository into a conventional Python package layout;
- rewrite current-facing documentation;
- consolidate CI;
- rename the current Bruno result record from certificate terminology to bound terminology;
- remove pre-release compatibility aliases if they exist only to preserve names introduced during the recent development sequence;
- simplify dependencies;
- strengthen native-only import and packaging tests.

### This task MUST NOT

- add composite hypothesis testing;
- add test inversion or flux confidence/compatibility regions;
- add Bayesian inference;
- add new observation-law models;
- change the stationary MFA objective;
- change the mathematics of the Bruno Type-II lower bound;
- add a Rényi-order grid or claim a finite grid solves the continuous order envelope;
- change the exact p-value definition;
- introduce pseudocounts, clipping, support repair, or effective-sample-size inference;
- weaken native FBA/FVA/EMU validation;
- infer atom mappings;
- independently sample FVA coordinates;
- change authoritative frozen scientific data merely to make tests pass;
- rewrite Git history or force-push.

---

# Non-negotiable scientific invariants

Preserve all of the following exactly.

## Simple binary testing convention

- `H0 = P0 = null`
- `H1 = P1 = alternative`
- Type I = `P0(decide H1)`
- Type II = `P1(decide H0)`
- reverse Rényi = `D_lambda(P1 || P0)`
- forward Rényi = `D_lambda(P0 || P1)`
- supplied finite real `lambda > 1` is used unchanged subject only to explicit numerical representability limits;
- no finite Rényi-order grid and no false claim of a globally optimised continuous-order envelope.

## Exact p-value convention

For a realised genuine-count observation `y`, preserve

`p(y) = P0{ LLR(Y) >= LLR(y) }`

with

`LLR(y) = log P1(y) - log P0(y)`.

This remains an exact discrete simple-null tail when the supported null outcome space can be enumerated. If the configured exact-enumeration limit is exceeded, fail explicitly. Do not silently substitute chi-square, Monte Carlo, saddlepoint or another approximation.

## Observation-law semantics

- genuine counts only;
- no conversion of peak areas, arbitrary intensities, percentages or normalised MIDs into pseudo-counts;
- exact support semantics;
- exact structural zeros;
- no pseudocounts;
- no probability clipping;
- independent product blocks only when independence is explicitly declared.

## Flux geometry

- FVA extrema are diagnostics, not jointly feasible flux vectors;
- complete feasible states are sampled from the constrained region;
- no independent coordinate sampling;
- canonical reaction order is preserved.

## EMU / mapping semantics

- preserve explicit authoritative atom mappings and declared direction/projection semantics;
- do not infer mappings from stoichiometry, names or formulas;
- preserve real-model projection and direction-activity logic;
- preserve native stationary and transient numerical checks.

---

# Required execution style

Use multiple focused agents/review passes where useful, but keep one coherent implementation owner.

At minimum perform these independent review roles before the final PR:

1. **Legacy-removal auditor** — identifies everything that exists only for mfapy/COBRApy/migration compatibility and checks no native dependency is accidentally removed.
2. **Native scientific-invariant auditor** — verifies the current native scientific paths and fixed testing conventions remain unchanged after structural cleanup.
3. **Packaging / CI / documentation auditor** — checks installability, root layout, links, British English in current-facing prose, dependency boundaries and redundant workflows.
4. **Final guardian** — independently examines the final diff and all test evidence and blocks completion if any scientific capability, native validation gate or exact testing semantic has regressed.

Do not use agents as a substitute for running the actual test suite.

Commit each coherent layer before moving to the next substantial layer and push it immediately.

---

# Stage 0 — Inventory before edits

Before deleting anything, build an explicit dependency/import inventory.

Use repository-native tools such as `git grep`, Python AST/import inspection and test collection to determine:

- current public native APIs;
- current CLI path;
- every module imported by the native pipeline;
- every test that exercises native production code;
- every file that depends on `cobra`, `cobrapy`, `mfapy`, `optlang`, `nlopt`, `vendor/`, `FLUXEMU_MFAPY_SOURCE`, or the old compatibility bridge;
- every workflow that installs compatibility extras or points at `vendor/`;
- every current documentation page that still presents compatibility or certificate terminology as current;
- whether any native source was actually copied/adapted from third-party source in a way that requires retained attribution/licence text.

Do not delete attribution that is legally required. If attribution is needed, retain the smallest correct licence/NOTICE/acknowledgement mechanism; do **not** retain an entire vendored source tree merely for attribution.

Record the inventory in the PR description or a temporary working note. The finished repository should not retain an internal migration-audit document solely because this task created it.

---

# Stage 1 — Remove the vendored and compatibility subsystem

The finished repository must have **no `vendor/` directory**.

Delete the vendored COBRApy and mfapy trees in their entirety after the inventory confirms no native code depends on them.

The current repository also contains a superseded compatibility surface outside `vendor/`. Audit and remove the whole subsystem rather than leaving dead wrappers.

Known legacy candidates include, but are not limited to:

- `codex/src/fluxemu/_mfapy.py`
- `codex/src/fluxemu/backends/`
- `codex/src/fluxemu/cobra_analysis.py`
- `codex/src/fluxemu/cobra_to_mfapy.py`
- `codex/src/fluxemu/compat/`
- `codex/src/fluxemu/forward.py`
- `codex/src/fluxemu/mfapy_model.py`

Also audit these older modules and remove them **only if** the import graph proves that they are migration/compatibility-only and not part of the native public/runtime path:

- `configuration.py`
- `flux_conversion.py`
- `isotope_metadata.py`
- `model_identity.py`
- `output.py`
- `validation.py`
- `toy.py`

Do not remove a module merely because it is old. Prove it is dead relative to the retained native package/tests first.

After deleting the compatibility modules:

- remove dead imports and exports;
- remove now-unused exception classes;
- remove compatibility-only helper types;
- remove compatibility-only environment variables;
- remove any runtime fallback that searches a vendored mfapy tree.

The final installed `fluxemu` package must not import or require COBRApy, mfapy, optlang or nlopt.

---

# Stage 2 — Remove compatibility-only tests and fixtures while preserving independent validation

Delete tests whose only purpose is to validate the retired compatibility bridge or the vendored packages.

Known candidates include:

- `test_cobra_analysis.py`
- `test_cobra_flux_projection.py`
- `test_cobra_to_mfapy.py`
- `test_mfapy_forward_regression.py`
- `test_mfapy_inverse_api.py`
- `tests/fixtures/official_mfapy_example_0.json`
- `test_native_highs_cobra_parity.py` if it is now solely an external COBRA oracle rather than a necessary native correctness gate.

Audit additional parity tests carefully.

Important distinction:

- **remove mfapy/COBRApy parity machinery**;
- **retain independent mathematical or literature-based validation**.

For example, if the Antoniewicz examples contain an independent direct-isotopomer solver or published frozen reference MIDs that validate the native EMU implementation without importing mfapy, retain those native-independent validations. If a test mixes an independent oracle with mfapy parity, split/rewrite it so the useful independent check survives without mfapy.

The final test suite should validate FluxEMU against its mathematics, explicit fixtures, published examples and independent solvers — not against a permanently bundled legacy runtime.

---

# Stage 3 — Remove migration workspace artefacts

The repository should no longer look like an internal migration workspace.

Audit and remove:

- `codex/handoff/` and its migration archive/sidecar, once native packaged real-model data and current acceptance tests no longer depend on it;
- old Codex task prompts once their work is fully represented in current code/docs;
- this Task 06 prompt itself at the very end, once the task has been read and executed;
- obsolete guardian/acceptance process documents that are not useful user-facing or scientific reproducibility material;
- superseded compatibility audits and migration reviews.

Do **not** remove authoritative scientific fixtures or reproducibility evidence that still directly validates the retained native implementation.

For the E. coli real-model acceptance path, remove any dependency on the historical handoff archive. The acceptance gate should run from the native packaged model/isotope/projection/activity data and frozen current references that are already in the repository.

---

# Stage 4 — Convert the repository to a conventional root layout

Unless the Stage 0 dependency audit finds a concrete reason not to, remove the historical `codex/` wrapper directory and make the repository a normal Python project.

Target layout:

- `pyproject.toml`
- `src/fluxemu/`
- `tests/`
- `docs/`
- `examples/`
- `benchmarks/`
- `results/` or another clearly named validation-results directory if those frozen outputs remain necessary
- `tools/`
- `.github/workflows/`
- `README.md`
- `AGENTS.md`

Move current native material from `codex/...` to the corresponding root path and update every import-independent path reference, workflow, documentation link, test fixture path and package-data rule.

Do not use copy-and-leave-behind. The final repository should contain one canonical location for each retained file.

Update `AGENTS.md` so future Codex work targets `src/fluxemu/` rather than `codex/src/fluxemu/`, and retain the important scientific invariants in concise form.

After flattening, installation should be the standard:

`python -m pip install .`

and optional scientific extras should be installable from the root, e.g. `.[mfa]` and `.[transient]`.

---

# Stage 5 — Dependency cleanup

Update `pyproject.toml` for the native-only package.

Remove compatibility-only dependency groups, especially:

- `compat`
- `mfapy`

Remove any dependency that becomes unused after compatibility deletion.

Retain only dependencies genuinely used by the native package and optional native features.

Expected retained categories include:

- NumPy;
- pandas if still genuinely used by retained public/native functionality;
- PyYAML if still required by the native input format;
- `highspy`;
- `python-libsbml`;
- SciPy only in the native optional features that actually need it (`mfa`, `transient`).

Do not add replacement libraries just to reproduce legacy behaviour.

Ensure package data declarations still include the authoritative carbon-transition YAML and native real-model JSON data.

Keep `src/fluxemu/__version__` and `pyproject.toml` version consistent. Do not invent a major release number or publish a package/tag in this task unless explicitly required to repair an existing inconsistency.

---

# Stage 6 — Bruno terminology clean-up

The user has explicitly requested that the Type-II results **not be called certificates**.

Make the current implementation consistently use **bound** terminology.

Required outcome:

- the actual canonical public result class is `BrunoOrderBound`;
- docstrings and user-facing diagnostics call it a Type-II lower bound;
- tests use bound terminology;
- current documentation uses bound terminology;
- fingerprints/schema strings should not continue using `certificate` merely because PR #18 originally did, unless changing the fingerprint would violate a genuine persisted-data contract. No such contract should be assumed — investigate it.

The current pre-release alias `BrunoOrderCertificate` may be removed if it serves only recent development compatibility and is not required by a published stable API. Prefer a clean final API over carrying pre-release terminology forever.

Do not indiscriminately rename unrelated scientific concepts whose formal name genuinely contains “certificate” (for example a direction-activity certificate, if that remains part of the canonical model). The instruction applies to the Bruno/testing **bound** terminology.

Keep `bruno_converse_at_order(...)` if the name remains mathematically appropriate; the issue is “certificate”, not “converse”.

---

# Stage 7 — Documentation clean-up

Rewrite the current-facing docs so they describe the software that exists now, not the migration history.

At minimum:

## README

Make `README.md` an actual public software README with:

- what FluxEMU does;
- the native pipeline;
- installation from repository root;
- a minimal stationary forward example/CLI invocation;
- stationary MFA capability;
- genuine-count observation-law/testing capability;
- exact p-value and Rényi Type-II bound distinction;
- links to the principal current docs;
- explicit scientific boundaries/limitations;
- acknowledgements/attribution only where actually required.

Do not lead with historical compatibility information.

## Known limitations

Rewrite `KNOWN_LIMITATIONS.md` so it is factually current.

It currently contains stale statements that p-values are not implemented and uses superseded certificate terminology. Fix all of that.

Keep genuine current limitations such as:

- local/multistart stationary MFA optimisation not proving a global optimum;
- no composite testing/inversion in this standalone repository;
- exact p-value enumeration limit;
- multinomial count semantics applying only to genuine counts;
- no native natural-abundance correction if still true;
- finite hit-and-run samples not proving mixing/independence;
- transient inverse MFA not implemented if still true;
- any current SBML metadata limitations.

Remove limitations that refer only to deleted compatibility stacks.

## API / architecture docs

Rewrite or replace `API_MAP.md` so it maps the **native** API only. Remove the huge historical COBRApy/mfapy source audit from current architecture docs.

Delete obsolete docs such as, where confirmed superseded:

- `MFAPY_ENGINEERING_COMPARISON.md`
- `MFAPY_OBSERVATION_SEMANTICS_AUDIT.md`
- `MFAPY_PATCH.md`
- `LEGACY_MODEL_REVIEW.md`
- `SIMPLE_BINARY_RENYI_CERTIFICATES.md`

Retain the new `SIMPLE_BINARY_RENYI_BOUNDS.md` (updated for any layout/API changes) and the TeX p-value technical note.

Audit all links after file moves/deletions.

## Language

Use British English in current-facing prose and CLI/user messages. Examples: `normalise`, `optimise`, `behaviour`, `modelling`.

Do not mechanically rename stable Python identifiers solely for spelling if doing so creates pointless API churn, but prefer British spelling for new/current user-facing text.

---

# Stage 8 — CI consolidation and native-only firewall

Remove every CI job that exists only to install or run COBRApy/mfapy compatibility oracles.

Examples include the current optional compatibility jobs and the canonical parity workflow that installs `.[compat,mfapy]`.

Consolidate overlapping workflows where this reduces duplication without hiding failures. The exact number of workflows is not prescribed; the goal is a small understandable native CI surface.

The final CI must cover at least:

1. base/native package clean install on Python 3.11 and 3.12;
2. native FBA/FVA/FastFVA/sampling;
3. stationary EMU;
4. stationary MFA with the `[mfa]` extra;
5. transient forward EMU with the `[transient]` extra;
6. multinomial observation laws;
7. simple binary Rényi bounds;
8. exact likelihood-ratio p-values;
9. real-model acceptance;
10. CLI/public pipeline;
11. package build/install sanity.

Add or retain a fresh-process import firewall proving the ordinary native package and base testing path do not import `cobra`, `optlang`, `mfapy` or `nlopt`.

Also add a repository-content gate that fails if the final tracked tree unexpectedly reintroduces:

- `vendor/`;
- compatibility-only source modules;
- `FLUXEMU_MFAPY_SOURCE`;
- `.[compat]` / `.[mfapy]` installation paths.

Do not make this grep gate so broad that required historical citation text or an unrelated formal “certificate” concept causes false failures.

---

# Stage 9 — Packaging and release-readiness checks

From a clean environment, verify the final root-layout project with no repository-specific `PYTHONPATH` hacks.

Run at least:

- `git diff --check`;
- `python -m compileall -q src tests examples tools` (adjust only for directories that actually remain);
- clean base installation from `.`;
- `python -m pip check`;
- full retained base/native test suite;
- full `[mfa]` tests;
- full `[transient]` tests;
- the simple binary example;
- stationary MFA example;
- public CLI on the native portability fixture;
- real-model acceptance;
- build a wheel and sdist with `python -m build` in an isolated release check;
- install the built wheel into a fresh environment and run a minimal import/CLI smoke test.

Check that the built wheel does not contain deleted vendor/compatibility files.

If a test fails because it depended on removed compatibility infrastructure, determine whether the test represented a genuine native invariant. If yes, replace it with an independent native oracle. If no, delete it. Do not keep compatibility code merely to make an obsolete test green.

---

# Stage 10 — Final repository hygiene

Before opening the PR, verify all of the following:

- no `vendor/` directory;
- no historical `codex/` wrapper if the flattening audit approved the move;
- no `handoff/` migration archive;
- no old agent prompt directory;
- no active runtime dependency on COBRApy/mfapy/optlang/nlopt;
- no compatibility extras in `pyproject.toml`;
- no current docs claiming p-values are absent;
- no current Bruno testing docs referring to Type-II bounds as certificates;
- no duplicate superseded testing guide;
- no broken documentation links introduced by moves/deletions;
- no orphan imports or empty packages;
- no stale CI references to deleted files;
- README install commands work from repository root;
- `AGENTS.md` reflects the final native-only layout and invariants;
- all final-head workflows are green.

Produce a concise before/after summary in the PR description including:

- major directories removed;
- major legacy modules/tests removed;
- dependency groups removed;
- final project layout;
- retained scientific capabilities;
- final test counts/workflows;
- final head SHA.

---

# Repository-level PR / branch housekeeping

There are two known stale open PRs, #10 and #12, targeting the historical acceptance branch rather than current `main`.

Do not merge them into `main`.

Once the final clean-up PR is ready and the current native tree clearly supersedes them, close them as superseded **if the authenticated environment permits GitHub PR management**. If not, explicitly report that they remain to be closed manually.

Do not delete arbitrary remote branches as part of the code rewrite. Instead include in the final report a short list of clearly superseded `codex/...` branches that can safely be deleted after the final PR is merged.

---

# Final guardian acceptance checklist

The final guardian must answer every item explicitly.

1. Is the final repository native-only?
2. Is `vendor/` gone?
3. Are mfapy/COBRApy compatibility runtime modules gone?
4. Are compatibility-only tests/fixtures gone?
5. Are compatibility dependency extras gone?
6. Are CI workflows native-only?
7. Does the repository install from its root using `pip install .`?
8. Is the repository in a conventional root layout unless a documented concrete blocker prevented flattening?
9. Does native FBA still pass?
10. Does native FVA/FastFVA still pass?
11. Does complete feasible-state sampling still pass without independent FVA-coordinate construction?
12. Does native stationary EMU still pass?
13. Does native transient forward EMU still pass?
14. Does stationary MFA still pass for KL and Rényi objectives?
15. Does explicit MID normalisation retain its existing semantics?
16. Do genuine-count multinomial laws retain exact support and count semantics?
17. Does the simple binary LLR retain `log P1 - log P0`?
18. Does the exact p-value retain `P0{LLR(Y) >= LLR(y_obs)}` and explicit enumeration-limit failure?
19. Are the fixed H0/H1, Type-I/Type-II and forward/reverse Rényi conventions unchanged?
20. Is the Bruno result now consistently called a Type-II lower **bound**, not a certificate?
21. Is there still no finite-grid/global-envelope claim?
22. Is the real-model acceptance programme retained and independent of the deleted migration handoff archive?
23. Is the authoritative carbon-transition library retained?
24. Are legally/scientifically required third-party attributions retained without bundling dead vendor trees?
25. Is `KNOWN_LIMITATIONS.md` current?
26. Are README/API docs current and using British English in user-facing prose?
27. Do clean base, `[mfa]`, `[transient]`, build/wheel and CLI checks pass?
28. Are all final-head GitHub workflows green?
29. Is the final diff free of unrelated new scientific features?
30. Is the worktree clean and the branch pushed?

If any answer is “no”, do not declare the task complete.

---

# Completion condition

Stop only when:

- the final native-only repository cleanup is implemented;
- all coherent commits are pushed;
- the complete final validation matrix passes;
- the final guardian approves all 30 items;
- a non-draft pull request to `main` is open with a clear summary and final SHA;
- the branch worktree is clean.

Do **not** merge the final PR yourself unless explicitly instructed after review.
