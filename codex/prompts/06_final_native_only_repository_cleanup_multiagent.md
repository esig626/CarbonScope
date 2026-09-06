# Task 06 — Final native-only FluxEMU repository clean-up and release hardening

## Execution environment

This prompt is intentionally safe for Codex Cloud worktrees.

- Work in the **current task checkout and current task branch**, whatever Codex calls it (for example `work`).
- **Do not fetch, switch branches, reset to another branch, or require a pre-existing task branch.**
- The checkout must contain this prompt because this file is committed on `main`.
- Before edits, run `git status --short --branch` and verify that commit `b72622d350c06443e9b3f23811919b5283fe8008` is an ancestor of `HEAD`. If it is not, stop and report that the checkout predates merged PR #19.
- Read `AGENTS.md` first and obey its scientific rules. This task is authorised to remove the read-only legacy `vendor/` tree as a whole and to update `AGENTS.md` when the repository layout is flattened. Do not edit third-party source inside `vendor/`; either retain it untouched during audit or delete it as a tree.

## Mission

Turn `esig626/fluxemu-standalone` into a clean, native-only, publication-ready Python repository. This is repository finalisation and hardening, **not a new scientific-method task**.

The final repository must preserve all current native scientific capabilities:

1. canonical model and SBML ingestion;
2. native HiGHS FBA;
3. native FVA and the VFFVA-style reusable/shared-memory FastFVA architecture;
4. complete feasible-state sampling, never independent FVA-coordinate sampling;
5. native stationary EMU;
6. native transient forward EMU;
7. stationary MFA with exact KL / finite positive-order Rényi objectives and explicit `normalise_mid(...)` preprocessing;
8. genuine-count multinomial observation laws;
9. simple-vs-simple log-likelihood-ratio evaluation;
10. exact simple-null likelihood-ratio p-values for enumerable multinomial spaces;
11. finite-order Bruno Type-II lower bounds;
12. the native E. coli real-model acceptance programme;
13. the authoritative carbon-transition library and all scientifically required atom-mapping data.

The finished repository should no longer look like a migration workspace and should no longer carry mfapy/COBRApy compatibility baggage.

## Non-negotiable scientific invariants

Preserve exactly:

- `H0 = P0 = null` and `H1 = P1 = alternative`.
- Type I = `P0(decide H1)` and Type II = `P1(decide H0)`.
- reverse Rényi = `D_lambda(P1 || P0)` and forward Rényi = `D_lambda(P0 || P1)`.
- any supplied finite real `lambda > 1` is used unchanged subject only to explicit numerical representability limits.
- no finite Rényi-order grid and no claim that a finite grid solves the continuous-order envelope.
- exact p-value definition `p(y)=P0{LLR(Y) >= LLR(y)}` with `LLR(y)=log P1(y)-log P0(y)`.
- exact-enumeration overflow must fail explicitly; do not silently substitute chi-square, Monte Carlo, saddlepoint or another approximation.
- genuine-count semantics only: no conversion of percentages, normalised MIDs, peak areas or arbitrary intensities into pseudo-counts; no effective sample size; no pseudocounts; no clipping or support repair; structural zeros remain exact.
- independent product blocks only when independence is explicitly declared.
- FVA extrema are diagnostics, never assembled into a flux vector.
- feasible-state sampling uses complete jointly feasible states and preserves canonical reaction order.
- explicit atom mappings, reaction directions, flux projections and direction-activity logic remain authoritative; never infer mappings from stoichiometry, names or formulae.
- preserve current native stationary/transient numerical validation and real-model frozen scientific references.

Do not add composite testing, test inversion, flux confidence regions, Bayesian inference, new observation laws, natural-abundance correction, experiment design, or any other new method in this task.

## Required review roles

Use focused parallel review agents where useful, with one implementation owner. At minimum perform these independent reviews before final completion:

1. **Legacy-removal auditor** — inventory mfapy/COBRApy/migration-only files and prove native code does not depend on them.
2. **Native scientific-invariant auditor** — verify every invariant above after structural changes.
3. **Packaging/CI/docs auditor** — verify installability, root layout, dependency cleanup, links, CI and British English in current-facing prose.
4. **Final guardian** — inspect the final diff and test evidence independently and block completion on any regression.

Agents do not replace real tests.

Commit each coherent layer before beginning the next substantial layer. Do not leave substantial work uncommitted.

## Stage 0 — Inventory before deletion

Before editing, build an explicit import/dependency inventory with `git grep`, AST/import inspection, test collection and targeted reads.

Identify:

- current public native APIs and CLI path;
- all modules imported by the native stationary pipeline, ensemble path, MFA, observation and testing layers;
- all files referring to `cobra`, `cobrapy`, `mfapy`, `optlang`, `nlopt`, `vendor/`, `FLUXEMU_MFAPY_SOURCE`, compatibility extras or legacy bridges;
- all tests/workflows/docs that exist only for compatibility/parity;
- all independent mathematical/literature validation that should survive removal of legacy runtimes;
- any attribution/licence obligations for code genuinely adapted from third-party work, especially the VFFVA architecture.

Do not retain an entire vendor tree merely for attribution. Preserve necessary acknowledgements/licences in a minimal correct form.

## Stage 1 — Remove the legacy runtime subsystem

The final repository must contain **no `vendor/` directory** and no runtime fallback to vendored mfapy/COBRApy.

Audit and remove compatibility-only source. Known candidates include:

- `codex/src/fluxemu/_mfapy.py`
- `codex/src/fluxemu/backends/`
- `codex/src/fluxemu/cobra_analysis.py`
- `codex/src/fluxemu/cobra_to_mfapy.py`
- `codex/src/fluxemu/compat/`
- `codex/src/fluxemu/forward.py`
- `codex/src/fluxemu/mfapy_model.py`

Also audit older modules such as `configuration.py`, `flux_conversion.py`, `isotope_metadata.py`, `model_identity.py`, `output.py`, `validation.py` and `toy.py`; remove them only if the import graph proves they are dead relative to retained native code/tests.

Remove dead exports, helpers, environment variables and exceptions created solely for the retired subsystem.

The installed `fluxemu` package must not require or import COBRApy, mfapy, optlang or nlopt.

## Stage 2 — Remove compatibility-only tests while retaining independent science validation

Delete tests/fixtures whose only purpose is the retired compatibility bridge. Known candidates include:

- `test_cobra_analysis.py`
- `test_cobra_flux_projection.py`
- `test_cobra_to_mfapy.py`
- `test_mfapy_forward_regression.py`
- `test_mfapy_inverse_api.py`
- `tests/fixtures/official_mfapy_example_0.json`
- `test_native_highs_cobra_parity.py` if it is solely a COBRA oracle.

Audit mixed parity tests carefully. If an Antoniewicz or other test contains both mfapy parity and an independent direct solver/published frozen reference, remove the mfapy part and retain the independent validation.

The final native implementation should be checked against mathematics, explicit fixtures, published examples and independent solvers, not against a bundled legacy runtime.

## Stage 3 — Remove migration artefacts

Audit and remove material that exists only because this repository was being migrated/developed:

- `codex/handoff/` and its archive/sidecar once current native real-model data/tests are self-contained;
- superseded mfapy/COBRA compatibility audits and migration reviews;
- obsolete guardian/process documents that are not useful user-facing or scientific reproducibility material;
- old completed Codex task prompts after their work is fully represented in code/docs;
- this Task 06 prompt itself as the final structural clean-up step, after it has been executed.

Do **not** remove authoritative carbon-transition data, current real-model packaged data, frozen scientific references, published-example fixtures or independent-oracle evidence still used by current tests.

Remove any real-model acceptance dependency on the historical handoff archive.

## Stage 4 — Flatten the repository to a conventional Python layout

Unless Stage 0 finds a concrete technical reason not to, remove the historical `codex/` wrapper and move retained files to a standard root layout:

- `pyproject.toml`
- `src/fluxemu/`
- `tests/`
- `docs/`
- `examples/`
- `benchmarks/`
- `results/` if frozen validation outputs remain necessary
- `tools/`
- `.github/workflows/`
- `README.md`
- `AGENTS.md`

Move, do not duplicate. Update every path in workflows, docs, tools, tests, package-data declarations and examples.

Update `AGENTS.md` in the same coherent structural commit so runtime code is thereafter under `src/fluxemu/` and all scientific invariants remain concise and explicit.

After flattening, the standard installation command must be:

`python -m pip install .`

with optional native extras such as `.[mfa]` and `.[transient]` where still required.

## Stage 5 — Dependency cleanup

Remove compatibility-only optional dependency groups, especially `compat` and `mfapy`, and remove dependencies unused after legacy deletion.

Retain only genuine native requirements. Expected categories include NumPy, pandas if still used, PyYAML if still used, `highspy`, `python-libsbml`, and SciPy only for native optional features that require it (`mfa`, `transient`).

Do not add replacement packages just to reproduce legacy compatibility behaviour.

Preserve package data for the authoritative carbon-transition YAML/data and native real-model JSON data.

Keep `src/fluxemu/__version__` and `pyproject.toml` version consistent. Do not publish a release/tag/PyPI package in this task.

## Stage 6 — Finish Bruno terminology migration

The Bruno testing result is a **Type-II lower bound**, not a certificate.

Required final state:

- canonical public result class is `BrunoOrderBound`;
- implementation, docstrings, tests, fingerprints/schema strings and user-facing output use `bound` terminology where referring to the Bruno result;
- remove the pre-release `BrunoOrderCertificate` compatibility alias if it exists only because of recent development and is not part of an established stable release contract;
- keep `bruno_converse_at_order(...)` if otherwise appropriate.

Do not rename unrelated formal concepts such as a direction-activity certificate merely because they contain the word “certificate”.

## Stage 7 — Rewrite current-facing documentation

Make `README.md` a concise public software README describing what FluxEMU does now, installation from root, the native pipeline, minimal stationary use, stationary MFA, genuine-count observation/testing, exact p-value versus Rényi Type-II lower-bound interpretation, principal docs and genuine limitations.

Rewrite `KNOWN_LIMITATIONS.md` so it is current. It must no longer say p-values are unimplemented or use Bruno certificate terminology. Retain genuine limits, including as applicable:

- local/multistart MFA does not prove a global optimum;
- no composite testing/inversion in this standalone repository;
- exact p-value enumeration limits;
- multinomial testing requires genuine counts;
- finite hit-and-run output does not prove mixing/independence;
- transient inverse MFA is absent;
- natural-abundance correction is absent if still true;
- current SBML/input metadata limitations.

Rewrite/replace `API_MAP.md` as a native-only API/architecture map. Remove historical COBRApy/mfapy source audits from current architecture docs.

Delete obsolete docs when confirmed superseded, including likely:

- `MFAPY_ENGINEERING_COMPARISON.md`
- `MFAPY_OBSERVATION_SEMANTICS_AUDIT.md`
- `MFAPY_PATCH.md`
- `LEGACY_MODEL_REVIEW.md`
- `SIMPLE_BINARY_RENYI_CERTIFICATES.md`

Retain and update `SIMPLE_BINARY_RENYI_BOUNDS.md` and `technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex`.

Audit all links after moves/deletions.

Use British English in current-facing prose and CLI/user messages (`normalise`, `optimise`, `behaviour`, `modelling`). Do not create pointless Python API churn solely for spelling.

## Stage 8 — Consolidate CI into native-only gates

Remove workflow jobs whose only purpose is installing COBRApy/mfapy or testing `vendor/` parity. In particular eliminate jobs using `FLUXEMU_MFAPY_SOURCE`, `.[compat]`, `.[mfapy]` or vendored package paths.

Consolidate redundant workflows where sensible, but retain strong native coverage for:

- clean root installation/import isolation;
- SBML/native model ingestion;
- HiGHS FBA/FVA/FastFVA;
- feasible-state sampling;
- stationary EMU and independent native validation;
- transient forward EMU;
- stationary MFA base and optional-SciPy paths;
- multinomial observation laws;
- simple binary LLR, exact p-value and Bruno bounds;
- real-model acceptance;
- example execution;
- Python 3.11/3.12 where already supported.

Add or retain an explicit import-isolation assertion that the base native installation does not import `cobra`, `mfapy`, `optlang` or `nlopt`.

## Stage 9 — Final native-only regression and repository audit

Run focused tests after each coherent implementation layer, then run the complete retained test suite in clean installed environments.

At final head, require at minimum:

1. `git diff --check` passes.
2. Python sources/tests/examples compile.
3. `python -m pip install .` succeeds in a clean environment and `pip check` passes.
4. Base native import succeeds without optional stacks.
5. Full retained native test suite passes.
6. `.[mfa]` tests pass.
7. `.[transient]` tests pass.
8. Native examples used in docs execute.
9. Real-model acceptance passes from native packaged data without handoff/vendor dependency.
10. Exact p-value tests pass and preserve the stated definition.
11. Bruno-bound tests preserve both Rényi directions and Type-I/II conventions.
12. Repository-wide search confirms no current runtime/workflow/docs dependency on `vendor/`, mfapy, COBRApy, compatibility extras, `FLUXEMU_MFAPY_SOURCE` or Bruno “certificate” terminology, except explicit historical acknowledgements that are intentionally retained for provenance/attribution.
13. No `vendor/`, `codex/`, migration handoff archive or obsolete compatibility runtime remains in final tree unless the final guardian documents a concrete blocker. The preferred outcome is complete removal.
14. Worktree is clean after final commit.

If a supposedly legacy file turns out to be necessary for native correctness, **do not delete functionality to satisfy the directory-cleanup goal**. Refactor the required native logic/data into an appropriately named native location and document the decision. Scientific correctness wins over cosmetic deletion.

## Publication behaviour

Complete the implementation even if shell GitHub credentials are unavailable.

- Commit all coherent work locally on the current Codex task branch.
- If the environment permits push/PR publication, publish a non-draft PR to `main` but **do not merge it**.
- If push/PR publication is blocked by credentials, do **not** stop before implementation. Finish all local work, tests, audits and commits, then report the final commit SHA and the publication blocker. The PR can be created externally afterwards.

## Stop condition

Stop only after:

- the legacy-removal auditor approves;
- the scientific-invariant auditor approves;
- the packaging/CI/docs auditor approves;
- the final guardian approves the final diff;
- all locally executable final acceptance gates pass;
- the worktree is clean and all work is committed;
- publication is completed if credentials permit, otherwise the exact publication blocker is reported.

Do not merge the final PR.
