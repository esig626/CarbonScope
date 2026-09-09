# Declarative hypothesis workflow: engineering validation record

Repository: `esig626/CarbonScope`; issue [#23](https://github.com/esig626/CarbonScope/issues/23).
Branch: `codex/end-to-end-hypothesis-workflow`.
Base main SHA: `47c18e8ecd80d45f9800fd20634158f9c67f5af6`.

This change adds orchestration around the repaired native and finite-testing layers already on main. It does not change numerical kernels, relax their validation contracts or claim a solution for continuous feasible flux families. No numerical-kernel defect was identified during this integration.

## Implemented contract

Schema-version-1 YAML names one common SBML/FBC model, ordered native experiment files, explicit H0/H1 reaction-bound restrictions, per-family hit-and-run settings, genuine-count blocks and requested testing procedures. Model, experiment and output paths in YAML resolve relative to the workflow file; explicit command-line overrides resolve from the working directory.

Hypothesis constraints intersect the original physical reaction bounds. The workflow preserves canonical reaction order and uses the entire constrained steady-state region without adding an objective-retention constraint. It validates both regions before sampling. Native audited FVA supplies region-distinction witnesses at the existing `1e-7` bound tolerance; redundant or numerically indistinguishable hypotheses refuse construction. FVA coordinates remain diagnostics and are never assembled into states.

Complete states are sampled and independently validated against their hypothesis regions. H0/P0/null and H1/P1/alternative roles, state order, experiment/target/replicate block order and mass-class order remain explicit. Fixed seeds reproduce the represented finite state families in the same numerical environment. Finite chain output does not establish mixing, sample independence, biological probabilities or coverage of a continuous region.

The public API is `fluxemu.run_hypothesis_testing_workflow(specification, *, model_path=None, output_directory=None)`, also available from `fluxemu.workflow`. `load_hypothesis_testing_spec(...)` loads validated declarations. Immutable structured results expose constrained families, the native stationary bridge, observable-law families, composite problem, typed procedure outcomes, refusals, relationship checks, provenance and optional output paths. Result construction and report export check bindings between declarations, families, laws, results and identities.

The CLI is `fluxemu test-hypotheses --specification PATH [--model PATH] [--output DIRECTORY]`. It requires an output destination from the CLI or YAML. Existing `fluxemu run` behavior and outputs remain unchanged. The Python API can return an in-memory result without persistence.

Native experiment YAML is checked before the existing reader can coerce scalars. Duplicate keys, unknown fields, boolean/numeric identifiers and invalid literal types fail explicitly. Quoted correction strings preserve the native semantics. These guards apply to the new workflow and do not alter the legacy forward command.

Count totals must be explicit positive integers in `1..2^63-1`. Booleans and floating-point totals are invalid. Normalised MIDs, percentages, peak areas, intensities and corrected fractions are never converted into counts or an inferred effective sample size. Multiple blocks require literal `independent_blocks: true`; replicate IDs and shared model origin do not imply independence. Structural zeros remain exact.

## Statistical outputs and refusal policy

| Procedure | Reported scientific quantity |
| --- | --- |
| `composite_converse` | Supplied-order composite Type-II lower bound, with finite order greater than one. |
| `exact_minimax` | Numerically validated unrestricted randomised minimax Type-II value for the represented finite classes and complete count space. |
| `candidate_score` | Finite-family Rényi vertex-pair candidate, with supplied order strictly between zero and one, support and moment diagnostics. |
| `analytical_score_bound` | Verified projected-score analytical guarantee and separately labelled constant-randomised/minimax upper guarantees. |
| `deterministic_score_error` | Directly enumerated achieved errors of the verified analytical threshold rule. |
| `calibrated_score_error` | Directly enumerated achieved errors within a fixed candidate score family. |

The runner checks applicable `converse <= minimax <= achieved Type-II error` relationships with a `2e-9` comparison allowance from the existing validation campaign. This does not modify primitive solver or certification tolerances. A candidate need not verify; calibration of a well-defined score can succeed when its analytical moment certificate fails. Calibration need not attain unrestricted minimax. No continuous-order optimisation or automatic least-favourable-pair claim is made.

Optional refusal categories are `enumeration_limit`, `numerical_limit`, `optimization_or_dependency_limit`, `score_verification` and `prerequisite_refused`. Reports retain the exact procedure/order, exception type and reason. Exact LP coefficient floors, minimum budgets, complete-space enumeration caps, support preservation and optimality validation remain unchanged; there is no clipping, coefficient dropping, cap enlargement, Monte Carlo or asymptotic fallback.

Invalid scientific inputs or failed model, feasible-region, sampling or EMU/law construction raise errors and yield CLI exit 2. Failed persistence also yields exit 2. A valid constructed problem with an optional statistical refusal yields exit 0 and `completed_with_refusals`; success does not imply every requested statistic was evaluated.

## Persisted report and reproducibility

`report.json` and `summary.txt` are deterministic outputs. The JSON includes the full canonical model, ordered experiments/tracers/targets/count declarations, H0/H1 specifications, effective bounds, generation settings, ordered states and laws, provenance fingerprints, requested procedures, labelled results and refusals. Procedure entries distinguish explicitly requested quantities, evaluated prerequisites and unused procedures.

Scientific identities exclude paths, timestamps and process IDs. Source byte digests and software/version/dependency/commit provenance are recorded separately. The runtime source digest covers executable Python and packaged authoritative YAML/JSON. Git dirty status is scoped to runtime source and `pyproject.toml`, so reports written elsewhere in the checkout do not alter it. Positive/negative infinity use JSON strings; NaN refuses export. Output files are fully serialised before replacement, and scientific input files are protected from overwrite.

The human summary states the hypotheses, finite sizes, observation design, successful and refused procedures, and limits of interpretation. It describes represented finite-class testing performance rather than a realised-data composite p-value, inferred true mechanism or compatibility region.

## Deterministic acceptance fixture

`tests/fixtures/hypothesis_workflow/` contains actual SBML/FBC, native isotope YAML and workflow YAML inputs. The compact model has canonical reactions `HX_U`, `HX_L`, `SINK`; two glucose feeds supply G6P and the sink is fixed at 10. Explicit authoritative hexokinase mappings propagate unlabelled and fully labelled glucose into the measured six-carbon pool. H0 restricts `HX_U` to `[7,8]`; H1 restricts it to `[2,3]`.

Each role requests two states with seeds 23/24, burn-in 8 and thinning 2. One genuine-count block has total 2 and ordered mass classes M+0 through M+6. The full count space contains 28 vectors; M+1 through M+5 have exact zero probabilities. No final probability law is hand-authored. Acceptance tests independently verify steady-state dilution and a closed-form minimax control derived from the ordered binary-support laws and their Neyman–Pearson rule.

| Accepted fixture quantity | Approximate value |
| --- | ---: |
| Converse, order 2 | 0.572530287383 |
| Represented finite minimax Type-II | 0.696682993429 |
| Candidate Rényi, order 0.5 | 0.424578437093; moments verified |
| Raw analytical exponential bound | 13.0809089664 |
| Separate minimax upper guarantee | 0.95 |
| Deterministic analytical-score achieved Type-II / Type-I | 1 / 0 |
| Calibrated achieved Type-II / Type-I | 0.696682993429 / 0.05 |

The raw analytical bound is uninformative in this fixture. Equality between calibrated and unrestricted minimax values is numerical evidence for this represented problem only. The fixture is software acceptance infrastructure, not the biological showcase of issue #24.

## Validation gates

The final runtime and regression source was committed locally as `951bf239f28403f05982df5c946e2cc76253584b`. The subsequent documentation commit changes no executable source or tests.

| Final local gate | Result |
| --- | --- |
| Complete Python 3.11.16 suite | **1469 passed**, 160.32 seconds |
| Complete Python 3.12.14 suite | **1469 passed**, 162.76 seconds |
| Four public examples on each version | All completed successfully |
| Clean-checkout CLI workflow | Passed from an independent clone of the committed runtime |
| Deterministic report/summary replay | Byte-identical across external output directories and repeated in-checkout output |
| Existing composite regressions and legacy `fluxemu run` | Passed in both complete suites; clean-checkout legacy CLI also passed |
| Source compilation and complete diff whitespace | Passed |
| Validation checkout cleanliness | Clean after generated outputs were removed |

The final suites include 145 new workflow tests in addition to the 1324 existing tests. Added tests cover strict schema failures, complete constrained-state feasibility, distinct-region witnesses, ordering/support, independent minimax and converse controls, explicit coefficient and enumeration refusals, failed moment verification with valid direct calibration, immutable result bindings, persistence and legacy CLI compatibility.

Local Python 3.11 used NumPy 2.4.6 and SciPy 1.17.1; Python 3.12 used NumPy 2.5.3 and SciPy 1.18.1. Both used highspy 1.12.0. The exact LP continues to use SciPy's bundled HiGHS without policy changes.

Review reproduced inconsistent caller replacements of workflow families/budgets/provenance and unsafe output symlink handling in the new orchestration code. The committed result-binding and atomic file-replacement guards reject those cases. The guards check retained consistency without rerunning EMU or statistical solvers; they do not authenticate a wholly fabricated but internally consistent collection of records.

Direct Git transport has no write credentials in this environment, so publication uses the connected GitHub API. Every published implementation tree was checked equal to its locally validated tree:

| Local implementation commit | Published implementation commit | Identical Git tree |
| --- | --- | --- |
| `3e2b2b4a4827a099e8bd902306944d507a518d66` | `13d84dddfc20705ea81ffea2ffbd64f360b589bc` | `7a909819db3603a72c5f19a4543b7211479b1716` |
| `951bf239f28403f05982df5c946e2cc76253584b` | `a49d7ac8d49ae12eb5697a63a383161748eea42c` | `4edc387197cdd9ce5550c08dff9ea5ed147d800c` |

Final branch SHA, GitHub Actions run ID and all Python 3.11/3.12 base-native/full-native job conclusions are checked after the documentation commit is published and delivered explicitly. CI results on an earlier SHA are not substituted for that final gate. This committed report records local evidence; it does not assert that its own subsequent CI run had completed before publication. No merge to main is authorised or performed.

## Remaining scientific limitations and non-goals

All results apply to represented finite classes of complete genuine-count laws. A sampled list does not certify continuous feasible-family coverage or uniform performance outside that list. Numerical LP/moment validation is floating-point evidence under declared policies, not symbolic or interval-arithmetic proof. Enumeration and coefficient resolution impose explicit small-problem limits. A converse constrains achievability; it does not establish an achieved procedure. An achieved procedure need not be unrestricted minimax.

This work does not implement realistic non-count LC-MS/GC-MS laws; new natural-abundance correction machinery; generic composite p-values; compatibility-set/test-inversion inference; continuous feasible flux-family optimisation; Bayesian nuisance integration; transient inverse MFA; transient composite testing; experiment-design optimisation; a GUI or notebook frontend; or the full biological showcase. It does not silently convexify finite classes or treat sampling frequencies as priors. Issues #24, #25, #26, #27 and #28 remain separate work.

## Changed files

The implementation/documentation scope relative to the base SHA, including this report, is:

```text
.github/workflows/native-ci.yml
README.md
docs/API_MAP.md
docs/EXPERIMENT_FORMAT.md
docs/HYPOTHESIS_WORKFLOW.md
docs/KNOWN_LIMITATIONS.md
docs/SCIENTIFIC_WORKFLOW.md
docs/STAGE1_NATIVE_WORKFLOW.md
examples/hypothesis_testing_workflow.py
results/hypothesis_workflow_validation/VALIDATION_REPORT.md
src/fluxemu/__init__.py
src/fluxemu/cli.py
src/fluxemu/workflow/__init__.py
src/fluxemu/workflow/_validation.py
src/fluxemu/workflow/_yaml.py
src/fluxemu/workflow/report.py
src/fluxemu/workflow/runner.py
src/fluxemu/workflow/schema.py
src/fluxemu/workflow/states.py
tests/fixtures/hypothesis_workflow/experiment.yaml
tests/fixtures/hypothesis_workflow/model.xml
tests/fixtures/hypothesis_workflow/workflow.yaml
tests/test_hypothesis_workflow_cli.py
tests/test_hypothesis_workflow_testing.py
tests/test_workflow_result_integrity.py
tests/test_workflow_schema.py
tests/test_workflow_states.py
tests/test_workflow_yaml.py
```
