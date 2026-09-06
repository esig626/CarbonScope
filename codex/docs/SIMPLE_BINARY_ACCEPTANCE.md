# Simple binary certificate acceptance evidence

Task: only `codex/prompts/05_simple_binary_renyi_certificates_bruno_multiagent.md`.
Branch: `codex/simple-binary-renyi-certificates`.
Source baseline: `217244a85b3c91e60e396842c0cde5e8b8f8c350`.
Local validation date: 2026-09-06, Python 3.12.13.

## Coherent milestones

| Milestone | Commit | Guardian decision and evidence |
| --- | --- | --- |
| A: source/API | `a74a2387a73233901151cf911d8e3fa4346ee630` | APPROVE; 108 schema tests, 446 combined schema/observation checks. |
| B: numerical core/oracle | `077379bea5afc5ce8cd8eb14c026fc22f7d74364` | APPROVE; 246 schema/numerical tests and 1,359 independently rerun oracle tests. |
| C: native bridge/LLR | `2cab2f7f9582bded66d0610fce0e7a6c934b1c2c` | APPROVE; 57 bridge/LLR tests; 2,000 combined testing/observation checks; executable example. |

Each milestone was committed and pushed before the next implementation began.
GitHub tree hashes and mirrored commit hashes were checked against the local
checkout. SIMPLE-TEST-GUARDIAN was the first specialist created and remains the
final acceptance authority. See its [audit record](SIMPLE_BINARY_GUARDIAN_AUDIT.md).

## Local regression groups

The final runtime is the C runtime: D adds documentation, CI, and import
isolation coverage. Both isolated installations were replaced by built wheels;
every installed Python module was compared byte-for-byte with the source.
`pip check` passes for both. The base environment has no SciPy; the MFA extra
has SciPy 1.18.1. Neither environment has COBRApy, optlang, mfapy, Matplotlib,
or JAX. No dependency declaration changed.

| Check | Result |
| --- | --- |
| All testing and observation tests, base wheel | 2,002 passed (1,664 testing + 338 observation), 58.53 seconds. |
| Base MFA schemas, preprocessing, divergence/import isolation and complete native Stage 1 gate | 572 passed, 1 expected skip. |
| All MFA tests with the MFA extra and shared native Stage 1 gate | 525 passed. |
| New fresh-process testing isolation | 2 passed, including the complete native example with optional imports blocked. |
| Both documented Python snippets | Executed successfully from the repository root. |
| Standalone synthetic example | Executed successfully, including support rejection and unequal totals. |
| Base and MFA-extra `pip check` | No broken requirements. |
| Compileall | All sources, tests, and examples compile. |
| `git diff --check` | Clean against the original baseline and for the current patch. |
| Vendor/MFA/observation source preservation | All 95 original blobs in these trees match their original Git SHA-1 hashes. |

The one base skip is the fitting-isolation check that requires optional SciPy.
It runs and passes in the MFA-extra group. Overlapping regression groups are
reported separately rather than counted as distinct scientific test cases.

Repository-root command groups, with `BASE` and `MFA` pointing to the Python
executables of the isolated base and MFA-extra installations:

```bash
"$BASE" -m pytest -q codex/tests/test_testing_*.py codex/tests/test_observation_*.py

"$BASE" -m pytest -q \
  codex/tests/test_mfa_schema.py \
  codex/tests/test_mfa_normalisation.py \
  codex/tests/test_mfa_divergence.py \
  codex/tests/test_mfa_import_isolation.py \
  codex/tests/test_cli.py \
  codex/tests/test_native_portability.py \
  codex/tests/test_standalone_model_contract.py \
  codex/tests/test_native_highs_flux_analysis.py \
  codex/tests/test_native_highs_vffva.py \
  codex/tests/test_native_highs_sampling.py \
  codex/tests/test_native_highs_import_isolation.py \
  codex/tests/test_native_emu_graph.py \
  codex/tests/test_native_emu_tracers.py \
  codex/tests/test_native_stationary_emu.py \
  codex/tests/test_native_stationary_analysis.py \
  codex/tests/test_native_stage1_ensemble_integration.py \
  codex/tests/test_native_stationary_batch_diagnostics.py \
  codex/tests/test_real_model_acceptance.py

"$MFA" -m pytest -q codex/tests/test_mfa_*.py \
  codex/tests/test_native_highs_flux_analysis.py \
  codex/tests/test_native_highs_vffva.py \
  codex/tests/test_native_highs_sampling.py \
  codex/tests/test_native_highs_import_isolation.py \
  codex/tests/test_native_stationary_emu.py \
  codex/tests/test_native_stationary_analysis.py \
  codex/tests/test_native_stage1_ensemble_integration.py \
  codex/tests/test_native_stationary_batch_diagnostics.py \
  codex/tests/test_real_model_acceptance.py

"$BASE" -m pip check
"$MFA" -m pip check
"$BASE" -m compileall -q codex/src codex/tests codex/examples
"$BASE" codex/examples/simple_binary_flux_discrimination.py
git diff --check 217244a85b3c91e60e396842c0cde5e8b8f8c350
```

## Independent scientific evidence

The oracle contains 24 single multinomial pairs and three explicitly
independent products with unequal totals. Seven epsilon values and seven
validation orders produce 1,323 certificate comparisons. All 7,032 deterministic
rejection regions are enumerated, giving 49,224 exact budget checks. The largest
space contains 12 atoms. These are test-only bounded enumerations, with no
production finite-order grid or support enumeration API.

Factorial/Fraction PMFs sum to one exactly. Independent Decimal full-law sums
check both Rényi directions, multinomial tensorisation, product additivity,
both component formulas, and the lower-certificate inequality against the
exact deterministic optimum. Boundary tests distinguish deterministic budgets
from randomized interpolation and from whole-atom LR-prefix searches. Explicit
monkeypatch guards verify that oracle helpers do not call production kernels
or derive the optimum from either bound formula. The numerical tolerances are
stated in the test module and [scientific guide](SIMPLE_BINARY_RENYI_CERTIFICATES.md).

The stationary fixture uses complete fixed states `v0=(2.5,7.5,10)` and
`v1=(5,5,10)` in declared `Z_IN,A_IN,M_OUT` order. At epsilon 0.05 and order 2,
totals 1, 4, 16, and 64 retain MIDs `(0.25,0.75)` and `(0.5,0.5)` while the lower
certificate changes to approximately 0.7418011, 0.6024768, 0.02540312, and
5.665084e-7. This demonstrates the declared-count operational transfer without
claiming actual or achievable errors. Controls cover identical/close laws,
matching structural zeros, support rejection, shared state IDs, unequal totals,
invalid states, and feasible states below the biological objective optimum.

LLR checks independently verify `log P1 - log P0`, correct infinite values,
joint zero/zero rejection, count validation, and a small signal at total
`10**15` lost by direct subtraction of rounded log PMFs.

## CI and stopping boundary

The new `simple-binary-renyi.yml` workflow installs the base wheel and runs all
testing/observation checks on Python 3.11 and 3.12, checks absent optional
dependencies, compiles, checks whitespace, and executes the native example.
Existing MFA, native Stage 1, observation, compatibility, and forward workflows
remain enabled. Exact final PR/head CI outcomes and the 25-point acceptance
decision are recorded by SIMPLE-TEST-GUARDIAN after the CI runs complete.

This task stops with the simple binary layer. Composite testing, inversion,
confidence regions, p-values, priors/Bayesian inference, new observation models,
transient MFA, JAX, experiment design, FVA performance work, and changes to the
MFA objective are not part of this implementation.
