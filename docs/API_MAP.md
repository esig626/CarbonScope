# Native API map

CarbonScope is organised around one canonical scientific model and a small set of native layers.

The project name is CarbonScope. The current Python package and CLI remain `fluxemu`, so the API names below are unchanged.

## Model and input

`fluxemu.model`

- immutable `FluxModel`, `IsotopeModel` and `CanonicalModel` records;
- explicit atom transitions, mapping branches, flux projections and direction activity semantics;
- model and experiment validation and deterministic fingerprints;
- `load_sbml_flux_model(...)` for the physical SBML Level 3 FBC model.

`fluxemu.native_io`

- `load_native_stationary_spec(...)` binds an experiment YAML to the physical flux model using the packaged authoritative carbon transition library.

`fluxemu.carbon_transitions`

- `load_default_library()` loads and strictly validates the curated mapping data;
- no automatic mapping inference or external compatibility runtime.

## Flux analysis

`fluxemu.flux_analysis`

- native HiGHS FBA;
- reference FVA and reusable VFFVA style FastFVA;
- complete feasible state sampling and geometry validation.

`fluxemu.analysis`

- `run_native_fba(...)`;
- `run_native_fva(...)`;
- `run_native_stationary_analysis(...)`;
- `run_native_stationary_ensemble(...)`;
- `run_native_stationary_ensemble_analysis(...)`.

FVA extrema are diagnostic only and are never combined into a flux state.

## EMU prediction

`fluxemu.emu`

- `compile_emu_plan(...)`;
- `evaluate_stationary(...)`;
- `compile_transient_emu_plan(...)`;
- `evaluate_transient(...)`.

`fluxemu.execution.CanonicalFluxState` is the immutable complete state record shared by native fitting, observation and testing layers.

## Stationary MFA

`fluxemu.mfa`

- `normalise_mid(...)` for explicit closure of external MID like vectors;
- `evaluate_stationary_mfa(...)`;
- `fit_stationary_mfa(...)`;
- exact KL and finite positive order Rényi divergence kernels.

The objective is a plain sum of `D_alpha(observed || predicted)` across declared observations.

## Observation laws

`fluxemu.observation`

- `MultinomialMIDLaw` for genuine fixed total isotopologue counts;
- stationary specification, evaluation and reproducible sampling records;
- exact multinomial KL and Rényi identities and independent product aggregation.

Count totals are explicit. No pseudo count conversion or effective sample size inference is provided.

## Simple binary testing

`fluxemu.testing`

- `SimpleBinaryLawPair` and `SimpleBinaryFluxHypotheses`;
- `log_likelihood_ratio(...)`;
- `likelihood_ratio_p_value(...)`;
- `BrunoOrderBound`;
- `bruno_converse_at_order(...)`;
- `evaluate_stationary_simple_hypotheses(...)`.

Roles are fixed: H0=P0=null and H1=P1=alternative. The exact p value is the P0 upper tail of the realised log likelihood ratio. The Bruno result is an order specific lower bound on optimal Type II error under the supplied Type I budget.

## Finite composite testing

`fluxemu.testing`

- `IndependentMIDProductLaw`: one complete observable law over an explicitly independent ordered set of genuine count MID blocks;
- `CompositeMIDLawFamily`: one explicit finite H0 or H1 family of complete product laws, with no implicit convex hull or member frequency prior;
- `CompositeBinaryTestingProblem`: aligned finite null and alternative classes with the same complete block identities, totals and mass class spaces;
- `composite_renyi_converse_at_order(...)`: order specific `lambda > 1` Type II lower bound from the minimum directed full product law Rényi separation;
- `exact_finite_composite_minimax(...)`: complete joint count space randomised minimax LP oracle, using SciPy and HiGHS through the `testing` extra and failing at an explicit product outcome cap;
- `CompositeRenyiScoreCandidate` and `composite_renyi_score_candidate(...)`: order `0 < lambda < 1` finite family vertex pair minimum plus support and uniform moment diagnostics;
- `verified_composite_renyi_score(...)`: returns that candidate only when both uniform composite moment inequalities verify over every represented member;
- `CompositeScoreBound` and `composite_score_bound_at_order(...)`: analytical threshold, score construction Type II bound, constant randomised test bound and represented minimax upper bound without joint outcome enumeration;
- `evaluate_composite_score_test(...)`: optional small space exact evaluation of the deterministic analytical threshold rule;
- `calibrate_composite_score_test(...)`: exact Type I calibration within the fixed verified upper score threshold family when the joint count space is enumerable;
- `CompositeFluxHypotheses`, `StationaryCompositeTestingResult`, and `evaluate_stationary_composite_hypotheses(...)`: finite complete flux state families mapped through native stationary EMU into one complete product observation law per state. Multiple blocks require explicit `independent_blocks=True`.

The exact represented minimax value, calibrated score family value, deterministic score error and analytical Rényi bounds are distinct quantities. A finite family vertex pair Rényi minimum is a candidate score, not automatically a joint convex class projection or a finite sample least favourable pair.

See `docs/COMPOSITE_TESTING.md` for the statistical contract and current scope.

## CLI

`fluxemu run --model MODEL.xml --experiment EXPERIMENT.yaml --output DIRECTORY`

runs the public native stationary SBML -> FBA/FVA -> EMU pipeline and writes fluxes, MIDs, diagnostics and a provenance manifest. Composite testing is currently a Python API; the CLI does not infer hypothesis families from model files, FVA ranges or sampling frequencies.
