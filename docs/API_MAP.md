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
- `DirichletMIDLaw` and `MIDCorrectionProvenance` for externally corrected
  continuous MIDs with explicit concentration, active face, replicate and
  correction provenance;
- `kl_dirichlet(...)`, `renyi_dirichlet(...)`,
  `DirichletLogLikelihoodScore`, log density, analytic covariance and
  reproducible continuous sampling;
- `implied_dirichlet_precision_from_uncertainty(...)`,
  `diagnose_dirichlet_covariance(...)`, and
  `estimate_dirichlet_precision_from_replicates(...)` for structured
  uncertainty, covariance-fit and scalar-concentration diagnostics;
- `StationaryDirichletObservationSpecification` and
  `evaluate_stationary_dirichlet_observation_families(...)` for joint H0/H1
  common-face construction from stationary EMU predictions.

Count totals are explicit. Dirichlet concentration is not a count. No pseudo
count conversion or effective sample size inference is provided. Same-data
plug-in concentration is diagnostic/model-conditional. See
[DIRICHLET_MID_OBSERVATION.md](DIRICHLET_MID_OBSERVATION.md).

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
- `exact_finite_composite_minimax(...)`: unrestricted complete joint count-space randomised minimax LP, using SciPy and HiGHS through the `testing` extra, with numerical feasibility/optimality validation and explicit enumeration or numerical refusals;
- `CompositeRenyiScoreCandidate` and `composite_renyi_score_candidate(...)`: order `0 < lambda < 1` finite family vertex pair minimum plus support and uniform moment diagnostics;
- `verified_composite_renyi_score(...)`: returns that candidate only when both uniform composite moment inequalities verify over every represented member;
- `CompositeScoreBound` and `composite_score_bound_at_order(...)`: analytical threshold, score construction Type II bound, constant randomised test bound and represented minimax upper bound without joint outcome enumeration;
- `evaluate_composite_score_test(...)`: optional complete small-space enumeration of the deterministic analytical threshold rule and its achieved errors;
- `calibrate_composite_score_test(...)`: enumerated Type I calibration within a fixed, well-defined candidate upper-score threshold family, returning achieved worst-case errors even when the analytical moment certificate fails;
- `CompositeFluxHypotheses`, `StationaryCompositeTestingResult`, and `evaluate_stationary_composite_hypotheses(...)`: finite complete flux state families mapped through native stationary EMU into one complete product observation law per state. Multiple blocks require explicit `independent_blocks=True`.

The parallel continuous API provides
`IndependentDirichletMIDProductLaw`, `DirichletCompositeMIDLawFamily`,
`DirichletCompositeBinaryTestingProblem`,
`composite_dirichlet_renyi_converse_at_order(...)`,
`composite_dirichlet_renyi_score_candidate(...)`,
`verified_composite_dirichlet_renyi_score(...)`,
`composite_dirichlet_score_bound_at_order(...)`, and
`evaluate_stationary_dirichlet_composite_hypotheses(...)`. The exact Dirichlet
minimax, deterministic-error and calibration entry points exist only to return
explicit continuous-observation refusals; no count LP, Gaussian score CDF or
simplex discretisation is reused.

The LP is a mathematically exact characterisation of the represented finite minimax value; its solution is validated numerically in floating-point arithmetic. A calibrated score returns an achieved error, which may strictly exceed the unrestricted minimax optimum. A valid converse is a lower bound: `converse <= beta_star <= achieved score error`. These quantities and the analytical score upper bounds remain distinct. A finite-family vertex-pair Rényi minimum does not by itself prove the uniform projected-moment inequalities or finite-sample least favourability.

These APIs cover the supplied finite lists only. Rigorous optimisation or bounds over a complete continuous feasible flux family, generic composite p-values and compatibility-region inversion are not implemented. See [COMPOSITE_TESTING.md](COMPOSITE_TESTING.md) for numerical policies and count semantics.

## Declarative hypothesis-testing workflow

`fluxemu.workflow`

- `WorkflowSpecification`, `HypothesisSpecification`, `ReactionBoundConstraint`, `StateGenerationSpecification` and `TestingSpecification`: immutable validated declarations for the scientific design;
- `load_hypothesis_testing_spec(path, *, model_path=None)` loads the schema-version-1 YAML and validates the common physical/isotope model, ordered experiments, genuine-count or corrected-MID Dirichlet declarations, H0/H1 constraints and numerical settings;
- `generate_hypothesis_state_families(...)` returns ordered H0/H1 `HypothesisStateFamily` records after native feasibility, region-distinction, sampling and complete-state checks;
- `run_hypothesis_testing_workflow(specification, *, model_path=None, output_directory=None)` accepts the validated specification or its YAML path, constructs and validates both constrained state families, evaluates native stationary EMU and the selected observation laws, evaluates supported requested finite procedures and optionally persists the report;
- `HypothesisTestingWorkflowResult` is an immutable record exposing `.specification`, `.null_family`, `.alternative_family`, `.stationary`, `.problem`, `.null_observation_family`, `.alternative_observation_family`, `.testing_results`, `.relationship_checks`, `.refusals`, `.provenance`, `.output_paths` and `.summary`;
- `ProcedureEvaluation` keeps the procedure, supplied order, whether it was explicitly requested, its evaluated/refused/not-requested status, a typed testing result or `TestingRefusal`;
- `WorkflowProvenance` binds the model, experiment, hypothesis, represented state-family, observation-family and testing-problem identities and records software provenance separately.

The orchestration entry points are also available from `fluxemu`. Passing `model_path` overrides only a file specification; a validated specification already contains its model. An explicit `output_directory` overrides the specification's output setting. With neither output setting, the Python API returns its structured result without persistence.

V1 hypothesis constraints intersect reaction bounds with the common model and use the full constrained steady-state region without retaining an objective fraction. Complete hit-and-run states retain declared H0/H1 and canonical reaction ordering. Scientific constraints are distinct from generated finite representations; the latter alone define the classes passed to the testing engine.

The runner preserves the existing numerical contracts. It reports optional enumeration, numerical certification, optimisation/dependency and score-verification refusals as first-class outcomes. Invalid scientific input or failed modelling, state generation or observation construction raises an error. Available converse/minimax/achieved comparisons are checked using the documented `2e-9` reporting comparison allowance; this does not relax any primitive's solver or certification tolerance.

See [HYPOTHESIS_WORKFLOW.md](HYPOTHESIS_WORKFLOW.md) for the complete schema and result contract.

## CLI

`fluxemu run --model MODEL.xml --experiment EXPERIMENT.yaml --output DIRECTORY`

runs the public native stationary SBML -> FBA/FVA -> EMU pipeline and writes fluxes, MIDs, diagnostics and a provenance manifest.

`fluxemu test-hypotheses --specification WORKFLOW.yaml [--model MODEL.xml] [--output DIRECTORY]`

runs the declarative stationary finite hypothesis workflow and writes `report.json` and `summary.txt` to the declared output directory. The CLI requires an output destination either in the specification or through `--output`. Exit 0 means the workflow completed, possibly with explicit optional statistical refusals; exit 2 means invalid input, failed workflow construction or failed persistence. Hypotheses are declared explicitly rather than inferred from FVA ranges or sampling frequencies. See [HYPOTHESIS_WORKFLOW.md](HYPOTHESIS_WORKFLOW.md).
