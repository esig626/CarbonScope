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

- `FiniteObservationLaw` for one complete genuine count law, including explicitly independent products;
- `FiniteCompositeHypotheses` for explicit finite null and alternative law classes on one common observation geometry;
- `projected_renyi_test(...)` for a supplied `0 < lambda < 1`, including direct uniform moment checks and finite threshold calibration;
- `composite_renyi_converse_at_order(...)` for an order specific class lower bound at `lambda > 1`;
- `solve_finite_minimax_test(...)` for the unrestricted randomised minimax optimum on an enumerable finite observation space;
- `ProjectedRenyiTestResult`, `CompositeRenyiConverse`, and `FiniteMinimaxTestResult` for the corresponding diagnostics and guarantees.

For an explicit finite problem the LP computes the numerical value of the exact minimax characterisation

```text
beta*(epsilon) = inf_phi sup_Q E_Q[1-phi]
```

subject to uniform `sup_P E_P[phi] <= epsilon`.

The projected threshold test is an achieved test and therefore gives an upper bound on this unrestricted optimum. The composite Rényi converse gives a lower bound. A finite sampled flux ensemble is not silently identified with the full continuous flux family.

See [COMPOSITE_TESTING.md](COMPOSITE_TESTING.md).

## CLI

`fluxemu run --model MODEL.xml --experiment EXPERIMENT.yaml --output DIRECTORY`

runs the public native stationary SBML -> FBA/FVA -> EMU pipeline and writes fluxes, MIDs, diagnostics and a provenance manifest.
