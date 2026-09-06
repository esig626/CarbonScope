# mfapy engineering comparison

This source audit preceded stationary MFA implementation. The reference is the
vendored mfapy snapshot in FluxEMU base commit
`6117a90c795fa0190626f6ab40c2ed26c7dfa1b4`; locations below refer to that unchanged
snapshot. All six required files were inspected as source, including executable
behavior rather than relying on function descriptions.

`vendor/mfapy` remains read-only third-party reference material. Production
FluxEMU must neither import it nor require an mfapy model file. Engineering
concepts are adapted to the existing canonical model, native affine feasible
geometry, complete `CanonicalFluxState`, and stationary EMU evaluator. No mfapy
source code or mutable model dictionaries are copied into the MFA runtime.

## Transfer decisions

The decisions distinguish retained workflow concepts from the statistical and
representation changes required for FluxEMU.

| mfapy mechanism | exact vendored source location | what mfapy does | FluxEMU decision | reason |
|---|---|---|---|---|
| independent flux coordinates | `vendor/mfapy/mfapy/metabolicmodel.py:561-1096`, `MetabolicModel.update`, especially `969-1068` | Row-reduces stacked stoichiometric, fixed-state, and reversible equalities; nonpivot columns select `independent_flux`; records positions and independent bounds. | REPLACE WITH EXISTING FLUXEMU NATIVE MECHANISM | Retain reduced-coordinate fitting, but generalize native sampling's `_ReducedFluxGeometry` / `_build_reduced_geometry`; a second `matrixinv` parameterization would duplicate existing affine geometry. |
| full flux reconstruction | `vendor/mfapy/mfapy/optimize.py:834-836`, `calc_MDV_residue_scipy`; `vendor/mfapy/mfapy/metabolicmodel.py:2962-3011`, `generate_state_dict` | Inserts fitted coordinates into `Rm_initial`, computes `matrixinv @ Rm`, then reconstructs reaction, pool, and reversible dictionaries. | REPLACE WITH EXISTING FLUXEMU NATIVE MECHANISM | Native `v = particular + basis @ theta` already produces the complete ordered reaction state required by `CanonicalFluxState` and native EMU; stationary fitting does not need mfapy pool-coordinate dictionaries. |
| feasible start generation | `vendor/mfapy/mfapy/optimize.py:29-156`, `initializing_Rm_fitting`, and `161-222`, `calc_protrude_scipy`; `vendor/mfapy/mfapy/metabolicmodel.py:3231-3407`, `generate_state` / `generate_initial_states` | Randomizes independent-coordinate boxes, minimizes bound protrusion with SLSQP, checks reconstructed bounds, retries up to three times; callers keep feasible starts and rank their RSS. | REPLACE WITH EXISTING FLUXEMU NATIVE MECHANISM | Reuse native complete feasible-state sampling and its affine geometry with optional retention. This avoids a third initializer and avoids treating independent interval draws as admissible complete states. |
| multistart fitting | `vendor/mfapy/mfapy/metabolicmodel.py:3410-3648`, `fitting_flux`, especially `3566-3585` and `3616-3648` | Fits a supplied list of starts, collects nonempty fitted states, and sorts them by RSS; low-level exceptions are returned as state values. | REUSE WITH ADAPTATION | Preserve independent starts, isolated optimization attempts, and best-result selection. Use deterministic seeded orchestration, retain every failed-start diagnostic, and select only independently validated complete states by exact divergence. |
| global -> local optimization | `vendor/mfapy/mfapy/optimize.py:516-712`, `fit_r_mdv_nlopt`; `716-796`, `fit_r_mdv_deep`, especially `784-792` | Runs NLopt `GN_CRS2_LM` globally, then repeats SciPy `SLSQP` followed by NLopt `LN_PRAXIS`; repeats default to three. | DEFER | First backend uses constrained SciPy SLSQP with distinct native feasible starts. A separate NLopt/global backend adds dependencies and behavior beyond the smallest validated core; the start orchestration boundary leaves room for a later global stage without changing the objective. No global-optimum guarantee is claimed. |
| multiple experiments | `vendor/mfapy/mfapy/metabolicmodel.py:3098-3157`, `set_experiment`; `vendor/mfapy/mfapy/optimize.py:436-450` and `846-864` | Stores tracer/target/MDV blocks by experiment name, concatenates observations in sorted experiment order, predicts each block from the same reconstructed state, and selects used entries. | REUSE WITH ADAPTATION | Preserve one shared state and multiple tracer experiments; use explicit experiment/target/replicate identities, declared ordering, one compiled native EMU plan per experiment, and the plain sum of per-observation MID divergences. |
| measured flux observations | `vendor/mfapy/mfapy/metabolicmodel.py:632-741`, `update`, and `2791-2847`, `set_constraint`; `vendor/mfapy/mfapy/optimize.py:436-450`, `fit_r_mdv_scipy`, and `846-864`, `calc_MDV_residue_scipy` | State values for reactions, metabolite pools, and reversible flux combinations precede MID observations. Type `fitting` sets `use=1`; matching predicted complete-state entries enter the same standard-deviation-weighted residual vector. Type `fixed` instead adds an exact reconstruction equality. | REUSE WITH ADAPTATION | Keep exact reaction/exchange observations through explicitly chosen canonical bounds: equal bounds mean an exact value; a supplied interval is a hard admissible interval. Soft measured-flux residuals, pool observations, and general observation laws are deferred, with the canonical constraint boundary as the extension point. Do not silently interpret standard deviations as bounds or add an undefined weighted term. |
| bounds/feasibility handling | `vendor/mfapy/mfapy/optimize.py:473-488`, `fit_r_mdv_scipy`; `678-683`, `fit_r_mdv_nlopt`; `838-844` and `868`, `calc_MDV_residue_scipy` | SLSQP/NLopt bound the independent variables; residual evaluation adds `100000` times total full-state bound excess to RSS. The SciPy COBYLA branch passes no explicit bounds. | REPLACE WITH EXISTING FLUXEMU NATIVE MECHANISM | Native affine equalities, explicit complete-state bound inequalities, optional retained-objective constraints, and original-model validation carry feasibility directly. A bound penalty must not become part of the scientific MID objective or repair an invalid final state. |
| RSS/covariance objective | `vendor/mfapy/mfapy/optimize.py:436-450`, `fit_r_mdv_scipy`; `802-868`, `calc_MDV_residue_scipy`; `870-935`, `calc_MDV_residue_nlopt` | Filters the combined state/MDV vector by use flags, constructs diagonal inverse covariance `1 / std**2`, and evaluates residual-transpose times inverse covariance times residual. | REPLACE FOR SCIENTIFIC REASON | The requested criterion is exactly `sum_j D_alpha(observed_j || predicted_j)`, with KL at `alpha == 1`, every finite positive real order accepted, and exact support conventions. RSS, hidden weights, covariance inputs, clipping, and pseudocounts would change that criterion. |
| goodness-of-fit/CI machinery | `vendor/mfapy/mfapy/metabolicmodel.py:4456-4534`, `goodness_of_fit` / `get_thres_confidence_interval`; `4692-4966`, `posterior_distribution`; `4972-6286`, `generate_ci_template` / `search_ci` | Computes chi-square thresholds and tail probabilities, F/chi-square RSS interval thresholds, fixed-flux profile/grid refits, and posterior sampling machinery. | REJECT | These statistical conclusions require assumptions outside MID-divergence fitting and are explicitly excluded from this task. No thresholds, p-values, confidence regions, posterior machinery, or profile fitting are transferred. |
| parallel execution | `vendor/mfapy/mfapy/metabolicmodel.py:3347-3376`, `generate_initial_states`; `3540-3648`, `fitting_flux`; `vendor/mfapy/mfapy/optimize.py:376-405`, `fit_r_mdv_scipy` | Uses joblib `Parallel` / `delayed` with configured `ncpus`, passes generated forward-function text for worker reconstruction, and tries to restrict MKL threads. Older Parallel Python paths are inert string blocks. | DEFER | Sequential reproducible multistart is sufficient for the correctness-first backend and preserves failure isolation without adding joblib, worker serialization, or nested BLAS management. Compiled native experiment plans and start diagnostics keep future parallel scheduling separable. |
| synthetic-data utilities | `vendor/mfapy/mfapy/metabolicmodel.py:3044-3078`, stationary `generate_mdv`; `vendor/mfapy/mfapy/mdv.py:433-476`, `MdvData.add_gaussian_noise`; `vendor/mfapy/mfapy/carbonsource.py:374-531`, `CarbonSource.generate_carbonsource_MDV` | Forward-simulates MDVs from a complete state and tracer. An optional separate utility adds absolute/relative Gaussian noise, clips negative draws, optionally normalizes each replicate, then averages replicates. | REUSE WITH ADAPTATION | Retain forward simulation of a known feasible truth using the existing native tracer and stationary EMU stack. Recovery fixtures are exact and noise-free; no noise generation, clipping, normalization, or hidden replicate averaging is transferred. |

## Verified workflow and differences that matter

`MetabolicModel.update` includes reaction fluxes, internal metabolite pool sizes,
and reversible combinations in the reconstructed state. Fixed entries become
equality rows, whereas `fitting` entries become observations. Independent
coordinate bounds therefore do not alone ensure all reconstructed coordinates
are admissible. This explains mfapy's separate initialization and full-state
bound-protrusion paths; it does not justify retaining their penalty formulation
when FluxEMU already has explicit feasible geometry.

The live initializer performs at most three random attempts using NumPy's global
random generator. Although `initial_search_iteration_max` is accepted, it is not
passed to the live SciPy minimize call at `optimize.py:117`; its use in an NLopt
alternative is inside a nonexecuted string. The reconstructed state is checked
against complete bounds with tolerance `0.0001` at lines `139-144`. Template
initialization instead minimizes absolute distance from the template; the
assembled `g` list is not passed as constraints (`196-218`). Native seeded
complete-state sampling is therefore a direct, better-matched replacement.

The SciPy local path uses SLSQP with `ftol=1e-9` and configurable `maxiter`
(default `1000`); its COBYLA option uses `tol=1e-9` and the same iteration limit
(`optimize.py:389-395,473-488`). NLopt uses `xtol_abs=1e-6`, configurable
`maxeval`, and independent bounds (`678-683`). The `deep` implementation actually
calls **LN_PRAXIS**, despite its docstring and logging saying LN_SBPLX. It passes
each stage's output directly into the next stage and returns the last one;
there is no best-intermediate-state retention or independent scientific
acceptance between stages (`784-796`). This source-specific behavior is not
copied into FluxEMU.

Low-level SciPy catches exceptions and returns the exception as `state`, with
initial sentinel loss `-1.0` and potentially empty vectors
(`optimize.py:411-414,508-514`); its normal return uses `res.message`, not a
complete success/feasibility record. NLopt similarly catches failures
(`708-713`). List fitting discards empty outputs and sorts the remainder by RSS
(`metabolicmodel.py:3620-3638`). FluxEMU preserves orchestration, improves the
record by retaining every attempt, and requires native independent validation
before selecting a fit. Optimizer termination messages are diagnostics, not
evidence that the science or model is valid.

`MdvData` distinguishes fragment selection from per-isotopologue `use` flags
(`mdv.py:250-286,306-348,411-431`). Its export orders fragments and isotopologue
indices by sorting (`478-522`), while its `check` rejects used zero entries
without verifying the complete simplex normalization (`377-409`). FluxEMU
instead preserves declared scientific ordering and selects explicit whole MID
observations. Every selected MID must be a complete validated probability
vector; individual mass classes are never silently dropped or renormalized.
Exact zero mass remains valid. Replicate identities remain explicit.

`CarbonSource` stores isotopomer distributions and projects them onto specified
EMU atom positions (`carbonsource.py:33-61,121-152,214-274,495-530`). Its optional
natural-isotope correction and unconditional final MID normalization are not
adopted as MFA data repair. Native tracer/EMU semantics already serve this role.
`mfapyio.load_metabolic_model_reactions` reads separate stoichiometry, reaction,
and explicit atom-map fields and assigns file-order indices
(`mfapyio.py:40-154`). `load_metabolic_model` composes reaction, metabolite,
reversible, and target-fragment parsers (`468-504`). This confirms that mfapy's
text-model/import stack is a representation boundary, not a missing fitting
dependency; FluxEMU retains its canonical model and existing explicit mapping
workflow.

## Native reuse and scope boundary

The existing implementation boundaries inspected before this work were
`flux_analysis/sampling.py` (`_ReducedFluxGeometry`, `_build_reduced_geometry`,
`_sampling_affine_nullspace`, `_chebyshev_center`, `sample_prepared_flux_states`,
and `validate_flux_states`), `flux_analysis/highs.py` (compiled canonical LP and
retained-objective semantics), `execution.CanonicalFluxState`, and
`emu.stationary.evaluate_stationary`. The MFA implementation must share these
mechanisms or carefully generalize them; it must not create an incompatible
independent-flux system. Biological objective retention becomes optional for
MFA, and remains a feasibility condition when explicitly requested.

Basic workflow coverage consists of complete feasible starts, reduced-state
reconstruction, multiple tracer experiments, constrained multistart fitting,
transparent per-observation loss, all-start diagnostics, exact synthetic
recovery, and original-model validation. Measured reaction/exchange values
declared as exact or bounded canonical constraints are supported through that
existing boundary. Soft measured-flux observations, optional global search,
parallel execution, and noisy-data generation are deliberate deferrals, not
silent omissions or substitutes for the required stationary core.

The fitted objective is a divergence between MID distributions. It is not an
experimental observation law `P(Y | v)` and is not called a likelihood. The
reference's statistical tests and uncertainty procedures are not evidence for
interpreting the new fitting criterion as those later layers.

## License and attribution

[`vendor/mfapy/LICENSE`](../../vendor/mfapy/LICENSE) identifies the snapshot as
MIT licensed, copyright 2018 Fumio Matsuda, and requires preservation of its
copyright and permission notice in copies or substantial portions. The vendor
license and source remain unchanged. This implementation transfers documented
engineering concepts and adapts native FluxEMU mechanisms; it does not copy
mfapy source into production or add mfapy as a dependency. Any later direct
source adaptation would need to retain the applicable notice and be reviewed
separately. These exact source references make the workflow lineage auditable.
