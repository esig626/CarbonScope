# Known limitations

This page describes the current native CarbonScope implementation. The runtime Python package and CLI are still named `fluxemu`.

## Stationary MFA

Stationary MFA uses constrained multistart SLSQP over native feasible flux geometry. Successful multistart optimisation does not prove a global optimum, stationarity of every rejected start, or unique flux identifiability. Distinct flux states may produce indistinguishable MIDs, including scale nonidentifiability in suitable networks.

Exact support is preserved. If a predicted MID assigns zero mass where the observed MID is positive, the relevant divergence may be infinite; CarbonScope does not add pseudocounts or clip support to make optimisation easier.

Soft measured flux likelihood terms, global optimisation, Bayesian inference and uncertainty or confidence regions are not implemented.

## Observation laws and exact p values

The multinomial observation layer is valid only when the measurements genuinely have fixed total isotopologue count semantics. Normalised MIDs, percentages, peak areas and arbitrary intensities do not define a count total and are not converted to pseudo counts.

The exact simple null likelihood ratio p value enumerates the positive probability null count space. Enumeration is protected by an explicit outcome limit. If the limit is exceeded, CarbonScope raises an error; it does not silently substitute chi square, Monte Carlo, saddlepoint or another approximation.

A generic composite p value is not implemented. The existing p value remains explicitly simple null and alternative specific.

## Simple Rényi Type II lower bounds

`bruno_converse_at_order(...)` evaluates the published simple binary lower bound at one caller supplied finite real order `lambda > 1`. CarbonScope does not perform an order search and does not claim that a finite order grid equals the continuous order envelope.

The Bruno theorem path requires exact mutual absolute continuity of the fixed observation law pair. Support mismatch is rejected rather than smoothed.

## Finite composite testing

Composite testing currently supports **explicit finite represented classes** of complete observable laws. A member may contain one genuine count multinomial MID block or an explicitly independent ordered product of multiple blocks. Corresponding H0 and H1 members must share the complete block identity, count total and mass class structure.

The stationary bridge maps finite tuples of complete feasible `CanonicalFluxState` records through the native observation chain into one product observation law per state. Multiple blocks require explicit `independent_blocks=True`.

The implementation does not silently convexify a finite family, use member sampling frequency as a prior, or pass hidden state IDs or flux coordinates into the decision rule. Continuous or implicitly parameterised mechanism classes, optimisation over a continuum of flux or nuisance states, random effects and observation kernel uncertainty are not yet implemented.

`composite_renyi_converse_at_order(...)` evaluates the order specific finite family converse at one supplied finite `lambda > 1` using full product law Rényi divergence. It does not optimise over the continuous order envelope.

`exact_finite_composite_minimax(...)` enumerates the **complete Cartesian product** of all declared MID block count spaces and constructs the unrestricted randomised minimax linear programme. Its mathematical characterisation is exact; a particular floating-point solution is accepted only after numerical feasibility and optimality checks under declared tolerances. This is a small problem or discretised oracle. The joint outcome count grows multiplicatively across blocks; exceeding the explicit cap raises `CompositeEnumerationLimitError` and no approximation or reduced support is substituted. The LP requires the optional `testing` SciPy extra.

A Type I budget below `MIN_EXACT_COMPOSITE_EPSILON = 1e-12` is rejected rather than silently enlarged. Null constraints are scaled by `1/epsilon`. HiGHS is configured with `small_matrix_value=1e-12`, and every nonzero scaled LP coefficient at or below that magnitude triggers refusal before solving. Positive probabilities must not disappear through solver coefficient dropping or floating-point underflow.

Consequently, full-support binary count laws necessarily refuse at `n >= 40`, and full-support ternary laws at `n >= 26`: their least probable extreme outcomes are no larger than `2^-n` and `3^-n`, respectively. Refusal can occur earlier for unbalanced laws and products. Count totals below these ceilings do not ensure acceptance. The validation report records the concrete laws and budgets used in boundary tests.

Accepted results require valid decision-variable bounds, primal and dual checks, objective/epigraph consistency and an acceptable optimality gap. Errors are recomputed directly as `E_P[phi]` and `E_Q[1-phi]`; solver variables are never clipped into range. The returned numerical tolerance and diagnostics delimit the floating-point claim, which is not a rigorous exact-arithmetic or interval certificate. See [COMPOSITE_TESTING.md](COMPOSITE_TESTING.md) for the refusal policy and result fields.

For `0 < lambda < 1`, `composite_renyi_score_candidate(...)` selects the minimum divergence **vertex pair** in the represented finite class. This object is only a candidate score. It is not called a joint convex class Rényi projection and is not automatically a finite blocklength least favourable pair.

Structural zeros are retained in candidate score construction. Uniform support and exponential moment conditions are checked directly over every represented H0 and H1 member; a finite list is not presumed convex. Candidate tie tolerance cannot relax the uniform inequalities. `verified_composite_renyi_score(...)` fails when those numerical gates do not establish the required conditions. A coordinate with `P*=Q*=0` is harmless only when every represented member is also zero there.

`composite_score_bound_at_order(...)` requires verified uniform moments and produces an analytical threshold and Type II guarantees without joint outcome enumeration. `evaluate_composite_score_test(...)` and `calibrate_composite_score_test(...)` require enumeration and therefore share the same combinatorial scalability limits as the finite minimax oracle. Calibration may use a well-defined candidate whose analytical moment gates fail: it computes the actual finite-family errors directly, without certifying the failed projected formula. Undefined scores and unrepresentable positive masses are refused.

Score calibration returns an achieved test and is optimal only within the fixed upper-score threshold family. Its worst-case Type II error can strictly exceed unrestricted minimax. Agreement with `exact_finite_composite_minimax(...)` establishes numerical equality only within the stated tolerances for that represented finite problem.

A sampled finite flux state family is exactly the class represented to these finite solvers. Rigorous optimisation or bounds over the complete continuous feasible flux family are not implemented, so a finite sample does not certify uniform testing performance over that larger family. Test inversion into flux compatibility or confidence regions is not implemented.

## Declarative hypothesis workflow

`fluxemu test-hypotheses` and `run_hypothesis_testing_workflow(...)` orchestrate the existing native layers from a common SBML/FBC model, separate H0/H1 constraints and ordered experiment/count declarations. V1 constraints only intersect explicit reaction lower/upper bounds with the original physical bounds; they do not relax the common model, introduce arbitrary coupled inequalities or infer pathway biology. H0 and H1 must define distinct feasible regions, although overlap is allowed.

Each finite family is sampled from its complete constrained steady-state region without a retained objective constraint. The native experiment's forward-analysis `fva_fraction_of_optimum` does not restrict the hypothesis region. Bounds fixing a single reaction are supported; a general objective-retention or coupled biological constraint requires a separately specified extension.

The workflow requires native stationary isotope experiments sharing the same authoritative canonical isotope model and genuine positive integer count totals. Multiple declared observation blocks require explicit independence, including blocks labelled as replicates. Neither a shared flux state nor separate replicate IDs establishes independence. The workflow does not provide correlated block laws, realistic non-count LC-MS/GC-MS models or biological replicate random effects.

Malformed scientific input, infeasible or indistinguishable hypothesis regions, failed state generation and incompatible/undefined native observation construction invalidate the workflow. Optional statistical procedures may instead produce explicit refused outcomes with reasons in an otherwise valid report. Enumeration and numerical certification limits remain unchanged, and a workflow that finishes successfully does not imply every requested statistic was available.

The output describes finite-class testing performance and provenance. It does not perform a realised-data composite decision, provide generic composite p-values, invert tests, identify a true flux or mechanism, or establish a continuous-family guarantee. A converse supplies an impossibility constraint; an achieved score procedure need not be unrestricted minimax. The compact acceptance fixture is software infrastructure, not a biological showcase. Issues #24–#28 remain separate work; this workflow does not implement their non-count observation, compatibility/inversion, continuous-family or other scientific extensions.

See [HYPOTHESIS_WORKFLOW.md](HYPOTHESIS_WORKFLOW.md) for the schema, result labels and refusal contract.

## Stationary and transient EMU scope

Atom transitions must be supplied explicitly through the authoritative mappings and model representation. CarbonScope does not infer mappings from stoichiometry, names, molecular formulae or an external database.

The native stationary path does not implement natural abundance correction; experiments requiring that capability must be preprocessed by a scientifically justified external procedure before entering the current native model.

The transient implementation is fixed flux forward simulation with explicit pool quantities, requested time points and an initial unlabelled internal state. Transient inverse MFA is not implemented.

## Flux sampling

Native feasible state sampling uses hit and run on a numerically reduced affine polytope. Every returned state is checked for bounds, mass balance and any explicitly retained objective constraint. The hypothesis workflow retains no objective constraint. FVA endpoints are never independently sampled or assembled into a state.

A finite burn in and thinned Markov chain is still correlated. CarbonScope does not claim that a particular finite sample proves convergence, independence, biological probability or adequate mixing for every genome scale geometry.

A sampled finite flux state set can be passed explicitly to composite testing or generated by the declarative workflow. CarbonScope does not claim that a finite sample is the entire underlying mechanism class. That modelling choice remains the caller's responsibility.

## Numerical and scalability limits

HiGHS, NumPy and SciPy computations are subject to finite binary64 precision. Numerically singular EMU systems, ambiguous reduced geometry and unrepresentable testing inputs fail explicitly rather than being silently repaired.

Exact count space procedures scale combinatorially within each multinomial block and multiplicatively across independent blocks. Explicit caps are safety boundaries, not evidence that larger problems are statistically approximated.

The packaged real model acceptance suite provides substantial software evidence, but it is not a proof that every genome scale model, tracer experiment or biological interpretation is valid.

## Model interchange

CarbonScope loads the physical flux model from SBML Level 3 FBC and keeps isotope semantics in explicit project data. The current isotope metadata conventions are project specific rather than a standardised SBML isotope package. External SBML tools therefore cannot be assumed to preserve those isotope semantics unless they are carried separately.
