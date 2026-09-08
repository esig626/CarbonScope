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

`exact_finite_composite_minimax(...)` enumerates the **complete Cartesian product** of all declared MID block count spaces and solves a randomised minimax linear programme. This is a small problem or discretised oracle. The joint outcome count grows multiplicatively across blocks; exceeding the explicit cap raises `CompositeEnumerationLimitError` and no approximation or reduced support is substituted. The LP requires the optional `testing` SciPy extra.

A Type I budget below `MIN_EXACT_COMPOSITE_EPSILON = 1e-12` is rejected as a numerical LP limit rather than silently enlarged. Null constraints are scaled by the budget before HiGHS optimisation and worst case errors are recomputed afterwards.

For `0 < lambda < 1`, `composite_renyi_score_candidate(...)` selects the minimum divergence **vertex pair** in the represented finite class. This object is only a candidate score. It is not called a joint convex class Rényi projection and is not automatically a finite blocklength least favourable pair.

Structural zeros are retained in candidate score construction. Uniform support and exponential moment conditions are checked directly over every represented H0 and H1 member. `verified_composite_renyi_score(...)` fails if those gates do not hold. A coordinate with `P*=Q*=0` is harmless only when every represented member is also zero there.

`composite_score_bound_at_order(...)` produces an analytical threshold and Type II guarantees without joint outcome enumeration. `evaluate_composite_score_test(...)` and `calibrate_composite_score_test(...)` require enumeration and therefore share the same combinatorial scalability limits as the exact minimax oracle.

Score calibration is optimal only within the fixed verified upper score threshold family. It is not labelled as unrestricted minimax equality unless comparison with `exact_finite_composite_minimax(...)` actually establishes equality for the represented finite problem.

A sampled finite flux state family is exactly the class represented to these finite solvers. CarbonScope does not claim that a finite sample is the complete underlying continuous or biological mechanism class. Test inversion into flux compatibility or confidence regions is not implemented.

## Stationary and transient EMU scope

Atom transitions must be supplied explicitly through the authoritative mappings and model representation. CarbonScope does not infer mappings from stoichiometry, names, molecular formulae or an external database.

The native stationary path does not implement natural abundance correction; experiments requiring that capability must be preprocessed by a scientifically justified external procedure before entering the current native model.

The transient implementation is fixed flux forward simulation with explicit pool quantities, requested time points and an initial unlabelled internal state. Transient inverse MFA is not implemented.

## Flux sampling

Native feasible state sampling uses hit and run on a numerically reduced affine polytope. Every returned state is checked for bounds, mass balance and retained objective feasibility. FVA endpoints are never independently sampled or assembled into a state.

A finite burn in and thinned Markov chain is still correlated. CarbonScope does not claim that a particular finite sample proves convergence, independence, biological probability or adequate mixing for every genome scale geometry.

A sampled finite flux state set can be passed explicitly to composite testing, but CarbonScope does not claim that a finite sample is the entire underlying mechanism class. That modelling choice remains the caller's responsibility.

## Numerical and scalability limits

HiGHS, NumPy and SciPy computations are subject to finite binary64 precision. Numerically singular EMU systems, ambiguous reduced geometry and unrepresentable testing inputs fail explicitly rather than being silently repaired.

Exact count space procedures scale combinatorially within each multinomial block and multiplicatively across independent blocks. Explicit caps are safety boundaries, not evidence that larger problems are statistically approximated.

The packaged real model acceptance suite provides substantial software evidence, but it is not a proof that every genome scale model, tracer experiment or biological interpretation is valid.

## Model interchange

CarbonScope loads the physical flux model from SBML Level 3 FBC and keeps isotope semantics in explicit project data. The current isotope metadata conventions are project specific rather than a standardised SBML isotope package. External SBML tools therefore cannot be assumed to preserve those isotope semantics unless they are carried separately.
