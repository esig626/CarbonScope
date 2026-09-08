# Known limitations

This page describes the current native CarbonScope implementation. The runtime Python package and CLI are still named `fluxemu`.

## Stationary MFA

Stationary MFA uses constrained multistart SLSQP over native feasible flux geometry. Successful multistart optimisation does not prove a global optimum, stationarity of every rejected start, or unique flux identifiability. Distinct flux states may produce indistinguishable MIDs, including scale nonidentifiability in suitable networks.

Exact support is preserved. If a predicted MID assigns zero mass where the observed MID is positive, the relevant divergence may be infinite; CarbonScope does not add pseudocounts or clip support to make optimisation easier.

Soft measured flux likelihood terms, global optimisation, Bayesian inference and uncertainty or confidence regions are not implemented.

## Observation laws and exact p values

The multinomial observation layer is valid only when the measurements genuinely have fixed total isotopologue count semantics. Normalised MIDs, percentages, peak areas and arbitrary intensities do not define a count total and are not converted to pseudo counts.

The exact simple null likelihood ratio p value enumerates the positive probability null count space. Enumeration is protected by an explicit outcome limit. If the limit is exceeded, CarbonScope raises an error; it does not silently substitute chi square, Monte Carlo, saddlepoint or another approximation.

## Composite testing scope

Composite testing is implemented for explicitly finite classes of genuine count observation laws on a common finite observation space.

`projected_renyi_test(...)` selects a Rényi projected pair from the explicitly supplied finite laws at one caller supplied order `0 < lambda < 1`. A finite list is not automatically a convex class, so CarbonScope does not silently invoke the stronger joint projection theorem. It evaluates the required exponential moments over every supplied law and reports separately whether the projected formula is certified for that finite class. Its calibrated threshold test is an achieved test, not a claim of unrestricted minimax optimality.

`solve_finite_minimax_test(...)` solves the unrestricted randomised minimax problem for the explicitly supplied finite classes when the complete observation space is enumerable. The linear programme is an exact characterisation of that finite problem. The numerical solution is a binary64 HiGHS computation checked against explicit feasibility tolerances, not an exact rational certificate.

A finite sampled flux ensemble is **not** the full continuous biological hypothesis family. The exact minimax value for 100 or 1000 supplied observation laws is exact only for those laws. CarbonScope does not promote it to a theorem about the entire feasible flux region. Full family certification requires the corresponding worst case, projection, or separation optimisation over the continuous family to be solved or rigorously bounded.

The current composite layer does not yet provide that continuous flux family optimisation, nuisance parameter calibration beyond explicitly supplied laws, test inversion into flux compatibility regions, or a generic observation law for continuous corrected MID measurements.

## Rényi Type II lower bounds

`bruno_converse_at_order(...)` evaluates the published simple binary lower bound at one caller supplied finite real order `lambda > 1`. CarbonScope does not perform an order search and does not claim that a finite order grid equals the continuous order envelope.

The simple Bruno theorem path requires exact mutual absolute continuity of the fixed observation law pair. Support mismatch is rejected rather than smoothed.

`composite_renyi_converse_at_order(...)` applies pairwise full law Rényi converses over explicitly supplied finite null and alternative classes and retains the strongest resulting class lower bound. It is order specific and does not claim a continuous order optimum.

## Stationary and transient EMU scope

Atom transitions must be supplied explicitly through the authoritative mappings and model representation. CarbonScope does not infer mappings from stoichiometry, names, molecular formulae or an external database.

The native stationary path does not implement natural abundance correction; experiments requiring that capability must be preprocessed by a scientifically justified external procedure before entering the current native model.

The transient implementation is fixed flux forward simulation with explicit pool quantities, requested time points and an initial unlabelled internal state. Transient inverse MFA is not implemented.

## Flux sampling

Native feasible state sampling uses hit and run on a numerically reduced affine polytope. Every returned state is checked for bounds, mass balance and retained objective feasibility. FVA endpoints are never independently sampled or assembled into a state.

A finite burn in and thinned Markov chain is still correlated. CarbonScope does not claim that a particular finite sample proves convergence, independence, biological probability or adequate mixing for every genome scale geometry.

## Numerical and scalability limits

HiGHS, NumPy and SciPy computations are subject to finite binary64 precision. Numerically singular EMU systems, ambiguous reduced geometry and unrepresentable testing inputs fail explicitly rather than being silently repaired.

Exact finite composite testing grows with the complete observation space. CarbonScope imposes an explicit outcome limit and fails rather than silently switching to an approximate testing procedure. Positive probability coefficients at or below the declared HiGHS matrix resolution are also rejected rather than dropped from the minimax LP.

The packaged real model acceptance suite provides substantial software evidence, but it is not a proof that every genome scale model, tracer experiment or biological interpretation is valid.

## Model interchange

CarbonScope loads the physical flux model from SBML Level 3 FBC and keeps isotope semantics in explicit project data. The current isotope metadata conventions are project specific rather than a standardised SBML isotope package. External SBML tools therefore cannot be assumed to preserve those isotope semantics unless they are carried separately.
