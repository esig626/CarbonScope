# Known limitations

This page describes the current native FluxEMU implementation.

## Stationary MFA

Stationary MFA uses constrained multistart SLSQP over native feasible flux geometry. Successful multistart optimisation does not prove a global optimum, stationarity of every rejected start, or unique flux identifiability. Distinct flux states may produce indistinguishable MIDs, including scale non-identifiability in suitable networks.

Exact support is preserved. If a predicted MID assigns zero mass where the observed MID is positive, the relevant divergence may be infinite; FluxEMU does not add pseudocounts or clip support to make optimisation easier.

Soft measured-flux likelihood terms, global optimisation, Bayesian inference and uncertainty/confidence regions are not implemented.

## Observation laws and exact p-values

The multinomial observation layer is valid only when the measurements genuinely have fixed-total isotopologue-count semantics. Normalised MIDs, percentages, peak areas and arbitrary intensities do not define a count total and are not converted to pseudo-counts.

The exact simple-null likelihood-ratio p-value enumerates the positive-probability null count space. Enumeration is protected by an explicit outcome limit. If the limit is exceeded, FluxEMU raises an error; it does not silently substitute chi-square, Monte Carlo, saddlepoint or another approximation.

A generic composite p-value is not implemented. The existing p-value remains explicitly simple-null and alternative-specific.

## Simple Rényi Type-II lower bounds

`bruno_converse_at_order(...)` evaluates the published simple-binary lower bound at one caller-supplied finite real order `lambda > 1`. FluxEMU does not perform an order search and does not claim that a finite order grid equals the continuous-order envelope.

The Bruno theorem path requires exact mutual absolute continuity of the fixed observation-law pair. Support mismatch is rejected rather than smoothed.

## Finite composite testing

Composite testing currently supports **explicit finite** H0 and H1 families of genuine-count categorical MID laws with a common count total and mass-class space. The stationary flux bridge maps finite tuples of complete feasible `CanonicalFluxState` records to such families.

The implementation does not silently convexify a finite family. Continuous or implicitly parameterised uncertainty classes, numerical optimisation over a continuum of flux states, nuisance distributions, random effects and observation-kernel uncertainty are not implemented.

`composite_renyi_converse_at_order(...)` evaluates the arbitrary-finite-class converse at one supplied finite `lambda > 1`. It does not optimise over the continuous-order envelope.

`exact_finite_composite_minimax(...)` enumerates the complete multinomial count space and solves a randomised minimax linear programme. Enumeration has an explicit outcome cap. Exceeding it raises `CompositeEnumerationLimitError`; no approximation or reduced support is substituted. The LP requires the optional `testing` SciPy extra.

The order-below-one projected path currently requires full support for every declared member. Structural zeros remain supported by the exact minimax and converse paths. A finite-family Rényi-minimising pair is accepted for projected testing only if both uniform composite moment inequalities are directly verified over every declared member.

A verified Rényi-minimising pair is **not** automatically a finite-blocklength least-favourable pair. `calibrate_composite_projected_test(...)` is optimal only within the fixed projected-score upper-threshold family. It is not labelled as unrestricted minimax equality unless comparison with `exact_finite_composite_minimax(...)` actually establishes equality for the represented finite problem.

The stationary composite bridge currently supports exactly one experiment/target/replicate genuine-count block. General non-identical independent product-block composite testing is not inferred from the simple product-law API.

Test inversion into flux compatibility/confidence regions is not implemented.

## Stationary and transient EMU scope

Atom transitions must be supplied explicitly through the authoritative mappings/model representation. FluxEMU does not infer mappings from stoichiometry, names, molecular formulae or an external database.

The native stationary path does not implement natural-abundance correction; experiments requiring that capability must be preprocessed by a scientifically justified external procedure before entering the current native model.

The transient implementation is fixed-flux forward simulation with explicit pool quantities, requested time points and an initial unlabelled internal state. Transient inverse MFA is not implemented.

## Flux sampling

Native feasible-state sampling uses hit-and-run on a numerically reduced affine polytope. Every returned state is checked for bounds, mass balance and retained-objective feasibility. FVA endpoints are never independently sampled or assembled into a state.

A finite burn-in/thinned Markov chain is still correlated. FluxEMU does not claim that a particular finite sample proves convergence, independence, biological probability, or adequate mixing for every genome-scale geometry.

A sampled finite flux-state set can be passed explicitly to composite testing, but FluxEMU does not claim that a finite sample is the entire underlying mechanism class. That modelling choice remains the caller's responsibility.

## Numerical and scalability limits

HiGHS, NumPy and SciPy computations are subject to finite binary64 precision. Numerically singular EMU systems, ambiguous reduced geometry and unrepresentable testing inputs fail explicitly rather than being silently repaired.

Exact count-space procedures scale combinatorially with count total and number of mass classes. Explicit caps are safety boundaries, not evidence that larger problems are statistically approximated.

The packaged real-model acceptance suite provides substantial software evidence, but it is not a proof that every genome-scale model, tracer experiment or biological interpretation is valid.

## Model interchange

FluxEMU loads the physical flux model from SBML Level 3 FBC and keeps isotope semantics in explicit FluxEMU data. The current metadata conventions are FluxEMU-specific rather than a standardised SBML isotope package. External SBML tools therefore cannot be assumed to preserve FluxEMU-specific isotope semantics unless those semantics are carried separately.
