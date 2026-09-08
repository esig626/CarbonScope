# FluxEMU repository rules

1. This repository contains the standalone FluxEMU software, not exploratory research.
2. Finite composite binary testing is a settled software requirement. Production scope is explicit finite H0/H1 families of complete genuine-count observation laws or complete feasible flux states mapped to those laws. A complete law may be an explicitly independent ordered product of MID count blocks. Do not silently convexify a finite family, infer a continuum uncertainty class, or import unresolved composite-testing research.
3. A finite-family Rényi-minimising vertex pair is a candidate score, not automatically a joint convex-class Rényi projection or a finite-blocklength least-favourable pair. Exact reduction requires a separately verified ordering/optimality result; otherwise report minimax, converse, score-bound, score-calibration, and actual-error quantities as distinct objects.
4. Keep runtime code under `src/fluxemu/`.
5. Do not vendor COBRApy, mfapy, or other compatibility runtimes. The production stack is native FluxEMU.
6. Do not infer atom mappings from stoichiometry, names, formulae, or external databases. Use explicit authoritative mappings.
7. Boolean symmetry must never generate scientific mapping branches or weights.
8. Never independently sample reaction FVA intervals; sample complete jointly feasible states and preserve canonical reaction order.
9. FVA extrema are diagnostics and must never be assembled into a flux vector.
10. Preserve exact structural zeros and support semantics in observation/testing code. Do not add pseudocounts, clipping, hidden normalisation, or inferred effective sample sizes.
11. Genuine-count multinomial laws apply only to genuine counts; never convert percentages, peak areas, arbitrary intensities, or normalised MIDs into counts.
12. Testing roles are fixed: H0=P0/null; H1=P1/alternative; Type I=P0(decide H1); Type II=P1(decide H0). Composite Type I and Type II errors are worst-case suprema over their declared H0/H1 families.
13. Independence between multiple MID count blocks must be declared explicitly. Do not infer joint-product semantics from shared model origin or data shape.
14. Simple Bruno and composite converse evaluation are order-specific for a supplied finite real lambda>1. Finite-family candidate-score evaluation is order-specific for a supplied 0<lambda<1. Do not substitute a finite grid or claim a continuous-order optimum unless a dedicated continuous optimisation has actually been performed and validated.
15. The exact simple-null p-value is P0{LLR(Y)>=LLR(y_obs)}. If exact enumeration exceeds its explicit limit, fail rather than silently switching methods.
16. Exact finite composite minimax evaluation must enumerate the complete declared joint count space and fail at its explicit cap rather than substitute an approximation. It is a small-problem/discretised oracle, not evidence that a sampled finite class exhausts a larger mechanism family.
17. Composite decision rules consume observable laws only. Flux coordinates, state IDs, class sampling frequencies, inverse-MFA estimates, priors, or averages must not enter the decision statistic unless a separately specified future model explicitly changes the statistical problem.
18. Preserve declared scientific ordering exactly.
19. Run focused tests before each coherent implementation commit and full relevant regression tests before publication.
20. Never leave substantial work only in an uncommitted working tree.
