# FluxEMU repository rules

1. This repository contains the standalone FluxEMU software, not exploratory research.
2. Finite composite binary testing is a settled software requirement. Production scope is explicit finite H0/H1 families of genuine-count MID laws or complete feasible flux states. Do not silently convexify a finite family, infer a continuum uncertainty class, or import unresolved composite-testing research.
3. A Rényi-minimising pair is not automatically a finite-blocklength least-favourable pair. Exact reduction requires a separately verified ordering/optimality result; otherwise report minimax, converse, projected-test, and calibration quantities as distinct objects.
4. Keep runtime code under `src/fluxemu/`.
5. Do not vendor COBRApy, mfapy, or other compatibility runtimes. The production stack is native FluxEMU.
6. Do not infer atom mappings from stoichiometry, names, formulae, or external databases. Use explicit authoritative mappings.
7. Boolean symmetry must never generate scientific mapping branches or weights.
8. Never independently sample reaction FVA intervals; sample complete jointly feasible states and preserve canonical reaction order.
9. FVA extrema are diagnostics and must never be assembled into a flux vector.
10. Preserve exact structural zeros and support semantics in observation/testing code. Do not add pseudocounts, clipping, hidden normalisation, or inferred effective sample sizes.
11. Genuine-count multinomial laws apply only to genuine counts; never convert percentages, peak areas, arbitrary intensities, or normalised MIDs into counts.
12. Testing roles are fixed: H0=P0/null; H1=P1/alternative; Type I=P0(decide H1); Type II=P1(decide H0). Composite Type I and Type II errors are worst-case suprema over their declared H0/H1 families.
13. Simple Bruno and composite converse evaluation are order-specific for a supplied finite real lambda>1. Composite projected achievability is order-specific for a supplied 0<lambda<1. Do not substitute a finite grid or claim a continuous-order optimum unless a dedicated continuous optimisation has actually been performed and validated.
14. The exact simple-null p-value is P0{LLR(Y)>=LLR(y_obs)}. If exact enumeration exceeds its explicit limit, fail rather than silently switching methods.
15. Exact finite composite minimax evaluation must enumerate the complete declared count space and fail at its explicit cap rather than substitute an approximation.
16. Preserve declared scientific ordering exactly.
17. Run focused tests before each coherent implementation commit and full relevant regression tests before publication.
18. Never leave substantial work only in an uncommitted working tree.
