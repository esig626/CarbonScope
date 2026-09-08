# FluxEMU repository rules

1. This repository contains the standalone CarbonScope software, not exploratory research.
2. Composite hypothesis testing is now a settled CarbonScope software requirement. Do not add topology-reconstruction, biological-analysis, Bayesian-inference, or other research methodology unless it has first become a settled software requirement.
3. Keep runtime code under `src/fluxemu/`.
4. Do not vendor COBRApy, mfapy, or other compatibility runtimes. The production stack is native CarbonScope.
5. Do not infer atom mappings from stoichiometry, names, formulae, or external databases. Use explicit authoritative mappings.
6. Boolean symmetry must never generate scientific mapping branches or weights.
7. Never independently sample reaction FVA intervals; sample complete jointly feasible states and preserve canonical reaction order.
8. FVA extrema are diagnostics and must never be assembled into a flux vector.
9. Preserve exact structural zeros and support semantics in observation/testing code. Do not add pseudocounts, clipping, hidden normalisation, or inferred effective sample sizes.
10. Genuine-count multinomial laws apply only to genuine counts; never convert percentages, peak areas, arbitrary intensities, or normalised MIDs into counts.
11. Testing conventions are fixed: H0=P0=null; H1=P1=alternative; Type I=P0(decide H1); Type II=P1(decide H0); reverse Rényi is D_lambda(P1||P0); forward Rényi is D_lambda(P0||P1).
12. Rényi-bound evaluation is order-specific for any supplied finite real lambda. Do not substitute a finite grid or claim it optimises the continuous-order envelope.
13. The exact simple-null p-value is P0{LLR(Y)>=LLR(y_obs)}. If exact enumeration exceeds its explicit limit, fail rather than silently switching methods.
14. A finite sampled flux ensemble is only a numerical representation of a larger hypothesis family. Never report finite-family testing as certified testing of the full continuous feasible family unless the missing optimisation has been solved or bounded.
15. Preserve declared scientific ordering exactly.
16. Run focused tests before each coherent implementation commit and full relevant regression tests before publication.
17. Never leave substantial work only in an uncommitted working tree.
