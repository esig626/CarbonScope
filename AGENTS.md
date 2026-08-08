# FluxEMU standalone repository rules

1. This repository contains the standalone FluxEMU software, not exploratory research.
2. Do not add composite-hypothesis-testing, topology-reconstruction, or biological-analysis research here.
3. Treat fluxemu-prototype as a separate research repository.
4. Keep runtime code under codex/src/fluxemu/.
5. Treat vendor/ as read-only third-party source.
6. Do not require an mfapy text model at runtime.
7. Do not create a user-maintained reaction-map CSV.
8. Do not infer atom mappings from stoichiometry.
9. Boolean symmetry must never generate scientific mapping branches or weights.
10. Do not independently sample reaction FVA intervals; sample complete feasible states.
11. Preserve declared scientific ordering exactly.
12. Run focused tests before every implementation commit.
13. Commit each coherent Cloud milestone before beginning the next one.
14. Never leave substantial Cloud work only in an uncommitted working tree.
