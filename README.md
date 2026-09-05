# FluxEMU

Standalone software development repository for FluxEMU.

This repository contains the reusable metabolic/isotope modelling package,
authoritative carbon-transition library, compatibility infrastructure, and
validation benchmarks.

Stage 1 supports the native public sequence:

```text
canonical model -> FBA -> FastFVA -> complete feasible flux states
                -> stationary EMU -> sample-indexed MID ensemble
```

Its production FastFVA engine is a HiGHS-native port/adaptation of the
shared-memory computational architecture in Marouen Ben Guebila's
[VFFVA](https://github.com/marouenbg/VFFVA/tree/7cf7b82505bf99aed38a2073e3ed308f79e95802),
audited at pinned commit `7cf7b82505bf99aed38a2073e3ed308f79e95802`.
FluxEMU retains its own more general linear-objective and objective-retention
semantics; it does not require the original VFFVA binary or its CPLEX, GLPK, or
MPI dependencies.

Install the standard native package from the repository root with:

```bash
python -m pip install ./codex
```

See the [Stage 1 native workflow](codex/docs/STAGE1_NATIVE_WORKFLOW.md) for the
public deterministic and ensemble APIs, dependency boundary, sampling
guarantees, validation rules, and reproducible FastFVA evidence.

Exploratory hypothesis testing, topology reconstruction, and biological
research remain in the separate fluxemu-prototype repository. Inverse MFA and
information-theoretic analysis are not part of this standalone Stage 1 engine.
