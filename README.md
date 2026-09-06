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

Stationary MFA is available as a separate native fitting layer with the
optional optimizer dependency:

```bash
python -m pip install './codex[mfa]'
python codex/examples/stationary_mfa_recovery.py
```

`fluxemu.fit_stationary_mfa` fits complete feasible states by minimizing the
plain sum of `D_alpha(observed MID || predicted MID)`, with exact KL at order
one and finite positive-real Rényi orders. Experimental fractions,
percentages, or non-negative intensity vectors can first be explicitly closed
to the probability simplex with `fluxemu.normalise_mid`; the divergence and
fitting layers never silently renormalise their inputs. See the
[stationary MFA workflow](codex/docs/STATIONARY_MFA_RENYI_CORE.md) for the public
API, support semantics, multistart diagnostics, and identifiable/non-identifiable
recovery examples, the
[explicit MID preprocessing guide](codex/docs/MID_PREPROCESSING.md) for experimental
input normalisation, and the
[mfapy engineering comparison](codex/docs/MFAPY_ENGINEERING_COMPARISON.md) for
the audited reference lineage.

Exploratory hypothesis testing, topology reconstruction, and biological
research remain in the separate fluxemu-prototype repository.
