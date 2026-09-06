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
optional optimiser dependency:

```bash
python -m pip install './codex[mfa]'
python codex/examples/stationary_mfa_recovery.py
```

`fluxemu.fit_stationary_mfa` fits complete feasible states by minimising the
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

For measurements with genuine isotopologue-count semantics,
`fluxemu.observation` provides an explicit fixed-total multinomial law,
raw count records, reproducible sampling, and a native stationary EMU bridge.
It also exposes the exact multinomial KL/Rényi identities. Count totals must
be supplied explicitly: percentages, peak areas, normalised MIDs, and arbitrary
intensities are never converted into pseudo-counts. This separate layer leaves
the existing MFA objective unchanged. See the
[stationary observation-law guide](codex/docs/STATIONARY_OBSERVATION_LAW.md)
for the API, likelihood identity, numerical boundaries, and synthetic example.

For two fixed feasible flux hypotheses, `fluxemu.testing` evaluates the
order-specific finite-sample Type-II lower bounds of Bruno, Vandenbroucque &
Esposito, arXiv:2601.09550v2. It reuses the stationary count-law bridge, checks
the theorem's mutual-absolute-continuity assumption exactly, and exposes both
Rényi directions, raw bound components, provenance, and count-sample
log-likelihood ratios.

For a realised genuine-count observation it can also report the exact
simple-null likelihood-ratio p-value
`P0{log(P1(Y)/P0(Y)) >= log(P1(y_obs)/P0(y_obs))}`. The p-value is a separate
sample-specific quantity: it is not a divergence and it is not a Type-II lower
bound. Exact p-value evaluation enumerates the positive-probability null count
space up to an explicit caller-controlled limit; FluxEMU does not silently
substitute a chi-square or Monte Carlo approximation when that limit is
exceeded.

Ordinary Rényi-bound evaluation accepts any finite real order greater than one
within the documented numerical limits; no grid or global order optimisation
is substituted. Run:

```bash
python codex/examples/simple_binary_flux_discrimination.py
```

See the [simple binary bounds guide](codex/docs/SIMPLE_BINARY_RENYI_BOUNDS.md)
for the exact error convention, p-value semantics, count boundary, public APIs,
and validation, and the
[technical p-value note](codex/docs/technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex)
for the mathematical bridge between likelihood-ratio p-values, KL divergence,
and Rényi divergence. This layer uses the base dependencies and leaves the
existing MFA fitting objective unchanged.

Exploratory composite hypothesis testing, topology reconstruction, and biological
research remain in the separate fluxemu-prototype repository.
