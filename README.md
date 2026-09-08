# CarbonScope

CarbonScope is a native Python framework for using carbon isotope tracing to interrogate metabolic hypotheses through forward modelling and finite-sample inference.

The scientific aim is not to treat one fitted flux vector as biological truth. CarbonScope is built around feasible families of metabolic states, the isotope distributions those states can generate, and the question of what an experiment can actually distinguish.

See [docs/SCIENTIFIC_WORKFLOW.md](docs/SCIENTIFIC_WORKFLOW.md) for the intended scientific workflow.

## Current scope

CarbonScope currently provides the following native workflow:

```text
SBML/FBC model
  -> canonical metabolic and isotope model
  -> HiGHS FBA / FVA
  -> complete jointly feasible flux states
  -> stationary or transient EMU prediction
  -> MID ensembles
  -> explicit observation laws for genuine counts
  -> simple binary inference
  -> finite-class composite inference
```

For finite composite problems, CarbonScope deliberately separates three different objects:

```text
projected Rényi test        -> achieved upper bound on beta*
finite minimax LP           -> numerical solution of the unrestricted finite problem
composite Rényi converse    -> lower bound on beta*
```

When all three calculations are available, they can be compared through

```text
Rényi converse <= beta* <= achieved projected-test error
```

This separation is intentional. The projected Rényi test is a principled explicit test, but it is not generally minimax optimal at finite sample size. The finite minimax LP searches over all randomised tests on the enumerated observation space and therefore provides the correct optimisation problem for the explicitly supplied finite law classes.

A finite sampled flux ensemble is still only a numerical representation of a larger continuous hypothesis family. CarbonScope does **not** claim that solving the finite problem certifies the complete continuous feasible flux region.

See [docs/COMPOSITE_TESTING.md](docs/COMPOSITE_TESTING.md).

## Runtime interface

The project name is CarbonScope. The existing Python package and command-line interface remain `fluxemu` for compatibility.

CarbonScope does not depend on COBRApy or mfapy at runtime.

## Installation

From the repository root:

```bash
python -m pip install .
```

Stationary MFA and transient integration use SciPy through optional extras:

```bash
python -m pip install '.[mfa]'
python -m pip install '.[transient]'
```

## Native stationary workflow

The command-line interface accepts an SBML Level 3 FBC model and a CarbonScope experiment YAML:

```bash
fluxemu run --model model.xml --experiment experiment.yaml --output results
```

The public Python orchestration API includes `run_native_fba`, `run_native_fva`, `run_native_stationary_analysis`, and complete-feasible-state ensemble functions in `fluxemu.analysis`.

FVA endpoints are diagnostics only. CarbonScope never combines independently optimised FVA coordinates into a flux vector. Ensemble calculations use complete jointly feasible states.

See [docs/STAGE1_NATIVE_WORKFLOW.md](docs/STAGE1_NATIVE_WORKFLOW.md) and [docs/EXPERIMENT_FORMAT.md](docs/EXPERIMENT_FORMAT.md).

## Stationary MFA

MFA is available as a computational tool, not as a claim that one fitted flux vector is uniquely true.

`fluxemu.fit_stationary_mfa` fits complete feasible flux states by minimising a plain sum of

```text
D_alpha(observed MID || predicted MID)
```

with exact KL at `alpha=1` and finite positive-real Rényi orders otherwise. External rounded fractions, percentages, or nonnegative intensity vectors can be explicitly closed to the probability simplex with `fluxemu.normalise_mid`; fitting and divergence functions never silently renormalise inputs.

The optimiser is constrained by native feasible flux geometry and uses multistart SLSQP. It does not claim a global optimum or unique identifiability.

See [docs/STATIONARY_MFA_RENYI_CORE.md](docs/STATIONARY_MFA_RENYI_CORE.md) and [docs/MID_PREPROCESSING.md](docs/MID_PREPROCESSING.md).

## Genuine-count observation laws

For measurements with genuine count semantics, `fluxemu.observation` provides explicit fixed-total multinomial laws

```text
Y | v ~ Multinomial(n, p_v)
```

with declared count totals, exact structural-zero support, reproducible sampling, and independent-product divergence identities. Percentages, peak areas, normalised MIDs, and arbitrary intensities are never converted into pseudo-counts and no effective sample size is inferred.

See [docs/STATIONARY_OBSERVATION_LAW.md](docs/STATIONARY_OBSERVATION_LAW.md).

## Simple binary inference

For two fixed feasible flux states, CarbonScope supports:

- `log_likelihood_ratio(...)`, with `LLR(y)=log P1(y)-log P0(y)`;
- `likelihood_ratio_p_value(...)`, the exact simple-null tail `P0{LLR(Y) >= LLR(y_obs)}` when the null count space is enumerable within the explicit limit;
- `bruno_converse_at_order(...)`, the order-specific finite-sample Bruno et al. lower bound on optimal Type II error under a declared Type I budget.

The conventions are fixed:

```text
H0 = P0 = null
H1 = P1 = alternative
Type I  = P0(decide H1)
Type II = P1(decide H0)
reverse Rényi = D_lambda(P1 || P0)
forward Rényi = D_lambda(P0 || P1)
```

A p-value is realised-data evidence under the fixed null. A Rényi Type II lower bound is a constraint on achievable testing performance for the fixed law pair. Neither is a substitute for the other.

CarbonScope evaluates any supplied finite real `lambda > 1` subject to explicit numerical limits. It does not replace the continuous Rényi-order envelope with a finite grid.

See [docs/SIMPLE_BINARY_RENYI_BOUNDS.md](docs/SIMPLE_BINARY_RENYI_BOUNDS.md) and [docs/technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex](docs/technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex).

## Composite inference

For explicitly finite null and alternative law classes, `fluxemu.testing` provides three complementary calculations.

### Projected Rényi test

`projected_renyi_test(...)` constructs a projected score at a supplied order `0 < lambda < 1`, checks the required uniform moment conditions over the declared finite classes, and calibrates a threshold test against the uniform Type I constraint.

The resulting worst-case Type II error is an **achieved upper bound** on the unrestricted minimax optimum. The projected test may be substantially suboptimal, so CarbonScope does not identify it with `beta*` unless an independent argument justifies that reduction.

### Composite Rényi converse

`composite_renyi_converse_at_order(...)` applies the finite-sample converse at one supplied order `lambda > 1` and gives a lower bound on the unrestricted minimax Type II error.

The null and alternative divergence directions remain explicit. No extra sample-size multiplier is inserted when the complete multinomial observation law already contains the genuine count total.

### Finite minimax LP

`solve_finite_minimax_test(...)` enumerates the complete finite observation space and solves

```text
minimise t

subject to
    E_P[phi] <= epsilon         for every P in C0
    E_Q[1 - phi] <= t           for every Q in C1
    0 <= phi(y) <= 1
```

over all randomised tests `phi` on that finite space.

The LP formulation is mathematically exact for the explicitly supplied finite law classes. The returned value is a floating-point numerical solution on the represented probability table, independently checked against declared feasibility and optimality tolerances. It is not an exact-arithmetic certificate.

See [docs/COMPOSITE_TESTING.md](docs/COMPOSITE_TESTING.md).

## Numerical limits of the finite LP

CarbonScope fails explicitly when the represented finite problem is outside the solver's declared numerical resolution. It does not delete tiny positive support or silently change the test.

Current important limits include:

- LP validation tolerance: `1e-7`;
- positive coefficient cutoff near `1e-12`;
- fully supported binary multinomial laws are refused from count total `n >= 40`;
- fully supported ternary multinomial laws are refused from count total `n >= 26`;
- unequal probabilities and independent products may reach the coefficient limit earlier;
- the default complete-geometry enumeration limit is finite and exceeding it raises an explicit error.

These are implementation limits, not mathematical impossibility results. The Rényi converse may remain available for problems that the finite LP refuses because it does not require full outcome enumeration.

## Continuous flux families remain unresolved

The current composite implementation is finite-class first.

If

```text
C0 = {P_v : v in V0}
C1 = {P_w : w in V1}
```

for continuous feasible flux regions `V0` and `V1`, testing a finite sampled subset does not certify the whole family.

A genuine continuous-family result would require global optimisation or rigorous bounds for quantities such as

```text
sup_{v in V0} E_{P_v}[phi]
sup_{w in V1} E_{P_w}[1 - phi]
```

and the corresponding Rényi projections or worst-case divergences over the complete families.

That problem is not currently solved by CarbonScope.

## Transient forward EMU

`fluxemu.emu` contains a native fixed-flux transient EMU integrator with explicit pool quantities, time points, and initial unlabelled internal state. This is forward simulation only; transient inverse MFA is not implemented.

## Validation

The composite testing layer has been subjected to adversarial validation using independent rational Neyman-Pearson calculations, independent finite LP constructions, direct PMF enumeration, metamorphic tests, stationary provenance checks, numerical pathology campaigns, and large seeded simulation campaigns on Python 3.11 and 3.12.

The validation is evidence for the finite implementation within its documented numerical tolerances. It does not convert floating-point computation into exact arithmetic, establish projected minimax optimality, or solve the continuous flux-family problem.

The wider test suite also includes analytical controls, exact finite-count enumeration, independent deterministic-test oracles, a direct full-isotopomer implementation of the Antoniewicz TCA benchmark, and the packaged E. coli acceptance model.

The production FastFVA architecture is a HiGHS-native adaptation of the shared-memory computational design of Marouen Ben Guebila's VFFVA. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

See [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) for the complete current limitations.
