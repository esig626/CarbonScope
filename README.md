# CarbonScope

CarbonScope is a native Python framework for using carbon isotope tracing to interrogate metabolic hypotheses through forward modelling and finite sample inference.

The scientific aim is not to treat one fitted flux vector as biological truth. CarbonScope is built around feasible families of metabolic states, the isotope distributions those states can generate, and the question of what an experiment can actually distinguish.

See [docs/SCIENTIFIC_WORKFLOW.md](docs/SCIENTIFIC_WORKFLOW.md) for the intended scientific workflow.

## Current scope

The current release provides the numerical and statistical foundations for that programme:

```text
SBML/FBC model
  -> canonical metabolic and isotope model
  -> HiGHS FBA / FVA
  -> complete jointly feasible flux states
  -> stationary or transient EMU prediction
  -> MID ensembles
  -> stationary MFA when required
  -> explicit observation laws for genuine counts
  -> simple binary likelihood ratio evidence
  -> finite sample Rényi bounds
  -> finite class composite tests
  -> exact minimax LP for tractable finite composite problems
```

Composite testing is currently implemented for explicitly finite classes of genuine count observation laws. This includes an explicit projected Rényi threshold test, order specific composite converse bounds, and the unrestricted minimax optimum for enumerable finite problems.

A sampled flux ensemble is still only a numerical representation of a larger continuous hypothesis family. CarbonScope does not claim that testing the sampled laws certifies the full continuous feasible flux region unless the corresponding worst case optimisation has itself been solved or bounded.

See [docs/COMPOSITE_TESTING.md](docs/COMPOSITE_TESTING.md).

## Runtime interface

The project has been renamed CarbonScope. The existing Python package and command line interface remain `fluxemu` for compatibility.

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

The command line interface accepts an SBML Level 3 FBC model and a CarbonScope experiment YAML:

```bash
fluxemu run --model model.xml --experiment experiment.yaml --output results
```

The public Python orchestration API includes `run_native_fba`, `run_native_fva`, `run_native_stationary_analysis`, and complete feasible state ensemble functions in `fluxemu.analysis`.

FVA endpoints are diagnostics only. CarbonScope never combines independently optimised FVA coordinates into a flux vector. Ensemble calculations use complete jointly feasible states.

See [docs/STAGE1_NATIVE_WORKFLOW.md](docs/STAGE1_NATIVE_WORKFLOW.md) and [docs/EXPERIMENT_FORMAT.md](docs/EXPERIMENT_FORMAT.md).

## Stationary MFA

MFA is available as a computational tool, not as a claim that one fitted flux vector is uniquely true.

`fluxemu.fit_stationary_mfa` fits complete feasible flux states by minimising a plain sum of

```text
D_alpha(observed MID || predicted MID)
```

with exact KL at `alpha=1` and finite positive real Rényi orders otherwise. External rounded fractions, percentages, or nonnegative intensity vectors can be explicitly closed to the probability simplex with `fluxemu.normalise_mid`; fitting and divergence functions never silently renormalise inputs.

The optimiser is constrained by native feasible flux geometry and uses multistart SLSQP. It does not claim a global optimum or unique identifiability.

See [docs/STATIONARY_MFA_RENYI_CORE.md](docs/STATIONARY_MFA_RENYI_CORE.md) and [docs/MID_PREPROCESSING.md](docs/MID_PREPROCESSING.md).

## Genuine count observation laws

For measurements with genuine count semantics, `fluxemu.observation` provides explicit fixed total multinomial laws

```text
Y | v ~ Multinomial(n, p_v)
```

with declared count totals, exact structural zero support, reproducible sampling and independent product divergence identities. Percentages, peak areas, normalised MIDs and arbitrary intensities are never converted into pseudo counts and no effective sample size is inferred.

See [docs/STATIONARY_OBSERVATION_LAW.md](docs/STATIONARY_OBSERVATION_LAW.md).

## Simple binary inference

For two fixed feasible flux states, CarbonScope supports:

* `log_likelihood_ratio(...)`, with `LLR(y)=log P1(y)-log P0(y)`;
* `likelihood_ratio_p_value(...)`, the exact simple null tail `P0{LLR(Y) >= LLR(y_obs)}` when the null count space is enumerable within the explicit limit;
* `bruno_converse_at_order(...)`, the order specific finite sample Bruno et al. lower bound on optimal Type II error under a declared Type I budget.

The conventions are fixed:

```text
H0 = P0 = null
H1 = P1 = alternative
Type I  = P0(decide H1)
Type II = P1(decide H0)
reverse Rényi = D_lambda(P1 || P0)
forward Rényi = D_lambda(P0 || P1)
```

A p value is realised data evidence under the fixed null. A Rényi Type II lower bound is a constraint on achievable testing performance for the fixed law pair. Neither is a substitute for the other.

CarbonScope evaluates any supplied finite real `lambda > 1` subject to explicit numerical limits. It does not replace the continuous Rényi order envelope with a finite grid.

See [docs/SIMPLE_BINARY_RENYI_BOUNDS.md](docs/SIMPLE_BINARY_RENYI_BOUNDS.md) and [docs/technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex](docs/technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex).

## Composite inference

For explicitly finite null and alternative law classes, `fluxemu.testing` now provides three complementary calculations:

* `projected_renyi_test(...)` constructs a projected Rényi score at a supplied order `0 < lambda < 1`, verifies its uniform exponential moments over the declared classes, and calibrates the best threshold test within that score family;
* `composite_renyi_converse_at_order(...)` gives an order specific lower bound on the unrestricted minimax Type II error;
* `solve_finite_minimax_test(...)` enumerates the complete finite observation space and solves the unrestricted randomised minimax test as a linear programme.

For a tractable finite problem this gives the useful sandwich

```text
Rényi converse lower bound <= beta* <= achieved projected test error
```

while the LP directly computes `beta*` for the explicitly supplied finite classes.

The LP formulation is exact, while the numerical optimum is subject to declared HiGHS feasibility tolerances and independent validation. Positive probability mass below the retained solver resolution is rejected rather than silently discarded.

See [docs/COMPOSITE_TESTING.md](docs/COMPOSITE_TESTING.md).

## Transient forward EMU

`fluxemu.emu` contains a native fixed flux transient EMU integrator with explicit pool quantities, time points and initial unlabelled internal state. This is forward simulation only; transient inverse MFA is not implemented.

## Validation

The test suite includes analytical controls, exact finite count enumeration, independent deterministic test oracles, a direct full isotopomer implementation of the Antoniewicz TCA benchmark, and the packaged E. coli acceptance model. Composite testing includes singleton reductions to the randomised Neyman Pearson optimum, worst case class controls, structural zero cases, enumeration limits, and direct comparison of projected tests, converse bounds, and the unrestricted minimax LP.

The production FastFVA architecture is a HiGHS native adaptation of the shared memory computational design of Marouen Ben Guebila's VFFVA. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

See [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) for the complete current limitations.
