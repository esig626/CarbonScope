# Stationary MFA with KL/Rényi objectives

FluxEMU's native stationary MFA layer fits complete feasible flux states against observed stationary MIDs while keeping isotope fitting separate from the biological FBA objective.

Install the optional optimiser dependency with:

```bash
python -m pip install '.[mfa]'
```

## Objective

For declared observations, FluxEMU minimises

```text
sum D_alpha(observed MID || predicted MID)
```

using natural logarithms. `alpha=1` is exact KL divergence. Other supported values are finite positive real Rényi orders. The supplied order is not snapped to a grid.

Exact support is preserved. A positive observed mass on a predicted zero can produce an infinite loss; FluxEMU does not add pseudocounts or clip probabilities.

## Explicit MID preprocessing

External rounded fractions, percentages and non-negative intensity vectors can be explicitly closed to the probability simplex with `fluxemu.normalise_mid(...)`. This helper preserves component order and structural zeros and discards only overall scale.

The divergence and fitting functions themselves never silently normalise inputs. See [MID_PREPROCESSING.md](MID_PREPROCESSING.md).

## Feasible geometry and optimisation

`fit_stationary_mfa(...)` uses native FluxEMU flux geometry and complete feasible-state sampling to initialise multistart SLSQP. It validates final candidates independently against physical feasibility and re-evaluates their predicted MIDs/objectives.

A biological FBA optimum is not imposed by default merely because FBA exists elsewhere in the package. Any retained objective constraint must be declared as part of the fitting problem.

## Diagnostics and limitations

Each attempted start is retained in diagnostics. Failed starts are not silently discarded, and an accepted start indicates numerical/feasibility acceptance rather than proof of global optimality.

Multistart SLSQP does not establish a global optimum or unique identifiability. Multiple flux states can yield the same MID, including absolute-scale ambiguity in suitable models.

Soft measured-flux likelihoods, global search, Bayesian inference and confidence/compatibility regions are outside the current core.

## Example

```bash
python examples/stationary_mfa_recovery.py
```

The example contains identifiable recovery and a non-identifiability control using the native model/EMU stack.
