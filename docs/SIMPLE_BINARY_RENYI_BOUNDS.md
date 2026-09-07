# Simple-binary Rényi bounds and exact likelihood-ratio p-values

FluxEMU's current testing layer compares two fixed observation laws induced by two fixed complete feasible flux states.

## Fixed convention

```text
H0 = P0 = null
H1 = P1 = alternative
Type I  = P0(decide H1)
Type II = P1(decide H0)
reverse Rényi = D_lambda(P1 || P0)
forward Rényi = D_lambda(P0 || P1)
```

These roles are not inferred or swapped internally.

## Realised log-likelihood ratio

For a realised genuine-count observation `y`,

```text
LLR(y) = log P1(y) - log P0(y).
```

Positive values favour H1 over H0; negative values favour H0.

Exact support is preserved. If only P1 assigns positive probability, the LLR is `+infinity`; if only P0 does, it is `-infinity`. If both complete laws assign zero probability to the supplied observation, the LLR is undefined and FluxEMU raises an explicit error.

## Exact simple-null p-value

`likelihood_ratio_p_value(...)` computes the non-randomised discrete tail

```text
p(y_obs) = P0{ LLR(Y) >= LLR(y_obs) }.
```

This is a sample-specific and fixed-alternative-specific p-value. It is not a divergence and not a Type-II lower bound.

Evaluation enumerates the complete positive-probability null count space. An explicit `max_outcomes` limit prevents accidental combinatorial explosion. If the limit is exceeded, FluxEMU raises `ExactPValueEnumerationLimitError`; it never silently falls back to chi-square, Monte Carlo, saddlepoint or another approximation.

## Bruno order-specific Type-II lower bound

For a Type-I budget `epsilon` and any caller-supplied finite real `lambda > 1`, `bruno_converse_at_order(...)` evaluates the Bruno, Vandenbroucque & Esposito finite-sample lower bound.

With

```text
D_rev = D_lambda(P1 || P0)
D_fwd = D_lambda(P0 || P1),
```

the two raw components are

```text
B_reverse = 1 - exp(((lambda-1)/lambda) * (log(epsilon) + D_rev))
B_forward = exp((lambda/(lambda-1))*log(1-epsilon) - D_fwd)
```

and the returned order-specific result is

```text
beta*(epsilon) >= max(B_reverse, B_forward).
```

The canonical result record is `BrunoOrderBound`.

The theorem path requires exact mutual absolute continuity. Mismatched support is rejected; FluxEMU does not smooth or repair it.

## No order-grid substitution

FluxEMU evaluates the bound at the supplied finite real order. It does not claim that a finite set of orders equals the continuous-order envelope and does not expose a grid search as a global bound.

## Interpretation

The quantities answer different questions:

- **LLR:** which fixed law does this realised data set favour?
- **p-value:** under P0, how often would LLR evidence at least this favourable to P1 occur?
- **Rényi divergence:** how separated are the complete laws in the chosen Rényi sense?
- **Type-II lower bound:** under the stated Type-I budget and Rényi order, how small can optimal Type-II error possibly be according to the bound?

See `technical_notes/P_VALUES_AND_INFORMATION_DIVERGENCE.tex` for the likelihood/KL/Rényi bridge.

## Example

```bash
python examples/simple_binary_flux_discrimination.py
```
