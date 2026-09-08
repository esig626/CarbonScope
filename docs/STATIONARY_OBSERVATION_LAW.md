# Stationary genuine count observation laws

CarbonScope separates MID fitting from the experimental observation law. For measurements that genuinely consist of fixed total isotopologue counts, the native baseline is

```text
Y | v ~ Multinomial(n, p_v)
```

where `p_v` is the stationary EMU predicted MID and `n` is an explicitly supplied count total.

## Count only boundary

The multinomial layer is for genuine counts only. CarbonScope does not convert percentages, normalised MIDs, peak areas or arbitrary intensities into pseudo counts and does not infer an effective sample size.

Exact structural zeros are retained. No pseudocounts, smoothing or clipping are introduced.

## Public API

`fluxemu.observation` provides:

* `MultinomialMIDLaw`;
* `MIDCountObservation`;
* stationary observation specification and result records;
* `evaluate_stationary_observation_laws(...)`;
* reproducible `sample_stationary_observations(...)`;
* exact multinomial KL and Rényi identities and independent product aggregation.

A stationary observation specification binds complete feasible flux states, experiments, target and replicate order and explicit count totals to the resulting laws.

## Exact identities

For a common total `n`,

```text
D_alpha(Mult(n,p) || Mult(n,q)) = n D_alpha(p || q)
```

for the supported KL and Rényi orders under the exact support conditions.

For observed counts `k`, with empirical proportions `k/n`, the multinomial negative log likelihood decomposes into a data only combinatorial term plus

```text
n D_KL(k/n || p).
```

These identities are used as mathematical validation, not as permission to invent a count total for noncount measurements.

## Products

Multiple observation blocks may be combined only when independence is explicitly declared. Each block retains its own count total and ordered mass class space; CarbonScope never fabricates a shared total.

## Numerical boundaries

At very large totals, use the raw count log PMF interface rather than reconstructing likelihoods from rounded empirical proportions. Inputs outside supported integer and numerical ranges fail explicitly.
