# MID preprocessing

CarbonScope divergence and MFA functions require probability vectors. They do not silently renormalise their inputs.

Use `fluxemu.normalise_mid(...)` when an external MID like vector is represented as rounded fractions, percentages or nonnegative intensities and should explicitly be closed to the probability simplex.

```python
from fluxemu import normalise_mid

mid = normalise_mid([24.9, 50.2, 24.9])
```

The helper:

* requires finite nonnegative components;
* requires positive total mass;
* preserves component order;
* preserves exact zero components;
* adds no pseudocounts;
* discards only the overall scale.

This operation does **not** turn an intensity vector into genuine count data. If an experiment has actual isotopologue counts, retain those raw counts and their declared total for `fluxemu.observation`. Percentages, peak areas and arbitrary intensities must never be assigned an invented effective sample size.

After explicit normalisation, the resulting vector may be used as an MFA observation. The fitting layer still applies exact KL and Rényi support semantics: if the observed distribution has positive mass where a prediction has zero mass, the divergence can be infinite.
