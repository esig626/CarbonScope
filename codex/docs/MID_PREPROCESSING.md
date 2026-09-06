# Explicit MID preprocessing

KL and Rényi divergence are defined here on probability vectors. The numerical
divergence kernel therefore expects vectors already closed to the probability
simplex to machine precision. Experimental files often contain rounded
fractions, percentages, or raw non-negative intensities instead.

FluxEMU provides an explicit preprocessing helper:

```python
from fluxemu import normalise_mid
from fluxemu.mfa import StationaryMIDObservation

fractions = normalise_mid((30.0, 70.0))
observation = StationaryMIDObservation("product-mid", fractions, "replicate-1")
```

`normalise_mid`:

- accepts any finite non-negative one-dimensional vector with positive total;
- preserves declared mass-class order and exact zero support;
- divides by the total explicitly and returns a probability MID at machine-simplex precision;
- performs no clipping, pseudocount insertion, support filling, or sign repair;
- fails rather than silently erasing positive support when the numerical dynamic range is too large.

The absolute input total is deliberately discarded. If raw ion counts or total
intensity will later be used in an experimental noise model, retain those raw
measurements separately alongside the normalised MID.

This preprocessing is intentionally separate from `kl_divergence`,
`renyi_divergence`, `evaluate_stationary_mfa`, and `fit_stationary_mfa`. Those
scientific objective functions never silently renormalise their inputs. The
strict machine-simplex check remains an internal correctness guard; users do
not need to hand-edit experimental fractions to sixteen-ULP accuracy before
fitting.
