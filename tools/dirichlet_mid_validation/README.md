# Dirichlet MID V1 validation tool

Run the deterministic full campaign from the repository root after installing
the testing extra:

```bash
python tools/dirichlet_mid_validation/run_campaign.py \
  --output results/dirichlet_mid_validation/summary.json
```

The production implementation is checked against independent SciPy and
80-digit mpmath formulas, direct sampling identities, analytic-versus-Monte
Carlo score moments, projected-test Type-I simulations, and compatible versus
deliberately non-Dirichlet synthetic replicate data. The fixed seed is recorded
in the output. Ill-conditioned binary64 Renyi evaluations must refuse rather
than clip or report unresolved digits. Monte Carlo evidence is a diagnostic
sanity check, not a proof.

`--scale` controls campaign size for development. The committed validation
record is produced with `--scale 1`.
