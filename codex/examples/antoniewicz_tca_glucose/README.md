# U-13C6 glucose → glycolysis → frozen Antoniewicz TCA benchmark

This is a numerical isotope-propagation benchmark, not a complete E. coli
energy model. It supplies 100% U-13C6 `glucose_ext` through a synthetic,
carbon-skeleton glycolytic layer into 100% M+2 AcCoA, then reuses the exact
validated eight-reaction Antoniewicz TCA transition system.

The original [`../antoniewicz_tca`](../antoniewicz_tca) benchmark remains a
separate frozen Table 6 regression and is neither modified nor replaced.

Run the reproducible build from `codex/`:

```bash
/workspace/.venv/bin/python examples/antoniewicz_tca_glucose/build_model.py
/workspace/.venv/bin/python -m pytest -q tests/test_antoniewicz_tca_glucose.py
```

The generated artefacts are:

- `antoniewicz_tca_glucose.xml` - annotated canonical COBRA/SBML model with
  directionality bounds; the generating flux state is stored separately.
- `stationary_mids.csv` — all required FluxEMU/mfapy stationary MIDs.
- `independent_tca_mids.csv` — full independent TCA MIDs from the reused
  176-state direct isotopomer solver using pure `#11` AcCoA.
- `timecourse_mids.csv` — mfapy `diffmdv` trajectory, automatically extended
  until it is within `1e-5` of stationary MIDs.

See [`GLYCOLYSIS_ATOM_MAPS.md`](GLYCOLYSIS_ATOM_MAPS.md),
[`FLUX_BALANCE.md`](FLUX_BALANCE.md),
[`STATIONARY_VALIDATION.md`](STATIONARY_VALIDATION.md), and
[`TIMECOURSE_VALIDATION.md`](TIMECOURSE_VALIDATION.md) for the maps, fixed
flux state, independent comparison, and dynamic recirculation evidence.
