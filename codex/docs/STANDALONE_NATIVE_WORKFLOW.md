# Standalone native workflow

## Deterministic CLI

`fluxemu run --model MODEL.xml --experiment EXPERIMENT.yaml --output DIR` is
the deterministic CLI. It uses the FluxEMU canonical model, native HiGHS
FBA/FastFVA, and native stationary EMU. Its FVA ranges are diagnostics; only the
complete FBA primal enters EMU. This CLI command does not perform sampling.

SBML input is Level 3 FBC v2 with explicit finite parameter-based reaction
bounds and one active linear maximize/minimize objective. Document species and
reaction order is preserved. Stoichiometry math, missing bounds, non-finite
bounds, and absent or ambiguous objectives fail explicitly.

The native experiment is YAML `schema_version: 1`.
`isotope_model.metabolites` maps model metabolite IDs to `carbon_count`, optional
`isotope_visible`, and optional `symmetry`.
`isotope_model.required_reactions` declares isotope-active model reaction IDs.
Every such reaction needs an `assignments` entry containing `reaction_id`, a
curated-library `transition_id`, explicit `forward` or `reverse` `direction`,
and a complete `metabolite_map` from transition participant IDs to model IDs.
No mapping is inferred. `experiment` contains tracer metabolite IDs and
isotopomer fractions plus target IDs, metabolites, and one-based carbon
positions; optional observation targets use precursor fragments.
`fva_fraction_of_optimum` is in `(0, 1]`.

Outputs are `fba_fluxes.csv`, `fva_ranges.csv`, `predicted_mids.json`,
`emu_diagnostics.json`, and `manifest.json` with content hashes and canonical
fingerprints. `sampling_performed` is false for this deterministic CLI route.

## Sampled Python API

The public Python API `run_native_stationary_ensemble` implements the complete
sampled Stage 1 sequence. It returns FBA, independent FVA extrema, validated
complete flux states, and sample-indexed MID predictions as distinct fields.
Sampler count, seed, burn-in, and thinning are Python API arguments; the native
YAML/CLI schema does not currently expose them. See
[Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md) for the exact API and
guarantees.

## Dependencies

Default dependencies are NumPy, pandas, PyYAML, highspy, and python-libSBML.
Native Stage 1 requires neither COBRApy/optlang nor mfapy/SciPy. SciPy is in the
`transient` extra and COBRApy is isolated in the `compat` extra. The `mfapy`
extra supplies a compatible SciPy dependency for an mfapy installation provided
separately; the extra does not distribute mfapy itself.
