# Standalone native workflow

`fluxemu run --model MODEL.xml --experiment EXPERIMENT.yaml --output DIR` now uses only the FluxEMU canonical model, native HiGHS FBA/FVA, and native stationary EMU. FVA ranges are diagnostics; the complete FBA primal alone enters EMU.

SBML input is Level 3 FBC v2 with explicit finite parameter-based reaction bounds and one active linear maximize/minimize objective. Document species and reaction order is preserved. Stoichiometry math, missing bounds, non-finite bounds, and absent/ambiguous objectives fail explicitly.

The experiment is YAML `schema_version: 1`. `isotope_model.metabolites` maps model metabolite IDs to `carbon_count`, optional `isotope_visible`, and optional `symmetry`. `isotope_model.required_reactions` declares isotope-active model reaction IDs. Every such reaction needs an `assignments` entry containing `reaction_id`, curated-library `transition_id`, explicit `forward`/`reverse` `direction`, and a complete `metabolite_map` from transition participant IDs to model IDs. No mapping is inferred. `experiment` contains tracer `metabolite_id` and isotopomer fractions plus target IDs, metabolites, and one-based carbon positions; optional observation targets use precursor fragments. `fva_fraction_of_optimum` is in `(0, 1]`.

Outputs are `fba_fluxes.csv`, `fva_ranges.csv`, `predicted_mids.json`, `emu_diagnostics.json`, and `manifest.json` with content hashes and canonical fingerprints. No sampling is performed.

Default dependencies are NumPy, pandas, PyYAML, highspy, and python-libSBML. SciPy is in the `transient` extra. COBRApy is isolated in the `compat` extra; mfapy parity support remains optional in `mfapy`.
