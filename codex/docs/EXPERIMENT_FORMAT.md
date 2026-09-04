# Historical COBRA/mfapy compatibility experiment format

> **Compatibility schema.** This document describes the original annotated
> COBRA SBML and mfapy-oriented YAML route. Its `sampler: achr|optgp` fields do
> not configure the native Stage 1 sampler. The native deterministic CLI loads
> its flux model from SBML through libSBML and uses a separate native experiment
> YAML schema, while native sampled execution is currently a Python API. See
> [Standalone native workflow](STANDALONE_NATIVE_WORKFLOW.md) and
> [Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md).

The compatibility route separates network topology from experiment settings:

- its compatibility input network is a COBRA SBML model;
- atom transitions and metabolite carbon properties are versioned JSON values
  in SBML notes; and
- tracer mixtures, requested fragments, sampling settings, tolerances, and
  output settings are YAML.

This compatibility schema does not use an mfapy text model or reaction-map input
file. Native Stage 1 requires neither COBRApy nor mfapy.

## SBML notes

Reaction metadata is stored under note key
`FLUXEMU_REACTION_METADATA_V1`; metabolite metadata uses
`FLUXEMU_METABOLITE_METADATA_V1`. The value is canonical JSON
(`sort_keys=True`, compact separators) escaped with Python `html.escape` before
COBRApy writes XHTML notes. FluxEMU unescapes before JSON decoding. This is a
prototype extension, not an SBML community standard.

Every reaction and metabolite must have a note, including explicitly excluded
boundary reactions. This prevents a missing annotation from being confused
with an intentional exclusion.

### Reaction JSON

```json
{
  "schema_version": 1,
  "kind": "reaction",
  "original_cobra_reaction_id": "v1",
  "directional_id": "v1:forward",
  "direction": "forward",
  "include_in_isotope_model": true,
  "substrates": [
    {"metabolite_id": "AcCoA", "atom_labels": ["A", "B"]},
    {"metabolite_id": "OAC", "atom_labels": ["C", "D", "E", "F"]}
  ],
  "products": [
    {"metabolite_id": "Cit", "atom_labels": ["F", "E", "D", "B", "A", "C"]}
  ]
}
```

Participant order is atom order. Matching single-character labels give the
explicit atom correspondence. FluxEMU never infers transitions from
stoichiometry. Included reactions in the first prototype must be separate,
nonnegative directional COBRA reactions. A reverse direction is another
reaction with its own atom map; a signed net flux is not split into invented
gross directions.

Excluded reactions set `include_in_isotope_model` to `false` and use empty
participant lists. `original_cobra_reaction_id` must equal the owning SBML
reaction ID, and `directional_id` must be unique.

### Metabolite JSON

```json
{
  "schema_version": 1,
  "kind": "metabolite",
  "original_cobra_metabolite_id": "AcCoA",
  "carbon_count": 2,
  "is_carbon_source": true,
  "is_excreted": false,
  "symmetry": false,
  "include_in_isotope_model": true
}
```

On the compatibility route, carbon sources enter mfapy's source template.
Excreted metabolites do not
participate in mfapy's internal steady-state equations. Symmetric metabolites
enable mfapy's carbon-order symmetry handling. Carbon-source and excreted
metabolites must be included in the isotope model.

Stable COBRApy round-trip tests cover reaction and metabolite metadata with
quotes, `<`, and `&`, and require both the escaped note string and decoded
objects to remain exact.

## Compatibility experiment YAML

The checked-in example is
`codex/examples/toy_experiment.yaml`. The compatibility schema is:

```yaml
schema_version: 1

tracers:
  - metabolite_id: AcCoA
    isotopomer_fractions:
      "#00": 0.50
      "#01": 0.25
      "#11": 0.25
    correction: no

targets:
  - fragment_id: Glue
    metabolite_id: Glu
    atom_positions: [1, 2, 3, 4, 5]
    analytical_method: intermediate
    formula: C5H10N2O3
    correction: no

fraction_of_optimum: 0.90
sample_count: 6
sampler: achr
seed: 1729

tolerances:
  bounds: 1.0e-7
  mass_balance: 1.0e-7
  objective_floor: 1.0e-7
  mid: 1.0e-8
  tracer_normalization: 1.0e-9

output:
  overwrite: false
  float_precision: 12
```

### Tracers

For this compatibility parser, each carbon-source metabolite must appear exactly
once. Isotopomer keys begin
with `#` and contain one `0`/`1` per tracked carbon in metabolic atom order.
Fractions must be finite, nonnegative, no greater than one, and sum to exactly
one within `tolerances.tracer_normalization`. `correction` is `yes` or `no` and
is passed to mfapy's source setter. YAML 1.1 may parse unquoted `yes`/`no` as
booleans; FluxEMU normalizes those values safely.

### Targets

For this compatibility parser, `fragment_id` is the public result ID.
`metabolite_id` must identify an
included isotope metabolite. `atom_positions` are unique one-based carbon
positions. The prototype accepts mfapy `intermediate` and `gcms` analytical
methods for one metabolite fragment. `formula` is retained for mfapy's isotope
correction matrices. Target-level natural-isotope addition is not enabled in
this forward-only prototype, so target `correction` must currently be `no`.

### Compatibility analysis and output

`fraction_of_optimum` is applied identically by FVA and by the persistent
sampling objective constraint. `sample_count` is a positive integer. `sampler`
is explicitly `achr` or `optgp`; OptGP runs with one process for an exact,
reproducible batch size. `seed` is a nonnegative 32-bit integer.

These fields belong only to the COBRApy compatibility sampler. They are not read
by `run_native_stationary_ensemble`, whose count, seed, fraction, burn-in,
thinning, and worker controls are explicit Python arguments. The native sampler
is random-direction hit-and-run and does not expose an ACHR/OptGP selector.

All tolerances must be finite and positive. Output precision is 1 through 17
decimal digits. `overwrite: false` refuses a nonempty destination. Setting it
to `true` explicitly authorizes replacement of files inside that one output
directory.

For readability the parser also accepts the documented short aliases
`metabolite`, `isotopomers`, `id`, and `atoms`, plus
`number_of_flux_samples`, `random_seed`, and `output_settings`. A file must use
only one spelling for each field; unknown keys are rejected.

## Explicit transient experiment YAML

Transient configuration uses `load_transient_experiment` and requires the top-level discriminator `experiment_mode: transient`; the stationary parser does not infer transient behaviour. Its `timecourse` mapping explicitly requires `time_points`, `initial_internal_mids: unlabelled`, and a `pool_quantities` list of `{metabolite_id, quantity}` records. The list form allows duplicate declarations to be rejected. See `PHASE2_NATIVE_TRANSIENT_EMU.md` for the complete V1 scientific semantics.

The transient `timecourse.tolerances` mapping is numerical execution policy.
`evaluate_configured_transient` forwards its validated `rtol`, `atol`, and `mid`
values unchanged; these settings are excluded from the scientific transient
fingerprint.
