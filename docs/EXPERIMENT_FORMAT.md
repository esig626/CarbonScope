# Native stationary experiment format

The public CLI reads an SBML Level 3 FBC physical model and a separate FluxEMU YAML file. The YAML declares isotope-visible metabolites, authoritative transition assignments, tracers and targets; FluxEMU never infers atom mappings.

A minimal structure is:

```yaml
schema_version: 1

isotope_model:
  metabolites:
    glucose_c:
      carbon_count: 6
    pyruvate_c:
      carbon_count: 3
  required_reactions: [PYK]
  assignments:
    - reaction_id: PYK
      transition_id: glycolysis.pyruvate_kinase
      direction: forward
      metabolite_map:
        phosphoenolpyruvate: pep_c
        pyruvate: pyruvate_c

experiment:
  tracers:
    - metabolite_id: glucose_c
      isotopomers:
        "#111111": 1.0
      correction: no
  targets:
    - target_id: pyruvate
      metabolite_id: pyruvate_c
      atom_positions: [1, 2, 3]
      analytical_method: native
      formula: C3H3O3
      correction: no

fva_fraction_of_optimum: 1.0
```

## Authoritative assignments

Each assignment names a physical reaction, an entry in the packaged carbon-transition library, the physical direction represented by the isotope reaction, and an explicit one-to-one mapping from canonical transition participants to model metabolite IDs.

The transition participants, carbon counts and physical stoichiometric direction must agree exactly. Missing or ambiguous assignments fail.

## Tracers

Tracer isotopomers use `#` followed by one binary digit per carbon atom, in the declared canonical atom order. Fractions must form a valid probability distribution. Native stationary correction currently requires `correction: no`.

## Targets

Targets declare an ID, isotope metabolite, one-based atom positions, analytical-method metadata, formula metadata and correction policy. Atom positions are ordered and are preserved exactly.

Observation targets that combine explicit precursor fragments may also be declared when required by the native model.

## FVA fraction

`fva_fraction_of_optimum` must lie in `(0, 1]`. It controls the retained biological-objective constraint for FVA/ensemble geometry. It does not turn FVA endpoints into a flux state.

## Genuine counts are separate

This stationary experiment file defines isotope prediction. Genuine-count observation totals used by `fluxemu.observation` are separate explicit declarations. FluxEMU never infers a count total from a normalised MID or intensity vector.
