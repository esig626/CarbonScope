# Native stationary experiment format

The public CLI reads an SBML Level 3 FBC physical model and a separate CarbonScope YAML file. The YAML declares isotope visible metabolites, authoritative transition assignments, tracers and targets; CarbonScope never infers atom mappings.

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
      correction: 'no'
  targets:
    - target_id: pyruvate
      metabolite_id: pyruvate_c
      atom_positions: [1, 2, 3]
      analytical_method: native
      formula: C3H3O3
      correction: 'no'

fva_fraction_of_optimum: 1.0
```

## Authoritative assignments

Each assignment names a physical reaction, an entry in the packaged carbon transition library, the physical direction represented by the isotope reaction, and an explicit one to one mapping from canonical transition participants to model metabolite IDs.

The transition participants, carbon counts and physical stoichiometric direction must agree exactly. Missing or ambiguous assignments fail.

## Tracers

Tracer isotopomers use `#` followed by one binary digit per carbon atom, in the declared canonical atom order. Fractions must form a valid probability distribution. Native stationary correction currently requires `correction: 'no'`. Quote the string so YAML does not interpret `no` as a boolean.

## Targets

Targets declare an ID, isotope metabolite, one based atom positions, analytical method metadata, formula metadata and correction policy. Atom positions are ordered and are preserved exactly.

Observation targets that combine explicit precursor fragments may also be declared when required by the native model.

## FVA fraction

`fva_fraction_of_optimum` must lie in `(0, 1]`. It controls the retained biological objective constraint in native forward FVA and ensemble analysis. It does not turn FVA endpoints into a flux state. The hypothesis-testing workflow validates this experiment field but samples the entire H0/H1 constrained steady-state regions without retaining an objective fraction; it does not reuse this field as a hypothesis constraint.

## Observation laws are separate

This stationary experiment file defines isotope prediction. Genuine count observation totals used by `fluxemu.observation` are separate explicit declarations. CarbonScope never infers a count total from a normalised MID or intensity vector.

The corrected-MID Dirichlet workflow also keeps noise outside this experiment
file. Its workflow declaration supplies block-specific concentration, source,
replicate semantics, independence and external correction provenance. The
native experiment's `correction: 'no'` records the forward isotope-model
policy; it is not evidence that measured data were externally corrected. A
Dirichlet workflow therefore requires its own explicit
`status: externally_corrected` audit record and does not add a correction
engine to the native experiment loader.

## Hypothesis-testing specification

The `fluxemu test-hypotheses` command consumes a separate schema-version-1 workflow YAML that refers to these native experiment files. The experiment file continues to define tracers, authoritative mappings, targets and mass-class order. The workflow file defines the common physical model, H0/H1 reaction-bound restrictions, finite-state sampling policy, observation-law blocks, independence and testing procedures.

All experiments in one workflow must use the same canonical isotope model. A
genuine-count block supplies its own integer `total_count`; measured fractions
and intensities are not accepted as count specifications. A corrected-MID
Dirichlet block instead supplies a positive concentration that is explicitly
not a count. Multiple blocks require explicit `independent_blocks: true`.

See [HYPOTHESIS_WORKFLOW.md](HYPOTHESIS_WORKFLOW.md) for both runnable YAML
branches using [the committed native experiment fixture](../tests/fixtures/hypothesis_workflow/experiment.yaml),
and [DIRICHLET_MID_OBSERVATION.md](DIRICHLET_MID_OBSERVATION.md) for the
continuous-law support and provenance contract. Observation declarations and
flux constraints remain separate from isotope prediction semantics.
