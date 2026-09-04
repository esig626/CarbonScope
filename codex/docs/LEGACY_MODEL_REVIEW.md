# Legacy model review

This review was performed only after the official mfapy Example 0 regression
and complete annotated COBRA toy pipeline passed. Everything under
`references/` remains unverified secondary material. The notebooks
were inspected statically; no notebook or legacy model was executed, repaired,
adopted, or certified.

## Observed facts

### `Fixed_FFW_Model.ipynb`

- Cell 0 hard-codes a macOS working directory.
- Cell 1 loads `Forward_Model/test_model.txt` through
  `mfapyio.load_metabolic_model`, constructs a `MetabolicModel`, calls
  `update`, and dispatches through `optimize.calc_MDV_from_flux` and
  `model.func`.
- It explicitly constructs its forward array in `model.reaction_ids` order,
  which confirms the legacy positional-order assumption.
- Missing flux IDs are replaced with zero while setting constraints and again
  while building the array.
- Targets are sorted alphabetically. Stored output is a wide, padded table
  (`Metabolite`, `m0` through `m6`) plus one plot per fragment; the notebook
  writes `mid_output.csv` and `mid_plots.png`.
- Cell 14 is an unexecuted alternative using `test_reduced.txt`. It calls
  `reconstruct`, while cell 1 only calls `update`.
- Cell 1 configures glucose as `#000000: 0.3`, `#100000: 0`,
  `#111111: 0.7`, correction `yes`. Its Gln/Leu/Ile/Val/Arg sources each
  receive only `#000...: 0.1`, so each mixture sums to 0.1 rather than one.
  Cell 14's glucose fractions sum to 0.99.

### `Forward_Simple.ipynb`

- Cell 0 also hard-codes a machine-specific working directory.
- Cells 1, 3, and 7 load `Forward_Model/test_2.txt` through the text parser and
  construct one mfapy model/function.
- Cell 2 builds a hard-coded array in `model.reaction_ids` order. Cells 3 and 7
  iterate a flux CSV, reuse the model/function/carbon source, and call forward
  EMU per row. Reusing the generated function is a sound pattern.
- Twelve GC-MS targets are hard-coded rather than supplied as experiment data.
  Batch output is transposed to rows named `<fragment>_m+<index>` and columns
  `run_0`, `run_1`, etc.; original input sample IDs are discarded.
- Glucose (`0.35` unlabeled, `0.65` uniformly labeled) and glutamine (`0.4`
  unlabeled, `0.6` uniformly labeled) mixtures both sum to one and use
  correction `yes`.
- Output paths differ between cells (`output/`, `Forward_Model/`, and
  `Forward_Model/output/`), and stored execution counts are out of order.

### Text models

`test_model.txt` contains 85 reactions (lines 2-86), 57 metabolites
(91-147), 16 reversible declarations (152-167), seven targets (172-178), and
`//End` at 179. `test_model_A.txt` contains 85 reactions (3-87), 72
metabolites (92-163), the same count of reversible declarations (168-183),
seven targets (188-194), and `//End` at 195. Reaction bounds are uniformly
0..1000; reversible aggregate bounds are -1000..1000.

`test_reduced.txt` contains 17 reactions (2-18), 15 metabolites (23-37), five
reversible declarations (42-46), and two targets (51-52). It has no `//End`
marker. `test_reduced_flux.txt` is not a flux vector: its header is
`Name Spectrum Select MDV Std`, followed by Pyr and Lac MID rows. It contains
no reaction state or complete flux distribution.

The text files mark carbon sources but contain no tracer isotopomer fractions:

- `test_model.txt`: SubsArg, SubsGlc, SubsGln, SubsIle, SubsLeu, SubsVal
  (lines 139-144);
- `test_model_A.txt`: the same plus SubsCO2 (153-159); and
- `test_reduced.txt`: only SubsGlc (line 37).

The models use explicit nonnegative directional pairs, such as PGI `r2`/`r3`,
and separately declare a reversible aggregate. Physical section order is the
mfapy reaction order; it is not numeric ID order.

## Variant incompatibilities

- The full model drains 15 precursors into one zero-carbon `Biomass`
  (reactions 61-75) and uses `rS` at line 86. Model A instead creates distinct
  carbon-containing excreted pseudo-metabolites (reactions 62-76) and omits
  `rS`.
- Model A adds `r101_CO2_ex` and carbon source SubsCO2 (reaction 78,
  metabolite 154); the full model has neither.
- `r102_PGA_ex` differs materially: full-model line 77 produces cofactors with
  no PGAEx, while Model A line 79 also produces PGAEx.
- Target sets and order differ. The full model uses Pyr, Lac, Mal, Suc, Cit,
  Ala, AKG; Model A uses Pyr, Mal, Cit, Ala, Asp, AKG, Lac.
- Both full variants contain 85 reactions, yet Model A's insertion shifts
  later positional order. Reduced `r108` is position 16 rather than 81/82.
- Reduced `r16_pyrdh` reuses an ID/KEGG tag but changes from
  `Pyr -> AcCOAmit + CO2in` with `ABC -> BC + A` to `Pyr -> Pyrex` with
  `ABC -> BCA`. Matching an ID alone is therefore insufficient.

No positional flux vector can be reused across variants. Stoichiometry,
directional role, participants, and atom mapping must all match.

## Doubts requiring scientific review

- The notebooks' flux CSV/hard-coded flux provenance is absent. They do not
  report solver status, objective, FVA, bounds, steady-state residuals, or an
  objective-floor constraint. A notebook comment calling values
  “COBRA-verified” is not verification evidence.
- `Forward_Simple` checks some missing columns, then still calls
  `.get(reaction_id, 0.0)`. Neither notebook rejects duplicates, nonfinite
  values, infeasible rows, or ambiguous mappings.
- Missing fragment/isotopologue cells become `NaN`; no MID presence, length,
  nonnegativity, finiteness, or normalization check is recorded.
- All text-model targets use formula `C5H10N2O3` despite fragments tracking
  three to six carbons. This might encode derivatization chemistry, but it
  needs independent analytical confirmation.
- `GC_Ala_260` maps to Pyr rather than Ala, making it structurally identical
  to `GC_Pyr` unless correction metadata justifies the distinction.
- Full stoichiometry and isotope stoichiometry omit different cofactors by
  design; isotope reaction strings must never replace COBRA stoichiometry.
- Suspicious labels include `(kegg:Glu)` on a SubsArg reaction,
  `(kegg:NADPH)` on `rS`, and inconsistent `keggPGA_ex` spelling. Arg-to-Glu
  loses a mapped carbon without a named carbon coproduct. These are doubts,
  not proposed corrections.
- Aggregate zero-carbon Biomass versus Model A's separate carbon sinks is a
  material modeling decision. No variant should be selected automatically.

## Migration requirements for any future adoption

1. Establish an authoritative SBML network, objective, reaction semantics, and
   boundaries independently of these files.
2. Review every legacy atom transition scientifically, then transfer it into
   versioned SBML notes. Do not copy isotope strings into COBRA stoichiometry.
3. Match reactions by full semantics, not ID or position; retain separate
   explicit directions and never infer gross exchange from a net value.
4. Recreate normalized tracer mixtures and requested targets in experiment
   YAML. The non-unit notebook mixtures must be rejected until a user supplies
   corrected intent.
5. Replace the flux CSV handoff with native, complete, jointly feasible
   `CanonicalFluxState` samples in memory, preserve real sample IDs, and emit
   mapping only as diagnostics. COBRApy may be used only as an optional parity
   oracle, not the production sampler.
6. Validate mappings, complete reaction order, finite values, bounds, `S v =
   0`, objective floor, fragment presence, and MID distributions before any
   numerical comparison.
7. Produce an independently reviewed regression expectation. None of the
   legacy files supplies a trusted complete flux/tracer/expected-MID triplet.
