# Historical compatibility integration API map

> **Compatibility audit, not native runtime architecture.** This document
> records the COBRApy 0.31.1 and mfapy 0.6.3 source contracts inspected on
> 2026-08-06. References to COBRApy samplers or mfapy execution below describe
> the legacy compatibility/parity path. The supported native Stage 1 pipeline
> imports neither package; see
> [Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md).

The historical audit compared the sources then present under `vendor/` with the
installed compatibility environment. The vendored trees were source and test
material, not standard native runtime dependencies.

## Current native public boundary

`fluxemu.run_native_stationary_analysis` preserves the deterministic
FBA-primal-to-MID path. `fluxemu.run_native_stationary_ensemble` provides FBA,
FastFVA, native complete feasible-state sampling, and one-batch stationary EMU.
Its result keeps the FBA optimum, independent FVA extrema, sampled complete
states, and conditional MID ensemble in distinct fields.

Low-level preparation, cold and reusable FVA, native hit-and-run sampling, and
independent sample validation are exported from `fluxemu.flux_analysis`.
COBRApy remains an optional projection/parity oracle and mfapy remains an
optional historical forward-backend oracle.

`fluxemu.fit_stationary_mfa` and `fluxemu.evaluate_stationary_mfa` directly
reexport native stationary fitting and objective evaluation from `fluxemu.mfa`.
That module exports the ordered MFA scientific records, exact KL/Rényi
divergences, validation, fingerprints, and complete result/start diagnostics.
Only fitting loads optional SciPy; native MFA imports neither compatibility
package. See [stationary MFA](STATIONARY_MFA_RENYI_CORE.md) for the current
contract and [mfapy engineering comparison](MFAPY_ENGINEERING_COMPARISON.md)
for the separate source audit of the retained MFA workflow mechanics.

## Legacy mfapy compatibility source

### Parser contract and constructor dictionaries

`mfapy.mfapyio.load_metabolic_model` delegates to four parsers and returns
`(reactions, reversible_reactions, metabolites, target_fragments)`
([`mfapyio.py:468-504`](../../vendor/mfapy/mfapy/mfapyio.py#L468-L504)). The
individual parsers and their emitted fields are:

| Object | Fields emitted by the parser | Source |
| --- | --- | --- |
| reaction | `stoichiometry`, `reaction`, `atommap`, `externalids`, `order`, `lb`, `ub` | [`mfapyio.py:40-154`](../../vendor/mfapy/mfapy/mfapyio.py#L40-L154) |
| reversible reaction | `forward`, `reverse`, `type`, `order`, `externalids`, `lb`, `ub` | [`mfapyio.py:271-369`](../../vendor/mfapy/mfapy/mfapyio.py#L271-L369) |
| metabolite | `C_number`, `symmetry`, `carbonsource`, `excreted`, `order`, `externalids`, `lb`, `ub` | [`mfapyio.py:156-269`](../../vendor/mfapy/mfapy/mfapyio.py#L156-L269) |
| target fragment | `type`, `atommap`, `use`, `order`, `formula` | [`mfapyio.py:371-466`](../../vendor/mfapy/mfapy/mfapyio.py#L371-L466) |

The forward constructor requires all reaction fields except `externalids`; all
five target fields; `C_number`, `symmetry`, `carbonsource`, `excreted`, and
`order` for metabolites; and `forward`, `reverse`, and `order` for reversible
entries. FluxEMU nevertheless supplies bounds and external IDs consistently.
Reaction bounds must be explicit because the constructor's missing-bound path
mistakenly writes through `self.metabolites[id]`
([`metabolicmodel.py:134-142`](../../vendor/mfapy/mfapy/metabolicmodel.py#L134-L142)).
The official metabolite parser increments its order counter twice; direct
dictionaries preserve its relative order but FluxEMU generates contiguous
orders itself.

`MetabolicModel.__init__` deep-copies the four dictionaries, runs `modelcheck`,
initializes carbon-source/symmetry/fragment data, and calls `reconstruct`
([`metabolicmodel.py:64-208`](../../vendor/mfapy/mfapy/metabolicmodel.py#L64-L208)).
mfapy prints model-check failures and returns a partially initialized object,
so FluxEMU validates before construction and checks the constructed object.

### Update, reconstruction, and the generated function

`MetabolicModel.update` rebuilds reaction/metabolite/reversible ordering, the
stoichiometric system, independent-variable data, and the inverse system
([`metabolicmodel.py:561-1096`](../../vendor/mfapy/mfapy/metabolicmodel.py#L561-L1096)).
`reconstruct` calls `update`, generates Python source, executes it, and stores
the functions as `model.func["calmdv"]` and `model.func["diffmdv"]`
([`metabolicmodel.py:1099-1133`](../../vendor/mfapy/mfapy/metabolicmodel.py#L1099-L1133)).
`generate_calmdv` is the forward-function generator
([`metabolicmodel.py:1137-2726`](../../vendor/mfapy/mfapy/metabolicmodel.py#L1137-L2726)).
The experimental optimized EMU branch is not used.

EMU topology is built as follows:

- mapped directional reactions become product-to-substrate atom relations
  ([`metabolicmodel.py:1289-1456`](../../vendor/mfapy/mfapy/metabolicmodel.py#L1289-L1456));
- reactions with atom map `nd`, or without `-->`, are excluded;
- required EMUs are traced backward from targets whose `use` is `use`
  ([`metabolicmodel.py:1459-1636`](../../vendor/mfapy/mfapy/metabolicmodel.py#L1459-L1636));
- required source EMUs populate `carbon_source_emu` at lines 1638-1644; and
- generated layer matrices solve `A X = B Y`, then assemble target MIDs
  ([`metabolicmodel.py:1753-2064`](../../vendor/mfapy/mfapy/metabolicmodel.py#L1753-L2064)).

Construction and code generation happen once in `reconstruct`. Calls through
`model.func` allocate call-local matrices, so one generated function can be
reused for every flux sample without reconstruction.

### Flux order and forward return value

The required reaction order is exactly `model.reaction_ids`, produced by
sorting reaction dictionaries on their `order` field
([`metabolicmodel.py:636-655`](../../vendor/mfapy/mfapy/metabolicmodel.py#L636-L655)).
Generated `calmdv` binds each bare reaction identifier to `r[i]` in that same
order ([`metabolicmodel.py:1759-1767`](../../vendor/mfapy/mfapy/metabolicmodel.py#L1759-L1767)).
Although `generate_mdv` builds a larger state vector containing metabolite and
reversible values ([`metabolicmodel.py:3044-3078`](../../vendor/mfapy/mfapy/metabolicmodel.py#L3044-L3078)),
steady-state `calmdv` reads only the leading reaction vector. FluxEMU therefore
passes a reaction-only vector in exact `model.reaction_ids` order.

`mfapy.optimize.calc_MDV_from_flux` dispatches to `func["calmdv"]` and sorts the
requested fragment IDs ([`optimize.py:288-330`](../../vendor/mfapy/mfapy/optimize.py#L288-L330)).
It returns `(mdv_vector, mdv_hash)`. The vector concatenates requested targets;
the hash contains all enabled targets plus the internal `X_list`. `X_list` is
added by the generated function at
[`metabolicmodel.py:2067-2073`](../../vendor/mfapy/mfapy/metabolicmodel.py#L2067-L2073)
and must not be exposed as a prediction.

### Carbon sources, atom maps, symmetry, and reversibility

Metabolites marked `carbonsource` seed constructor templates
([`metabolicmodel.py:123-130`](../../vendor/mfapy/mfapy/metabolicmodel.py#L123-L130)).
`generate_carbon_source_template` returns an initially unlabeled
`CarbonSource` ([`metabolicmodel.py:3013-3042`](../../vendor/mfapy/mfapy/metabolicmodel.py#L3013-L3042)).
Its `generate_dict` returns copies of source-EMU MDVs
([`carbonsource.py:100-119`](../../vendor/mfapy/mfapy/carbonsource.py#L100-L119)).
`set_all_isotopomers` requires `2**C` values and a unit sum
([`carbonsource.py:121-152`](../../vendor/mfapy/mfapy/carbonsource.py#L121-L152));
`set_each_isotopomer` reverses bit order for the internal index and only rejects
sums above one ([`carbonsource.py:214-274`](../../vendor/mfapy/mfapy/carbonsource.py#L214-L274)).
FluxEMU independently requires every configured tracer mixture to sum to one.

Atom maps are explicit strings. Carbon counts and target positions are checked
at [`metabolicmodel.py:412-454`](../../vendor/mfapy/mfapy/metabolicmodel.py#L412-L454),
while mapping labels must be unique per side and every product label must occur
on the substrate side
([`metabolicmodel.py:523-556`](../../vendor/mfapy/mfapy/metabolicmodel.py#L523-L556)).
Mapping labels are single characters; metabolite order supplies atom order and
labels supply correspondence. No transition is inferred from stoichiometry.
Target positions are one-based and colon-separated.

Symmetry is enabled only by `symmetry == "symmetry"`
([`metabolicmodel.py:117-121`](../../vendor/mfapy/mfapy/metabolicmodel.py#L117-L121)).
Symmetric EMUs are canonicalized against reversed carbon positions
([`metabolicmodel.py:1207-1214`](../../vendor/mfapy/mfapy/metabolicmodel.py#L1207-L1214));
both orientations are traced and half-weighted in the generated matrices.

Reversible entries do not replace directional fluxes. They add a net state
variable constrained as `forward reaction(s) - reverse reaction(s) - net = 0`
([`metabolicmodel.py:937-969`](../../vendor/mfapy/mfapy/metabolicmodel.py#L937-L969)).
Forward and reverse fields may contain `+`-separated reaction IDs, but missing
references are silently skipped. The initial FluxEMU runtime always constructs
an empty reversible dictionary and represents both directions only as separate,
nonnegative reactions. It never derives two gross fluxes from one signed net
value. The official direct-dictionary regression still exercises mfapy's
nonempty reversible-entry contract.

### Identifier restrictions and optional nlopt

mfapy strips underscores from metabolite IDs and prefixes IDs that do not start
with a letter ([`metabolicmodel.py:230-249`](../../vendor/mfapy/mfapy/metabolicmodel.py#L230-L249)).
It emits reaction IDs as bare Python assignment targets and target IDs inside
generated string literals. Python keywords, punctuation, collisions after
mutation, and generated-local names can therefore fail or corrupt execution.
FluxEMU generates conservative deterministic internal IDs for reactions,
metabolites, and targets, detects collisions, and retains the original IDs in
memory and diagnostics.

mfapy 0.6.3 imports `optimize` at package import, and upstream `optimize.py`
imported absent `nlopt` unconditionally. Forward EMU only calls the generated
function; active `nlopt.opt` construction is confined to `fit_r_mdv_nlopt`.
The minimal local repair makes that import optional and raises a clear
`ImportError` at the nlopt fitting entry point. Details and verification are in
`MFAPY_PATCH.md`.

### Trusted official reference

The smallest official forward example is
[`Example_0_toymodel.py:1-107`](../../vendor/mfapy/sample/Example_0_toymodel.py#L1-L107),
with network definition
[`Example_0_toymodel_model.txt:13-45`](../../vendor/mfapy/sample/Example_0_toymodel_model.txt#L13-L45)
and complete state
[`Example_0_toymodel_status.csv:1-20`](../../vendor/mfapy/sample/Example_0_toymodel_status.csv#L1-L20).
Its current upstream exact forward test is
[`test_metabolicmodel.py:358-374`](../../vendor/mfapy/tests/test_metabolicmodel.py#L358-L374).
The shipped script contains a filename-case mismatch and calls obsolete
`set_constrain`; the regression uses the actual filename and current
`set_constraint` method without changing its model, state, tracer, or expected
result.

## Legacy COBRApy compatibility source

### SBML, notes, and annotations

The public SBML interfaces are `read_sbml_model`
([`sbml.py:396-455`](../../vendor/cobrapy/src/cobra/io/sbml.py#L396-L455)) and
`write_sbml_model`
([`sbml.py:1136-1181`](../../vendor/cobrapy/src/cobra/io/sbml.py#L1136-L1181)).
Model notes/annotations are read at lines 624-647 and written at 1224-1235;
metabolite data at 665-705 and 1309-1328; reaction data at 812-826 and
1352-1364. The notes parser/serializer supports `<p>key: value</p>` dictionaries
([`sbml.py:1651-1699`](../../vendor/cobrapy/src/cobra/io/sbml.py#L1651-L1699)).
Because the serializer directly interpolates values into XML, FluxEMU stores
canonical JSON after `html.escape` and applies `html.unescape` before
`json.loads`. This preserves XML-sensitive characters exactly across a stable
COBRApy write/read round trip.

Annotations are parsed at `sbml.py:1750-1800` and written at
[`sbml.py:1832-1910`](../../vendor/cobrapy/src/cobra/io/sbml.py#L1832-L1910).
They are appropriate for identifiers.org/MIRIAM values, not nested isotope
metadata. Existing model, reaction, and metabolite notes and annotations remain
on their normal COBRApy objects; FluxEMU adds one namespaced note entry.

### FBA and status

`Model.optimize` is the public FBA entry point
([`model.py:1204-1237`](../../vendor/cobrapy/src/cobra/core/model.py#L1204-L1237)).
Solutions expose status, objective value, and reaction-indexed net fluxes
([`solution.py:22-84`](../../vendor/cobrapy/src/cobra/core/solution.py#L22-L84));
solution assembly follows model reaction order at `solution.py:138-210`.
Status validation and specialized errors are in
[`solver.py:525-590`](../../vendor/cobrapy/src/cobra/util/solver.py#L525-L590) and
[`exceptions.py:1-50`](../../vendor/cobrapy/src/cobra/exceptions.py#L1-L50).
Linear objective coefficients come from
[`solver.py:71-105`](../../vendor/cobrapy/src/cobra/util/solver.py#L71-L105).
FluxEMU requests a solution with `raise_error=False` so it can wrap every
non-optimal status consistently, then explicitly requires status `optimal`, a
finite objective and flux vector, and a nonempty linear objective map.

### Compatibility FVA and a persistent sampling floor

`flux_variability_analysis` and its minimum/maximum DataFrame contract are at
[`variability.py:92-147`](../../vendor/cobrapy/src/cobra/flux_analysis/variability.py#L92-L147).
It first optimizes and creates a fractional old-objective constraint at lines
232-259, then performs independent minimizations/maximizations at 279-317.
Those columns are bounds, not one jointly feasible vector.

FVA runs inside a model context, so its constraint does not persist after the
call. The legacy COBRApy sampling adapter applies the same numerical threshold
with `fix_objective_as_constraint`
([`solver.py:469-522`](../../vendor/cobrapy/src/cobra/util/solver.py#L469-L522))
before constructing the sampler. Passing the already calculated bound avoids a
second optimum calculation. Objective direction is retained so maximization
uses a lower floor and minimization uses an upper ceiling.

### Compatibility complete feasible sampling and validation

Stable COBRApy 0.31.1 supports ACHR and OptGP; the compatibility adapter chooses
one explicitly.
ACHR construction, warmup, seed, and sampling are in
[`achr.py:91-168`](../../vendor/cobrapy/src/cobra/sampling/achr.py#L91-L168).
OptGP construction and sampling are in
[`optgp.py:105-198`](../../vendor/cobrapy/src/cobra/sampling/optgp.py#L105-L198),
with chain seeds at lines 216-251. The compatibility adapter uses one OptGP
process so the exact requested batch size and deterministic seed are preserved.

`HRSampler` copies the already constrained model, normalizes the seed, and
materializes equality/inequality matrices
([`hr_sampler.py:171-278`](../../vendor/cobrapy/src/cobra/sampling/hr_sampler.py#L171-L278)).
Thus the explicit objective constraint remains active during warmup and every
sample. The sampler returns all reactions in model order, i.e. complete jointly
feasible distributions rather than independent FVA draws.

The COBRApy sampler's own `validate` checks steady-state equalities and reaction
bounds but not general inequalities such as the objective floor
([`hr_sampler.py:496-576`](../../vendor/cobrapy/src/cobra/sampling/hr_sampler.py#L496-L576)).
The compatibility adapter additionally verifies exact unique columns, finite
values, all bounds, `S v = 0`, and the linear objective threshold per sample.
The stoichiometric matrix helper preserves model metabolite/reaction order
([`array.py:20-87`](../../vendor/cobrapy/src/cobra/util/array.py#L20-L87)).
Reaction chemical formula balance (`Reaction.check_mass_balance`) is distinct
from steady-state sample balance
([`reaction.py:1403-1431`](../../vendor/cobrapy/src/cobra/core/reaction.py#L1403-L1431)).

### Identifier and order preservation

Models store metabolites and reactions in ordered `DictList` containers with
duplicate rejection
([`dictlist.py:25-77`](../../vendor/cobrapy/src/cobra/core/dictlist.py#L25-L77)).
SBML reaction document order is retained while loading (`sbml.py:818-924,958`),
and FBA fluxes, FVA rows, ACHR columns, and OptGP columns derive from
`model.reactions`. The compatibility adapter captures that order immediately
after loading and requires exact equality during every conversion. Original
COBRA IDs remain the public/output IDs; only mfapy receives generated internal
IDs.
