# FluxEMU integration API map

This audit was performed against the source checked into `/workspace/vendor` on
2026-08-06. Runtime execution uses COBRApy 0.31.1 from
`/workspace/.venv/lib/python3.11/site-packages/cobra`; the vendored COBRApy tree
is source and test material only. mfapy is the vendored 0.6.3 source.

## mfapy

### Parser contract and constructor dictionaries

`mfapy.mfapyio.load_metabolic_model` delegates to four parsers and returns
`(reactions, reversible_reactions, metabolites, target_fragments)`
([`mfapyio.py:468-504`](/workspace/vendor/mfapy/mfapy/mfapyio.py:468)). The
individual parsers and their emitted fields are:

| Object | Fields emitted by the parser | Source |
| --- | --- | --- |
| reaction | `stoichiometry`, `reaction`, `atommap`, `externalids`, `order`, `lb`, `ub` | [`mfapyio.py:40-154`](/workspace/vendor/mfapy/mfapy/mfapyio.py:40) |
| reversible reaction | `forward`, `reverse`, `type`, `order`, `externalids`, `lb`, `ub` | [`mfapyio.py:271-369`](/workspace/vendor/mfapy/mfapy/mfapyio.py:271) |
| metabolite | `C_number`, `symmetry`, `carbonsource`, `excreted`, `order`, `externalids`, `lb`, `ub` | [`mfapyio.py:156-269`](/workspace/vendor/mfapy/mfapy/mfapyio.py:156) |
| target fragment | `type`, `atommap`, `use`, `order`, `formula` | [`mfapyio.py:371-466`](/workspace/vendor/mfapy/mfapy/mfapyio.py:371) |

The forward constructor requires all reaction fields except `externalids`; all
five target fields; `C_number`, `symmetry`, `carbonsource`, `excreted`, and
`order` for metabolites; and `forward`, `reverse`, and `order` for reversible
entries. FluxEMU nevertheless supplies bounds and external IDs consistently.
Reaction bounds must be explicit because the constructor's missing-bound path
mistakenly writes through `self.metabolites[id]`
([`metabolicmodel.py:134-142`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:134)).
The official metabolite parser increments its order counter twice; direct
dictionaries preserve its relative order but FluxEMU generates contiguous
orders itself.

`MetabolicModel.__init__` deep-copies the four dictionaries, runs `modelcheck`,
initializes carbon-source/symmetry/fragment data, and calls `reconstruct`
([`metabolicmodel.py:64-208`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:64)).
mfapy prints model-check failures and returns a partially initialized object,
so FluxEMU validates before construction and checks the constructed object.

### Update, reconstruction, and the generated function

`MetabolicModel.update` rebuilds reaction/metabolite/reversible ordering, the
stoichiometric system, independent-variable data, and the inverse system
([`metabolicmodel.py:561-1096`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:561)).
`reconstruct` calls `update`, generates Python source, executes it, and stores
the functions as `model.func["calmdv"]` and `model.func["diffmdv"]`
([`metabolicmodel.py:1099-1133`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:1099)).
`generate_calmdv` is the forward-function generator
([`metabolicmodel.py:1137-2726`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:1137)).
The experimental optimized EMU branch is not used.

EMU topology is built as follows:

- mapped directional reactions become product-to-substrate atom relations
  ([`metabolicmodel.py:1289-1456`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:1289));
- reactions with atom map `nd`, or without `-->`, are excluded;
- required EMUs are traced backward from targets whose `use` is `use`
  ([`metabolicmodel.py:1459-1636`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:1459));
- required source EMUs populate `carbon_source_emu` at lines 1638-1644; and
- generated layer matrices solve `A X = B Y`, then assemble target MIDs
  ([`metabolicmodel.py:1753-2064`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:1753)).

Construction and code generation happen once in `reconstruct`. Calls through
`model.func` allocate call-local matrices, so one generated function can be
reused for every flux sample without reconstruction.

### Flux order and forward return value

The required reaction order is exactly `model.reaction_ids`, produced by
sorting reaction dictionaries on their `order` field
([`metabolicmodel.py:636-655`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:636)).
Generated `calmdv` binds each bare reaction identifier to `r[i]` in that same
order ([`metabolicmodel.py:1759-1767`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:1759)).
Although `generate_mdv` builds a larger state vector containing metabolite and
reversible values ([`metabolicmodel.py:3044-3078`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:3044)),
steady-state `calmdv` reads only the leading reaction vector. FluxEMU therefore
passes a reaction-only vector in exact `model.reaction_ids` order.

`mfapy.optimize.calc_MDV_from_flux` dispatches to `func["calmdv"]` and sorts the
requested fragment IDs ([`optimize.py:288-330`](/workspace/vendor/mfapy/mfapy/optimize.py:288)).
It returns `(mdv_vector, mdv_hash)`. The vector concatenates requested targets;
the hash contains all enabled targets plus the internal `X_list`. `X_list` is
added by the generated function at
[`metabolicmodel.py:2067-2073`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:2067)
and must not be exposed as a prediction.

### Carbon sources, atom maps, symmetry, and reversibility

Metabolites marked `carbonsource` seed constructor templates
([`metabolicmodel.py:123-130`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:123)).
`generate_carbon_source_template` returns an initially unlabeled
`CarbonSource` ([`metabolicmodel.py:3013-3042`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:3013)).
Its `generate_dict` returns copies of source-EMU MDVs
([`carbonsource.py:100-119`](/workspace/vendor/mfapy/mfapy/carbonsource.py:100)).
`set_all_isotopomers` requires `2**C` values and a unit sum
([`carbonsource.py:121-152`](/workspace/vendor/mfapy/mfapy/carbonsource.py:121));
`set_each_isotopomer` reverses bit order for the internal index and only rejects
sums above one ([`carbonsource.py:214-274`](/workspace/vendor/mfapy/mfapy/carbonsource.py:214)).
FluxEMU independently requires every configured tracer mixture to sum to one.

Atom maps are explicit strings. Carbon counts and target positions are checked
at [`metabolicmodel.py:412-454`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:412),
while mapping labels must be unique per side and every product label must occur
on the substrate side
([`metabolicmodel.py:523-556`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:523)).
Mapping labels are single characters; metabolite order supplies atom order and
labels supply correspondence. No transition is inferred from stoichiometry.
Target positions are one-based and colon-separated.

Symmetry is enabled only by `symmetry == "symmetry"`
([`metabolicmodel.py:117-121`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:117)).
Symmetric EMUs are canonicalized against reversed carbon positions
([`metabolicmodel.py:1207-1214`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:1207));
both orientations are traced and half-weighted in the generated matrices.

Reversible entries do not replace directional fluxes. They add a net state
variable constrained as `forward reaction(s) - reverse reaction(s) - net = 0`
([`metabolicmodel.py:937-969`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:937)).
Forward and reverse fields may contain `+`-separated reaction IDs, but missing
references are silently skipped. The initial FluxEMU runtime always constructs
an empty reversible dictionary and represents both directions only as separate,
nonnegative reactions. It never derives two gross fluxes from one signed net
value. The official direct-dictionary regression still exercises mfapy's
nonempty reversible-entry contract.

### Identifier restrictions and optional nlopt

mfapy strips underscores from metabolite IDs and prefixes IDs that do not start
with a letter ([`metabolicmodel.py:230-249`](/workspace/vendor/mfapy/mfapy/metabolicmodel.py:230)).
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
[`Example_0_toymodel.py:1-107`](/workspace/vendor/mfapy/sample/Example_0_toymodel.py:1),
with network definition
[`Example_0_toymodel_model.txt:13-45`](/workspace/vendor/mfapy/sample/Example_0_toymodel_model.txt:13)
and complete state
[`Example_0_toymodel_status.csv:1-20`](/workspace/vendor/mfapy/sample/Example_0_toymodel_status.csv:1).
Its current upstream exact forward test is
[`test_metabolicmodel.py:358-374`](/workspace/vendor/mfapy/tests/test_metabolicmodel.py:358).
The shipped script contains a filename-case mismatch and calls obsolete
`set_constrain`; the regression uses the actual filename and current
`set_constraint` method without changing its model, state, tracer, or expected
result.

## COBRApy

### SBML, notes, and annotations

The public SBML interfaces are `read_sbml_model`
([`sbml.py:396-455`](/workspace/vendor/cobrapy/src/cobra/io/sbml.py:396)) and
`write_sbml_model`
([`sbml.py:1136-1181`](/workspace/vendor/cobrapy/src/cobra/io/sbml.py:1136)).
Model notes/annotations are read at lines 624-647 and written at 1224-1235;
metabolite data at 665-705 and 1309-1328; reaction data at 812-826 and
1352-1364. The notes parser/serializer supports `<p>key: value</p>` dictionaries
([`sbml.py:1651-1699`](/workspace/vendor/cobrapy/src/cobra/io/sbml.py:1651)).
Because the serializer directly interpolates values into XML, FluxEMU stores
canonical JSON after `html.escape` and applies `html.unescape` before
`json.loads`. This preserves XML-sensitive characters exactly across a stable
COBRApy write/read round trip.

Annotations are parsed at `sbml.py:1750-1800` and written at
[`sbml.py:1832-1910`](/workspace/vendor/cobrapy/src/cobra/io/sbml.py:1832).
They are appropriate for identifiers.org/MIRIAM values, not nested isotope
metadata. Existing model, reaction, and metabolite notes and annotations remain
on their normal COBRApy objects; FluxEMU adds one namespaced note entry.

### FBA and status

`Model.optimize` is the public FBA entry point
([`model.py:1204-1237`](/workspace/vendor/cobrapy/src/cobra/core/model.py:1204)).
Solutions expose status, objective value, and reaction-indexed net fluxes
([`solution.py:22-84`](/workspace/vendor/cobrapy/src/cobra/core/solution.py:22));
solution assembly follows model reaction order at `solution.py:138-210`.
Status validation and specialized errors are in
[`solver.py:525-590`](/workspace/vendor/cobrapy/src/cobra/util/solver.py:525) and
[`exceptions.py:1-50`](/workspace/vendor/cobrapy/src/cobra/exceptions.py:1).
Linear objective coefficients come from
[`solver.py:71-105`](/workspace/vendor/cobrapy/src/cobra/util/solver.py:71).
FluxEMU requests a solution with `raise_error=False` so it can wrap every
non-optimal status consistently, then explicitly requires status `optimal`, a
finite objective and flux vector, and a nonempty linear objective map.

### FVA and a persistent sampling floor

`flux_variability_analysis` and its minimum/maximum DataFrame contract are at
[`variability.py:92-147`](/workspace/vendor/cobrapy/src/cobra/flux_analysis/variability.py:92).
It first optimizes and creates a fractional old-objective constraint at lines
232-259, then performs independent minimizations/maximizations at 279-317.
Those columns are bounds, not one jointly feasible vector.

FVA runs inside a model context, so its constraint does not persist after the
call. For sampling, FluxEMU applies the same numerical threshold with
`fix_objective_as_constraint`
([`solver.py:469-522`](/workspace/vendor/cobrapy/src/cobra/util/solver.py:469))
before constructing the sampler. Passing the already calculated bound avoids a
second optimum calculation. Objective direction is retained so maximization
uses a lower floor and minimization uses an upper ceiling.

### Complete feasible sampling and validation

Stable 0.31.1 supports ACHR and OptGP; FluxEMU always chooses one explicitly.
ACHR construction, warmup, seed, and sampling are in
[`achr.py:91-168`](/workspace/vendor/cobrapy/src/cobra/sampling/achr.py:91).
OptGP construction and sampling are in
[`optgp.py:105-198`](/workspace/vendor/cobrapy/src/cobra/sampling/optgp.py:105),
with chain seeds at lines 216-251. FluxEMU uses one OptGP process so the exact
requested batch size and deterministic seed are preserved.

`HRSampler` copies the already constrained model, normalizes the seed, and
materializes equality/inequality matrices
([`hr_sampler.py:171-278`](/workspace/vendor/cobrapy/src/cobra/sampling/hr_sampler.py:171)).
Thus the explicit objective constraint remains active during warmup and every
sample. The sampler returns all reactions in model order, i.e. complete jointly
feasible distributions rather than independent FVA draws.

The sampler's own `validate` checks steady-state equalities and reaction bounds
but not general inequalities such as the objective floor
([`hr_sampler.py:496-576`](/workspace/vendor/cobrapy/src/cobra/sampling/hr_sampler.py:496)).
FluxEMU additionally verifies exact unique columns, finite values, all bounds,
`S v = 0`, and the linear objective threshold per sample. The stoichiometric
matrix helper preserves model metabolite/reaction order
([`array.py:20-87`](/workspace/vendor/cobrapy/src/cobra/util/array.py:20)).
Reaction chemical formula balance (`Reaction.check_mass_balance`) is distinct
from steady-state sample balance
([`reaction.py:1403-1431`](/workspace/vendor/cobrapy/src/cobra/core/reaction.py:1403)).

### Identifier and order preservation

Models store metabolites and reactions in ordered `DictList` containers with
duplicate rejection
([`dictlist.py:25-77`](/workspace/vendor/cobrapy/src/cobra/core/dictlist.py:25)).
SBML reaction document order is retained while loading (`sbml.py:818-924,958`),
and FBA fluxes, FVA rows, ACHR columns, and OptGP columns derive from
`model.reactions`. FluxEMU captures that order immediately after loading and
requires exact equality during every conversion. Original COBRA IDs remain the
public/output IDs; only mfapy receives generated internal IDs.
