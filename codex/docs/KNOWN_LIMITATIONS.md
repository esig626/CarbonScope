# Known limitations

The native Stage 1 limitations are described first. Later mfapy and COBRApy
notes apply only to historical compatibility/parity paths and are not native
runtime requirements. See [Stage 1 native workflow](STAGE1_NATIVE_WORKFLOW.md)
for the supported public pipeline.

## Explicit directional isotope semantics

A direct isotope mapping without a `FluxProjectionRule` requires its physical
reaction to be directionally unambiguous from canonical bounds: nonnegative
bounds support `forward`, nonpositive bounds support `reverse`, and the declared
isotope direction must agree. Native canonical models may instead represent an
isotope-active directional component of signed or sign-spanning physical fluxes
with an explicit `positive_part` projection, explicit covered physical
directions, and (where supplied) a direction-activity certificate. FluxEMU does
not infer a projection or invent simultaneous forward and reverse gross fluxes.

## Supplied atom mappings

Every included reaction must provide ordered substrate/product participants
and explicit atom correspondence. FluxEMU does not infer carbon transitions
from stoichiometry, names, formulas, or external databases. The legacy
mfapy/SBML-note compatibility encoding uses single-character mapping labels and
unit isotope stoichiometry; those encoding restrictions are not reasons to
infer or repair a native mapping.

## Prototype SBML extension

The `FLUXEMU_*_V1` note keys and escaped canonical JSON are a private prototype
extension, not a standardized SBML package. Stable COBRApy 0.31.1 exact
round-trip behavior is tested, including XML-sensitive characters. Other SBML
tools may rewrite or discard XHTML notes. A future implementation should
evaluate a formal extension/package and schema version migration.

Every reaction and metabolite must explicitly declare inclusion or exclusion.
This deliberate strictness avoids silently treating missing metadata as an
exclusion but adds annotation work.

## Target and isotope correction scope

Native Stage 1 supports declared stationary targets and observation targets
over explicit one-based atom positions. Natural-abundance correction is not
implemented by the native stationary engine, so native experiments require
`correction: no`. More elaborate mfapy compound-fragment, MS/MS, correction,
and INST-MFA forms are compatibility capabilities and are not exposed by the
native Stage 1 path.

## Sampling and scalability

Native Stage 1 uses random-direction hit-and-run in an orthonormal basis of the
numerically reduced affine hull, starting from a HiGHS Chebyshev centre. It uses
FVA for collapsed-coordinate facial reduction, numerical-degeneracy and
unboundedness checks, and provenance digest binding; reaction-wise FVA intervals
are never independently sampled or combined. At fraction one—or any case where
the represented retained bound exactly equals the optimum—the chain runs on the
optimal face. Other smaller fractions retain the declared maximization or
minimization inequality. Zero-dimensional regions return the same unique vector
under distinct sample IDs.

The transition kernel has relative-volume uniform stationary target on the
numerically reduced retained affine polytope. A finite burn-in/thinned output
remains a correlated Markov chain: FluxEMU does not claim proven convergence,
adequate mixing, independence, representativeness, or biological probability.
Fixed-seed replay is scoped to the same algorithm and numerical environment.

Every returned state is independently checked for exact reaction order and
membership, finite values, bounds, `S v = 0`, and retained-objective feasibility.
Unbounded or numerically ambiguous geometry fails explicitly. States are not
clipped or repaired. A state that later makes stationary EMU singular or
undefined stops the batch with its sample ID; it is not dropped or resampled.

The sampler and native stationary EMU have acceptance coverage on the bundled
95-reaction model, but this is software evidence rather than proof of mixing or
biological validity at genome scale.

## Legacy mfapy compatibility constraints

mfapy 0.6.3 generates Python source containing reaction IDs as local variable
names and mutates metabolite IDs. FluxEMU isolates it behind deterministic hash
IDs and rejects collisions. The upstream editable-package metadata does not
correctly expose its nested `mfapy` package, so this workspace prototype falls
back to the repository's `vendor/mfapy` tree (or `FLUXEMU_MFAPY_SOURCE`) in the
audited compatibility workspace. The `mfapy` optional dependency group supplies
SciPy compatibility but does not distribute mfapy itself.

`nlopt` is absent and unnecessary for forward EMU. The local patch makes its
import optional and raises only when nlopt fitting is called. Fitting,
parameter estimation, confidence intervals, and INST-MFA are outside scope.
mfapy reports some construction failures by printing and returning a partial
object; FluxEMU prevalidates its input and checks for a usable `calmdv` function.

## Legacy toy validation and scientific scope

The toy network derives its isotope topology and exact trusted MID regression
from mfapy's official Example 0, but its COBRA boundary reactions, objective,
and sampled feasible region were created specifically to test this adapter.
Passing the regression and CLI tests establishes software behavior only. It
does not validate a biological model, tracer experiment, correction formula,
or scientific interpretation.

## Legacy models

Files under `references/` are unverified secondary material. They are
not runtime inputs, are not certified, and were inspected only after the
official prototype passed. Known structural doubts and required migration work
are separated in `LEGACY_MODEL_REVIEW.md`; no legacy model is silently repaired
or adopted.
