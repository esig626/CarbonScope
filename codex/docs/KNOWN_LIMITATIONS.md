# Known limitations

## Explicit directional reactions

The prototype accepts isotope reactions only when the COBRA lower bound is
nonnegative. Forward and reverse gross fluxes must be represented by separate
reactions, each with its own directionally correct atom map. A signed net flux
is never split into simultaneous forward and reverse values. The runtime mfapy
reversible-reaction dictionary is intentionally empty for the toy; the trusted
official regression separately covers mfapy's nonempty reversible dictionary.

## Supplied atom mappings

Every included reaction must provide ordered substrate/product participants
and explicit atom-label correspondence in SBML notes. FluxEMU does not infer
carbon transitions from stoichiometry, names, formulas, or external databases.
mfapy 0.6.3 treats mapping labels as single characters, so this prototype
rejects multi-character labels. It also limits included isotope reactions to
unit stoichiometric coefficients; molecule-level expansion for coefficients
greater than one is future work.

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

The prototype supports a single-metabolite `intermediate` or `gcms` target with
unique one-based atom positions. mfapy can express more complicated compound
fragments, MS/MS, and INST-MFA, but those forms are not exposed here. Tracer
natural-isotope correction is passed through to mfapy. Target-level natural
isotope addition is currently required to be `no`, because enabling it after
mfapy construction would regenerate the forward function and violate the
one-generation-per-run invariant.

## Sampling and scalability

ACHR and one-process OptGP are supported. One-process OptGP preserves exact
sample count and deterministic chain seeding but does not exploit parallel
sampling. The package creates a dense stoichiometric matrix for independent
sample validation and mfapy generates/executes dense layer-solvers; neither
choice has been profiled on a genome-scale isotope network. The complete
prototype must be validated on progressively larger trusted networks before
biological-scale use.

COBRApy's sampler validator does not check general inequality constraints, so
FluxEMU evaluates the objective floor independently. Numerical acceptance is
controlled by the experiment tolerances; this is validation, not proof of
mixing quality or representative sampling.

## mfapy constraints

mfapy 0.6.3 generates Python source containing reaction IDs as local variable
names and mutates metabolite IDs. FluxEMU isolates it behind deterministic hash
IDs and rejects collisions. The upstream editable-package metadata does not
correctly expose its nested `mfapy` package, so this workspace prototype falls
back to `/workspace/vendor/mfapy` (or `FLUXEMU_MFAPY_SOURCE`).

`nlopt` is absent and unnecessary for forward EMU. The local patch makes its
import optional and raises only when nlopt fitting is called. Fitting,
parameter estimation, confidence intervals, and INST-MFA are outside scope.
mfapy reports some construction failures by printing and returning a partial
object; FluxEMU prevalidates its input and checks for a usable `calmdv` function.

## Toy validation and scientific scope

The toy network derives its isotope topology and exact trusted MID regression
from mfapy's official Example 0, but its COBRA boundary reactions, objective,
and sampled feasible region were created specifically to test this adapter.
Passing the regression and CLI tests establishes software behavior only. It
does not validate a biological model, tracer experiment, correction formula,
or scientific interpretation.

## Legacy models

Files under `/workspace/references` are unverified secondary material. They are
not runtime inputs, are not certified, and were inspected only after the
official prototype passed. Known structural doubts and required migration work
are separated in `LEGACY_MODEL_REVIEW.md`; no legacy model is silently repaired
or adopted.
