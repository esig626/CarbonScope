# Vendored mfapy measurement-semantics audit

Audited against current-main baseline
`5969ec008dffb37e73a04647656823d02e20e548`. Vendored source is read-only.
This audit covers the observed-MDV paths in `vendor/mfapy/mfapy/mdv.py`,
`optimize.py`, and `metabolicmodel.py`; it does not infer a measurement law
from a fitting criterion.

## Source evidence

All source locations below are relative to `vendor/mfapy/mfapy/`.

| Mechanism | Exact source evidence | Interpretation |
| --- | --- | --- |
| Stored measurements | `mdv.py:78-86`, `109-135` | Each fragment/mass class has `id`, floating `ratio`, `std`, `use`, and `data`; the API calls `ratio` relative abundance. Fragment collections are sorted. |
| File input | `mdv.py:673-689` | Input columns are fragment, mass-class index, use flag, ratio, and SD. Ratio and SD are parsed as floats. The integer mass-class index is not an ion count. |
| SD assignment | `mdv.py:350-375` | Absolute SD is the supplied value; relative SD is ratio times supplied value. Neither defines an observed count total. |
| Gaussian simulation | `mdv.py:433-472` | For each replicate/class, relative noise is `(Z * stdev + 1) * ratio`; absolute noise is `Z * stdev + ratio`, with `Z` drawn by global `numpy.random.randn()`. Negative values are set to zero. By default, each replicate is divided by its fragment total; disabling `normalize` skips that closure. |
| Replicate storage and averaging | `mdv.py:457-476` | `iteration` becomes the replicate count; each class's `ratio` becomes its replicate mean and `data` retains its replicate values. The line computing empirical SD is commented out, so adding noise does not update the stored SD. |
| Exported replicate payload | `mdv.py:503-522` | Ratios, SDs, use flags, and IDs are exported in sorted fragment/class order. Replicate rows are stacked into `data` only when there are at least three replicates; otherwise the initialized zero array is returned. |
| Experiment registration | `metabolicmodel.py:3118-3130` | `set_experiment` receives `rawdata` but stores only ratios, SDs, use flags, IDs, targets, source labeling, and measurement-number metadata in the stationary experiment record. The replicate payload is not stored there. |
| SD-weighted fit | `optimize.py:436-450`, `846-868`; `metabolicmodel.py:4412-4428` | Measured values and SDs are concatenated across sorted experiments and filtered by use flags. The inverse covariance is diagonal, `1 / std**2`; the residual criterion is `res.T @ covinv @ res`, with a separate boundary penalty in the optimizer. It is weighted RSS, not a multinomial log-PMF. |
| Other density calls | `metabolicmodel.py:4776`, `4828-4835`, `4864-4873` | There are chi-square PDF/survival-function calls evaluated on RSS in separate flux-sampling/statistical routines. These are not an observed-MDV PMF/PDF and must not be transferred into this task. |

The audited observed-MDV storage, simulation, registration, and fitting paths
contain no literal isotopologue-count vector with a fixed observed total and
no multinomial PMF. Searches for count/ion/multinomial/Poisson/likelihood and
PMF/log-PDF identifiers in the three files, followed by inspection of the
actual density calls, found no implemented generative PMF/PDF evaluator for
the observed MDV vector. The Gaussian perturbation procedure is a simulator
heuristic; the source does not supply the probability law of the resulting
clipped, optionally closed vector. Weighted RSS alone does not establish that
law. This conclusion is scoped to the inspected vendored paths.

## Transfer decision

| mfapy mechanism | reuse | adapt | reject | reason |
| --- | --- | --- | --- | --- |
| Experiment, fragment, mass-class and replicate identities | Workflow concept | Explicit immutable IDs and declared order | Vendor's automatic sorting | Auditability is useful; FluxEMU must preserve scientific declaration order. |
| Separate replicate values | Workflow concept | Retain every raw count vector and total | Replicate averaging as the sole observation | Averaging loses individual count totals and identities. |
| Ratio/SD measurement interface | — | Document its different semantics | Treating ratios/SDs as counts or inferring effective `n` | Relative abundance and SD do not establish literal count observations. |
| Gaussian noise, clipping and optional closure | — | — | As the new rigorous observation law | The implemented perturbation heuristic supplies neither count semantics nor a resulting MDV-density evaluator. |
| Diagonal SD-weighted RSS | — | — | In this observation layer | Existing FluxEMU MID-divergence MFA remains unchanged; no additional optimizer or noise model is needed. |
| mfapy runtime and statistical routines | — | — | Runtime dependency or inference features | Reuse native stationary EMU; vendor remains a read-only reference and inference is outside scope. |

## Existing FluxEMU preprocessing boundary

`codex/docs/MID_PREPROCESSING.md:18-28` states that `normalise_mid(...)`
explicitly closes a finite non-negative vector, preserves order and exact
zero support, inserts no pseudocounts, and deliberately discards its absolute
total. Lines 30-35 keep this preprocessing separate from divergence and MFA
objectives, which never silently renormalise their inputs.

A probability composition therefore cannot identify a count total. Only
genuine integer count observations, with raw counts and their positive total
retained separately, qualify for the fixed-total multinomial observation
law. Percentages, normalized MIDs, peak areas, arbitrary intensities, and
mfapy-style noisy ratios must never be rounded/rescaled into pseudo-counts,
and SDs must never be used to invent an effective sample size. Native EMU
probabilities supply the law's `p`; an explicit count specification supplies
its `n`. Neither this audit nor the new law changes the existing MFA
objective.
