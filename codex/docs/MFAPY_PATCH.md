# Minimal mfapy forward-only dependency patch

## Outcome

The vendored mfapy forward EMU implementation works without `nlopt`. The only
local mfapy change makes `nlopt` optional at import time and raises a clear
dependency error if an nlopt-backed fitting routine is actually requested.
No forward calculation, EMU construction, optimizer algorithm, or result
format was changed.

The source identifies itself as mfapy 0.6.3 in
`/workspace/vendor/mfapy/setup.py` lines 22-32. The runtime environment at
`/workspace/.venv` does not contain `nlopt`.

## Source evidence

Before this patch, `/workspace/vendor/mfapy/mfapy/optimize.py` imported
`nlopt` unconditionally. Both `/workspace/vendor/mfapy/mfapy/__init__.py`
lines 4-8 and `/workspace/vendor/mfapy/mfapy/metabolicmodel.py` line 32 import
that module, so `import mfapy` failed before a caller could reach forward EMU.

Forward EMU does not use nlopt:

- `MetabolicModel.reconstruct` generates and stores `model.func["calmdv"]`
  and `model.func["diffmdv"]` in `metabolicmodel.py` lines 1099-1133.
- `optimize.calc_MDV_from_flux` only dispatches to those stored functions in
  `optimize.py` lines 288-330.
- Active `nlopt.opt` construction occurs inside `fit_r_mdv_nlopt`, around
  current lines 661-690. The earlier nlopt example in
  `initializing_Rm_fitting` is inside a disabled triple-quoted block.

Thus nlopt is a fitting dependency, not a forward-EMU dependency.

## Changed file and lines

Only `/workspace/vendor/mfapy/mfapy/optimize.py` is changed.

At current lines 23-27, the unconditional import is replaced by:

```python
try:
    import nlopt as nlopt
except ModuleNotFoundError:
    nlopt = None
```

At current lines 556-561, `fit_r_mdv_nlopt` checks the optional dependency
before entering its broad optimizer exception handler:

```python
if nlopt is None:
    raise ImportError(
        "mfapy nlopt fitting requires the optional 'nlopt' package; "
        "forward EMU calculation does not require it"
    )
```

Placing the guard before the fitting function's broad `try` is intentional:
the missing dependency is raised to the caller rather than returned as an
optimizer state value.

## Scope and compatibility

- When nlopt is installed, it is imported normally and the existing fitting
  implementation is unchanged.
- When nlopt is absent, importing mfapy, constructing a model, generating a
  carbon source, and calculating forward MDVs continue to work.
- Calling `fit_r_mdv_nlopt` without nlopt raises an actionable `ImportError`.
- Other nlopt import failures are not hidden; only `ModuleNotFoundError` is
  treated as the optional package being absent.
- mfapy's dependency declaration is unchanged. This patch concerns runtime
  import behavior for the vendored forward-only integration.

## Verification

The deterministic fixture is
`/workspace/codex/tests/fixtures/official_mfapy_example_0.json`. It records the
official Example 0 parser dictionaries, tracer distribution, reaction order,
complete flux vector, requested target, and full-precision expected MID.

The focused FluxEMU regression command was:

```bash
/workspace/.venv/bin/pytest -q \
  /workspace/codex/tests/test_mfapy_forward_regression.py
```

Result:

```text
5 passed in 1.48s
```

Those tests prove:

1. the official text-parser route reproduces the expected forward MID;
2. all four parser dictionaries equal the checked-in direct fixture;
3. direct `MetabolicModel` construction reproduces the same result while
   `mfapy.mfapyio.load_metabolic_model` is monkeypatched to raise;
4. a clean subprocess imports mfapy while nlopt is unavailable; and
5. invoking nlopt-backed fitting reports the optional dependency clearly.

The exact expected `Glue` MID remains:

```text
[0.3463541666666666,
 0.26953124999999994,
 0.27083333333333326,
 0.08072916666666666,
 0.028645833333333332,
 0.003906249999999999]
```

A focused upstream construction/forward check also passed:

```bash
cd /workspace/vendor/mfapy
/workspace/.venv/bin/pytest -q tests/test_metabolicmodel.py \
  -k 'load_metabolic_model_construction or calmdv'
```

```text
2 passed, 16 deselected
```
