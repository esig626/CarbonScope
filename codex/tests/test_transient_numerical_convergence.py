"""GitHub-only diagnostic for residual native/mfapy transient differences.

This module deliberately does not change the production solver or the existing
historical parity gates.  It reuses mfapy's generated ODE system, replacing only
its numerical integrator so that GitHub CI can determine whether the residual
native/frozen difference is integration error or a structural equation mismatch.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PARITY_PATH = Path(__file__).with_name("test_native_transient_parity.py")


def _load_parity_module():
    name = "_fluxemu_transient_parity_helpers"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PARITY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PARITY = _load_parity_module()


SOLVERS = (
    ("odeint-historical", "odeint", 1.0e-3, 1.0e-3),
    ("odeint-tight", "odeint", 1.0e-6, 1.0e-9),
    ("odeint-high", "odeint", 1.0e-9, 1.0e-12),
    ("LSODA-high", "LSODA", 1.0e-9, 1.0e-12),
    ("RK45-high", "RK45", 1.0e-9, 1.0e-12),
    ("DOP853-high", "DOP853", 1.0e-11, 1.0e-13),
)


@contextmanager
def _mfapy_integrator(mode: str, rtol: float, atol: float):
    scipy = pytest.importorskip("scipy", reason="transient convergence diagnostic requires SciPy")
    original = scipy.integrate.odeint

    def replacement(func, y0, timepoints, args=(), **kwargs):
        times = np.asarray(timepoints, dtype=float)
        initial = np.asarray(y0, dtype=float)
        assert times.ndim == 1 and times.size >= 2
        assert np.all(np.diff(times) > 0.0)
        kwargs = dict(kwargs)
        kwargs.pop("rtol", None)
        kwargs.pop("atol", None)
        if mode == "odeint":
            result = original(
                func,
                initial,
                times,
                args=args,
                rtol=rtol,
                atol=atol,
                **kwargs,
            )
            result = np.asarray(result, dtype=float)
        else:
            def rhs(time, state):
                return np.asarray(func(state, time, *args), dtype=float)

            solved = scipy.integrate.solve_ivp(
                rhs,
                (float(times[0]), float(times[-1])),
                initial,
                t_eval=times,
                method=mode,
                rtol=rtol,
                atol=atol,
            )
            assert solved.success, solved.message
            assert np.array_equal(solved.t, times)
            result = np.asarray(solved.y.T, dtype=float)
        assert result.shape == (times.size, initial.size)
        assert np.isfinite(result).all()
        return result

    scipy.integrate.odeint = replacement
    try:
        yield
    finally:
        scipy.integrate.odeint = original


def _run_historical_builder(builder, directory: str, times, mode: str, rtol: float, atol: float):
    with _mfapy_integrator(mode, rtol, atol):
        if directory == "antoniewicz_tca":
            output = builder.run_timecourse()
            frame = output
        else:
            output = builder.run_timecourse(timepoints=times)
            frame, returned_times, _ = output
            assert tuple(returned_times) == tuple(times)
    mids = PARITY._frame_mids(frame)
    assert mids
    assert all(np.isfinite(value).all() for value in mids.values())
    return mids


def _worst(first, second, target_ids):
    assert first.keys() == second.keys()
    allowed = set(target_ids)
    best = None
    for (time, target), values in first.items():
        if target not in allowed:
            continue
        reference = second[(time, target)]
        delta = np.abs(np.asarray(values) - np.asarray(reference))
        index = int(np.argmax(delta))
        candidate = {
            "max_abs_difference": float(delta[index]),
            "time": float(time),
            "target": target,
            "isotopologue": index,
            "first_value": float(values[index]),
            "second_value": float(reference[index]),
        }
        if best is None or candidate["max_abs_difference"] > best["max_abs_difference"]:
            best = candidate
    assert best is not None
    return best


def _run_case(directory: str, yaml_name: str, reactions, transitions, fluxes, pool_size: float, benchmark: str):
    pytest.importorskip("scipy", reason="transient convergence diagnostic requires SciPy")
    base = ROOT / "examples" / directory
    frozen = PARITY._frozen(base / "timecourse_mids.csv")
    times = tuple(dict.fromkeys(time for time, _ in frozen))
    model, experiment = PARITY._canonical(
        directory,
        yaml_name,
        reactions,
        transitions,
        times,
        pool_size,
    )
    native_result = PARITY.evaluate_transient(
        PARITY.compile_transient_emu_plan(model, experiment),
        fluxes,
        rtol=PARITY.RTOL,
        atol=PARITY.ATOL,
    )
    native = PARITY._predictions(native_result)
    assert native.keys() == frozen.keys()
    shared_targets, terminal_targets = PARITY._target_semantics(model, experiment)
    builder = PARITY._load_historical_builder(directory)

    rows = []
    references = {}
    for label, mode, rtol, atol in SOLVERS:
        reference = _run_historical_builder(builder, directory, times, mode, rtol, atol)
        assert reference.keys() == native.keys()
        references[label] = reference
        worst = _worst(native, reference, shared_targets)
        rows.append({
            "label": label,
            "solver": mode,
            "rtol": rtol,
            "atol": atol,
            **worst,
        })

    frozen_worst = _worst(native, frozen, shared_targets)
    agreements = {
        "odeint_high_vs_lsoda": _worst(
            references["odeint-high"], references["LSODA-high"], shared_targets
        )["max_abs_difference"],
        "odeint_high_vs_dop853": _worst(
            references["odeint-high"], references["DOP853-high"], shared_targets
        )["max_abs_difference"],
        "lsoda_vs_dop853": _worst(
            references["LSODA-high"], references["DOP853-high"], shared_targets
        )["max_abs_difference"],
    }
    coarse = next(row for row in rows if row["label"] == "odeint-historical")
    tight = next(row for row in rows if row["label"] == "odeint-tight")
    high = next(row for row in rows if row["label"] == "odeint-high")
    payload = {
        "benchmark": benchmark,
        "shared_semantics_target_ids": shared_targets,
        "terminal_target_ids": terminal_targets,
        "native_solver": "RK45",
        "native_rtol": PARITY.RTOL,
        "native_atol": PARITY.ATOL,
        "native_vs_frozen_historical_worst": frozen_worst,
        "solver_runs": rows,
        "high_accuracy_solver_agreement": agreements,
        "coarse_to_tight_sequence": [
            coarse["max_abs_difference"],
            tight["max_abs_difference"],
            high["max_abs_difference"],
        ],
        "high_accuracy_reduces_difference": (
            high["max_abs_difference"] < coarse["max_abs_difference"]
        ),
    }
    print("FLUXEMU_TRANSIENT_CONVERGENCE " + json.dumps(payload, sort_keys=True))

    # Diagnostic-only assertions: execution, shape/key integrity and finite output.
    # The existing parity tests remain the scientific pass/fail gates.
    assert all(np.isfinite(row["max_abs_difference"]) for row in rows)
    assert all(np.isfinite(value) for value in agreements.values())


def test_antoniewicz_mfapy_equation_solver_convergence():
    _run_case(
        "antoniewicz_tca",
        "experiment_timecourse.yaml",
        PARITY.TCA_REACTIONS,
        PARITY.TCA_TRANSITIONS,
        PARITY.TCA_FLUXES,
        1.0,
        "antoniewicz-table-5-transient",
    )


def test_glucose_tca_mfapy_equation_solver_convergence():
    _run_case(
        "antoniewicz_tca_glucose",
        "experiment_u13c6_glucose_timecourse.yaml",
        PARITY.UPSTREAM_REACTIONS + PARITY.TCA_REACTIONS,
        PARITY.UPSTREAM_TRANSITIONS + PARITY.TCA_TRANSITIONS,
        {**PARITY.UPSTREAM_FLUXES, **PARITY.TCA_FLUXES},
        100.0,
        "glucose-to-tca-transient",
    )
