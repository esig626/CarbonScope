"""Exercise installed public MFA boundaries behind fresh-process import guards."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "stationary_mfa_recovery.py"


def _isolated(code: str, *, block_scipy: bool = True) -> None:
    blocked = ("cobra", "optlang", "mfapy") + (("scipy",) if block_scipy else ())
    guard = f"""
import importlib.abc
import sys

blocked = {blocked!r}

class RejectOptionalStacks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in blocked:
            raise ModuleNotFoundError('optional stack blocked: ' + fullname)
        return None

sys.meta_path.insert(0, RejectOptionalStacks())
"""
    check = "\nassert not any(name.split('.', 1)[0] in blocked for name in sys.modules)\n"
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(guard) + textwrap.dedent(code) + check],
        check=True,
        env=os.environ.copy(),
        timeout=60,
    )


def test_public_imports_and_pure_divergence_need_no_optional_stack() -> None:
    _isolated("""
        import fluxemu
        import fluxemu.mfa as mfa
        from fluxemu.mfa import schema, stationary

        assert fluxemu.fit_stationary_mfa is mfa.fit_stationary_mfa
        assert fluxemu.evaluate_stationary_mfa is mfa.evaluate_stationary_mfa
        assert mfa.fit_stationary_mfa is stationary.fit_stationary_mfa
        assert mfa.evaluate_stationary_mfa is stationary.evaluate_stationary_mfa
        assert mfa.StationaryMFAProblem is schema.StationaryMFAProblem
        assert mfa.kl_divergence((0.3, 0.7), (0.3, 0.7)) == 0.0
        assert mfa.renyi_divergence((0.3, 0.7), (0.3, 0.7), 0.73) == 0.0
        assert mfa.DivergenceObjectiveConfig(alpha=1.37).alpha == 1.37
        assert callable(fluxemu.run_native_stationary_ensemble)
    """)


def test_native_objective_and_schema_work_without_scipy() -> None:
    _isolated(f"""
        import runpy
        from fluxemu import evaluate_stationary_mfa
        from fluxemu.mfa import validate_stationary_mfa_problem

        example = runpy.run_path({str(EXAMPLE)!r})
        problem, truth = example['build_mixture_problem']()
        validate_stationary_mfa_problem(problem)
        evaluated = evaluate_stationary_mfa(problem, truth)
        assert evaluated.total_loss == 0.0
        assert evaluated.state is truth
        assert evaluated.components[0].predicted == problem.experiments[0].observations[0].fractions
    """)


def test_missing_scipy_fails_only_at_fit_with_extra_install_hint() -> None:
    _isolated(f"""
        import runpy
        from fluxemu import fit_stationary_mfa
        from fluxemu.exceptions import AnalysisError

        example = runpy.run_path({str(EXAMPLE)!r})
        problem, truth = example['build_mixture_problem']()
        try:
            fit_stationary_mfa(problem, initial_states=(truth,))
        except AnalysisError as error:
            assert 'requires SciPy' in str(error), str(error)
            assert 'fluxemu[mfa]' in str(error), str(error)
        else:
            raise AssertionError('fitting succeeded with its backend blocked')
    """)


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="requires the mfa extra")
def test_native_fit_with_scipy_never_imports_compatibility_backends() -> None:
    _isolated(f"""
        import runpy
        from fluxemu import fit_stationary_mfa
        from fluxemu.execution import CanonicalFluxState

        example = runpy.run_path({str(EXAMPLE)!r})
        problem, truth = example['build_mixture_problem']()
        start = CanonicalFluxState('nontruth', (('Z_IN', 2.0), ('A_IN', 8.0), ('M_OUT', 10.0)))
        fitted = fit_stationary_mfa(problem, initial_states=(start,))
        assert fitted.total_loss < 1e-10
        assert fitted.validation.valid
        assert fitted.start_diagnostics[0].accepted
        assert 'scipy.optimize' in sys.modules
    """, block_scipy=False)
