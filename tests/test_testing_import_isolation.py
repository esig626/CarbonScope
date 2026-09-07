"""Run the complete public simple-testing path without optional stacks."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "simple_binary_flux_discrimination.py"


def _without_optional_stacks(code):
    guard = """
        import importlib.abc
        import sys

        blocked = ('scipy', 'cobra', 'optlang', 'mfapy', 'nlopt', 'matplotlib', 'jax')

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
        check=True, env=os.environ.copy(), timeout=60,
    )


def test_public_law_bound_and_llr_execute_without_optional_stacks():
    _without_optional_stacks("""
        import math
        import fluxemu
        from fluxemu.observation import MIDCountObservation, MultinomialMIDLaw
        from fluxemu.testing import (
            BrunoOrderBound, SimpleBinaryLawPair,
            bruno_converse_at_order, log_likelihood_ratio,
        )

        pair = SimpleBinaryLawPair(
            null=MultinomialMIDLaw(4, (0.25, 0.75)),
            alternative=MultinomialMIDLaw(4, (0.5, 0.5)),
        )
        result = bruno_converse_at_order(pair, epsilon=0.05, order=1.7)
        assert isinstance(result, BrunoOrderBound)
        assert result.order == 1.7 and 0 < result.type_ii_lower_bound < 1
        assert result.reverse_renyi > result.forward_renyi > 0
        assert len(result.fingerprint) == 64
        assert math.isclose(log_likelihood_ratio(pair, MIDCountObservation((1, 3), 4)),
                            math.log(16 / 27), rel_tol=1e-14)
    """)


def test_complete_native_flux_example_executes_without_optional_stacks():
    _without_optional_stacks(f"""
        import runpy
        runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')
    """)
