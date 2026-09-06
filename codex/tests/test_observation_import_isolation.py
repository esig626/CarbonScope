"""Exercise the explicit count law in a fresh process without optional stacks."""

import os
import subprocess
import sys
import textwrap


def test_count_law_public_import_and_execution_need_no_optional_stack():
    code = """
        import importlib.abc
        import math
        import sys

        blocked = ('scipy', 'cobra', 'optlang', 'mfapy', 'matplotlib')

        class RejectOptionalStacks(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.', 1)[0] in blocked:
                    raise ModuleNotFoundError('optional stack blocked: ' + fullname)
                return None

        sys.meta_path.insert(0, RejectOptionalStacks())
        import fluxemu
        from fluxemu.observation import (
            MIDCountObservation, MultinomialMIDLaw,
            independent_product_kl, independent_product_renyi,
            kl_multinomial, multinomial_count_constant, renyi_multinomial,
        )

        p = MultinomialMIDLaw(10, (0.25, 0.75))
        q = MultinomialMIDLaw(10, (0.5, 0.5))
        observation = MIDCountObservation((3, 7), 10)
        assert math.isfinite(p.log_pmf(observation))
        assert math.isfinite(multinomial_count_constant(observation))
        assert kl_multinomial(p, q) > 0
        assert renyi_multinomial(p, q, 0.73) > 0
        assert independent_product_kl((p,), (q,)) == kl_multinomial(p, q)
        assert independent_product_renyi((p,), (q,), 1.3) == renyi_multinomial(p, q, 1.3)
        first = p.sample(seed=626)
        assert first == p.sample(seed=626)
        assert sum(first.counts) == 10
        assert not any(name.split('.', 1)[0] in blocked for name in sys.modules)
    """
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        env=os.environ.copy(),
        timeout=60,
    )
