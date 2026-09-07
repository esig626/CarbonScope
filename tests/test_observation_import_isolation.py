"""Exercise the explicit count law in a fresh process without optional stacks."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "stationary_mfa_recovery.py"


def _run_without_optional_stacks(code):
    guard = """
        import importlib.abc
        import sys

        blocked = ('scipy', 'cobra', 'optlang', 'mfapy', 'matplotlib')

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


def test_count_law_public_import_and_execution_need_no_optional_stack():
    _run_without_optional_stacks("""
        import math
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
    """)


def test_native_stationary_law_and_count_generation_need_no_optional_stack():
    _run_without_optional_stacks(f"""
        import runpy
        import fluxemu
        from fluxemu.observation import (
            StationaryCountSpecification,
            StationaryObservationExperiment,
            StationaryObservationSpecification,
            evaluate_stationary_observation_laws,
            sample_stationary_observations,
        )

        fixture = runpy.run_path({str(EXAMPLE)!r})
        problem, truth = fixture['build_mixture_problem']()
        specification = StationaryObservationSpecification(
            problem.model,
            (StationaryObservationExperiment(
                'base-count-experiment', problem.experiments[0].experiment,
                (StationaryCountSpecification('O-mid', 1000, 'base-replicate'),),
            ),),
        )
        result = evaluate_stationary_observation_laws(specification, (truth,))
        samples = sample_stationary_observations(result, seed=626)
        assert result.validation.valid
        assert result.components[0].predicted_mid == problem.experiments[0].observations[0].fractions
        assert samples[0].component is result.components[0]
        assert samples[0].component.replicate_id == 'base-replicate'
        assert sum(samples[0].observation.counts) == 1000
        assert samples == sample_stationary_observations(result, seed=626)
    """)
