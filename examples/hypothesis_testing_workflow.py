"""Run the native SBML-to-testing acceptance example through the public API.

From a source checkout with the testing extra installed::

    python examples/hypothesis_testing_workflow.py --output results/hypotheses

This deliberately small synthetic example demonstrates software semantics;
its two sampled states per role do not exhaust either continuous flux region.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from fluxemu import run_hypothesis_testing_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    specification = (
        Path(__file__).resolve().parents[1]
        / "tests" / "fixtures" / "hypothesis_workflow" / "workflow.yaml"
    )
    if arguments.output is not None:
        result = run_hypothesis_testing_workflow(
            specification, output_directory=arguments.output,
        )
        print(result.summary)
        return
    with TemporaryDirectory(prefix="fluxemu-hypotheses-") as temporary:
        result = run_hypothesis_testing_workflow(
            specification, output_directory=Path(temporary),
        )
        print(result.summary)


if __name__ == "__main__":
    main()
