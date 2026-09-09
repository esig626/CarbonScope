"""Command-line entry point for FluxEMU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .exceptions import FluxEMUError, InputValidationError
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fluxemu")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run the complete forward-EMU pipeline")
    run.add_argument("--model", required=True, type=Path, help="SBML Level 3 FBC model")
    run.add_argument(
        "--experiment", required=True, type=Path, help="FluxEMU experiment YAML"
    )
    run.add_argument("--output", required=True, type=Path, help="output directory")
    hypotheses = subcommands.add_parser(
        "test-hypotheses",
        help="generate finite flux hypotheses and evaluate declared count or Dirichlet tests",
    )
    hypotheses.add_argument(
        "--specification", required=True, type=Path,
        help="CarbonScope hypothesis workflow YAML",
    )
    hypotheses.add_argument(
        "--model", type=Path, help="override the common SBML Level 3 FBC model path",
    )
    hypotheses.add_argument(
        "--output", type=Path, help="override the report output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    namespace = parser.parse_args(arguments)
    if namespace.command == "test-hypotheses":
        from fluxemu import load_hypothesis_testing_spec, run_hypothesis_testing_workflow

        try:
            specification = load_hypothesis_testing_spec(
                namespace.specification, model_path=namespace.model,
            )
            destination = (namespace.output if namespace.output is not None
                           else specification.output_directory)
            if destination is None:
                raise InputValidationError(
                    "test-hypotheses requires --output or specification output.directory"
                )
            result = run_hypothesis_testing_workflow(
                specification, output_directory=destination,
            )
        except (FluxEMUError, OSError) as error:
            print(f"fluxemu: {error}", file=sys.stderr)
            return 2
        print(result.summary)
        return 0
    if namespace.command != "run":  # pragma: no cover - argparse guarantees this
        parser.error(f"unknown command {namespace.command!r}")
    try:
        result = run_pipeline(
            namespace.model,
            namespace.experiment,
            namespace.output,
            cli_arguments=arguments,
        )
    except FluxEMUError as error:
        print(f"fluxemu: {error}", file=sys.stderr)
        return 2
    print(
        f"FluxEMU native stationary analysis completed: "
        f"{len(result.analysis.mids.forward.values)} MID rows -> "
        f"{Path(namespace.output).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
