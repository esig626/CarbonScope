"""Command-line entry point for FluxEMU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .exceptions import FluxEMUError
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fluxemu")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run the complete forward-EMU pipeline")
    run.add_argument("--model", required=True, type=Path, help="COBRA SBML model")
    run.add_argument(
        "--experiment", required=True, type=Path, help="FluxEMU experiment YAML"
    )
    run.add_argument("--output", required=True, type=Path, help="output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    namespace = parser.parse_args(arguments)
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
        f"FluxEMU completed: {len(result.sampling.samples)} samples, "
        f"{len(result.forward.mids)} MID rows -> {Path(namespace.output).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
