"""Regenerate the checked-in annotated toy SBML model."""

from pathlib import Path

from fluxemu.toy import write_toy_sbml


if __name__ == "__main__":
    write_toy_sbml(Path(__file__).with_name("toy_model.xml"))
